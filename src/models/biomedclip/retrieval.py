import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import os
import argparse
import logging
import torch
import pandas as pd
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer
from src.datasets.rocov2 import ROCOv2DataModule
from src.utils.retrieval_metrics import compute_retrieval_metrics, log_retrieval_metrics
from src.utils.tools import setup_logging
from src.adapters import inject_mona_variant_to_open_clip


def get_args():
    parser = argparse.ArgumentParser("BiomedCLIP Text-Image Retrieval on ROCOv2")

    # Model related
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to finetuned model checkpoint (if None, use pretrained BiomedCLIP)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        help="Model name for open_clip",
    )

    # MONA adapter related
    parser.add_argument(
        "--mona_weights",
        type=str,
        default=None,
        help="Path to pretrained MONA weights (if provided, inject and load MONA adapters)",
    )
    parser.add_argument("--mona_bottleneck", type=int, default=64, help="MONA bottleneck dimension")
    parser.add_argument("--mona_layers", type=int, default=None, help="Number of layers to inject MONA (None=all)")
    parser.add_argument(
        "--mona_variant",
        type=str,
        default="freq_enhanced",
        choices=["baseline", "fractional", "noise_aware", "freq_enhanced", "hybrid"],
        help="MONA variant type",
    )

    # LoRA adapter related
    parser.add_argument("--lora_weights", type=str, default=None, help="Path to pretrained LoRA weights")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    # Data related
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data/rocov2_cache",
        help="Directory to cache ROCOv2 dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for debugging, None=all)",
    )

    # Retrieval settings
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10],
        help="K values for Recall@K metrics",
    )

    # Experiment settings
    parser.add_argument("--exp", type=str, default="biomedclip_retrieval", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # Output settings
    parser.add_argument(
        "--save_features",
        action="store_true",
        help="Save extracted features to disk",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: runs/{exp})",
    )

    return parser.parse_args()


def load_model(args):
    """Load BiomedCLIP model (pretrained or finetuned with optional MONA/LoRA adapters)"""
    model, _ = create_model_from_pretrained(args.model_name, cache_dir="ckpt")
    tokenizer = get_tokenizer(args.model_name)

    if args.ckpt is not None and os.path.exists(args.ckpt):
        logging.info(f"Loading finetuned weights from: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        logging.info("✓ Finetuned weights loaded")

    elif args.lora_weights is not None and os.path.exists(args.lora_weights):
        from src.adapters import inject_lora_to_biomedclip

        logging.info(f"Injecting LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})")
        inject_lora_to_biomedclip(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0)

        checkpoint = torch.load(args.lora_weights, map_location="cpu", weights_only=True)
        lora_state_dict = checkpoint.get("lora_state_dict", checkpoint)

        model_dict = model.state_dict()
        lora_params_loaded = 0
        for name, param in lora_state_dict.items():
            if name in model_dict:
                model_dict[name] = param
                lora_params_loaded += 1
        assert lora_params_loaded > 0, "No LoRA parameters loaded"
        model.load_state_dict(model_dict)
        logging.info(f"✓ Loaded {lora_params_loaded} LoRA parameters from {args.lora_weights}")

    elif args.mona_weights is not None and os.path.exists(args.mona_weights):
        logging.info(f"Injecting MONA adapters (variant: {args.mona_variant})")
        inject_mona_variant_to_open_clip(
            model, variant=args.mona_variant, bottleneck_dim=args.mona_bottleneck, num_layers=args.mona_layers
        )

        checkpoint = torch.load(args.mona_weights, map_location="cpu", weights_only=True)
        mona_state_dict = checkpoint.get("mona_state_dict", checkpoint)

        model_dict = model.state_dict()
        mona_params_loaded = 0
        for name, param in mona_state_dict.items():
            if name in model_dict:
                model_dict[name] = param
                mona_params_loaded += 1
        assert mona_params_loaded > 0, "No MONA parameters loaded"
        model.load_state_dict(model_dict)
        logging.info(f"✓ Loaded {mona_params_loaded} MONA parameters from {args.mona_weights}")

    model.float()
    model.to(args.device)
    model.eval()

    return model, tokenizer


@torch.no_grad()
def extract_features(model, tokenizer, dataloader, args):
    """
    Extract image and text features from the dataset

    Returns:
        image_features: [N, D] tensor
        text_features: [N, D] tensor
        captions: List of N captions
    """
    all_image_features = []
    all_text_features = []
    all_captions = []

    for batch in tqdm(dataloader, desc="Extracting features", disable=True):
        images, captions, _ = batch
        images = images.to(args.device)

        # Tokenize captions
        texts = tokenizer(captions, context_length=256).to(args.device)

        # Extract features
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        # Store features
        all_image_features.append(image_features.cpu())
        all_text_features.append(text_features.cpu())
        all_captions.extend(captions)

    # Concatenate all features
    image_features = torch.cat(all_image_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)

    return image_features, text_features, all_captions


def save_results(args, metrics, image_features=None, text_features=None, captions=None):
    """Save retrieval results to disk"""
    import datetime
    import shutil

    # Create a sub-folder for single test with time and rSum
    backup_folder = os.path.join(
        args.output_dir,
        f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_rsum={metrics['rsum']:.2f}",
    )
    os.makedirs(backup_folder, exist_ok=True)

    # Create results dataframe
    metric_names = []
    metric_values = []

    # Image-to-Text metrics
    for k in args.k_values:
        metric_names.append(f"I2T_R@{k}")
        metric_values.append(metrics[f"i2t_r@{k}"])
    metric_names.extend(["I2T_MedR", "I2T_MeanR"])
    metric_values.extend([metrics["i2t_medr"], metrics["i2t_meanr"]])

    # Text-to-Image metrics
    for k in args.k_values:
        metric_names.append(f"T2I_R@{k}")
        metric_values.append(metrics[f"t2i_r@{k}"])
    metric_names.extend(["T2I_MedR", "T2I_MeanR"])
    metric_values.extend([metrics["t2i_medr"], metrics["t2i_meanr"]])

    # Summary metric
    metric_names.append("rSum")
    metric_values.append(metrics["rsum"])

    df = pd.DataFrame(
        {
            "Metric": metric_names,
            "Value": metric_values,
        }
    )

    # Save results as CSV
    csv_path = os.path.join(backup_folder, "results.csv")
    df.to_csv(csv_path, index=False, float_format="%.2f")
    logging.info(f"Results saved to: {csv_path}")

    # Move log file to backup folder
    log_path = os.path.join(args.output_dir, "log.log")
    if os.path.exists(log_path):
        shutil.move(log_path, os.path.join(backup_folder, "log.log"))

    # Log formatted results
    result_str = f"\n{'=' * 50}\n"
    result_str += "Image-to-Text Retrieval:\n"
    for k in args.k_values:
        result_str += f"  R@{k}: {metrics[f'i2t_r@{k}']:.2f}%\n"
    result_str += f"  MedR: {metrics['i2t_medr']:.1f}\n"
    result_str += f"  MeanR: {metrics['i2t_meanr']:.1f}\n\n"

    result_str += "Text-to-Image Retrieval:\n"
    for k in args.k_values:
        result_str += f"  R@{k}: {metrics[f't2i_r@{k}']:.2f}%\n"
    result_str += f"  MedR: {metrics['t2i_medr']:.1f}\n"
    result_str += f"  MeanR: {metrics['t2i_meanr']:.1f}\n\n"

    result_str += f"rSum: {metrics['rsum']:.2f}\n"
    result_str += f"{'=' * 50}\n"
    logging.info(result_str)

    # Save features if requested
    if args.save_features and image_features is not None and text_features is not None:
        features_file = os.path.join(backup_folder, "features.pth")
        torch.save(
            {
                "image_features": image_features,
                "text_features": text_features,
                "captions": captions,
                "metrics": metrics,
            },
            features_file,
        )
        logging.info(f"Features saved to: {features_file}")


def main():
    args = get_args()

    if args.output_dir is None:
        args.output_dir = f"runs/{args.exp}/test"
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args, args.output_dir)

    # Set random seed
    torch.manual_seed(args.seed)

    # Load model
    model, tokenizer = load_model(args)

    # Load dataset
    data_module = ROCOv2DataModule(args, cache_dir=args.cache_dir, max_samples=args.max_samples, seed=args.seed)

    # Get appropriate dataloader
    if args.split == "train":
        dataloader = data_module.train_dataloader(shuffle=False)
    elif args.split == "validation":
        dataloader = data_module.val_dataloader()
    else:  # test
        dataloader = data_module.test_dataloader()

    # Extract features
    image_features, text_features, captions = extract_features(model, tokenizer, dataloader, args)

    # Compute retrieval metrics
    metrics = compute_retrieval_metrics(image_features, text_features, k_values=args.k_values, normalize=True)

    # Log metrics
    log_retrieval_metrics(metrics, prefix=args.split)

    # Save results
    save_results(args, metrics, image_features, text_features, captions)

    logging.info("✓ Retrieval evaluation complete")


if __name__ == "__main__":
    main()
