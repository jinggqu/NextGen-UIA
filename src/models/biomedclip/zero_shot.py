import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import os
import logging
import argparse
import random
import datetime
import shutil

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from open_clip import create_model_from_pretrained, get_tokenizer
from torchmetrics import ROC
from src.datasets import zero_shot as dataset_cls_zero_shot
from src.utils.tools import MetricAccumulator, setup_logging
from src.adapters import inject_mona_variant_to_open_clip
from src.models.zero_shot_prompt import LN_PROMPTS_ENSEMBLE, BREAST_PROMPTS_ENSEMBLE

LESION_TYPES = [
    "benign",
    "malignant",
]

_ROC = ROC(task="binary").cuda() if torch.cuda.is_available() else ROC(task="binary")


def get_args():
    """Get arguments from command line"""

    parser = argparse.ArgumentParser("Adaptation of Visual Foundation Model for Medical Ultrasound Image Analysis")

    # Data related
    parser.add_argument("--exp", type=str, default="biomedclip_zero_shot")
    parser.add_argument("--dataset", type=str, default="LN-INT", help="Dataset name")
    parser.add_argument("--img_size", type=int, default=224, help="Image width and height")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--num_workers", type=int, default=8)

    # Augmentation related
    parser.add_argument("--strong_augs", default=False, action=argparse.BooleanOptionalAction, help="Use strong augs")
    parser.add_argument("--weak_augs", default=False, action=argparse.BooleanOptionalAction, help="Use weak augs")

    # Model related
    parser.add_argument("--mona_weights", type=str, default=None, help="Path to finetuned MONA checkpoint")
    parser.add_argument(
        "--mona_variant",
        type=str,
        default="freq_enhanced",
        choices=["baseline", "fractional", "noise_aware", "freq_enhanced", "hybrid"],
        help="MONA variant type",
    )
    parser.add_argument("--mona_bottleneck", type=int, default=64, help="MONA bottleneck dimension")
    parser.add_argument("--mona_layers", type=int, default=None, help="Number of layers to inject MONA (None=all)")
    parser.add_argument("--lora_weights", type=str, default=None, help="Path to finetuned LoRA checkpoint")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)

    # Inference related
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


criterion = torch.nn.CrossEntropyLoss()


def save_zero_shot_artifacts(args, stats, df, fig, fig_name):
    """Persist zero-shot evaluation artifacts in a timestamped folder."""
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    backup_folder = os.path.join(
        args.test_snapshot_path,
        f"{timestamp}_acc={stats['acc'] * 100:.2f}",
    )
    os.makedirs(backup_folder, exist_ok=True)

    csv_path = os.path.join(backup_folder, "results.csv")
    df.to_csv(csv_path, index=False, float_format="%.2f")
    logging.info(f"Results saved to: {csv_path}")

    if fig is not None:
        fig.savefig(os.path.join(backup_folder, f"{fig_name}.png"), bbox_inches="tight")
        plt.close(fig)

    log_path = os.path.join(args.test_snapshot_path, "log.log")
    if os.path.exists(log_path):
        shutil.move(log_path, os.path.join(backup_folder, "log.log"))

    return backup_folder


def prepare_model(args):
    logging.info("Loading BiomedCLIP model...")

    clip_model, _ = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", cache_dir="./ckpt"
    )
    tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

    if args.lora_weights:
        from src.adapters import inject_lora_to_biomedclip

        logging.info(f"Injecting LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})")
        inject_lora_to_biomedclip(clip_model, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0)

        checkpoint = torch.load(args.lora_weights, map_location="cpu", weights_only=True)
        lora_state_dict = checkpoint.get("lora_state_dict", checkpoint)

        model_dict = clip_model.state_dict()
        lora_params_loaded = 0
        for name, param in lora_state_dict.items():
            if name in model_dict:
                model_dict[name] = param
                lora_params_loaded += 1
        assert lora_params_loaded > 0, "No LoRA parameters loaded"
        clip_model.load_state_dict(model_dict)
        logging.info(f"✓ Loaded {lora_params_loaded} LoRA parameters from {args.lora_weights}")

    elif args.mona_weights:
        logging.info(f"Injecting MONA adapters (variant: {args.mona_variant})")
        inject_mona_variant_to_open_clip(
            clip_model,
            variant=args.mona_variant,
            bottleneck_dim=args.mona_bottleneck,
            num_layers=args.mona_layers,
        )

        checkpoint = torch.load(args.mona_weights, map_location="cpu", weights_only=True)
        mona_state_dict = checkpoint.get("mona_state_dict", checkpoint)

        model_dict = clip_model.state_dict()
        mona_params_loaded = 0
        for name, param in mona_state_dict.items():
            if name in model_dict:
                model_dict[name] = param
                mona_params_loaded += 1
        assert mona_params_loaded > 0, "No MONA parameters loaded"
        clip_model.load_state_dict(model_dict)
        logging.info(f"✓ Loaded {mona_params_loaded} MONA parameters from {args.mona_weights}")

    clip_model.float()
    clip_model.to(args.device)
    clip_model.eval()
    return clip_model, tokenizer


@torch.no_grad()
def test(args):
    logging.info("=" * 50)
    logging.info("BiomedCLIP Zero-shot Classification")
    logging.info("=" * 50)

    # Model initialization
    model, tokenizer = prepare_model(args)

    # Data initialization
    dm = dataset_cls_zero_shot.DataModule(args)
    testloader = dm.test_dataloader()
    accumulator = MetricAccumulator(type="cls", criterion=criterion, num_classes=args.num_classes)

    if "ln" in args.dataset.lower():
        prompt_ensemble = LN_PROMPTS_ENSEMBLE
    elif "busi" in args.dataset.lower():
        prompt_ensemble = BREAST_PROMPTS_ENSEMBLE
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    text_features_ensemble = {}
    for class_name in LESION_TYPES:
        prompts = prompt_ensemble[class_name]

        # Tokenize prompts
        text_tokens = tokenizer(prompts).to(args.device)

        # Encode text
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_features_ensemble[class_name] = text_features

    benign_proto = text_features_ensemble["benign"].mean(dim=0)
    malignant_proto = text_features_ensemble["malignant"].mean(dim=0)
    proto_sim = (benign_proto @ malignant_proto).item()

    if proto_sim > 0.95:
        logging.warning(f"Text prompts very similar: {proto_sim:.4f}")

    all_image_feats = []
    all_labels = []
    for _, sampled_batch in enumerate(testloader):
        images, labels, _ = sampled_batch
        images, labels = images.to(args.device), labels.to(args.device)

        # Encode images
        with torch.no_grad():
            image_features = model.encode_image(images)  # [B, 512]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [B, 512]

            if len(all_image_feats) < 10:
                all_image_feats.append(image_features.cpu())
                all_labels.append(labels)

            # Compute similarity with each class ensemble
            ensemble_logits = []

            for class_name in LESION_TYPES:
                # Get all text features for this class [N_prompts, 512]
                text_feats = text_features_ensemble[class_name]

                # Compute similarity: [B, 512] @ [512, N_prompts] = [B, N_prompts]
                similarities = 100.0 * image_features @ text_feats.T

                # Average across all prompts for this class: [B, N_prompts] -> [B]
                avg_similarity = similarities.mean(dim=1)  # [B]
                ensemble_logits.append(avg_similarity)

            # Stack to create [B, 2] logits tensor
            logits = torch.stack(ensemble_logits, dim=1)  # [B, 2]
            preds = logits  # Pass logits to accumulator

        accumulator.update(preds.detach(), labels.detach())

    stats = accumulator.compute()

    all_image_feats = torch.cat(all_image_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if len(all_image_feats) > 10:
        cov = all_image_feats.T @ all_image_feats / len(all_image_feats)
        eigenvalues, _ = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.flip(0).abs()
        top_eigenvalue_ratio = eigenvalues[0] / eigenvalues.sum()

        if top_eigenvalue_ratio > 0.95:
            logging.warning(f"Features may be collapsed (ratio={top_eigenvalue_ratio:.4f})")

    # Plot ROC curve
    fpr, tpr, _ = _ROC(
        torch.softmax(accumulator.all_preds, dim=1)[:, 1].to(_ROC.device),
        accumulator.all_labels.to(_ROC.device),
    )
    stats = accumulator.compute()
    accumulator.reset()

    fig = plt.figure(figsize=(4, 4), dpi=600)
    plt.plot(fpr.cpu().numpy(), tpr.cpu().numpy(), linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title(f"BiomedCLIP Zero-shot\nAUC = {stats['auc']:.4f}", fontsize=14)
    plt.tight_layout()

    df = pd.DataFrame(
        {
            "Metric": ["Acc", "Rec", "Pre", "F1", "AUC"],
            "Mean": [stats["acc"] * 100, stats["rec"] * 100, stats["pre"] * 100, stats["f1"] * 100, stats["auc"] * 100],
        }
    )

    result_str = f"\n{'=' * 50}\n"
    result_str += df.to_string(index=False, float_format="%.2f") + "\n"
    result_str += f"{'=' * 50}\n"
    logging.info(result_str)

    fig_name = "roc_curve_biomedclip_zero_shot"

    save_zero_shot_artifacts(args, stats, df, fig, fig_name)

    return "Testing Finished!"


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.test_snapshot_path = f"runs/{args.exp}/{args.dataset}/test"
    os.makedirs(args.test_snapshot_path, exist_ok=True)

    setup_logging(args, args.test_snapshot_path)
    test(args)


if __name__ == "__main__":
    main()
