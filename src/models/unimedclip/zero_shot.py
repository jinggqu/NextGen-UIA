import sys
from pathlib import Path

# Add project root to path
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
from open_clip import create_model_and_transforms
from open_clip.tokenizer import HFTokenizer
from torchmetrics import ROC
from src.datasets import zero_shot as dataset_cls_zero_shot
from src.utils.tools import MetricAccumulator, setup_logging
import matplotlib.pyplot as plt
from src.adapters import inject_mona_variant_to_open_clip
from src.models.zero_shot_prompt import LN_PROMPTS_ENSEMBLE, BREAST_PROMPTS_ENSEMBLE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LESION_TYPES = [
    "benign",
    "malignant",
]


_ROC = ROC(task="binary").cuda() if torch.cuda.is_available() else ROC(task="binary")


def get_args():
    """Get arguments from command line"""
    parser = argparse.ArgumentParser("Zero-shot Classification with UniMedCLIP")

    # Data related
    parser.add_argument("--exp", type=str, default="unimedclip_zero_shot")
    parser.add_argument("--dataset", type=str, default="LN-INT", help="Dataset name")
    parser.add_argument("--img_size", type=int, default=224, help="Image width and height")
    parser.add_argument("--num_workers", type=int, default=8)

    # Model related
    parser.add_argument("--version", type=str, default="ViT-B-16-quickgelu")
    parser.add_argument("--ckpt", type=str, default="ckpt/unimed_clip_vit_b16.pt")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)

    # MONA adapter related (for loading fine-tuned model)
    parser.add_argument("--mona_variant", type=str, default="noise_aware", help="MONA variant")
    parser.add_argument("--mona_weights", type=str, default=None, help="Path to fine-tuned MONA checkpoint")
    parser.add_argument("--mona_bottleneck", type=int, default=64, help="MONA bottleneck dimension")

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
    """Load UniMedCLIP and inject MONA adapters"""
    # Create model without pretrained weights
    model, _, _ = create_model_and_transforms(
        model_name=args.version,
        pretrained=False,  # Don't load yet, will load manually
        precision="amp",
        force_quick_gelu=True,
    )

    # Load UniMedCLIP checkpoint manually
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load only the visual encoder weights (strict=False to ignore text_encoder keys)
    # Filter to only visual and logit_scale keys
    visual_state_dict = {k: v for k, v in state_dict.items() if k.startswith("visual.") or k == "logit_scale"}
    model.load_state_dict(visual_state_dict, strict=False)

    text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    tokenizer = HFTokenizer(text_encoder_name, context_length=77, **{})

    if args.mona_weights:
        logging.info(f"Loading fine-tuned MONA adapters from {args.mona_weights}...")

        # Inject MONA adapters
        model, mona_count = inject_mona_variant_to_open_clip(
            model, variant=args.mona_variant, bottleneck_dim=args.mona_bottleneck
        )

        # Load checkpoint
        checkpoint = torch.load(args.mona_weights, map_location="cpu")
        mona_state_dict = checkpoint

        # Load MONA parameters
        model.load_state_dict(mona_state_dict, strict=False)
        logging.info(f"✓ Loaded MONA adapters ({mona_count} layers)")

    model.eval()
    model = model.to(args.device)
    return model, tokenizer


@torch.no_grad()
def test(args):
    logging.info("=" * 50)
    logging.info("UniMedCLIP Zero-shot Classification")
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

    for _, sampled_batch in enumerate(testloader):
        images, labels, _ = sampled_batch
        images, labels = images.to(args.device), labels.to(args.device)

        # Encode images
        with torch.no_grad():
            image_features = model.encode_image(images)  # [B, 512]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [B, 512]

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

    all_image_feats = []
    all_labels = []

    for _, sampled_batch in enumerate(testloader):
        images, labels, _ = sampled_batch
        images = images.to(args.device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_feats.append(image_features.cpu())
            all_labels.append(labels)

        if len(all_image_feats) >= 10:  # Sample first 10 batches
            break

    all_image_feats = torch.cat(all_image_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # benign_idx = all_labels == 0
    # malignant_idx = all_labels == 1

    # benign_img_feats = all_image_feats[benign_idx]
    # malignant_img_feats = all_image_feats[malignant_idx]

    # Compute class means and similarity
    # benign_img_mean = benign_img_feats.mean(dim=0)
    # malignant_img_mean = malignant_img_feats.mean(dim=0)
    # img_class_sim = (benign_img_mean @ malignant_img_mean).item()

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
    plt.title(f"UniMedCLIP Zero-shot\nAUC = {stats['auc']:.4f}", fontsize=14)
    plt.tight_layout()

    fig_name = "roc_curve_unimedclip_zero_shot"

    df = pd.DataFrame(
        {
            "Metric": ["Acc", "Rec", "Pre", "F1", "AUC"],
            "Mean": [stats["acc"] * 100, stats["rec"] * 100, stats["pre"] * 100, stats["f1"] * 100, stats["auc"] * 100],
        }
    )

    result_str = "=" * 50 + "\n"
    result_str += "UniMedCLIP Zero-shot Results\n"
    result_str += "=" * 50 + "\n"
    result_str += df.to_string(index=False, float_format="%.2f") + "\n"
    result_str += "=" * 50 + "\n"

    logging.info(result_str)

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
