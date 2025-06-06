import os
import logging
import argparse
import random

import numpy as np
import pandas as pd
import torch
import clip
from tqdm import tqdm
from torchmetrics import ROC
import dataset_cls_zero_shot
from tools import MetricAccumulator, setup_logging
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LESION_TYPES = [
    "benign",
    "malignant",
]
LN_PROMPT = torch.cat(
    [
        clip.tokenize(
            f"""Benign: Oval shape, preserved echogenic hilum, thin homogeneous cortex. Malignant: Round, lost hilum, thickened/heterogeneous cortex, microcalcifications, irregular margins.The abnormality in this lymph node scan is {c}."""
        )
        for c in LESION_TYPES
    ]
).to(DEVICE)

BUSI_PROMPT = torch.cat(
    [
        clip.tokenize(
            f"""Benign: Oval shape, smooth margins, parallel orientation, homogeneous hypoechoic echotexture, posterior enhancement. Malignant: Irregular shape, spiculated margins, non-parallel orientation, heterogeneous hypoechoic echotexture, microcalcifications, posterior shadowing. The abnormality in this breast scan is {c}."""
        )
        for c in LESION_TYPES
    ]
).to(DEVICE)

_ROC = ROC(task="binary").cuda()


def get_args():
    """Get arguments from command line"""

    parser = argparse.ArgumentParser("Adaptation of Visual Foundation Model for Medical Ultrasound Image Analysis")

    # Data related
    parser.add_argument("--exp", type=str, default="clip_mona_cls_zero_shot")
    parser.add_argument("--dataset", type=str, default="LN-1", help="Dataset name")
    parser.add_argument("--img_size", type=int, default=224, help="Image width and height")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--num_workers", type=int, default=8)

    # Augmentation related
    parser.add_argument("--strong_augs", default=True, action=argparse.BooleanOptionalAction, help="Use strong augs")
    parser.add_argument("--weak_augs", default=True, action=argparse.BooleanOptionalAction, help="Use weak augs")

    # Model related
    parser.add_argument("--version", type=str, default="ViT-B/16")
    parser.add_argument("--ckpt", type=str, default="ckpt/ViT-B-16.pt")
    parser.add_argument("--finetuned_weights", type=str, default=None)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)

    # Training related
    parser.add_argument("--deterministic", default=True, action="store_true", help="Whether use deterministic training")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--caption_type", type=str, choices=["refined", "truncated"])

    return parser.parse_args()


# Loss functions
criterion = torch.nn.CrossEntropyLoss()


def prepare_model(args):
    clip_model, _ = clip.load(args.ckpt, device=args.device)

    if args.finetuned_weights:
        mona_state_dict = torch.load(args.finetuned_weights, weights_only=True)
        model_dict = clip_model.state_dict()
        for name, param in mona_state_dict.items():
            if name in model_dict:
                model_dict[name] = param
        clip_model.load_state_dict(model_dict)

    # Convert model to float32 to prevent NaN output
    # when model.encode_image() and model.encode_text() are called
    # https://github.com/openai/CLIP/issues/144
    clip_model.float()

    clip_model.eval()
    return clip_model.to(args.device)


@torch.no_grad()
def test(args):
    print("Start testing")
    # Model initialization
    model = prepare_model(args)

    # Data initialization
    dm = dataset_cls_zero_shot.DataModule(args)
    testloader = dm.test_dataloader()
    accumulator = MetricAccumulator(type="cls", criterion=criterion, num_classes=args.num_classes)

    # Pre-encode text features for zero-shot classification
    with torch.no_grad():
        if "ln" in args.dataset.lower():
            text_features = model.encode_text(LN_PROMPT)
        elif "busi" in args.dataset.lower():
            text_features = model.encode_text(BUSI_PROMPT)
        else:
            raise ValueError(f"Dataset {args.dataset} not supported")

    for _, sampled_batch in enumerate(tqdm(testloader, ncols=70)):
        images, labels, _ = sampled_batch
        images, labels = images.to(args.device), labels.to(args.device)

        # Zero-shot classification approach
        with torch.no_grad():
            image_features = model.encode_image(images)  # [B, 512]
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)  # [2, 512]
            similarity = (100.0 * image_features @ text_features_norm.T).softmax(dim=-1)  # [B, 2]
            preds = similarity  # Use similarity scores directly as predictions

        accumulator.update(preds.detach(), labels.detach())

    # Plot ROC curve
    roc_dir = args.test_snapshot_path

    fpr, tpr, _ = _ROC(
        torch.softmax(accumulator.all_preds, dim=0)[:, 1],
        accumulator.all_labels,
    )
    stats = accumulator.compute()
    accumulator.reset()

    fig = plt.figure(figsize=(4, 4), dpi=600)
    plt.plot(fpr.cpu().numpy(), tpr.cpu().numpy())
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.5)
    plt.title(f"AUC = {stats['auc']:.4f}")

    fig_name = "roc_curve"
    if args.finetuned_weights:
        if args.caption_type == "refined":
            fig_name = fig_name + "_refined"
        elif args.caption_type == "truncated":
            fig_name = fig_name + "_truncated"
    else:
        fig_name = fig_name + "_raw"

    plt.savefig(os.path.join(roc_dir, fig_name + ".png"), bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(
        {
            "Metric": ["Acc", "Rec", "Pre", "F1", "AUC"],
            "Mean": [stats["acc"] * 100, stats["rec"] * 100, stats["pre"] * 100, stats["f1"] * 100, stats["auc"] * 100],
        }
    )

    result_str = "\n" + "=" * 40 + "\n"
    result_str += df.to_string(index=False, float_format="%.2f") + "\n"
    result_str += "=" * 40 + "\n"
    logging.info(result_str)

    return "Testing Finished!"


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.test_snapshot_path = f"runs/{args.exp}/{args.dataset}"
    os.makedirs(args.test_snapshot_path, exist_ok=True)

    setup_logging(args, args.test_snapshot_path)
    test(args)
