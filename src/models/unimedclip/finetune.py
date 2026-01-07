"""
UniMedCLIP Fine-tuning with Frequency-Enhanced MONA and Loss Comparison
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import os
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.datasets import finetune as dataset_finetune
from open_clip import create_model_and_transforms
from open_clip.tokenizer import HFTokenizer
from src.utils.tools import model_summary, setup_logging
from src.losses import InfoNCELoss
from src.adapters import inject_mona_variant_to_open_clip


def get_args():
    """Get arguments from command line"""
    parser = argparse.ArgumentParser("UniMedCLIP Fine-tuning with Frequency-Enhanced MONA")

    # Data related
    parser.add_argument("--img_size", type=int, default=224, help="Image width and height")
    parser.add_argument("--num_workers", type=int, default=8)

    # Augmentation related
    parser.add_argument("--strong_augs", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--weak_augs", default=False, action=argparse.BooleanOptionalAction)

    # Model related
    parser.add_argument("--version", type=str, default="ViT-B-16-quickgelu")
    parser.add_argument("--ckpt", type=str, default="ckpt/unimed_clip_vit_b16.pt")
    parser.add_argument("--mona_variant", type=str, default="noise_aware", help="MONA variant")
    parser.add_argument("--exp", type=str, default="unimedclip_finetune")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--mona_bottleneck", type=int, default=64, help="MONA bottleneck dimension")
    parser.add_argument("--mona_layers", type=int, default=None, help="Number of layers to inject MONA (None=all)")

    # Loss parameters
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")

    # Training related
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1_adam", type=float, default=0.9)
    parser.add_argument("--beta2_adam", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    return parser.parse_args()


def prepare_model(args):
    """Load UniMedCLIP and inject MONA adapters"""
    # Create model without pretrained weights
    model, _, _ = create_model_and_transforms(
        model_name=args.version,
        pretrained=False,  # Don't load yet, will load manually
        precision="amp",
        device=args.device,
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
    tokenizer = HFTokenizer(text_encoder_name, context_length=77, **{})  # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Inject Noise-Aware MONA adapters
    model, mona_count = inject_mona_variant_to_open_clip(
        model, variant=args.mona_variant, bottleneck_dim=args.mona_bottleneck, num_layers=args.mona_layers
    )

    # Only train MONA parameters
    for name, param in model.named_parameters():
        if "mona" in name.lower():
            param.requires_grad = True

    # Convert model to float32
    model.float()
    model.to(args.device)

    return model, tokenizer


def train(args):
    """Train UniMedCLIP with MONA adapters"""
    logging.info("=" * 50)
    logging.info(f"Training: {args.exp}")
    logging.info("=" * 50)

    model, tokenizer = prepare_model(args)
    model.train()

    writer = SummaryWriter(args.train_snapshot_path + "/log")
    logging.info(model_summary({"model": model}))
    logging.info("Start training")

    # Data initialization
    dm = dataset_finetune.DataModule(args)
    trainloader, valloader = dm.train_dataloader(), dm.val_dataloader()

    # Loss function
    criterion = InfoNCELoss(temperature=args.temperature)

    # Optimizer (only MONA parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(args.beta1_adam, args.beta2_adam),
        weight_decay=args.weight_decay,
    )

    max_epoch = args.epochs
    max_iters = len(trainloader) * max_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=args.lr_min)

    iter_num = 0
    best_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        model.train()
        criterion.train()
        train_loss = 0.0

        for sampled_batch in trainloader:
            images, texts = sampled_batch
            images = images.to(args.device)

            # Tokenize texts
            text_tokens = tokenizer(texts).to(args.device)

            # Forward pass
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute loss
            loss = criterion(image_features, text_features)

            # Check for invalid loss values
            if not torch.isfinite(loss):
                logging.warning(f"Non-finite loss detected at iteration {iter_num}, skipping batch")
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            iter_num += 1

            # Logging
            writer.add_scalar("train/loss", loss.item(), iter_num)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iter_num)

        train_loss /= len(trainloader)

        # Validation
        model.eval()
        criterion.eval()
        val_loss = 0.0

        with torch.no_grad():
            for sampled_batch in valloader:
                images, texts = sampled_batch
                images = images.to(args.device)

                text_tokens = tokenizer(texts).to(args.device)

                image_features = model.encode_image(images)
                text_features = model.encode_text(text_tokens)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                loss = criterion(image_features, text_features)

                # Skip non-finite validation losses
                if torch.isfinite(loss):
                    val_loss += loss.item()

        val_loss /= len(valloader)

        # Logging
        writer.add_scalar("val/loss", val_loss, epoch_num)

        logging.info(
            f"Epoch {epoch_num + 1}/{max_epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, Best={best_loss:.4f}"
        )

        # Early stopping and checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch_num
            patience_counter = 0

            # Save best model (only MONA parameters)
            save_path = os.path.join(args.train_snapshot_path, "best_model.pth")
            mona_state_dict = {}
            for name, param in model.named_parameters():
                if "mona" in name.lower():
                    mona_state_dict[name] = param.data

            torch.save(mona_state_dict, save_path)
            logging.info(f"Best model saved at epoch {epoch_num + 1}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logging.info(f"\nEarly stopping at epoch {epoch_num + 1}")
            logging.info(f"Early stopping triggered at epoch {epoch_num + 1}")
            break

    logging.info(f"\n✓ Training completed! Best loss: {best_loss:.4f} (epoch {best_epoch + 1})")

    writer.close()
    return "Training Finished!"


def run_zero_shot_evaluation(args):
    """Run zero-shot evaluation after fine-tuning"""
    import subprocess

    logging.info("=" * 50)
    logging.info("Starting Zero-shot Evaluation on Fine-tuned Model")
    logging.info("=" * 50)

    # Path to best fine-tuned checkpoint
    best_model_path = os.path.join(args.train_snapshot_path, "best_model.pth")

    if not os.path.exists(best_model_path):
        logging.error(f"Best model checkpoint not found at {best_model_path}")
        return

    # Path to zero-shot evaluation script
    zero_shot_script = Path(__file__).parent / "zero_shot.py"

    # Datasets to evaluate on
    datasets = ["LN-INT", "LN-EXT", "BUSI"]

    for dataset in datasets:
        logging.info(f"\nEvaluating on {dataset} dataset...")

        cmd = [
            "python",
            str(zero_shot_script),
            "--exp",
            f"{args.exp}",
            "--dataset",
            dataset,
            "--img_size",
            str(args.img_size),
            "--mona_weights",
            best_model_path,
            "--mona_bottleneck",
            str(args.mona_bottleneck),
            "--batch_size",
            "64",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]

        logging.info(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info(result.stdout)
            if result.stderr:
                logging.warning(result.stderr)
        except subprocess.CalledProcessError as e:
            logging.error(f"Zero-shot evaluation failed for {dataset}")
            logging.error(f"Error: {e.stderr}")

    logging.info("Zero-shot Evaluation Completed")


def main():
    """Main entry point for the script"""
    args = get_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Setup paths and logging
    args.train_snapshot_path = f"runs/{args.exp}"
    os.makedirs(args.train_snapshot_path, exist_ok=True)

    setup_logging(args, args.train_snapshot_path)

    # Train
    train(args)

    # Run zero-shot classification evaluation
    run_zero_shot_evaluation(args)


if __name__ == "__main__":
    main()
