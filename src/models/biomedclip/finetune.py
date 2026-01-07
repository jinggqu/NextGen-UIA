"""
BiomedCLIP Fine-tuning with Multiple Methods

Supports:
- Full fine-tuning (all parameters or selected layers)
- MONA adapter fine-tuning (various variants)
- LoRA adapter fine-tuning
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import os
import math
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.datasets import finetune as dataset_finetune
from open_clip import create_model_from_pretrained, get_tokenizer
from src.utils.tools import model_summary, setup_logging
from src.losses import InfoNCELoss
from src.adapters import inject_mona_variant_to_open_clip, inject_lora_to_biomedclip


def get_args():
    parser = argparse.ArgumentParser("BiomedCLIP Fine-tuning")

    # Data related
    parser.add_argument("--img_size", type=int, default=224, help="Image width and height")
    parser.add_argument("--num_workers", type=int, default=8)

    # Augmentation related
    parser.add_argument("--strong_augs", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--weak_augs", default=False, action=argparse.BooleanOptionalAction)

    # Model related
    parser.add_argument("--exp", type=str, default="biomedclip_finetune")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to finetuned model checkpoint")

    # Fine-tuning method selection
    parser.add_argument(
        "--method",
        type=str,
        default="full",
        choices=["full", "mona", "lora"],
        help="Fine-tuning method: full (all params), mona (MONA adapter), lora (LoRA adapter)",
    )

    # Full fine-tuning options
    parser.add_argument(
        "--tune_text_encoder",
        default=False,
        action="store_true",
        help="Tune text encoder parameters during finetuning (if not set, only train image encoder)",
    )
    parser.add_argument(
        "--tune_layers",
        type=str,
        default="all",
        choices=["last3", "last6", "last9", "all"],
        help="Which ViT layers to train for full fine-tuning",
    )

    # MONA options
    parser.add_argument(
        "--mona_variant",
        type=str,
        default="freq_enhanced",
        choices=["baseline", "fractional", "noise_aware", "freq_enhanced", "hybrid"],
        help="MONA variant type",
    )
    parser.add_argument("--mona_bottleneck", type=int, default=64, help="MONA bottleneck dimension")
    parser.add_argument("--mona_layers", type=int, default=None, help="Number of layers to inject MONA (None=all)")

    # LoRA options
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (bottleneck dimension)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability")
    parser.add_argument("--lora_layers", type=int, default=None, help="Number of layers to inject LoRA (None=all)")

    # Loss parameters
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")

    # Training related
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1_adam", type=float, default=0.9)
    parser.add_argument("--beta2_adam", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch size = batch_size * accumulation_steps)",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")

    return parser.parse_args()


def prepare_model(args):
    """Load BiomedCLIP and configure based on fine-tuning method"""
    model, _ = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", cache_dir="./ckpt"
    )
    tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

    model.float()

    if args.method == "full":
        _setup_full_finetuning(model, args)
    elif args.method == "mona":
        _setup_mona_finetuning(model, args)
    elif args.method == "lora":
        _setup_lora_finetuning(model, args)

    model.to(args.device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    return model, tokenizer


def _setup_full_finetuning(model, args):
    if not args.tune_text_encoder:
        for param in model.text.parameters():
            param.requires_grad = False
        logging.info("Text encoder frozen")

    if args.tune_layers != "all":
        num_blocks = len(model.visual.trunk.blocks)
        for param in model.visual.parameters():
            param.requires_grad = False

        layer_map = {"last3": 3, "last6": 6, "last9": 9}
        layers = layer_map.get(args.tune_layers, 0)
        if layers > 0:
            start_layer = num_blocks - layers
            for i in range(start_layer, num_blocks):
                for param in model.visual.trunk.blocks[i].parameters():
                    param.requires_grad = True
            logging.info(f"Tuning ViT layers {start_layer}-{num_blocks - 1} ({layers}/{num_blocks} layers)")

    if args.lr > 1e-5:
        args.lr = 1e-6
        logging.info(f"Adjusted learning rate to {args.lr} for full fine-tuning")


def _setup_mona_finetuning(model, args):
    for param in model.parameters():
        param.requires_grad = False

    inject_mona_variant_to_open_clip(
        model, variant=args.mona_variant, bottleneck_dim=args.mona_bottleneck, num_layers=args.mona_layers
    )

    for name, param in model.named_parameters():
        if "mona" in name.lower():
            param.requires_grad = True

    logging.info(f"MONA variant: {args.mona_variant}, bottleneck: {args.mona_bottleneck}")


def _setup_lora_finetuning(model, args):
    for param in model.parameters():
        param.requires_grad = False

    inject_lora_to_biomedclip(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_layers=args.lora_layers,
        tune_text_encoder=args.tune_text_encoder,
    )

    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    logging.info(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")


def _save_checkpoint(model, args, save_path):
    if args.method == "mona":
        state_dict = {name: param.data.clone() for name, param in model.named_parameters() if "mona" in name.lower()}
        torch.save(state_dict, save_path)
    elif args.method == "lora":
        state_dict = {name: param.data.clone() for name, param in model.named_parameters() if "lora" in name.lower()}
        torch.save(state_dict, save_path)
    else:
        torch.save(model.state_dict(), save_path)


def train(args):
    logging.info("=" * 50)
    logging.info(f"Training: {args.exp}")
    logging.info(f"Method: {args.method}")
    logging.info("=" * 50)

    model, tokenizer = prepare_model(args)
    model.train()

    writer = SummaryWriter(args.train_snapshot_path + "/log")
    logging.info(model_summary({"model": model}))

    if args.method == "full":
        logging.info(f"Text encoder: {'Trainable' if args.tune_text_encoder else 'Frozen'}")
        logging.info(f"ViT tuning: {args.tune_layers}")
    elif args.method == "mona":
        logging.info(f"MONA variant: {args.mona_variant}")
    elif args.method == "lora":
        logging.info(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")

    logging.info(f"Gradient accumulation steps: {args.accumulation_steps}")
    logging.info(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    logging.info(
        f"Gradient clipping: {'Enabled (max_norm=' + str(args.grad_clip) + ')' if args.grad_clip > 0 else 'Disabled'}"
    )
    logging.info(f"Learning rate: {args.lr}")
    logging.info("Start training")

    dm = dataset_finetune.DataModule(args)
    trainloader, valloader = dm.train_dataloader(), dm.val_dataloader()

    criterion = InfoNCELoss(temperature=args.temperature).to(args.device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(args.beta1_adam, args.beta2_adam),
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(len(trainloader) / args.accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    logging.info(f"Updates per epoch: {updates_per_epoch}")
    logging.info(f"Total expected updates: {total_updates}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=args.lr_min)

    update_count = 0
    best_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    optimizer.zero_grad()

    for epoch_num in tqdm(range(args.epochs), ncols=70):
        model.train()
        criterion.train()

        epoch_train_loss = 0.0
        epoch_batch_count = 0
        update_cycle_loss = 0.0
        update_cycle_batch_count = 0

        for batch_idx, sampled_batch in enumerate(trainloader):
            images, texts = sampled_batch
            images = images.to(args.device)
            text_tokens = tokenizer(texts).to(args.device)
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            loss = criterion(image_features, text_features)

            if not torch.isfinite(loss):
                logging.warning(
                    f"Non-finite loss detected at batch {batch_idx} in epoch {epoch_num + 1}, skipping batch"
                )
                continue

            scaled_loss = loss / args.accumulation_steps
            scaled_loss.backward()

            loss_item = loss.item()
            update_cycle_loss += loss_item
            update_cycle_batch_count += 1
            epoch_train_loss += loss_item
            epoch_batch_count += 1

            if ((batch_idx + 1) % args.accumulation_steps == 0) or (batch_idx + 1 == len(trainloader)):
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_count += 1

                avg_update_loss = update_cycle_loss / update_cycle_batch_count if update_cycle_batch_count > 0 else 0
                writer.add_scalar("train/loss_per_update", avg_update_loss, update_count)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], update_count)

                update_cycle_loss = 0.0
                update_cycle_batch_count = 0

        model.eval()
        criterion.eval()
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for sampled_batch in valloader:
                images, texts = sampled_batch
                images = images.to(args.device)
                text_tokens = tokenizer(texts).to(args.device)
                image_features = model.encode_image(images)
                text_features = model.encode_text(text_tokens)
                loss = criterion(image_features, text_features)

                if torch.isfinite(loss):
                    val_loss += loss.item()
                    val_batch_count += 1
                else:
                    logging.warning(f"Non-finite validation loss detected in epoch {epoch_num + 1}, skipping batch")

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0.0
        avg_epoch_train_loss = epoch_train_loss / epoch_batch_count if epoch_batch_count > 0 else 0.0

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch_num
            save_path = os.path.join(args.train_snapshot_path, "best_model.pth")
            _save_checkpoint(model, args, save_path)
            logging.info(f"\nBest model saved at epoch {epoch_num + 1} with validation loss {best_loss:.4f}")
        else:
            patience_counter += 1

        writer.add_scalar("val/loss_per_epoch", avg_val_loss, epoch_num + 1)
        writer.add_scalar("train/loss_per_epoch", avg_epoch_train_loss, epoch_num + 1)

        logging.info(
            f"Epoch {epoch_num + 1}: Train={avg_epoch_train_loss:.4f}, Val={avg_val_loss:.4f}, Best={best_loss:.4f}"
        )

        if patience_counter >= args.patience:
            logging.info(
                f"\nEarly stopping at epoch {epoch_num + 1} as validation loss did not improve for {args.patience} epochs."
            )
            break

    logging.info(f"\n✓ Training completed! Best validation loss: {best_loss:.4f} at epoch {best_epoch + 1}")

    writer.close()
    return "Training Finished!"


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.train_snapshot_path = f"runs/{args.exp}"
    os.makedirs(args.train_snapshot_path, exist_ok=True)
    setup_logging(args, args.train_snapshot_path)

    train(args)


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
