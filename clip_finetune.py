import os
import logging
import argparse
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import dataset_finetune as dataset_finetune
from models.openai import clip
from tools import model_summary, setup_logging


def get_args():
    """Get arguments from command line"""

    parser = argparse.ArgumentParser("Adaptation of Visual Foundation Model for Medical Ultrasound Image Analysis")

    # Data related
    parser.add_argument("--img_size", type=int, default=224, help="Image width and height")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--num_workers", type=int, default=8)

    # Augmentation related
    parser.add_argument("--strong_augs", default=False, action=argparse.BooleanOptionalAction, help="Use strong augs")
    parser.add_argument("--weak_augs", default=False, action=argparse.BooleanOptionalAction, help="Use weak augs")

    # Model related
    parser.add_argument("--exp", type=str, default="clip_mona_finetune")
    # Available models: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    parser.add_argument("--version", type=str, default="ViT-B/16")
    parser.add_argument("--ckpt", type=str, default="ckpt/ViT-B-16.pt")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)

    # Training related
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--caption_type", type=str, choices=["refined", "truncated"])

    return parser.parse_args()


criterion = torch.nn.CrossEntropyLoss()


def prepare_model(args):
    # Build model
    model, _ = clip.load(args.ckpt, device=args.device)

    # Freeze CLIP model and only train the Mona layers
    for name, param in model.named_parameters():
        if "mona" not in name:
            param.requires_grad = False

    # Convert model to float32 to prevent NaN output
    # when model.encode_image() and model.encode_text() are called
    # https://github.com/openai/CLIP/issues/144
    model.float()
    model.to(args.device)

    return model


def train(args):
    # Model initialization
    model = prepare_model(args)
    model.train()
    logging.info(model_summary({"model": model}))
    writer = SummaryWriter(args.train_snapshot_path + "/log")
    logging.info("Start training")

    # Data initialization
    dm = dataset_finetune.DataModule(args)
    trainloader, valloader = dm.train_dataloader(), dm.val_dataloader()

    # Optimizer initialization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    max_epoch = args.epochs
    max_iters = len(trainloader) * max_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=args.lr_min)

    iter_num = 0
    best_val_loss = float("inf")
    iterator = tqdm(range(max_epoch), ncols=70)
    patience_counter = 0
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            image, text = sampled_batch
            image = image.to(args.device)
            text_tokens = clip.tokenize(text, truncate=True).to(args.device)

            optimizer.zero_grad()

            image_features, text_features = model(image, text_tokens)
            ground_truth = torch.arange(image.shape[0], dtype=torch.long, device=args.device)

            loss = (criterion(image_features, ground_truth) + criterion(text_features, ground_truth)) / 2
            loss.backward()

            optimizer.step()
            scheduler.step()

            if iter_num % 10 == 0:
                writer.add_scalar(f"{args.exp}/train_loss", loss.item(), iter_num)
                writer.add_scalar(f"{args.exp}/lr", scheduler.get_last_lr()[0], iter_num)
                writer.flush()
            iter_num += 1

        # Validation (every 10 epochs)
        if epoch > 0 and epoch % 10 == 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for _, sampled_batch in enumerate(valloader):
                    image, text = sampled_batch
                    image = image.to(args.device)
                    text_tokens = clip.tokenize(text, truncate=True).to(args.device)

                    image_features, text_features = model(image, text_tokens)
                    ground_truth = torch.arange(image.shape[0], dtype=torch.long, device=args.device)

                    loss = (criterion(image_features, ground_truth) + criterion(text_features, ground_truth)) / 2
                    val_loss += loss.item()

            val_loss /= len(valloader)

            writer.add_scalar(f"{args.exp}/val_loss", val_loss, iter_num)
            writer.flush()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_save_path = os.path.join(args.train_snapshot_path, f"{args.caption_type}.pth")
                mona_state_dict = {}
                for name, param in model.named_parameters():
                    if "mona" in name.lower():
                        mona_state_dict[name] = param.data
                torch.save(mona_state_dict, ckpt_save_path)
                patience_counter = 0
            else:
                patience_counter += 1

            logging.info(f"\titer: {iter_num}, loss: {val_loss:.4f}, patience: {patience_counter}")

            # Switch back to train mode
            model.train()

            if patience_counter >= args.patience:
                logging.info(f"Early stopping at iter {iter_num} with patience {patience_counter}")
                iterator.close()
                writer.close()
                torch.cuda.empty_cache()
                return "CLIP Finetuning Finished!"

    writer.close()
    torch.cuda.empty_cache()
    return "CLIP Finetuning Finished!"


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Logging
    snapshot_path = f"runs/{args.exp}"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    args.train_snapshot_path = snapshot_path

    setup_logging(args, args.train_snapshot_path)
    train(args)
