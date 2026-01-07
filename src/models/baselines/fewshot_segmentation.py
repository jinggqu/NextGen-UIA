"""
Few-shot Segmentation with UNet Baseline

This script performs segmentation with limited training data to evaluate
the data efficiency of traditional UNet compared to CLIP-based methods.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


import os
import shutil
import logging
import argparse
import random
import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.datasets.fewshot_segmentation import FewShotDataModule
from src.datasets import segmentation as dataset_seg
from monai.losses import DiceCELoss
from src.utils.tools import MetricAccumulator, model_summary, setup_logging, visualize_seg
from src.third_party.unet import UNet


def get_args():
    """Get arguments from command line"""

    parser = argparse.ArgumentParser("Few-shot Segmentation with UNet")

    # Data related
    parser.add_argument("--exp", type=str, default="fewshot_unet_seg")
    parser.add_argument("--dataset", type=str, default="LN-INT", help="Dataset name")
    parser.add_argument("--img_size", type=int, default=224, help="Image width and height")
    parser.add_argument("--num_workers", type=int, default=8)

    # Few-shot related
    parser.add_argument("--train_ratio", type=float, default=0.1, help="Percentage of training data to use (0.0-1.0)")

    # Augmentation related
    parser.add_argument("--strong_augs", default=True, action=argparse.BooleanOptionalAction, help="Use strong augs")
    parser.add_argument("--weak_augs", default=True, action=argparse.BooleanOptionalAction, help="Use weak augs")

    # Model related
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)

    # Training related
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (10 * N epochs)")

    # Testing related
    parser.add_argument("--test", default=False, action="store_true", help="Load local checkpoint for testing")
    return parser.parse_args()


# Loss functions
criterion = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=1e-8, smooth_dr=1e-8)


def prepare_model(args):
    model = UNet(in_channels=args.in_channels, num_classes=args.num_classes).to(args.device)
    return model


def train(args):
    # Model initialization
    model = prepare_model(args)
    model.train()
    logging.info(model_summary({"model": model}))
    writer = SummaryWriter(args.train_snapshot_path + "/log")
    logging.info("Start few-shot training (UNet)")

    # Data initialization with few-shot sampling
    dm = FewShotDataModule(args, train_ratio=args.train_ratio)
    trainloader, valloader, testloader = dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()

    # Log few-shot info
    logging.info(f"Training samples: {dm.sampled_train_count} / {dm.original_train_count}")

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
    best_val_dice = 0.0
    patience_counter = 0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            images, labels, _ = sampled_batch
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if iter_num % 10 == 0:
                writer.add_scalar(f"{args.exp}/train_loss", loss.item(), iter_num)
                writer.add_scalar(f"{args.exp}/lr", scheduler.get_last_lr()[0], iter_num)
                writer.flush()
            iter_num += 1

        # Validation (every 10 epochs or last epoch)
        if (epoch > 0 and epoch % 10 == 0) or (epoch == max_epoch - 1):
            model.eval()
            accumulator = MetricAccumulator(type="seg", criterion=criterion, num_classes=args.num_classes)
            with torch.no_grad():
                for _, sampled_batch in enumerate(valloader):
                    images, labels, _ = sampled_batch
                    images, labels = images.to(args.device), labels.to(args.device)
                    preds = model(images)
                    accumulator.update(preds.detach(), labels.detach())

            # Log validation images for visualization
            # Convert to single-channel binary mask
            pred_images = torch.argmax(torch.softmax(preds[:4], dim=1), dim=1, keepdim=True)

            writer.add_images(f"{args.exp}/input_images", images[:4], iter_num)
            writer.add_images(f"{args.exp}/label_images", labels[:4], iter_num)
            writer.add_images(f"{args.exp}/pred_images", pred_images, iter_num)
            writer.flush()

            # Calculate metrics statistics
            stats = accumulator.compute()
            accumulator.reset()

            writer.add_scalar(f"{args.exp}/val_loss", stats["loss"], iter_num)
            writer.add_scalar(f"{args.exp}/val_dice", stats["dice_mean"], iter_num)
            writer.add_scalar(f"{args.exp}/val_iou", stats["iou_mean"], iter_num)
            writer.add_scalar(f"{args.exp}/val_hd95", stats["hd95_mean"], iter_num)
            writer.add_scalar(f"{args.exp}/val_asd", stats["asd_mean"], iter_num)
            writer.flush()

            # Save best model
            if stats["dice_mean"] > best_val_dice:
                patience_counter = 0
                best_val_dice = stats["dice_mean"]
                save_best = os.path.join(args.train_snapshot_path, "best_model.pth")
                torch.save(model.state_dict(), save_best)
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logging.info(f"\nEarly stopping at epoch {epoch + 1}")
                break

            logging.info(
                f"\titer: {iter_num}, loss: {stats['loss']:.4f}, dice: {stats['dice_mean'] * 100:.2f}, "
                f"iou: {stats['iou_mean'] * 100:.2f}, hd95: {stats['hd95_mean']:.2f}, asd: {stats['asd_mean']:.2f}"
            )

            # Testing
            with torch.no_grad():
                for _, sampled_batch in enumerate(testloader):
                    images, labels, _ = sampled_batch
                    images, labels = images.to(args.device), labels.to(args.device)
                    preds = model(images)
                    accumulator.update(preds.detach(), labels.detach())

            # Calculate metrics statistics
            stats = accumulator.compute()
            accumulator.reset()

            writer.add_scalar(f"{args.exp}/test_loss", stats["loss"], iter_num)
            writer.add_scalar(f"{args.exp}/test_dice", stats["dice_mean"], iter_num)
            writer.add_scalar(f"{args.exp}/test_iou", stats["iou_mean"], iter_num)
            writer.add_scalar(f"{args.exp}/test_hd95", stats["hd95_mean"], iter_num)
            writer.add_scalar(f"{args.exp}/test_asd", stats["asd_mean"], iter_num)
            writer.flush()

            # Switch back to train mode
            model.train()

    writer.close()
    return "Few-shot UNet Segmentation Training Finished!"


@torch.no_grad()
def test(args):
    logging.info("Start testing")
    # Model initialization
    model = prepare_model(args)
    saved_best = os.path.join(args.train_snapshot_path, "best_model.pth")
    model.load_state_dict(torch.load(saved_best))
    model.eval()

    # Data initialization (use full test set)
    dm = dataset_seg.DataModule(args)
    testloader = dm.test_dataloader()

    # Create visualization folder
    viz_path = args.test_snapshot_path + "/viz"
    if os.path.exists(viz_path):
        shutil.rmtree(viz_path)
    os.makedirs(viz_path)

    accumulator = MetricAccumulator(type="seg", criterion=criterion, num_classes=args.num_classes)
    with torch.no_grad():
        for _, sampled_batch in enumerate(tqdm(testloader, ncols=70)):
            images, labels, file_names = sampled_batch
            images, labels = images.to(args.device), labels.to(args.device)
            preds = model(images)
            accumulator.update(preds.detach(), labels.detach())
            visualize_seg(images, labels, preds, file_names, viz_path)

    # Calculate metric statistics
    stats = accumulator.compute()
    accumulator.reset()

    df = pd.DataFrame(
        {
            "Metric": ["Dice", "IoU", "HD95", "ASD"],
            "Mean": [stats["dice_mean"] * 100, stats["iou_mean"] * 100, stats["hd95_mean"], stats["asd_mean"]],
            "Std": [stats["dice_std"] * 100, stats["iou_std"] * 100, stats["hd95_std"], stats["asd_std"]],
        }
    )

    result_str = f"\n{'=' * 50}\n"
    result_str += df.to_string(index=False, float_format="%.2f") + "\n"
    result_str += f"{'=' * 50}\n"
    logging.info(result_str)

    # Create a sub-folder for single test with time and IoU
    backup_folder = os.path.join(
        args.test_snapshot_path,
        f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_iou={stats['iou_mean'] * 100:.2f}",
    )
    os.makedirs(backup_folder)

    # Save results as CSV
    csv_path = os.path.join(backup_folder, "results.csv")
    df.to_csv(csv_path, index=False, float_format="%.2f")
    logging.info(f"Results saved to: {csv_path}")

    # Backup the best model, log, and visualization
    shutil.copy(saved_best, os.path.join(backup_folder, "best_model.pth"))
    shutil.move(viz_path, os.path.join(backup_folder, "viz"))
    shutil.move(os.path.join(args.test_snapshot_path, "log.log"), os.path.join(backup_folder, "log.log"))

    return "Testing Finished!"


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Logging
    snapshot_path_list = [
        f"runs/{args.exp}/{args.dataset}/train",
        f"runs/{args.exp}/{args.dataset}/test",
    ]
    for path in snapshot_path_list:
        if not os.path.exists(path):
            os.makedirs(path)

    args.train_snapshot_path = snapshot_path_list[0]
    args.test_snapshot_path = snapshot_path_list[1]

    if not args.test:
        setup_logging(args, args.train_snapshot_path)
        train(args)

    # Test
    setup_logging(args, args.test_snapshot_path)
    test(args)


if __name__ == "__main__":
    main()
