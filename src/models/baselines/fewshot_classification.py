"""
Few-shot Classification with ResNet Baseline

This script performs classification with limited training data to evaluate
the data efficiency of traditional ResNet compared to CLIP-based methods.
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
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.datasets.fewshot_classification import FewShotDataModule
from src.datasets import classification as dataset_cls
from monai.losses import FocalLoss
from matplotlib import pyplot as plt
from src.utils.tools import MetricAccumulator, model_summary, setup_logging, plot_roc_curve


def get_args():
    """Get arguments from command line"""

    parser = argparse.ArgumentParser("Few-shot Classification with ResNet")

    # Data related
    parser.add_argument("--exp", type=str, default="fewshot_resnet_cls")
    parser.add_argument("--dataset", type=str, default="LN-INT", help="Dataset name")
    parser.add_argument("--img_size", type=int, default=224, help="Image width and height")
    parser.add_argument("--num_workers", type=int, default=8)

    # Few-shot related
    parser.add_argument("--train_ratio", type=float, default=0.1, help="Percentage of training data to use (0.0-1.0)")
    parser.add_argument(
        "--shots_per_class", type=int, default=None, help="Number of samples per class (overrides train_ratio)"
    )

    # Augmentation related
    parser.add_argument("--strong_augs", default=True, action=argparse.BooleanOptionalAction, help="Use strong augs")
    parser.add_argument("--weak_augs", default=True, action=argparse.BooleanOptionalAction, help="Use weak augs")

    # Model related
    # Available models: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    parser.add_argument("--version", type=str, default="resnet18")
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
criterion = FocalLoss(to_onehot_y=True)


def prepare_model(args):
    torch.hub.set_dir("./ckpt/")

    if args.version == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif args.version == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif args.version == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif args.version == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif args.version == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    else:
        raise ValueError(f"Invalid model version: {args.version}")

    model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
    model.to(args.device)

    return model


def train(args):
    # Model initialization
    model = prepare_model(args)
    model.train()
    logging.info(model_summary({"model": model}))
    writer = SummaryWriter(args.train_snapshot_path + "/log")
    logging.info("Start few-shot training (ResNet)")

    # Data initialization with few-shot sampling
    dm = FewShotDataModule(args, train_ratio=args.train_ratio, shots_per_class=args.shots_per_class)
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
    best_val_acc = 0.0
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
            accumulator = MetricAccumulator(type="cls", criterion=criterion, num_classes=args.num_classes)
            with torch.no_grad():
                for _, sampled_batch in enumerate(valloader):
                    images, labels, _ = sampled_batch
                    images, labels = images.to(args.device), labels.to(args.device)
                    preds = model(images)
                    accumulator.update(preds.detach(), labels.detach())

            # Log roc curve for visualization
            roc_dir = os.path.join(args.train_snapshot_path, "roc_curves")
            fig, auc = plot_roc_curve(accumulator.all_preds, accumulator.all_labels, iter_num, roc_dir)
            writer.add_figure(f"{args.exp}/val_roc_curve", fig, iter_num)
            writer.add_scalar(f"{args.exp}/val_auc", auc, iter_num)
            writer.flush()
            plt.close(fig)

            # Calculate metrics statistics
            stats = accumulator.compute()
            accumulator.reset()

            writer.add_scalar(f"{args.exp}/val_loss", stats["loss"], iter_num)
            writer.add_scalar(f"{args.exp}/val_acc", stats["acc"], iter_num)
            writer.add_scalar(f"{args.exp}/val_pre", stats["pre"], iter_num)
            writer.add_scalar(f"{args.exp}/val_rec", stats["rec"], iter_num)
            writer.add_scalar(f"{args.exp}/val_f1", stats["f1"], iter_num)
            writer.add_scalar(f"{args.exp}/val_auc", stats["auc"], iter_num)
            writer.flush()

            # Save best model
            if stats["acc"] > best_val_acc:
                patience_counter = 0
                best_val_acc = stats["acc"]
                save_best = os.path.join(args.train_snapshot_path, "best_model.pth")
                torch.save(model.state_dict(), save_best)
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logging.info(f"\nEarly stopping at epoch {epoch + 1}")
                break

            logging.info(
                f"\titer: {iter_num}, loss: {stats['loss']:.4f}, acc: {stats['acc'] * 100:.2f}, "
                f"rec: {stats['rec'] * 100:.2f}, pre: {stats['pre'] * 100:.2f}, f1: {stats['f1'] * 100:.2f}, "
                f"auc: {stats['auc'] * 100:.2f}"
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
            writer.add_scalar(f"{args.exp}/test_acc", stats["acc"], iter_num)
            writer.add_scalar(f"{args.exp}/test_pre", stats["pre"], iter_num)
            writer.add_scalar(f"{args.exp}/test_rec", stats["rec"], iter_num)
            writer.add_scalar(f"{args.exp}/test_f1", stats["f1"], iter_num)
            writer.add_scalar(f"{args.exp}/test_auc", stats["auc"], iter_num)
            writer.flush()

            # Switch back to train mode
            model.train()

    writer.close()
    return "Few-shot ResNet Classification Training Finished!"


@torch.no_grad()
def test(args):
    logging.info("Start testing")
    # Model initialization
    model = prepare_model(args)
    saved_best = os.path.join(args.train_snapshot_path, "best_model.pth")
    model.load_state_dict(torch.load(saved_best))
    model.eval()

    # Data initialization (use full test set)
    dm = dataset_cls.DataModule(args)
    testloader = dm.test_dataloader()

    # Create visualization folder
    viz_path = args.test_snapshot_path + "/viz"
    if os.path.exists(viz_path):
        shutil.rmtree(viz_path)
    os.makedirs(viz_path)

    accumulator = MetricAccumulator(type="cls", criterion=criterion, num_classes=args.num_classes)
    with torch.no_grad():
        for _, sampled_batch in enumerate(tqdm(testloader, ncols=70)):
            images, labels, _ = sampled_batch
            images, labels = images.to(args.device), labels.to(args.device)
            preds = model(images)
            accumulator.update(preds.detach(), labels.detach())

    # Calculate metric statistics
    stats = accumulator.compute()

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

    # Create a sub-folder for single test with time and acc
    backup_folder = os.path.join(
        args.test_snapshot_path,
        f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_acc={stats['acc'] * 100:.2f}",
    )
    os.makedirs(backup_folder)

    # Save results as CSV
    csv_path = os.path.join(backup_folder, "results.csv")
    df.to_csv(csv_path, index=False, float_format="%.2f")
    logging.info(f"Results saved to: {csv_path}")

    # Plot ROC curve
    fig, _ = plot_roc_curve(accumulator.all_preds, accumulator.all_labels, -1, backup_folder)
    plt.close(fig)

    # Clear accumulator after plotting
    accumulator.reset()

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
