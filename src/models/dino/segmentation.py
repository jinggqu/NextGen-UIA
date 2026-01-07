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
from src.datasets import segmentation as dataset_seg
from src.third_party.dino.dinov2 import DINOV2Encoder, load_pretrained_weights, setup_decoders
from src.third_party.dino import vision_transformer as vit
from monai.losses import DiceCELoss
from src.utils.tools import MetricAccumulator, model_summary, setup_logging, visualize_seg


def get_args():
    """Get arguments from command line"""

    parser = argparse.ArgumentParser("Adaptation of Visual Foundation Model for Medical Ultrasound Image Analysis")

    # Data related
    parser.add_argument("--exp", type=str, default="dino_seg")
    parser.add_argument("--dataset", type=str, default="LN-INT", help="Dataset name")
    parser.add_argument("--img_size", type=int, default=518, help="Image width and height")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size")
    parser.add_argument("--num_workers", type=int, default=8)

    # Augmentation related
    parser.add_argument("--strong_augs", default=True, action=argparse.BooleanOptionalAction, help="Use strong augs")
    parser.add_argument("--weak_augs", default=True, action=argparse.BooleanOptionalAction, help="Use weak augs")

    # Model related
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--decoder_type", type=str, default="unet")

    # Training related
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=24)
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
    # Build model
    model = vit.__dict__["vit_base"](img_size=args.img_size, patch_size=args.patch_size)
    n_last_blocks = 5 if args.decoder_type == "unet" else 1
    feature_model = DINOV2Encoder(model, n_last_blocks=n_last_blocks)
    pretrained_weights = "ckpt/dinov2_vitb14_pretrain.pth"
    checkpoint_key = "student"
    load_pretrained_weights(feature_model, pretrained_weights, checkpoint_key)

    for param in feature_model.parameters():
        param.requires_grad = False

    decoders, optim_param_groups = setup_decoders(
        embed_dim=768,
        learning_rates=[args.lr],  # Temporarily only one learning rate
        num_classes=args.num_classes,
        decoder_type=args.decoder_type,
        image_size=args.img_size,
        patch_size=args.patch_size,
    )

    return feature_model.to(args.device), decoders, optim_param_groups


def train(args):
    # Model initialization
    feature_model, decoders, optim_param_groups = prepare_model(args)
    feature_model.eval()
    decoders.train()
    logging.info(model_summary({"feature_model": feature_model, "decoders": decoders}))

    writer = SummaryWriter(args.train_snapshot_path + "/log")
    logging.info("Start training")

    # Data initialization
    dm = dataset_seg.DataModule(args)
    trainloader, valloader, testloader = dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()

    # Optimizer initialization
    optimizer = torch.optim.AdamW(
        optim_param_groups,
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
            with torch.no_grad():
                features = feature_model(images)
            outputs = decoders(features)
            name = list(outputs.keys())[0]
            preds = outputs[name]

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
            decoders.eval()
            accumulator = MetricAccumulator(type="seg", criterion=criterion, num_classes=args.num_classes)
            with torch.no_grad():
                for _, sampled_batch in enumerate(valloader):
                    images, labels, _ = sampled_batch
                    images, labels = images.to(args.device), labels.to(args.device)
                    features = feature_model(images)
                    outputs = decoders(features)
                    name = list(outputs.keys())[0]
                    preds = outputs[name]
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
                torch.save(decoders.state_dict(), save_best)

            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logging.info(f"\nEarly stopping at epoch {epoch + 1}")
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
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
                    features = feature_model(images)
                    outputs = decoders(features)
                    name = list(outputs.keys())[0]
                    preds = outputs[name]
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
            decoders.train()

    writer.close()
    return "Segmentation Training Finished!"


@torch.no_grad()
def test(args):
    logging.info("Start testing")
    # Model initialization
    feature_model, decoders, _ = prepare_model(args)
    saved_best = os.path.join(args.train_snapshot_path, "best_model.pth")
    decoders.load_state_dict(torch.load(saved_best, weights_only=True), strict=False)
    decoders.eval()

    # Data initialization
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
            features = feature_model(images)
            outputs = decoders(features)
            name = list(outputs.keys())[0]
            preds = outputs[name]
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
