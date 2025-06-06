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
import dataset_seg
from models.openai import clip
from models.openai.clip_adapter import CLIPAdapter
from monai.losses import DiceCELoss
from tools import MetricAccumulator, model_summary, setup_logging, visualize_seg


def get_args():
    """Get arguments from command line"""

    parser = argparse.ArgumentParser("Adaptation of Visual Foundation Model for Medical Ultrasound Image Analysis")

    # Data related
    parser.add_argument("--exp", type=str, default="clip_mona_seg")
    parser.add_argument("--dataset", type=str, default="LN-1", help="Dataset name")
    parser.add_argument("--img_size", type=int, default=224, help="Image width and height")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--num_workers", type=int, default=8)

    # Augmentation related
    parser.add_argument("--strong_augs", default=True, action=argparse.BooleanOptionalAction, help="Use strong augs")
    parser.add_argument("--weak_augs", default=True, action=argparse.BooleanOptionalAction, help="Use weak augs")

    # Model related
    # Available models: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    parser.add_argument("--version", type=str, default="ViT-B/16")
    parser.add_argument("--ckpt", type=str, default="ckpt/ViT-B-16.pt")
    parser.add_argument("--finetuned_weights", type=str, default=None)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--reduce_dim", type=int, default=512)

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
    parser.add_argument("--caption_type", type=str, choices=["refined", "truncated"])

    # Testing related
    parser.add_argument("--test", default=False, action="store_true", help="Load local checkpoint for testing")
    return parser.parse_args()


# Loss functions
criterion = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=1e-8, smooth_dr=1e-8)


def prepare_model(args):
    # Build model and load official CLIP pretrained weights
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

    # Create CLIPAdapter for segmentation
    adapter = CLIPAdapter(
        clip_model=clip_model,
        extract_layers=[3, 6, 9],  # Extract features from these transformer layers
        reduce_dim=args.reduce_dim,
        num_classes=args.num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        task="seg",
    )

    # Ensure model is on the correct device
    adapter.to(args.device)
    adapter.freeze_clip_backbone()

    return adapter


def train(args):
    # Model initialization
    model = prepare_model(args)
    model.train()
    logging.info(model_summary({"model": model}))
    writer = SummaryWriter(args.train_snapshot_path + "/log")
    logging.info("Start training")

    # Data initialization
    dm = dataset_seg.DataModule(args)
    trainloader, valloader = dm.train_dataloader(), dm.val_dataloader()

    # Optimizer initialization
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    max_epoch = args.epochs
    max_iters = len(trainloader) * max_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=args.lr_min)

    iter_num = 0
    best_val_dice = 0.0
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

        # Validation (every 10 epochs)
        if epoch > 0 and epoch % 10 == 0:
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
                best_val_dice = stats["dice_mean"]
                save_best = os.path.join(args.train_snapshot_path, "adapter_components.pth")
                adapter_state_dict = {
                    "reduces": model.reduces.state_dict(),
                    "blocks": model.blocks.state_dict(),
                    "seg_head": model.seg_head.state_dict(),
                }
                mona_state_dict = {}
                for name, param in model.named_parameters():
                    if "mona" in name:
                        mona_state_dict[name] = param.data
                adapter_state_dict["mona"] = mona_state_dict
                torch.save(adapter_state_dict, save_best)

            logging.info(
                f"\titer: {iter_num}, loss: {stats['loss']:.4f}, dice: {stats['dice_mean'] * 100:.2f}, "
                f"iou: {stats['iou_mean'] * 100:.2f}, hd95: {stats['hd95_mean']:.2f}, asd: {stats['asd_mean']:.2f}"
            )

            # Switch back to train mode
            model.train()

    writer.close()
    torch.cuda.empty_cache()
    return "Segmentation Training Finished!"


@torch.no_grad()
def test(args):
    print("Start testing")
    # Model initialization
    model = prepare_model(args)
    saved_best = os.path.join(args.train_snapshot_path, "adapter_components.pth")
    adapter_state_dict = torch.load(saved_best)

    # Load the adapter components
    model.reduces.load_state_dict(adapter_state_dict["reduces"])
    model.blocks.load_state_dict(adapter_state_dict["blocks"])
    model.seg_head.load_state_dict(adapter_state_dict["seg_head"])

    # Load the mona components
    mona_state_dict = adapter_state_dict["mona"]
    for name, param in model.named_parameters():
        if "mona" in name:
            param.data = mona_state_dict[name]

    model.eval()

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

    result_str = "\n" + "=" * 40 + "\n"
    result_str += df.to_string(index=False, float_format="%.2f") + "\n"
    result_str += "=" * 40 + "\n"
    logging.info(result_str)

    # Create a sub-folder for single test with time and IoU
    backup_folder = os.path.join(
        args.test_snapshot_path,
        f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_iou={stats['iou_mean'] * 100:.2f}",
    )
    os.makedirs(backup_folder)

    # Backup the best model, log, and visualization
    shutil.copy(saved_best, os.path.join(backup_folder, "adapter_components.pth"))
    shutil.move(viz_path, os.path.join(backup_folder, "viz"))
    shutil.move(os.path.join(args.test_snapshot_path, "log.txt"), os.path.join(backup_folder, "log.txt"))

    return "Testing Finished!"


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_name = f"{args.exp}"
    model_name = model_name + "_" + ("raw" if args.finetuned_weights is None else "mona")
    model_name = model_name + "_" + ("na" if args.caption_type is None else args.caption_type)
    args.model_name = model_name

    # Logging
    snapshot_path_list = [
        f"runs/{args.exp}/{args.dataset}/train/{model_name}",
        f"runs/{args.exp}/{args.dataset}/test/{model_name}",
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
