import logging
import os
from pathlib import Path
import sys
import gc

import PIL.Image as Image
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ROC
from monai.metrics import (
    compute_dice as _dice,
    compute_iou as _iou,
    compute_hausdorff_distance as _hd95,
    compute_average_surface_distance as _asd,
    PSNRMetric,
    SSIMMetric,
)
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"

_ssim = SSIMMetric(spatial_dims=2, reduction="none")
_psnr = PSNRMetric(max_val=1.0, reduction="none")

_acc = Accuracy(task="binary").to(device)
_pre = Precision(task="binary").to(device)
_rec = Recall(task="binary").to(device)
_f1 = F1Score(task="binary").to(device)
_auc = AUROC(task="binary").to(device)
_roc = ROC(task="binary").to(device)


def setup_logging(args, log_path):
    """Setup unified logging configuration"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    os.makedirs(log_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_path, "log.log"),
        filemode="w",
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


def get_logger(name=None):
    """Get a logger instance for consistent logging"""
    return logging.getLogger(name)


def format_params(num):
    if num >= 1e6:
        return f"{num / 1e6:.1f} M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f} K"
    else:
        return str(num)


def model_summary(model_dict):
    summary_data = []

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for name, model in model_dict.items():
        model_params = sum(p.numel() for p in model.parameters())
        trainable_params_for_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params_for_model = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        total_params += model_params
        trainable_params += trainable_params_for_model
        non_trainable_params += non_trainable_params_for_model

        mode = "train" if trainable_params_for_model > 0 else "eval"

        summary_data.append(
            [
                name,
                model.__class__.__name__,
                format_params(trainable_params_for_model),
                format_params(non_trainable_params_for_model),
                format_params(model_params),
                mode,
            ]
        )

    df = pd.DataFrame(
        summary_data, columns=["Name", "Type", "Trainable Params", "Non-trainable Params", "Total Params", "Mode"]
    ).T

    return_str = f"\n{'=' * 50}\n"
    return_str += df.to_string(header=False) + "\n"
    return_str += f"{'=' * 50}\n"
    return return_str


class MetricAccumulator:
    """
    Accumulate metrics for a given type of task
    type: "seg" | "cls" | "recon"
    """

    def __init__(self, type="seg", criterion=None, num_classes=2):
        self.type = type
        self.criterion = criterion
        self.num_classes = num_classes
        self.all_preds = None
        self.all_labels = None

        if self.type == "seg":
            self.dice_list = np.array([])
            self.iou_list = np.array([])
            self.hd95_list = np.array([])
            self.asd_list = np.array([])
            self.seg_loss_list = np.array([])

    def update(self, preds, labels):
        if self.type == "seg":
            metrics = self._compute_segmentation_metrics_by_batch(preds, labels)
            self.dice_list = np.append(self.dice_list, metrics["dice"])
            self.iou_list = np.append(self.iou_list, metrics["iou"])
            self.hd95_list = np.append(self.hd95_list, metrics["hd95"])
            self.asd_list = np.append(self.asd_list, metrics["asd"])
            self.seg_loss_list = np.append(self.seg_loss_list, metrics["loss"])
        else:
            if self.all_preds is None:
                self.all_preds = preds
                self.all_labels = labels
            else:
                if len(preds.shape) == 1:
                    preds = preds.unsqueeze(0)
                self.all_preds = torch.cat([self.all_preds, preds], dim=0)
                self.all_labels = torch.cat([self.all_labels, labels], dim=0)

    def compute(self):
        if self.type == "seg":
            dice_mean, dice_std = (
                np.mean(self.dice_list[np.isfinite(self.dice_list)]),
                np.std(self.dice_list[np.isfinite(self.dice_list)]),
            )
            iou_mean, iou_std = (
                np.mean(self.iou_list[np.isfinite(self.iou_list)]),
                np.std(self.iou_list[np.isfinite(self.iou_list)]),
            )
            hd95_mean, hd95_std = (
                np.mean(self.hd95_list[np.isfinite(self.hd95_list)]),
                np.std(self.hd95_list[np.isfinite(self.hd95_list)]),
            )
            asd_mean, asd_std = (
                np.mean(self.asd_list[np.isfinite(self.asd_list)]),
                np.std(self.asd_list[np.isfinite(self.asd_list)]),
            )
            loss_mean = np.mean(self.seg_loss_list[np.isfinite(self.seg_loss_list)])

            return {
                "dice_mean": dice_mean,
                "dice_std": dice_std,
                "iou_mean": iou_mean,
                "iou_std": iou_std,
                "hd95_mean": hd95_mean,
                "hd95_std": hd95_std,
                "asd_mean": asd_mean,
                "asd_std": asd_std,
                "loss": loss_mean,
            }

        elif self.type == "cls":
            return self._compute_classification_metrics(self.all_preds, self.all_labels)
        elif self.type == "recon":
            return self._compute_reconstruction_metrics(self.all_preds, self.all_labels)
        else:
            raise ValueError(f"Invalid metric type: {self.type}")

    def _compute_segmentation_metrics_by_batch(self, preds, labels):
        loss = self.criterion(preds.float(), labels.float()).item()

        preds = F.one_hot(torch.argmax(preds, dim=1), num_classes=self.num_classes)
        preds = preds.permute(0, 3, 1, 2).float()

        dice = _dice(preds, labels, include_background=False).cpu().numpy()
        iou = _iou(preds, labels, include_background=False).cpu().numpy()
        hd95 = _hd95(preds, labels, include_background=False, percentile=95).cpu().numpy()
        asd = _asd(preds, labels, include_background=False).cpu().numpy()

        # To prevent GPU memory leak when compute MONAI hausdorff distance and ASD
        # https://github.com/Project-MONAI/MONAI/issues/7480#issuecomment-2010472963
        gc.collect()

        return {
            "dice": dice,
            "iou": iou,
            "hd95": hd95,
            "asd": asd,
            "loss": loss,
        }

    def _compute_classification_metrics(self, preds, labels):
        loss = self.criterion(preds.float(), labels.long()).item()

        probs = torch.softmax(preds, dim=1)[:, 1]

        acc = _acc(probs, labels).item()
        pre = _pre(probs, labels).item()
        rec = _rec(probs, labels).item()
        f1 = _f1(probs, labels).item()
        auc = _auc(probs, labels).item()

        return {
            "acc": acc,
            "rec": rec,
            "pre": pre,
            "f1": f1,
            "auc": auc,
            "loss": loss,
        }

    def _compute_reconstruction_metrics(self, preds, labels):
        loss = self.criterion(preds.float(), labels.float()).item()

        # Limit the range of predictions and labels to [0,1]
        preds = torch.clamp(preds, 0.0, 1.0)
        labels = torch.clamp(labels, 0.0, 1.0)

        ssim = _ssim(preds, labels).cpu().numpy()
        psnr = _psnr(preds, labels).cpu().numpy()

        ssim_mean, ssim_std = np.mean(ssim), np.std(ssim)
        psnr_mean, psnr_std = np.mean(psnr), np.std(psnr)

        return {
            "ssim_mean": ssim_mean,
            "ssim_std": ssim_std,
            "psnr_mean": psnr_mean,
            "psnr_std": psnr_std,
            "loss": loss,
        }

    def reset(self):
        self.all_preds = None
        self.all_labels = None

        if self.type == "seg":
            self.dice_list = np.array([])
            self.iou_list = np.array([])
            self.hd95_list = np.array([])
            self.asd_list = np.array([])
            self.seg_loss_list = np.array([])


def visualize_pretrain(images, labels, model, viz_path=None, names=None):
    images, labels = images.to(device), labels.to(device)
    model.eval()
    with torch.no_grad():
        loss, pred, mask = model(images)
        recon_image = model.unpatchify(pred)

        # Visualize the results
        assert viz_path is not None, "Visualization path not provided during testing"
        assert names is not None, "File names not provided during visualization"
        # Save the reconstructed images
        for i, name in enumerate(names):
            Image.fromarray((recon_image[i][0].cpu().numpy() * 255).astype(np.uint8), mode="L").save(
                f"{viz_path}/{os.path.splitext(name)[0]}.png", mode="L"
            )


def visualize_seg(images, labels, preds, file_names, viz_path):
    """
    Visualize the segmentation results
    Args:
        images: torch.Tensor, shape: (B, C, H, W)
        labels: torch.Tensor, shape: (B, C, H, W), single-channel binary mask
        preds: torch.Tensor, shape: (B, C, H, W), must be one-hot encoding
        file_names: List[str],
        viz_path: str, path to save the visualization images
    """
    if preds.shape[1] > 1:
        preds = torch.argmax(preds, dim=1)

    for i, file_name in enumerate(file_names):
        # Convert to numpy, scale to 0-255 and convert to uint8
        image_i = (images[i].cpu().numpy() * 255).astype(np.uint8)
        labels_i = (labels[i].cpu().numpy() * 255).astype(np.uint8)
        preds_i = (preds[i].cpu().numpy() * 255).astype(np.uint8)

        # Create RGB image with Red - Ground truth, Green - Predicted
        rgb_image = np.zeros((image_i.shape[1], image_i.shape[2], 3), dtype=np.uint8)
        rgb_image[:, :, 0] = np.maximum(rgb_image[:, :, 0], labels_i)
        rgb_image[:, :, 1] = np.maximum(rgb_image[:, :, 1], preds_i)

        # Get the file basename
        file_basename = str(Path(file_name).stem)

        # Save the overlay images with ground truth and predicted images, but without input
        Image.fromarray(rgb_image).save(f"{viz_path}/{file_basename}.png", mode="RGB")

        # Save the overlay images with input, ground truth and predicted images
        rgb_image[:, :, 0] = np.maximum(image_i[0], labels_i)
        rgb_image[:, :, 1] = np.maximum(image_i[0], preds_i)
        rgb_image[:, :, 2] = image_i[0]
        Image.fromarray(rgb_image).save(f"{viz_path}/{file_basename}_overlay.png", mode="RGB")

        # Save only the predicted images (binary mask)
        Image.fromarray(preds_i.squeeze()).save(f"{viz_path}/{file_basename}_pred.png", mode="L")


def plot_roc_curve(probs, all_labels, iter_num, save_dir):
    """
    Generate and save ROC curve for current model state

    Args:
        all_preds: torch.Tensor, shape: (B, N), logits
        all_labels: torch.Tensor, shape: (N,), one-hot encoding
        iter_num: current iteration number (for filename), any number less than 1 will not include it in the filename
        save_dir: directory to save the ROC curve plot
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get ROC curve data
    probs = torch.softmax(probs, dim=1)[:, 1]
    fpr, tpr, thresholds = _roc(probs, all_labels)

    # Calculate AUC
    auc = _auc(probs, all_labels).item()

    # Create figure
    fig = plt.figure(figsize=(4, 4), dpi=600)
    ax = fig.add_subplot(111)
    ax.plot(fpr.cpu().numpy(), tpr.cpu().numpy())
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.5)
    ax.set_title(f"AUC = {auc:.4f}")

    if iter_num > 0:
        fig_save_path = os.path.join(save_dir, f"roc_curve_iter_{iter_num}.png")
    else:
        fig_save_path = os.path.join(save_dir, "roc_curve.png")

    fig.savefig(fig_save_path, bbox_inches="tight")

    return fig, auc
