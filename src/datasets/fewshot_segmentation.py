"""
Few-shot Segmentation Dataset Module

This module provides data loading utilities for few-shot learning experiments
where only a small percentage of training data is available.
"""

import random
from torch.utils.data import DataLoader
from .segmentation import USDataset


class FewShotDataModule:
    """
    Data module for few-shot segmentation experiments.

    Supports ratio-based sampling: Use a percentage of training data (e.g., 1%, 5%, 10%)

    Args:
        args: Arguments containing dataset configuration
        train_ratio: Percentage of training data to use (0.0-1.0)
    """

    def __init__(self, args, train_ratio=None):
        self.args = args
        self.train_ratio = train_ratio

        # Load all image names
        train_image_names, val_image_names, test_image_names = [], [], []
        with open(f"../data/NextGen-UIA/segmentation/{self.args.dataset}/train.txt", "r") as f:
            train_image_names = f.read().splitlines()
        with open(f"../data/NextGen-UIA/segmentation/{self.args.dataset}/val.txt", "r") as f:
            val_image_names = f.read().splitlines()
        with open(f"../data/NextGen-UIA/segmentation/{self.args.dataset}/test.txt", "r") as f:
            test_image_names = f.read().splitlines()

        # Sample training data
        sampled_train_names = self._sample_training_data(train_image_names)

        # Log sampling info
        print("  Few-shot sampling:")
        print(f"    - Original training samples: {len(train_image_names)}")
        print(f"    - Sampled training samples: {len(sampled_train_names)}")
        if self.train_ratio is not None:
            print(f"    - Train ratio: {self.train_ratio * 100:.1f}%")

        # Initialize datasets
        self.train_dataset = USDataset(self.args, sampled_train_names, "train")
        self.val_dataset = USDataset(self.args, val_image_names, "val")
        self.test_dataset = USDataset(self.args, test_image_names, "test")

        # Store original count for reference
        self.original_train_count = len(train_image_names)
        self.sampled_train_count = len(sampled_train_names)

    def _sample_training_data(self, train_image_names):
        """
        Sample training data based on ratio.
        """
        if self.train_ratio is not None and self.train_ratio < 1.0:
            n_samples = max(1, int(len(train_image_names) * self.train_ratio))
            sampled = random.sample(train_image_names, n_samples)
            random.shuffle(sampled)
            return sampled
        else:
            return train_image_names

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=min(self.args.batch_size, len(self.train_dataset)),
            num_workers=self.args.num_workers,
            persistent_workers=True if self.args.num_workers > 0 else False,
            shuffle=True,
            drop_last=len(self.train_dataset) > self.args.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=True if self.args.num_workers > 0 else False,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=True if self.args.num_workers > 0 else False,
            shuffle=False,
            drop_last=False,
        )
