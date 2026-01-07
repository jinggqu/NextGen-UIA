"""
Few-shot Classification Dataset Module

This module provides data loading utilities for few-shot learning experiments
where only a small percentage of training data is available.
"""

import csv
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from .classification import USDataset


class FewShotDataModule:
    """
    Data module for few-shot classification experiments.

    Supports two sampling strategies:
    1. Ratio-based: Use a percentage of training data (e.g., 1%, 5%, 10%)
    2. Shot-based: Use K samples per class (e.g., 1-shot, 5-shot, 10-shot)

    Args:
        args: Arguments containing dataset configuration
        train_ratio: Percentage of training data to use (0.0-1.0)
        shots_per_class: Number of samples per class (overrides train_ratio if set)
        stratified: Whether to maintain class balance when sampling by ratio
    """

    def __init__(self, args, train_ratio=None, shots_per_class=None, stratified=True):
        self.args = args
        self.train_ratio = train_ratio
        self.shots_per_class = shots_per_class
        self.stratified = stratified

        # Load labels for stratified sampling
        self.label_dict = {}
        with open(f"../data/NextGen-UIA/classification/{self.args.dataset}/labels.csv", "r") as file:
            reader = csv.reader(file)
            self.label_dict = {str(row[0]): int(row[1]) for row in reader}

        # Load all image names
        train_image_names, val_image_names, test_image_names = [], [], []
        with open(f"../data/NextGen-UIA/classification/{self.args.dataset}/train.txt", "r") as f:
            train_image_names = f.read().splitlines()
        with open(f"../data/NextGen-UIA/classification/{self.args.dataset}/val.txt", "r") as f:
            val_image_names = f.read().splitlines()
        with open(f"../data/NextGen-UIA/classification/{self.args.dataset}/test.txt", "r") as f:
            test_image_names = f.read().splitlines()

        # Sample training data
        sampled_train_names = self._sample_training_data(train_image_names)

        # Log sampling info
        print("  Few-shot sampling:")
        print(f"    - Original training samples: {len(train_image_names)}")
        print(f"    - Sampled training samples: {len(sampled_train_names)}")
        if self.shots_per_class is not None:
            print(f"    - Shots per class: {self.shots_per_class}")
        elif self.train_ratio is not None:
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
        Sample training data based on ratio or shots_per_class.
        """
        if self.shots_per_class is not None:
            return self._sample_k_shot(train_image_names)
        elif self.train_ratio is not None:
            if self.stratified:
                return self._sample_stratified_ratio(train_image_names)
            else:
                return self._sample_random_ratio(train_image_names)
        else:
            return train_image_names

    def _sample_k_shot(self, train_image_names):
        """
        Sample K samples per class (K-shot learning).
        """
        # Group images by class
        class_to_images = defaultdict(list)
        for img_name in train_image_names:
            label = self.label_dict.get(img_name, 0)
            class_to_images[label].append(img_name)

        # Sample K images from each class
        sampled = []
        for label, images in class_to_images.items():
            k = min(self.shots_per_class, len(images))
            sampled.extend(random.sample(images, k))

        random.shuffle(sampled)
        return sampled

    def _sample_stratified_ratio(self, train_image_names):
        """
        Sample a percentage of training data while maintaining class balance.
        """
        # Group images by class
        class_to_images = defaultdict(list)
        for img_name in train_image_names:
            label = self.label_dict.get(img_name, 0)
            class_to_images[label].append(img_name)

        # Sample proportionally from each class
        sampled = []
        for label, images in class_to_images.items():
            n_samples = max(1, int(len(images) * self.train_ratio))
            sampled.extend(random.sample(images, n_samples))

        random.shuffle(sampled)
        return sampled

    def _sample_random_ratio(self, train_image_names):
        """
        Randomly sample a percentage of training data (may not preserve class balance).
        """
        n_samples = max(1, int(len(train_image_names) * self.train_ratio))
        sampled = random.sample(train_image_names, n_samples)
        random.shuffle(sampled)
        return sampled

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
