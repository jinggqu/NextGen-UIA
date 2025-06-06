import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
from PIL import ImageOps, ImageFilter, ImageEnhance


# region Augmentation
def img_aug_identity(img, scale=None):
    return img


def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)


def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)


def img_aug_blur(img, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def img_aug_contrast(img, scale=[0.5, 1.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Contrast(img).enhance(v)


def img_aug_brightness(img, scale=[0.5, 1.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Brightness(img).enhance(v)


def img_aug_sharpness(img, scale=[0.5, 2.0]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Sharpness(img).enhance(v)


def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.posterize(img, v)


def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.solarize(img, v)


class RandomResizedCrop(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, image):
        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.8, 1.2), ratio=(1.0, 1.0))
        image = F.resized_crop(image, i, j, h, w, size=(self.img_size, self.img_size))
        return image


class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image = F.hflip(image)
        return image


class RandomVerticalFlip(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image = F.vflip(image)
        return image


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return image


def get_strong_aug_list():
    op_list = [
        (img_aug_identity, None),
        (img_aug_autocontrast, None),
        (img_aug_equalize, None),
        (img_aug_blur, [0.75, 1.25]),
        (img_aug_contrast, [0.75, 1.25]),
        (img_aug_brightness, [0.75, 1.25]),
        (img_aug_sharpness, [0.75, 1.25]),
        (img_aug_posterize, [4, 8]),
        (img_aug_solarize, [1, 256]),
    ]
    return op_list


class StrongAugmentation:
    def __init__(self):
        self.augment_list = get_strong_aug_list()
        self.total_ops = len(self.augment_list)

    def __call__(self, img):
        max_num = np.random.randint(0, high=self.total_ops + 1)
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            img = op(img, scales)
        return img


class WeakAugmentation(object):
    def __init__(self, args):
        self.augment_list = [
            RandomResizedCrop(args.img_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            Identity(),
        ]
        self.total_ops = len(self.augment_list)

    def __call__(self, image):
        max_num = np.random.randint(0, high=self.total_ops + 1)
        ops = random.choices(self.augment_list, k=max_num)
        for op in ops:
            image = op(image)
        return image


# endregion Augmentation


class USDataset(Dataset):
    def __init__(self, args, jsonl, dataloader_type):
        self.args = args
        self.jsonl = jsonl
        self.strong_aug = StrongAugmentation()
        self.weak_aug = WeakAugmentation(args)
        self.dataloader_type = dataloader_type
        self.transform = T.Compose(
            [
                T.Resize(args.img_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(args.img_size),
                T.ToTensor(),
                T.ConvertImageDtype(torch.float32),
            ]
        )

    def __len__(self):
        return len(self.jsonl)

    def __getitem__(self, idx):
        line_dict = self.jsonl[idx]

        image_path = os.path.join("../data/NextGen-UIA/finetune/images", line_dict["image"])

        if self.args.caption_type == "refined":
            caption = line_dict["refined"]
        else:
            caption = line_dict["caption"]

        image = Image.open(image_path)

        # Apply augmentation
        if self.dataloader_type == "train":
            if self.args.strong_augs and self.args.weak_augs:
                if random.random() < 0.5:
                    image = self.strong_aug(image)
                    image = self.weak_aug(image)
            elif self.args.strong_augs:
                image = self.strong_aug(image)
            elif self.args.weak_augs:
                image = self.weak_aug(image)

        image = self.transform(image)

        return image, caption


class DataModule:
    def __init__(self, args):
        self.args = args

        with open("../data/NextGen-UIA/finetune/us_caption_train_qwen3_8b.jsonl", "r") as f:
            train_jsonl = [json.loads(line) for line in f]
        with open("../data/NextGen-UIA/finetune/us_caption_val_qwen3_8b.jsonl", "r") as f:
            val_jsonl = [json.loads(line) for line in f]

        # Initialize datasets with appropriate indices
        self.train_dataset = USDataset(self.args, train_jsonl, "train")
        self.val_dataset = USDataset(self.args, val_jsonl, "val")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )
