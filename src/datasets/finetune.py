import os
import re
import logging
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image


class FinetuneDataset(Dataset):
    def __init__(self, args, data_df, caption_key, img_key, img_path):
        self.args = args
        self.data_df = data_df
        self.caption_key = caption_key
        self.img_key = img_key
        self.transform = T.Compose(
            [
                T.Resize(args.img_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(args.img_size),
                T.ToTensor(),
                T.ConvertImageDtype(torch.float32),
            ]
        )

        # Pre-build image path mapping for fast lookup during training
        img_paths = img_path if isinstance(img_path, list) else [img_path]
        self.image_path_map = {}
        for idx in range(len(data_df)):
            row = data_df.iloc[idx]
            filename = row[img_key]
            image_filename = os.path.basename(filename)

            # Find the correct path for this image
            for path in img_paths:
                candidate_path = os.path.join(path, image_filename)
                if os.path.exists(candidate_path):
                    self.image_path_map[idx] = candidate_path
                    break

            if idx not in self.image_path_map:
                logging.warning(f"Image {image_filename} not found in any path during initialization")

        logging.info(f"Built image path mapping for {len(self.image_path_map)} images")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        data_dict = self.data_df.iloc[idx]

        # Use pre-built path mapping for fast lookup (no file system operations)
        image_path = self.image_path_map.get(idx)
        if image_path is None:
            filename = data_dict[self.img_key]
            raise FileNotFoundError(f"Image {filename} path not found in pre-built mapping")

        caption = data_dict[self.caption_key]
        image = Image.open(image_path)

        # Convert to RGB if needed (some medical images might be grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image)
        return image, caption


class DataModule:
    def __init__(self, args):
        self.args = args

        # Load MedPix and PMC-CURD CSV datasets
        medpix_path = "/root/project/data/NextGen-UIA/finetune/medpix_dataset"
        pmc_curd_path = "/root/project/data/NextGen-UIA/finetune/pmc_curd_dataset"
        medpix_csv_path = os.path.join(medpix_path, "medpix_dataset.csv")
        medpix_img_path = os.path.join(medpix_path, "images")
        pmc_curd_csv_path = os.path.join(pmc_curd_path, "pmc_curd_dataset.csv")
        pmc_curd_img_path = os.path.join(pmc_curd_path, "images")

        medpix_df = pd.read_csv(medpix_csv_path, sep=",")
        pmc_curd_df = pd.read_csv(pmc_curd_csv_path, sep=",")
        df = pd.concat([medpix_df, pmc_curd_df])
        caption_key, img_key = "Caption", "filename"

        logging.info(f"Loaded {len(df)} samples from MedPix and PMC-CURD datasets")

        # Filter invalid captions
        # The MedPixdataset was cleaned by removing special characters, trimming leading and trailing white spaces,
        # and excluding samples with captions shorter than 20 characters.
        clean_pattern = re.compile(
            r"[^A-Za-z0-9\s\.,;:\(\)\[\]\{\}\/_\-+\*=<>@&\|\\\^'\"`~\$?#!…\u00B1\u00B0\u00B5\u03BC\u2264\u2265\u2248\u2192\u2013\u2014\u2022]"
        )
        df[caption_key] = df[caption_key].apply(lambda x: clean_pattern.sub("", str(x)))
        df[caption_key] = df[caption_key].apply(lambda x: x.strip())
        df = df[df[caption_key].str.len() > 20]
        logging.info(f"After caption filtering: {len(df)} samples")

        # Remove row if image does not exist (check with full path)
        def check_image_exists(img_name):
            if os.path.isabs(img_name):
                return os.path.exists(img_name)
            else:
                medpix_full_path = os.path.join(medpix_img_path, img_name)
                pmc_curd_full_path = os.path.join(pmc_curd_img_path, img_name)
                return os.path.exists(medpix_full_path) or os.path.exists(pmc_curd_full_path)

        df = df[df[img_key].apply(check_image_exists)]

        # Shuffle the dataframe and split into train and validation sets (90% train, 10% val)
        df = df.sample(frac=1, random_state=self.args.seed).reset_index(drop=True)

        split_idx = int(len(df) * 0.9)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        logging.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        # Initialize datasets with both image paths
        img_paths = [medpix_img_path, pmc_curd_img_path]
        self.train_dataset = FinetuneDataset(self.args, train_df, caption_key, img_key, img_paths)
        self.val_dataset = FinetuneDataset(self.args, val_df, caption_key, img_key, img_paths)

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
