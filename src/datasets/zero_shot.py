import os
import csv

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image


class USDataset(Dataset):
    def __init__(self, args, image_names, dataloader_type):
        self.args = args
        self.image_names = image_names
        self.dataloader_type = dataloader_type
        self.transform = T.Compose([T.ToTensor(), T.ConvertImageDtype(torch.float32)])

        with open(f"../data/NextGen-UIA/classification/{self.args.dataset}/labels.csv", "r") as file:
            reader = csv.reader(file)
            self.label_dict = {str(row[0]): int(row[1]) for row in reader}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join("../data/NextGen-UIA/all/images", self.image_names[idx])

        image = Image.open(image_path).convert("L")
        label = self.label_dict[self.image_names[idx]]

        # To tensor whether transform is applied or not
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)

        # Adaptation for pre-trained ViT
        if self.args.in_channels == 3:
            image = image.repeat(3, 1, 1)

        return image, label, self.image_names[idx]


class DataModule:
    def __init__(self, args):
        self.args = args

        test_image_names = []
        with open(f"../data/NextGen-UIA/classification/{self.args.dataset}/train.txt", "r") as f:
            test_image_names.extend(f.read().splitlines())
        with open(f"../data/NextGen-UIA/classification/{self.args.dataset}/val.txt", "r") as f:
            test_image_names.extend(f.read().splitlines())
        with open(f"../data/NextGen-UIA/classification/{self.args.dataset}/test.txt", "r") as f:
            test_image_names.extend(f.read().splitlines())

        # Initialize datasets with appropriate indices
        self.test_dataset = USDataset(self.args, test_image_names, "test")

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=True if self.args.num_workers > 0 else False,
            shuffle=False,
            drop_last=False,
        )
