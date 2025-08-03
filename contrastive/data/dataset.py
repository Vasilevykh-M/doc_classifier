import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from data.transform import get_train_transform, get_val_transform


class MyDataset(Dataset):
    def __init__(self, root_dir, transform, samples, processor):
        self.root_dir = Path(root_dir).glob('**/*')
        self.transform = transform
        self.samples_count = samples
        self.samples = []
        self.processor = processor

        self.class_to_idx = {}
        classes = [x for x in self.root_dir if x.is_dir()]
        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        for class_name in self.class_to_idx:
            class_dir = Path(class_name).glob('*.jpg')

            for file_name in class_dir:
                self.samples.append((file_name, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = cv2.imread(img_path)
        images = None
        for _ in range(self.samples_count):
            transformed = self.transform(image=image)
            image_ = transformed["image"]
            image_ =  self.processor(Image.fromarray(image_)).unsqueeze(0)
            images = image_ if images is None else torch.cat((images, image_), dim=0)

        return images, torch.tensor([class_idx for _ in range(self.samples_count)])

def build_data(data_path, target_size, batch_size, samples, processor):
    data_train = MyDataset(f"{data_path}/train", get_train_transform(target_size), samples, processor)
    data_val = MyDataset(f"{data_path}/val", get_val_transform(target_size), samples, processor)

    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=1)

    return (dataloader_train, dataloader_val)
