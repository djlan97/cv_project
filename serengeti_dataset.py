import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class SerengetiDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        specie = self.img_labels.iloc[idx, 1]
        descrizione = self.img_labels.iloc[idx, 2:5]
        descrizione = descrizione.to_list()
        descrizione = torch.tensor(descrizione, dtype=torch.float)
        emptyimg = self.img_labels.iloc[idx, 5]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            specie = self.target_transform(specie)
            descrizione = self.target_transform(descrizione)
            emptyimg = self.target_transform(emptyimg)

        return image, specie, descrizione, emptyimg
