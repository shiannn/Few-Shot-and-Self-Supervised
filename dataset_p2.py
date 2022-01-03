import os
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from config_p2 import LABEL2CIDX

class OfficeHomeDataset(Dataset):
    def __init__(self, csv_path, data_dir, is_valid=False, is_test=False):
        self.data_dir = data_dir
        self.is_valid = is_valid 
        self.is_test = is_test
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.label2cidx = LABEL2CIDX
        #print(self.label2cidx)
        self.transform = transforms.Compose([
            lambda x: Image.open(x),
            transforms.Resize(size=160),
            transforms.RandomCrop(size=128),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            lambda x: Image.open(x),
            transforms.Resize(size=160),
            transforms.CenterCrop(size=128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.target_transform = transforms.Compose([
            lambda x: self.label2cidx[x],
        ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        if not self.is_test:
            if self.is_valid:
                image = self.test_transform(os.path.join(self.data_dir, path))
            else:
                image = self.transform(os.path.join(self.data_dir, path))
            label = self.data_df.loc[index, "label"]
            label = self.target_transform(label)
            return image, label
        else:
            image = self.test_transform(os.path.join(self.data_dir, path))
            return image

    def __len__(self):
        return len(self.data_df)
    
# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.labels = np.unique(self.data_df.loc[:,"label"].values)
        self.label2cidx = {label:idx for idx, label in enumerate(self.labels)}

        self.transform = transforms.Compose([
            lambda x: Image.open(x),
            transforms.Resize(size=128),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.target_transform = transforms.Compose([
            lambda x: self.label2cidx[x],
        ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.data_df)

if __name__ == '__main__':
    from config_p2 import (
        TRAIN_CSV, TRAIN_ROOT, VAL_CSV, VAL_ROOT,
        FINE_TUNE_TRAIN_ROOT, FINE_TUNE_TRAIN_CSV
    )
    dataset = OfficeHomeDataset(FINE_TUNE_TRAIN_CSV, FINE_TUNE_TRAIN_ROOT)
    img, label = dataset[0]
    print(img.shape, label)