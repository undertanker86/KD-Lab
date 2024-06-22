import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
import pytorch_lightning as pl

class KFoldDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, num_folds=5):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_folds = num_folds

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        self.dataset = ImageFolder(root=self.data_dir, transform=self.transform)
        self.kf = KFold(n_splits=self.num_folds, shuffle=True)

    def setup(self, stage=None):
        self.splits = list(self.kf.split(self.dataset))

    def train_dataloader(self, fold_idx):
        train_idx, val_idx = self.splits[fold_idx]
        train_subset = Subset(self.dataset, train_idx)
        return DataLoader(train_subset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self, fold_idx):
        train_idx, val_idx = self.splits[fold_idx]
        val_subset = Subset(self.dataset, val_idx)
        return DataLoader(val_subset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
