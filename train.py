import torch
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torchvision.transforms import v2 as v2_transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import torch.nn.functional as F
import argparse
import os
import sys


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, dataset_name):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # Download the dataset
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        transform = transforms.Compose([
            v2_transforms.AutoAugment(policy=v2_transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        ])

        if self.dataset_name == 'cifar10':
            self.train_dataset = datasets.CIFAR10(self.data_dir, train=True, transform=transform, download=True)
            self.val_dataset = datasets.CIFAR10(self.data_dir, train=False, transform=transform, download=True)
        elif self.dataset_name == 'cifar100':
            self.train_dataset = datasets.CIFAR100(self.data_dir, train=True, transform=transform, download=True)
            self.val_dataset = datasets.CIFAR100(self.data_dir, train=False, transform=transform, download=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    

class CIFARModel(pl.LightningModule):
    def __init__(self, dataset_name:str, model,seed:int=42):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.model = model
        self.warmup_epochs = 5
        self.seed = seed
        self.lr = 0.1
    

    def setup(self, stage=None):
        if self.dataset_name == 'cifar10':
            self.num_classes = 10
        elif self.dataset_name == 'cifar100':
            self.num_classes = 100

        pl.seed_everything(self.seed, workers=True)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False        
    
    def forward(self, x):
        return self.model(x)
   
  

    def train_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor | os.Mapping[str, torch.Any] | None:
        x,y = batch
        y_hat = self.model
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)