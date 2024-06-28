import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate, optimizer, lr_scheduler, max_epoch):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.lr_scheduler = lr_scheduler
        self.learning_rate = learning_rate

        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=100)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=100)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=100)
    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9 , weight_decay=5e-4)
        elif self.optimizer == "adam_wav2vec2.0":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-6) # wav2vec2,0's optimizer set up on Adam. (Need to verify)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-6) # distilBert's optimzer setup on Adam
        elif self.optimizer == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-4)
        else:
            raise NotImplementedError
        if self.lr_scheduler == "":
            return optimizer
        elif self.lr_scheduler == "linear_decay_with_warm_up":
            def lr_lambda(current_epoch): # Copied from https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
                if current_epoch < self.num_lr_warm_up_epoch:
                    return float(current_epoch+1) / float(max(1, self.num_lr_warm_up_epoch)) # current_epoch+1 to prevent lr=0 in epoch 0
                return max(
                    0.0, float(self.max_epoch - current_epoch) / float(max(1, self.max_epoch - self.num_lr_warm_up_epoch)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.lr_scheduler == "cosine_warmup_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=9, T_mult=1, eta_min=1e-6)
        elif self.lr_scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.train_dataloader())//256, epochs=self.max_epoch)
        elif self.lr_scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
        elif self.lr_scheduler == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90, 120], gamma=0.2)
        elif self.lr_scheduler == "cosine_annealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch, eta_min=1e-6)
        elif self.lr_scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=5, verbose=True)
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]


class Cifar100DataModule(L.LightningDataModule):
    def __init__(
        self, data_path="./", batch_size=64, num_workers=0, height_width=(32, 32),
        train_transform=None, test_transform=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_workers = num_workers
        self.height_width = height_width
        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_data(self):
        datasets.CIFAR100(root=self.data_path, download=True)

        if self.train_transform is None:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize(self.height_width),
                    transforms.ToTensor(),
                ]
            )

        if self.test_transform is None:
            self.test_transform = transforms.Compose(
                [
                    transforms.Resize(self.height_width),
                    transforms.ToTensor(),
                ]
            )
            return

    def setup(self, stage=None):
        train = datasets.CIFAR100(
            root=self.data_path,
            train=True,
            transform=self.train_transform,
            download=False,
        )

        self.test = datasets.CIFAR100(
            root=self.data_path,
            train=False,
            transform=self.test_transform,
            download=False,
        )

        self.train, self.valid = random_split(train, lengths=[45000, 5000])

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader


class Fer2013DataModule(L.LightningDataModule):
    def __init__(
        self, train_path="./",val_path="./",test_path="./", batch_size=64, num_workers=0, height_width=(48, 48),
        train_transform=None, test_transform=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.num_workers = num_workers
        self.height_width = height_width
        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_data(self):
        datasets.ImageFolder(root=self.data_path)

        if self.train_transform is None:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize(self.height_width),
                    transforms.ToTensor(),
                ]
            )

        if self.test_transform is None:
            self.test_transform = transforms.Compose(
                [
                    transforms.Resize(self.height_width),
                    transforms.ToTensor(),
                ]
            )
            return None
        
    def setup(self, stage=None):
        self.train = datasets.ImageFolder(
            root=self.train_path,
            transform=self.train_transform
        )

        self.valid = datasets.ImageFolder(
            root=self.val_path,
            transform=self.test_transform
        )

        self.test = datasets.ImageFolder(
            root=self.test_path,
            transform=self.test_transform
        )



    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
    
class FerPlusDataModule(L.LightningDataModule):
    def __init__(
        self, train_path="./",val_path="./",test_path="./" ,batch_size=64, num_workers=0, height_width=(48, 48),
        train_transform=None, test_transform=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.num_workers = num_workers
        self.height_width = height_width
        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_data(self):
        datasets.ImageFolder(root=self.data_path)

        if self.train_transform is None:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize(self.height_width),
                    transforms.ToTensor(),
                ]
            )

        if self.test_transform is None:
            self.test_transform = transforms.Compose(
                [
                    transforms.Resize(self.height_width),
                    transforms.ToTensor(),
                ]
            )
            return None
        
    def setup(self, stage=None):
        self.train = datasets.ImageFolder(
            root=self.data_path,
            transform=self.train_transform
        )

        self.val = datasets.ImageFolder(
            root=self.val_path,
            transform=self.test_transform
        )

        self.test = datasets.ImageFolder(
            root=self.test_path,
            transform=self.test_transform
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader