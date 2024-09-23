import lightning as L
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import time
from helper import LightningModel, Fer2013DataModule, Cifar100DataModule
import timm
import argparse

from torchvision import transforms
CIFAR100MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100STD = (0.2675, 0.2565, 0.2761)


def train():
    train_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100MEAN, CIFAR100STD),

        ]
    )
    L.seed_everything(2024)
    dm = Cifar100DataModule(
        height_width=(32, 32),
        batch_size=256,
        train_transform=train_transform,
        num_workers=4
    )
    pytorch_model = torchvision.models.resnet18(weights=None)
    pytorch_model.fc = torch.nn.Linear(512, 100)
    pytorch_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(
        3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # pytorch_model.

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    # save top 1 model
    callbacks = [ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc")]

    trainer = L.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=[3],
        callbacks=callbacks,
        logger=WandbLogger(project="BYOT"),
        deterministic=True,
    )

    trainer.fit(model=lightning_model, datamodule=dm)
    trainer.test(lightning_model, datamodule=dm, ckpt_path='best')


if __name__ == '__main__':
    import torch.nn as nn
    import timm

    class Resnet34Fer(nn.Module):
        def __init__(self, model_name, pretrained=False, num_classes=7):
            super().__init__()
            self.resnet = timm.create_model(
                model_name, pretrained=pretrained, num_classes=num_classes)
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(
                3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.resnet.maxpool = nn.Identity()
            self.resnet.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            return self.resnet(x)

    # Load pre-trained model
    model = Resnet34Fer("resnet18", pretrained=False, num_classes=7)
    # # Replace final layers with custom head
    # model.global_pool = custom_head.pool
    # model.fc = custom_head.fc
    x = torch.randn(1, 1, 48, 48)
    out = model(x)
    print(out.shape)

# Use the modified model for training or inference
