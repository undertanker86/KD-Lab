import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from src.model.Customodel import MobileNetV2CBAM

# from torchmetrics import Accuracy
from torchmetrics.classification import Accuracy
from torchvision import datasets, transforms
from torchvision.transforms import v2 as v2_transforms


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


class Ferdatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_folder="/kaggle/input/fer2013/train",
        test_folder="/kaggle/input/fer2013/test",
        batch_size=64,
        num_workers=4,
    ) -> None:
        super().__init__()
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.sampler = torch.utils.data.sampler.WeightedRandomSampler

    def setup(self, stage=None):
        self.set_up_transforms()
        self.train_dataset = datasets.ImageFolder(
            self.train_folder, transform=self.train_transforms
        )
        self.test_dataset = datasets.ImageFolder(
            self.test_folder, transform=self.test_transforms
        )
        self.weights = make_weights_for_balanced_classes(
            self.train_dataset.imgs, len(self.train_dataset.classes)
        )
        self.weights = torch.DoubleTensor(self.weights)

    def set_up_transforms(self):
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler(self.weights, len(self.train_dataset)),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class Fermodel(pl.LightningModule):
    def __init__(self, 
                 num_classes=7,
                 scheduler_method="CosineAnnealing",
                 optimize_method="Adam",
                 max_epoch=100,
                 num_gpu_used=1,
                 learning_rate = 0.001,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters
        self.model = MobileNetV2CBAM()
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.num_classes = num_classes


        self.optimize_method = optimize_method
        self.scheduler_method = scheduler_method

        if num_gpu_used == 1:
            self.max_epoch = max_epoch
            self.lr = learning_rate

        else:
            self.register_buffer("max_epoch", torch.tensor(max_epoch))
            self.register_buffer("lr", torch.tensor(learning_rate))


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        pred = out.argmax(dim=1, keepdim=True)
        acc = self.accuracy.update(pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        pred = out.argmax(dim=1, keepdim=True)
        loss = self.loss(out, y)
        acc = self.val_accuracy.update(pred, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.max_epoch, eta_min=0, last_epoch=-1)
        return [optimizer], [scheduler]

def train(
    train_folder: str = "/kaggle/input/fer2013/train",
    test_folder: str = "/kaggle/input/fer2013/test",
    batch_size: int = 128,
    num_workers: int = 2,
    num_gpu_used: int = 2,
    max_epoch: int = 100,
    learning_rate: float = 0.001,
    dataset_name: str = "fer2013",
    optimize_method: str = "adam",
    scheduler_method: str = "CosineAnnealing",
    model = 'resnet18',
    checkpoint_dir: str = "checkpoints",
    accelerator: str = "gpu",
    debug: bool = False,
    ckpt_path:str = None):
    pl.seed_everything(42)
    datamodule = Ferdatamodule(batch_size=batch_size, train_folder=train_folder, test_folder=test_folder, num_workers=num_workers)
    model = Fermodel(num_classes=7, scheduler_method=scheduler_method, optimize_method=optimize_method, max_epoch=max_epoch, num_gpu_used=num_gpu_used, learning_rate = learning_rate)
    logger = WandbLogger(project="BYOT")
    model_check_point = ModelCheckpoint(
        checkpoint_dir,
        filename = f"resnet18_separable_{dataset_name}",
        monitor = "val_teacher_accuracy",
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(max_epochs=100, devices=num_gpu_used, accelerator="gpu",strategy="ddp", logger=logger,
    log_every_n_steps=10,
    callbacks=[model_check_point, lr_monitor])
                         
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", type=str, default="/kaggle/input/fer2013/train")
    parser.add_argument("--test_folder", type=str, default="/kaggle/input/fer2013/test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_gpu_used", type=int, default=2)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dataset_name", type=str, default="fer2013")
    parser.add_argument("--optimize_method", type=str, default="adam")
    parser.add_argument("--scheduler_method", type=str, default="CosineAnnealing")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()
    train(**vars(parser.parse_args()))