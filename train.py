import argparse
import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from resnet_2021 import resnet18_cbam
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from torchvision.transforms import v2 as v2_transforms

from src.method import BYOT, KD, Similarity


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

    # def test_dataloader(self):
    #     return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    

class CIFARModel(pl.LightningModule):
    def __init__(self,
                
                 num_gpu_used:int,
                 max_epoch:int,
                 learning_rate:float,
                 num_lr_warm_up_epoch:int,
                 temperature:float, 
                 dataset_name:str,
                 optimize_method:str,
                 scheduler_method:str,
                 final_loss_coeff_dict:dict,
                 model = resnet18_cbam,
                 seed:int=42,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.model = resnet18_cbam(pretrained=False)
        self.seed = seed
        self.logger = WandbLogger()


        self.optimize_method = optimize_method
        self.scheduler_method = scheduler_method
        self.final_loss_coeff_dict = final_loss_coeff_dict
        if num_gpu_used == 1:
            self.max_epoch = max_epoch
            self.lr = learning_rate
            self.num_lr_warm_up_epoch = num_lr_warm_up_epoch
            self.temperature = temperature
        else:
            self.register_buffer("max_epoch", torch.tensor(max_epoch))
            self.register_buffer("lr", torch.tensor(learning_rate))
            self.register_buffer("num_lr_warm_up_epoch", torch.tensor(num_lr_warm_up_epoch))
            self.register_buffer("temperature", torch.tensor(temperature))


     
    
    def forward(self, x):
        return self.model(x)
   
    def train_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, outputs_feature = self.forward(inputs)

        loss = torch.FloatTensor([0.]).to(self.device)
        loss += self.criterion(outputs[0], labels)
        teacher_output = outputs[0].detach()
        teacher_feature = outputs_feature[0].detach()

        for index in range(1, len(outputs)):
            loss += KD(outputs_feature[index], teacher_feature, self.temperature)
            loss += self.criterion(teacher_output, outputs[index]) * (1 - self.final_loss_coeff_dict[1])

        loss /= len(outputs)
        acc  = (outputs[0].argmax(dim=1) == labels).float().mean()
        acc1 = (outputs[1].argmax(dim=1) == labels).float().mean()
        acc2 = (outputs[2].argmax(dim=1) == labels).float().mean()
        acc3 = (outputs[3].argmax(dim=1) == labels).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('train_acc1', acc1)
        self.log('train_acc2', acc2)
        self.log('train_acc3', acc3)
        return loss
    
    def configure_optimizers(self):
        if self.optimize_method == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimize_method == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimize_method == "adam_wav2vec2.0":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-6) # wav2vec2,0's optimizer set up on Adam. (Need to verify)
        elif self.optimize_method == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-6) # distilBert's optimzer setup on Adam
        elif self.optimize_method == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-6)
        else:
            raise NotImplementedError
        if self.scheduler_method == "":
            return optimizer
        elif self.scheduler_method == "linear_decay_with_warm_up":
            def lr_lambda(current_epoch): # Copied from https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
                if current_epoch < self.num_lr_warm_up_epoch:
                    return float(current_epoch+1) / float(max(1, self.num_lr_warm_up_epoch)) # current_epoch+1 to prevent lr=0 in epoch 0
                return max(
                    0.0, float(self.max_epoch - current_epoch) / float(max(1, self.max_epoch - self.num_lr_warm_up_epoch)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.scheduler_method == "cosine_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=9, T_mult=1, eta_min=1e-6)
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor | os.Mapping[str, torch.Any] | None:
        inputs, labels = batch
        outputs, outputs_feature = self.forward(inputs)
        acc = (outputs[0].argmax(dim=1) == labels).float().mean()
        acc1 = (outputs[1].argmax(dim=1) == labels).float().mean()
        acc2 = (outputs[2].argmax(dim=1) == labels).float().mean()
        acc3 = (outputs[3].argmax(dim=1) == labels).float().mean()
        self.log('val_acc1', acc1)
        self.log('val_acc2', acc2)
        self.log('val_acc3', acc3)

        self.log('val_acc', acc)

    def criterion(self, outputs, labels):
        return torch.nn.CrossEntropyLoss()(outputs, labels)



    

def train(
    data_dir: str = "/data",
    batch_size: int = 512,
    num_workers: int = 2,
    num_gpu_used: int = 1,
    max_epoch: int = 100,
    learning_rate: float = 0.01,
    num_lr_warm_up_epoch: int = 10,
    temperature: float = 3.0,
    dataset_name: str = "cifar100",
    optimize_method: str = "adam",
    scheduler_method: str = "cosine_anneal",
    final_loss_coeff_dict: dict = {"kd": 0.5, "ce": 0.5},
    checkpoint_dir: str = "checkpoints",
    accelerator: str = "gpu",
    debug: bool = False
):
    datamodule = CIFARDataModule(data_dir, batch_size, num_workers, dataset_name)
    datamodule.setup()
    model = CIFARModel(num_gpu_used, max_epoch, learning_rate, num_lr_warm_up_epoch, temperature, dataset_name, optimize_method, scheduler_method, final_loss_coeff_dict)
    pl.seed_everything(42)
    model_check_point = ModelCheckpoint(
        checkpoint_dir,
        filename = f"resnet18_cbam-{final_loss_coeff_dict}",
        monitor = "val_acc",
    )
    logger = WandbLogger(project="BYOT")
    trainer = pl.Trainer(
        accelerator=accelerator, 
        devices=num_gpu_used, 
        logger=logger, 
        callbacks=[model_check_point],
        log_every_n_steps=2,
        max_epochs = 2 if debug else max_epoch,
    )
    trainer.fit(model, datamodule=datamodule)

    
    
if __name__ == '__main__':
    train(debug=True)