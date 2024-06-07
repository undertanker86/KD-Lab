import argparse
import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from src.model.resnet_2021 import TripleAuxResNet
# from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms
from torchvision.transforms import v2 as v2_transforms

from src.distil_loss import BYOT, DistilKL, Similarity


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, dataset_name):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None



    def setup(self, stage=None):
        transform = transforms.Compose([
            v2_transforms.AutoAugment(policy=v2_transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        



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
                 alpha:float,
                 model = 'resnet18',
                 pretrained:bool = True,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.model =  TripleAuxResNet(resnet_model=model, num_classes=100, pretrained=pretrained)
        self.criterion = torch.nn.CrossEntropyLoss()


        self.optimize_method = optimize_method
        self.scheduler_method = scheduler_method
        self.alpha = alpha
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
        """One foward pass of self-distilation training"""
        student1,student2,student3,teacher_logits = self.model(x)
        softlabel = F.softmax(teacher_logits/self.temperature, dim=1)
        return student1,student2,student3,softlabel,teacher_logits
   
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        student1,student2,student3, softlabel,teacher_logits = self.forward(inputs)
        
        student1_kl_loss = F.kl_div(F.log_softmax(student1/self.temperature, dim=1), softlabel, reduction='mean')
        student2_kl_loss = F.kl_div(F.log_softmax(student2/self.temperature, dim=1), softlabel, reduction='mean')
        student3_kl_loss = F.kl_div(F.log_softmax(student3/self.temperature, dim=1), softlabel, reduction='mean')
        student1_ce_loss = F.cross_entropy(student1, labels, reduction='mean')
        student2_ce_loss = F.cross_entropy(student2, labels, reduction='mean')
        student3_ce_loss = F.cross_entropy(student3, labels, reduction='mean')
        student1_loss = student1_kl_loss + student1_ce_loss
        student2_loss = student2_kl_loss + student2_ce_loss
        student3_loss = student3_kl_loss + student3_ce_loss
        teacher_loss = F.cross_entropy(teacher_logits, labels, reduction='mean')
        student_loss = student1_loss + student2_loss + student3_loss
        loss = self.alpha * student_loss + (1 - self.alpha) * teacher_loss
        train_accuracy = accuracy(teacher_logits, labels,task="multiclass", num_classes=100)
        layer1_accuracy = accuracy(student1, labels,task="multiclass", num_classes=100)
        layer2_accuracy = accuracy(student2, labels,task="multiclass", num_classes=100)
        layer3_accuracy = accuracy(student3, labels,task="multiclass", num_classes=100)

        self.log("train_accuracy", train_accuracy)
        self.log("layer1_accuracy", layer1_accuracy)
        self.log("layer2_accuracy", layer2_accuracy)
        self.log("layer3_accuracy", layer3_accuracy)
        self.log("layer1_loss", student1_loss)
        self.log("layer2_loss", student2_loss)
        self.log("layer3_loss", student3_loss)
        self.log("train_loss", loss)
        return loss


    
    def configure_optimizers(self):
        if self.optimize_method == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimize_method == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9 , weight_decay=5e-4)
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
        elif self.scheduler_method == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.train_dataloader())//256, epochs=self.max_epoch)
        elif self.scheduler_method == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
        elif self.scheduler_method == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90, 120], gamma=0.2)
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor | os.Mapping[str, torch.Any] | None:
        inputs, labels = batch
        student1,student2,student3,softlabel,teacher_logits = self.forward(inputs)

        student1_accuracy = accuracy(student1, labels,task="multiclass", num_classes=100)
        student2_accuracy = accuracy(student2, labels,task="multiclass", num_classes=100)
        student3_accuracy = accuracy(student3, labels,task="multiclass", num_classes=100)

        teacher_accuracy = accuracy(teacher_logits, labels,task="multiclass", num_classes=100)

        self.log("val_layer1_accuracy", student1_accuracy)
        self.log("val_layer2_accuracy", student2_accuracy)
        self.log("val_layer3_accuracy", student3_accuracy)

        self.log("val_teacher_accuracy", teacher_accuracy)
        return teacher_accuracy

        
        
        




    

def train(
    data_dir: str = "data",
    batch_size: int = 256,
    num_workers: int = 2,
    num_gpu_used: int = 2,
    max_epoch: int = 100,
    learning_rate: float = 0.01,
    num_lr_warm_up_epoch: int = 10,
    temperature: float = 3.0,
    dataset_name: str = "cifar100",
    optimize_method: str = "sgd",
    scheduler_method: str = "cosine_anneal",
    alpha: float = 0.5,
    model = 'resnet18',
    checkpoint_dir: str = "checkpoints",
    accelerator: str = "gpu",
    debug: bool = False
):
    datamodule = CIFAR100DataModule(data_dir, batch_size, num_workers, dataset_name)
    datamodule.setup()
    model = CIFARModel(num_gpu_used, max_epoch, 
                       learning_rate, num_lr_warm_up_epoch, 
                       temperature, dataset_name, 
                       optimize_method, scheduler_method, 
                       alpha, model)
    pl.seed_everything(42)
    model_check_point = ModelCheckpoint(
        checkpoint_dir,
        filename = f"resnet18_separable_{dataset_name}",
        monitor = "val_teacher_accuracy",
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = WandbLogger(project="BYOT")
    trainer = pl.Trainer(
        accelerator=accelerator, 
        devices=num_gpu_used, 
        logger=logger, 
        callbacks=[model_check_point,lr_monitor],
        log_every_n_steps=2,
        max_epochs = 2 if debug else max_epoch,
    )
    trainer.fit(model, datamodule=datamodule)

    
    
if __name__ == '__main__':
    train(debug=False)