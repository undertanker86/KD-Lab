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
# from src.model.resnet_2021 import TripleAuxResNet
from src.model.resnet_fer import TripleAuxResNetFer
# from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms
from torchvision.transforms import v2 as v2_transforms

from src.distil_loss import BYOT, DistilKL, Similarity
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight 

class Ferdatamodule(pl.LightningDataModule):
    def __init__(self,train_folder='kaggle/input/fer2013/train', 
                 test_folder='kaggle/input/fer2013/test',
                 batch_size=64, num_workers=4) -> None:
        super().__init__()
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.sampler = torch.utils.data.WeightedRandomSampler

    def setup(self, stage=None):
        self.set_up_transforms()
        self.train_dataset = datasets.ImageFolder(self.train_folder)
        self.test_dataset = datasets.ImageFolder(self.test_folder)
        self.weights = make_weights_for_balanced_classes(self.train_dataset.imgs, len(self.train_dataset.classes))
        self.weights = torch.DoubleTensor(self.weights)

    def set_up_transforms(self):
        self.train_transforms = transforms.Compose([
            transforms.Resize(224,224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize(224,224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.sampler(self.weights), num_workers=self.num_workers, transforms=self.train_transforms)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, transforms=self.test_transforms)
    



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
                 pretrained:bool = False,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.model =  TripleAuxResNetFer(resnet_model=model, num_classes=7, pretrained=pretrained)
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
        train_accuracy = accuracy(teacher_logits, labels)

        layer1_accuracy = accuracy(student1, labels)
        layer2_accuracy = accuracy(student2, labels)
        layer3_accuracy = accuracy(student3, labels)

        self.log("train_accuracy", train_accuracy, sync_dist=True , on_epoch=True)
        self.log("layer1_accuracy", layer1_accuracy, sync_dist=True, on_epoch=True)
        self.log("layer2_accuracy", layer2_accuracy, sync_dist=True, on_epoch=True)
        self.log("layer3_accuracy", layer3_accuracy, sync_dist=True, on_epoch=True)
        self.log("layer1_loss", student1_loss)
        self.log("layer2_loss", student2_loss)
        self.log("layer3_loss", student3_loss)
        self.log("train_loss", loss)
        
        return loss

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)
    
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
        elif self.scheduler_method == "cosine_warmup_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=9, T_mult=1, eta_min=1e-6)
        elif self.scheduler_method == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.train_dataloader())//256, epochs=self.max_epoch)
        elif self.scheduler_method == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
        elif self.scheduler_method == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90, 120], gamma=0.2)
        elif self.scheduler_method == "cosine_annealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=280, eta_min=1e-6)
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor | os.Mapping[str, torch.Any] | None:
        inputs, labels = batch
        student1,student2,student3,softlabel,teacher_logits = self.forward(inputs)

        student1_accuracy = float((torch.max(student1, 1)[1].eq(labels)).cpu().sum()) / len(labels)
        student2_accuracy = float((torch.max(student2, 1)[1].eq(labels)).cpu().sum()) / len(labels)
        student3_accuracy = float((torch.max(student3, 1)[1].eq(labels)).cpu().sum()) / len(labels)
        teacher_accuracy = float((torch.max(teacher_logits, 1)[1].eq(labels)).cpu().sum()) / len(labels)

        self.log("val_layer1_accuracy", student1_accuracy,sync_dist=True)
        self.log("val_layer2_accuracy", student2_accuracy,sync_dist=True)
        self.log("val_layer3_accuracy", student3_accuracy,sync_dist=True)

        self.log("val_teacher_accuracy", teacher_accuracy,sync_dist=True)
        return teacher_accuracy
        
def train(
    data_dir: str = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    num_gpu_used: int = 2,
    max_epoch: int = 100,
    learning_rate: float = 0.01,
    num_lr_warm_up_epoch: int = 10,
    temperature: float = 4.0,
    dataset_name: str = "cifar100",
    optimize_method: str = "sgd",
    scheduler_method: str = "cosine_annealingLR",
    alpha: float = 0.3,
    model = 'resnet18',
    checkpoint_dir: str = "checkpoints",
    accelerator: str = "gpu",
    debug: bool = False,
    ckpt_path:str = None
    ):

    
    datamodule = Ferdatamodule(batch_size, num_workers)
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
        callbacks=[model_check_point, lr_monitor],
        log_every_n_steps=2,
        max_epochs = 2 if debug else max_epoch,
        strategy='ddp_find_unused_parameters_true',

    
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)