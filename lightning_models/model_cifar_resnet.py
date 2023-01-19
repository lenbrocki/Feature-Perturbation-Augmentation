from pytorch_lightning import LightningModule, Trainer

import os
import torch
import numpy as np
import cupy as cp
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST
import random
import matplotlib.pyplot as plt
from torchinfo import summary
from numba import njit
from tqdm import tqdm
from sklearn import model_selection
from perturbations import mask_batch_rand, mask_batch_rect

seed_everything(7)

from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR

import torchvision.models as models

def create_model():
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model



# augment can be 'rand' or 'rect'
class CIFAR_ResNet(LightningModule):
    def __init__(self, learning_rate=0.01, max_perturb=0.0, max_box=4,
                 p_box=0.1, num_workers=8, batch_size=64, label_smoothing=0.0, 
                 s_low=0.02, s_high=0.4, augment = 'rand', plot=False):

        super().__init__()

        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.plot = plot

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (3, 32, 32)
        self.channels, self.width, self.height = self.dims
        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        
        # Define augmentation

        self.augment = augment
        self.s_low = 0.02
        self.s_high = 0.4
        self.r1 = 0.3
        
        # to save computing time we precompute the random masks used for perturbation
        # number of precomputed batches of random masks
        self.aug_batch = 200
        # maximum density of pixels set to 0 in mask 
        self.max_perturb = max_perturb
        # chance that pixel set to 0 is turned into box
        self.p_box = p_box
        # max length of side of box
        self.max_box = max_box

        # Define PyTorch model
        self.model = create_model()

        self.accuracy = Accuracy()
        self.criterion = nn.CrossEntropyLoss()
        self.cnt = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # use perturbation as data augmentation
        idx = random.randrange(0, self.aug_batch)
        
        if(self.augment != 'none'):
            masks = torch.as_tensor(self.masks[idx][:len(x)], device=self.device)
            #only perturb with 0.5 probability
            if(random.random() > 0.5 or self.plot):
                x_perturbed = x*masks
            else:
                x_perturbed = x
        else:
            x_perturbed = x
            
        #for plotting
        if(self.plot):
            t = x_perturbed[0].detach().cpu().numpy()
            t = t/2+0.5
            t = np.moveaxis(t,[0,1,2],[2,0,1])
            plt.axis("off")
            plt.imshow(t)
            plt.savefig(f"graphics/examplecifar/{self.cnt}.png")
            self.cnt += 1            

        logits = self(x_perturbed)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        #loss = F.nll_loss(logits, y)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        num_gpus = self.trainer.num_devices
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, 
                                    momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {
                'scheduler': MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1),
                'monitor': 'val_loss',
                'name': 'log_lr'
         }
        return [optimizer], [lr_scheduler]

    ####################
    # DATA RELATED HOOKS
    ####################
    
    def get_masks(self):
        print('Creating masks for perturbation...')
        mm = np.zeros((self.aug_batch, self.batch_size, self.channels, self.width, self.height)
                      , dtype=np.float32)
        for j in tqdm(range(self.aug_batch)):
            if(self.augment == 'rand'):
                mm[j] = mask_batch_rand(self.batch_size, self.channels, self.width, 
                                       self.height, self.max_perturb, self.max_box, self.p_box)
            if(self.augment == 'rect'):
                mm[j] = mask_batch_rect(self.batch_size, self.channels, self.width, 
                       self.height, self.s_low, self.s_high, self.r1)
        return mm

    def prepare_data(self):
        # download
        CIFAR10(root='./data', train=True, download=True)
        CIFAR10(root='./data', train=False, download=True)

    def setup(self, stage=None):
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            #create masks for augmentation
            if(self.augment != 'none'):
                self.masks = self.get_masks()
                
            cifar_train = CIFAR10(root='./data', train=True, transform=self.train_transform)
            cifar_val = CIFAR10(root='./data', train=True, transform=self.test_transform)
            
            num_train = len(cifar_train)
            indices = list(range(num_train))
            train_indices, val_indices = model_selection.train_test_split(indices, test_size=0.1)
            self.train_data = torch.utils.data.Subset(cifar_train, train_indices)
            self.val_data = torch.utils.data.Subset(cifar_val, val_indices)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = CIFAR10(root='./data', train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)