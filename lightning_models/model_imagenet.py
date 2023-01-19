from pytorch_lightning import LightningModule, Trainer

import os
import torch
import numpy as np
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageNet
import random
import matplotlib.pyplot as plt
from torchinfo import summary
from numba import njit
from tqdm import tqdm
from sklearn import model_selection
import timm
from torchvision.models import resnet50, ResNet50_Weights
from perturbations import mask_batch_rand, mask_batch_rect

seed_everything(7)

from torch.optim.lr_scheduler import OneCycleLR, StepLR

import torchvision.models as models

# augment can be 'rand' or 'rect'
class ImgNet_ResNet(LightningModule):
    def __init__(self, learning_rate=0.05, max_perturb=0.0, max_box=4, 
                 p_box=0.1, num_workers=8, batch_size=64, 
                 s_low=0.02, s_high=0.4, augment = 'rand', r1=0.3, finetune=False):

        super().__init__()

        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_dir = "/home/lbrocki/AugmentData/data/"

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (3, 224, 224)
        self.channels, self.width, self.height = self.dims
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Define augmentation
        # number of precomputed batches of random masks
        self.augment = augment
        self.s_low = s_low
        self.s_high = s_high
        self.aug_batch = 200
        # maximum density of pixels set to 0 in mask 
        self.max_perturb = max_perturb
        # chance that pixel set to 0 is turned into box
        self.p_box = p_box
        # max length of side of box
        self.max_box = max_box
        self.cnt = 0

        # Define PyTorch model
        #self.model = timm.create_model('inception_resnet_v2')
        self.model = resnet50(weights=None)

        self.accuracy = Accuracy()
        self.criterion = nn.CrossEntropyLoss()
        self.r1 = r1
        self.finetune = finetune

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # use perturbation as data augmentation
        idx = random.randrange(0, self.aug_batch)
        
        if(self.augment != 'none'):
            masks = torch.as_tensor(self.masks[idx][:len(x)])
            masks = masks.type_as(x)
            #only perturb with 0.5 probability
            if(random.random() > 0.5):
                x_perturbed = x*masks
            else:
                x_perturbed = x
        else:
            x_perturbed = x
            
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
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, 
                                    momentum=0.9, weight_decay=1e-4)
    
        lr_scheduler = {
            'scheduler': StepLR(optimizer, step_size=30),
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
        print('preparing data ..')
        ImageNet(root=self.data_dir, split='train')
        ImageNet(root=self.data_dir, split='val')
        print('done!')

    def setup(self, stage=None):
        if(self.augment != 'none'):
                self.masks = self.get_masks()
                
        imgnet_train = ImageNet(root=self.data_dir,
                                split='train', transform = self.train_transform)
        imgnet_test = ImageNet(root=self.data_dir, 
                               split='val', transform = self.test_transform)
        num_full = len(imgnet_train)
        indices_full = list(range(num_full))
        
        num_test = len(imgnet_test)
        indices_test = list(range(num_test))
                
        # we only use 10% of validation set (5000 images) for validation and evaluation of importance estimators
        rest_indices, test_sub_indices = model_selection.train_test_split(indices_test, 
                                                                          test_size=0.1,
                                                                          random_state=2)
        np.save('/home/lbrocki/AugmentTrain/test_idx_val_imgnet.npy', test_sub_indices)
        
        self.train_data = torch.utils.data.Subset(imgnet_train, indices_full)
        self.val_data = torch.utils.data.Subset(imgnet_test, test_sub_indices)
        self.test_data = torch.utils.data.Subset(imgnet_test, indices_test)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)
