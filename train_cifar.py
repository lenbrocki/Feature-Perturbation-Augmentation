import csv
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model_cifar_resnet import CIFAR_ResNet
from pytorch_lightning.callbacks import TQDMProgressBar
from matplotlib_inline.backend_inline import set_matplotlib_formats
from pytorch_lightning.callbacks import LearningRateMonitor

from tqdm import tqdm
import torch

# augment can be set to "rand", "rect" or "none"
augment = "rand"

def train_rand(max_perturb, p_box, max_box):
    model = CIFAR_ResNet(learning_rate=0.01, 
                         batch_size=64, 
                         augment=augment, 
                         max_perturb=max_perturb, 
                         p_box=p_box, 
                         max_box=max_box
    )
    savename = f"cifar/{augment}"
    logger = CSVLogger("logs", name=savename)
    trainer = Trainer(
        gpus=1,
        strategy="ddp_find_unused_parameters_false",
        sync_batchnorm=True,
        max_epochs=90,
        callbacks=[LearningRateMonitor(logging_interval="step"),
                   TQDMProgressBar(refresh_rate=10)],
        logger=logger,
        precision=16
    )
    trainer.fit(model)
    torch.save(model.state_dict(), f"weights/{savename}.pt")

train_rand(0.25, 0.1, 3)
