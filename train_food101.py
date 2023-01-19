import csv
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model_food101 import Food101_ResNet
from pytorch_lightning.callbacks import TQDMProgressBar
from matplotlib_inline.backend_inline import set_matplotlib_formats
from pytorch_lightning.callbacks import LearningRateMonitor

from tqdm import tqdm
import torch
import numpy as np

steps = 20000
steps_per_epoch = 75750/256
epochs = round(steps/steps_per_epoch)

augment = "rect"

model = Food101_ResNet(
    learning_rate=0.1, 
    weight_decay=0.0005,
    batch_size=64,  
    augment=augment, 
    max_perturb=0.3, 
    max_box=10,
    p_box=0.01,
    max_epochs=epochs,
    P=0.5
)

savename = f'food101/{augment}'

logger = CSVLogger("logs", name=savename)
trainer = Trainer(
    gpus=4,
    strategy="ddp_find_unused_parameters_false",
    sync_batchnorm=True,
    max_epochs=epochs,
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10)
    ],
    logger=logger,
    precision=16
)

trainer.fit(model)
torch.save(model.state_dict(), f"weights/{savename}.pt")