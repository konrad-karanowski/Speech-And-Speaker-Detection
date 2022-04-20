from typing import *


import torch
from torch import nn
import pytorch_lightning as pl



class SiameseModel(nn.Module):

    def __init__(self) -> None:
        super(SiameseModel, self).__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class SiameseModel(pl.LightningModule):

    def __init__(self, config) -> None:
        super(SiameseModel, self).__init__()
        self.config = config
        


    def configure_optimizers(self):
        return super().configure_optimizers()

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


    def training_step(self, *args, **kwargs):
        pass


    def validation_step(self, *args, **kwargs):
        pass

    def test_step(self, *args, **kwargs):
        pass
