from typing import *

import torch 
from torch import nn

from models.backbones.backbone import Backbone

class CatalunaBackbone(Backbone):

    def __init__(self, input_size: Tuple[int, int, int]) -> None:
        """Backbone inspired by PAC paper

        Args:
            input_size (Tuple[int, int, int]): Input spectrogram shape.
        """
        super(CatalunaBackbone, self).__init__(input_size)
        self.fe = nn.Sequential(
            nn.Conv2d(1, 128, (3, 3), (1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(128, 256, (3, 3), (1, 1)),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(256, 512, (3, 3), (1, 1)),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1)),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fe = self.fe(x)
        flatt = self.flatten(fe)
        return flatt
