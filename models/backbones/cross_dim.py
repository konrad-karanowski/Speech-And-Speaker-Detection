from typing import *

import torch 
from torch import nn

from models.backbones.backbone import Backbone

class WidthCrossBackbone(Backbone):

    """
    Smaller backbone treating different dimensions in separate way.
    """

    def __init__(self, input_size: Tuple[int, int, int]) -> None:
        super(WidthCrossBackbone, self).__init__(input_size)

        self.fe = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 3)),
            nn.MaxPool2d(kernel_size=(1, 3)),
            #nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=(1, 7)),
            nn.MaxPool2d(kernel_size=(1, 4)),
            #nn.GELU(),

            nn.Conv2d(128, 256, kernel_size=(1, 10)),
            #nn.GELU(),

            nn.Conv2d(256, 512, kernel_size=(7, 2)),
            #nn.GELU(),
        )
        self.gap = nn.AvgPool2d(1, 27)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fe = self.fe(x)
        res = self.gap(fe).squeeze(-1).squeeze(-1)
        return res
