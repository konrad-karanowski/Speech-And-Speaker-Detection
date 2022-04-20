from typing import *


import torch
from torch import nn


class Backbone:

    @property
    def final_dim_size(self) -> int:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

