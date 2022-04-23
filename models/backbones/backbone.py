from typing import *

import torch
from torch import nn


class Backbone(nn.Module):

    def __init__(self, input_size: Tuple[int, int, int]):
        super(Backbone, self).__init__()
        self.input_size = input_size
        self.embedding_dim_size = None

    def embedding_size(self) -> int:
        if self.embedding_dim_size is None:
            with torch.no_grad():
                sample = torch.rand(1, *self.input_size)
                sample_embedding = self.forward(sample)
                self.embedding_dim_size = sample_embedding.shape[1]
        return self.embedding_dim_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CustomBackbone(Backbone):

    def __init__(self, input_size: Tuple[int, int, int]) -> None:
        super(CustomBackbone, self).__init__(input_size)
        self.fe = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(3),
            nn.GELU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3),
            nn.GELU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3),
            nn.GELU()
        )
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fe = self.fe(x)
        flatt = self.flatten(fe)
        return flatt
