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


class SelfAttentionPooling(nn.Module):
    """
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 16)
        
    def forward(self, batch_rep: torch.Tensor) -> torch.Tensor:
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


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


class CatalunaBackbone(Backbone):

    def __init__(self, input_size: Tuple[int, int, int]) -> None:
        super(CatalunaBackbone, self).__init__(input_size)
        self.fe = nn.Sequential(
            nn.Conv2d(1, 128, (3, 3), (1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, (3, 3), (1, 1)),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, (3, 3), (1, 1)),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1)),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.MaxPool2d(2, 2)
        )
        # self.sap = SelfAttentionPooling(16)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fe = self.fe(x)
        flatt = self.flatten(fe)
        return flatt
