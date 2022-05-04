from typing import *
import abc

import torch
from torch import nn


class Backbone(nn.Module):

    """
    Base interface for backbone, allowing dynamic embedding's size checking 
    """

    def __init__(self, input_size: Tuple[int, int, int]) -> None:
        super(Backbone, self).__init__()
        self.input_size = input_size
        self.embedding_dim_size = None

    def embedding_size(self) -> int:
        """Returns embedding size. Calculates it if necessary.

        Returns:
            int: Embedding size.
        """
        if self.embedding_dim_size is None:
            with torch.no_grad():
                sample = torch.rand(1, * self.input_size)
                sample_embedding = self.forward(sample)
                self.embedding_dim_size = sample_embedding.shape[1]
        return self.embedding_dim_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

