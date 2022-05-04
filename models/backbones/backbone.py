from typing import *

import torch
from torch import nn


class Backbone(nn.Module):

    """
    Base interface for backbone, allowing dynamic embedding's size checking 
    """

    def __init__(self, input_size: Tuple[int, int, int]):
        super(Backbone, self).__init__()
        self.input_size = input_size
        self.embedding_dim_size = None

    def embedding_size(self) -> int:
        if self.embedding_dim_size is None:
            with torch.no_grad():
                sample = torch.rand(1, * self.input_size)
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
