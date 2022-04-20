from typing import *

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class TripletDataset(Dataset):

    def __init__(self, data: pd.DataFrame, transforms: Compose) -> None:
        super(TripletDataset, self).__init__()
        self.data = data
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def _read_img(self, path: str) -> torch.Tensor:
        img = np.load(path)
        return self.transforms(img)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        anchor = self.data.iloc[index]
        while True:
            positive = (self.data[self.data['label'] == anchor['label']]).sample(n=1).iloc[0]
            if positive.name != anchor.name:
                break
        negative = (self.data[self.data['label'] != anchor['label']]).sample(n=1).iloc[0]
        return {
            'anchor': self._read_img(anchor['path']),
            'positive': self._read_img(positive['path']),
            'negative': self._read_img(negative['path']),
            'positive_label': anchor['label'],
            'negative_label': negative['label']
        }
