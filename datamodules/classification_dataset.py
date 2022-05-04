from typing import * 

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class ClassificationDataset(Dataset):

    def __init__(self, data: pd.DataFrame, transforms: Compose) -> None:
        super(ClassificationDataset, self).__init__()
        self.data = data
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def _read_img(self, path: str) -> torch.Tensor:
        img = np.load(path)
        return self.transforms(img).float()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.data.iloc[index]
        x = self._read_img(sample['path'])
        y_speaker = int(sample['speaker_id'])
        y_label = int(sample['label'])
        
        return {
            'x': x,
            'y_speaker': y_speaker,
            'y_label': y_label
        }