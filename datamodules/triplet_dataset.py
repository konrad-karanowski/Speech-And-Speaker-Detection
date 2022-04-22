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
        return self.transforms(img).float()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Ignore multiple datasets, due to time consuming process
        """
        anchor = self.data.iloc[index]

        positive_label = (self.data[self.data['label'] == anchor['label']]).sample(n=1).iloc[0]
        negative_label = (self.data[self.data['label'] != anchor['label']]).sample(n=1).iloc[0]

        positive_speaker = (self.data[self.data['speaker_id'] == anchor['speaker_id']]).sample(n=1).iloc[0]
        negative_speaker = (self.data[self.data['speaker_id'] != anchor['speaker_id']]).sample(n=1).iloc[0]
        
        return {
            'anchor': self._read_img(anchor['path']), 'anchor_target': anchor['label'], 'anchor_user': anchor['speaker_id'],
            'pos_label': self._read_img(positive_label['path']), 'pos_label_target': positive_label['label'],
            'neg_label': self._read_img(negative_label['path']), 'neg_label_target': negative_label['label'],
            'pos_speaker': self._read_img(positive_speaker['path']), 'pos_speaker_user': positive_speaker['speaker_id'],
            'neg_speaker': self._read_img(negative_speaker['path']), 'neg_speaker_user': negative_speaker['speaker_id']
        }
