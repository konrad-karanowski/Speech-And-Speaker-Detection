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
        self.queue = 0

    def __len__(self) -> int:
        return len(self.data)

    def _read_img(self, path: str) -> torch.Tensor:
        img = np.load(path)
        return self.transforms(img).float()

    def _get_balanced(self, label: str, speaker: str) -> pd.Series:
        if self.queue % 2 == 0:
            sample = self.data[self.data['label'] == label].sample(n=1).iloc[0]
        else:
            sample = self.data[self.data['speaker_id'] == speaker].sample(n=1).iloc[0]
        self.queue += 1
        return sample 

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        anchor = self.data.iloc[index]
        
        sample = self._get_balanced(label=anchor['label'], speaker=anchor['speaker_id'])

        label_target = int(sample['label'] == anchor['label'])
        speaker_target = int(sample['speaker_id'] == anchor['speaker_id'])
        
        return {
            'anchor': self._read_img(anchor['path']),
            'anchor_label': anchor['label'],
            'anchor_speaker': anchor['speaker_id'],
            'sample': self._read_img(sample['path']),
            'sample_label': sample['label'],
            'sample_speaker': sample['speaker_id'],
            'label_target': label_target,
            'speaker_target': speaker_target
        }

