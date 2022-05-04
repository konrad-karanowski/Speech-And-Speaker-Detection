from typing import *

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class TripletDataset(Dataset):

    def __init__(self, data: pd.DataFrame, transforms: Compose) -> None:
        """Dataset for triplet margin loss learning.

        Args:
            data (pd.DataFrame): Dataframe with data for the dataset.
            transforms (Compose): Transforms to apply on samples.
        """
        super(TripletDataset, self).__init__()
        self.data = data
        self.transforms = transforms

    def __len__(self) -> int:
        """Returns the size of the dataset.

        Returns:
            int: Size of the dataset.
        """
        return len(self.data)

    def _read_spectrogram(self, path: str) -> torch.Tensor:
        """Reads spectrogram from precomputed file, stored as np.array.

        Args:
            path (str): Path to the spectrogram.

        Returns:
            torch.Tensor: Spectrogram as tensor after transforms.
        """
        img = np.load(path)
        return self.transforms(img).float()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Returns sample (anchor) with positive and negative samples, along all dimensions (label, speaker).

        Args:
            index (int): Index of sample.

        Returns:
            Dict[str, torch.Tensor]: Sample (anchor) with positive and negative samples, along all dimensions (label, speaker) and their labels.
        """
        anchor = self.data.iloc[index]

        positive_label = (self.data[self.data['label'] == anchor['label']]).sample(n=1).iloc[0]
        negative_label = (self.data[self.data['label'] != anchor['label']]).sample(n=1).iloc[0]

        positive_speaker = (self.data[self.data['speaker_id'] == anchor['speaker_id']]).sample(n=1).iloc[0]
        negative_speaker = (self.data[self.data['speaker_id'] != anchor['speaker_id']]).sample(n=1).iloc[0]
        
        return {
            'anchor': self._read_spectrogram(anchor['path']), 'anchor_target': anchor['label'], 'anchor_user': anchor['speaker_id'],
            'pos_label': self._read_spectrogram(positive_label['path']), 'pos_label_target': positive_label['label'],
            'neg_label': self._read_spectrogram(negative_label['path']), 'neg_label_target': negative_label['label'],
            'pos_speaker': self._read_spectrogram(positive_speaker['path']), 'pos_speaker_user': positive_speaker['speaker_id'],
            'neg_speaker': self._read_spectrogram(negative_speaker['path']), 'neg_speaker_user': negative_speaker['speaker_id']
        }
