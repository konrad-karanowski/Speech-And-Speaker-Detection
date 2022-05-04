from typing import * 

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class PairwiseDataset(Dataset):

    def __init__(self, data: pd.DataFrame, transforms: Compose) -> None:
        """Pairwise dataset for easier evaluation of the model. Keeps balance of classes in the dataset.

        Args:
            data (pd.DataFrame): Dataframe with data for the dataset.
            transforms (Compose): Transforms to apply on samples.
        """
        super(PairwiseDataset, self).__init__()
        self.data = data
        self.transforms = transforms
        self.queue = 0

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

    def _get_balanced(self, label: str, speaker: str) -> pd.Series:
        """Keeps balance in samples by alternating between drawing the same label as sample's and the same speaker.

        Args:
            label (str): Label of anchor.
            speaker (str): Speaker of anchor.

        Returns:
            pd.Series: Choosen sample.
        """
        if self.queue % 2 == 0:
            sample = self.data[self.data['label'] == label].sample(n=1).iloc[0]
        else:
            sample = self.data[self.data['speaker_id'] == speaker].sample(n=1).iloc[0]
        self.queue += 1
        return sample 

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Returns anchor and another sample. Keeps balance of classes.

        Args:
            index (int): Index of sample.

        Returns:
            Dict[str, torch.Tensor]: Anchor, another sample, their labels and speakers.
        """
        anchor = self.data.iloc[index]
        
        sample = self._get_balanced(label=anchor['label'], speaker=anchor['speaker_id'])

        label_target = int(sample['label'] == anchor['label'])
        speaker_target = int(sample['speaker_id'] == anchor['speaker_id'])
        
        return {
            'anchor': self._read_spectrogram(anchor['path']),
            'anchor_label': anchor['label'],
            'anchor_speaker': anchor['speaker_id'],
            'sample': self._read_spectrogram(sample['path']),
            'sample_label': sample['label'],
            'sample_speaker': sample['speaker_id'],
            'label_target': label_target,
            'speaker_target': speaker_target
        }

