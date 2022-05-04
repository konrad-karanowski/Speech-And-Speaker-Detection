from typing import *

import numpy as np
from torch.utils.data import DataLoader

from datamodules.base import BaseSpeechDataModule
from datamodules.transforms import default_transform
from datamodules.fine_tuning.classification_dataset import ClassificationDataset

class FineTuneSpeechDatamodule(BaseSpeechDataModule):

    def __init__(self, **kwargs) -> None:
        """Datamodule for fine-tuning model. 
        Simplifies task for binary classification: Does the audio contain target word and is the word spoken by target speaker.
        """
        super(FineTuneSpeechDatamodule, self).__init__(**kwargs)

    def setup(self) -> None:
        """Setup dataset. Simplify task for binary classification.
        """
        super(FineTuneSpeechDatamodule, self).setup()
        self.data['label'] = np.where(self.data['label'] == self.hparams.target_label, 1, 0)
        self.data['speaker_id'] = np.where(self.data['speaker_id'] == self.hparams.target_speaker, 1, 0)

    def train_dataloader(self) -> DataLoader:
        """Creates train dataloader (for multitask binary classification).

        Returns:
            DataLoader: ClassificationDataset's dataloader.
        """
        return DataLoader(
            ClassificationDataset(self.data[self.data['split'] == 'train'], transforms=default_transform()),
            shuffle=True,
            batch_size=self.hparams.batch_size
        )

    def val_dataloader(self) -> DataLoader:
        """Creates validation dataloader (for multitask binary classification).

        Returns:
            DataLoader: ClassificationDataset's dataloader.
        """
        return DataLoader(
            ClassificationDataset(self.data[self.data['split'] == 'val'], transforms=default_transform()),
            shuffle=False,
            batch_size=self.hparams.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        """Creates test dataloader (for multitask binary classification).

        Returns:
            DataLoader: ClassificationDataset's dataloader.
        """
        return DataLoader(
            ClassificationDataset(self.data[self.data['split'] == 'test'], transforms=default_transform()),
            shuffle=False,
            batch_size=self.hparams.batch_size
        )
    