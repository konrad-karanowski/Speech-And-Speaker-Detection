from typing import *

from torch.utils.data import DataLoader

from datamodules.base import BaseSpeechDataModule
from datamodules.transforms import default_transform
from datamodules.pre_training.triplet_dataset import TripletDataset
from datamodules.pre_training.pairwise_dataset import PairwiseDataset


class SiameseSpeechDatamodule(BaseSpeechDataModule):

    def __init__(self, **kwargs):
        """Datamodule for training siamese model. 
        """
        super(SiameseSpeechDatamodule, self).__init__(**kwargs)

    def train_dataloader(self) -> DataLoader:
        """Creates train dataloader (for triplet margin learning).

        Returns:
            DataLoader: TripletDataset's dataloader.
        """
        return DataLoader(
            TripletDataset(self.data[self.data['split'] == 'train'], transforms=default_transform()),
            shuffle=True,
            batch_size=self.hparams.batch_size
        )

    def val_dataloader(self) -> DataLoader:
        """Creates validation dataloader (for triplet margin learning).

        Returns:
            DataLoader: TripletDataset's dataloader.
        """
        return DataLoader(
            TripletDataset(self.data[self.data['split'] == 'val'], transforms=default_transform()),
            shuffle=False,
            batch_size=self.hparams.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        """Creates test dataloader (for pairwise comparison). 

        Returns:
            DataLoader: PairwiseDataset's dataloader.
        """
        return DataLoader(
            PairwiseDataset(self.data[self.data['split'] == 'test'], transforms=default_transform()),
            shuffle=False,
            batch_size=self.hparams.batch_size
        )

