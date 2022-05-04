from typing import *
import abc
import os
import shutil

import tqdm
import hydra
import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import create_df, get_classes


class BaseSpeechDataModule(pl.LightningDataModule):

    PROPER_CLASSES = [
        'zero',
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine',
        'other'
    ]

    def __init__(self, **kwargs) -> None:
        """Base datamodule for speech including preprocessing data, creating spectrograms and dataloaders.
        """
        super(BaseSpeechDataModule, self).__init__()
        self.save_hyperparameters()
        self.data = None

    def prepare_data(self) -> None:
        """Base pipeline for preparing data:

        - Download data if not exists TODO
        - Create dataframe from wav dataset
        - Create spectrograms using stft
        - Create dataframe from image dataset
        """
        data_dir = os.path.join(self.hparams.root, self.hparams.data_dir)
        img_dir = os.path.join(self.hparams.root, self.hparams.img_dir)

        self._maybe_download_data(data_dir)
        self._maybe_create_spectrograms(data_dir, img_dir)

        self.data = create_df(
            classes=get_classes(img_dir),
            path=img_dir
        )
        if self.hparams.keep_proper_classes:
            self._keep_only_propper()

    def _keep_only_propper(self) -> None:
        self.data['label'] = np.where(self.data['label'].isin(self.PROPER_CLASSES), self.data['label'], 'other')

    @property
    def input_size(self) -> Tuple[int, int, int]:
        """Returns size of data sample. 

        Raises:
            ValueError: If there is no prepared_data for sampling.

        Returns:
            Tuple[int, int, int]: Spectrogram input size.
        """
        if self.data is None:
            raise ValueError('There is no data in this datamodule. Call .prepare_data() first!')
        sample_shape = np.load(self.data.iloc[0]['path']).shape
        if len(sample_shape) == 2:
            return 1, sample_shape[0], sample_shape[1]
        return sample_shape

    def _maybe_download_data(self, data_dir: str) -> None:
        # TODO, do this 
        if os.path.exists(data_dir):
            return

    def _maybe_create_spectrograms(self, data_dir: str, img_dir: str) -> None:
        """Create spectrograms from audio files and stores them as numpy arrays.

        Args:
            data_dir (str): Combined path to raw audio data dir.
            img_dir (str): Combined path to spectrograms data dir.
        """
        if os.path.exists(os.path.join(img_dir)):
            return

        try:
            # create df
            classes = get_classes(data_dir)
            df = create_df(classes, data_dir)
            # create dirs 
            os.makedirs(img_dir, exist_ok=True)
            for cls in classes:
                os.makedirs(os.path.join(img_dir, cls), exist_ok=True)

            preprocess_method = hydra.utils.instantiate(self.hparams.process_audio_method, _partial_=True)
            spectogram_method = hydra.utils.instantiate(self.hparams.spectrogram_method, _partial_=True)

            for _, item in tqdm.tqdm(df.iterrows(), total=len(df), desc='Create spectrogram from wav files...'):
                path = item['path']
                audio, sr = librosa.load(path)
                processed_audio = preprocess_method(
                    signal=audio,
                    sr=sr,
                )
                spectrogram = spectogram_method(audio=processed_audio, sr=sr)
                new_path = path.replace(self.hparams.data_dir, self.hparams.img_dir).replace('.wav', '.npy')
                np.save(new_path, spectrogram)
        except Exception as e:
            print(e)
            shutil.rmtree(img_dir)

    def setup(self) -> None:
        """Setup data for experiment
        """
        train_df, remain = train_test_split(self.data, train_size=self.hparams.train_vs_rest_size, random_state=1,
                                            stratify=self.data['label'])
        val_df, test_df = train_test_split(remain, train_size=self.hparams.val_vs_test_size, random_state=1,
                                           stratify=remain['label'])

        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'

        self.data = pd.concat([train_df, val_df, test_df], ignore_index=True)

    @abc.abstractmethod
    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abc.abstractmethod
    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abc.abstractmethod
    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError
