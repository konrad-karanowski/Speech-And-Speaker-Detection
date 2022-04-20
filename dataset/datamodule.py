from typing import *

import shutil
import os
import pytorch_lightning as pl
import pandas as pd
import tqdm
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset.transforms import default_transform
from dataset.triplet_dataset import TripletDataset
from utils import create_df, get_classes, create_spectrogram, create_mel_spectrogram, preprocess_signal


class SpeechDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super(SpeechDataModule, self).__init__()
        self.config = config
        self.data = None

    def prepare_data(self) -> None:
        """Base pipeline for preparing data:

        - Download data if not exists TODO
        - Create dataframe from wav dataset
        - Create spectrograms using stft
        - Create dataframe from image dataset
        """
        data_dir = os.path.join(self.config.root, self.config.data_dir)
        img_dir = os.path.join(self.config.root, self.config.img_dir)

        self._maybe_download_data(data_dir)
        self._maybe_create_spectrograms(data_dir, img_dir)

        self.data = create_df(
            classes=get_classes(img_dir),
            path=img_dir
        )

    def _maybe_download_data(self, data_dir: str) -> None:
        # TODO, do this 
        if os.path.exists(data_dir):
            return

    def _maybe_create_spectrograms(self, data_dir: str, img_dir: str) -> None:
        """Create spectrograms from audio files

        Args:
            data_dir (str): combined path to raw audio data
            img_dir (str): combined path to spectrograms data
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

            for _, item in tqdm.tqdm(df.iterrows(), total=len(df), desc='Create spectrogram from wav files...'):
                path = item['path']
                audio, sr = librosa.load(path)
                processed_audio = preprocess_signal(
                    signal=audio,
                    sr=sr,
                    target_sr=self.config.target_sr,
                    target_num_samples=self.config.target_num_samples,
                    res_type=self.config.res_type
                )
                # TODO add different spectrograms
                spectrogram = create_mel_spectrogram(
                    audio=processed_audio,
                    sr=sr,
                    frame_size=self.config.frame_size,
                    hop_size=self.config.hop_size,
                    window_function=self.config.window_function
                )
                new_path = path.replace(self.config.data_dir, self.config.img_dir).replace('.wav', '.npy')
                np.save(new_path, spectrogram)
        except Exception as e:
            print(e)
            shutil.rmtree(img_dir)

    def setup(self) -> None:
        """Setup data for experiment
        """
        train_df, remain = train_test_split(self.data, test_size=0.3, random_state=1, stratify=self.data['label'])
        val_df, test_df = train_test_split(remain, test_size=0.8, random_state=1, stratify=remain['label'])

        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'

        self.data = pd.concat([train_df, val_df, test_df], ignore_index=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            TripletDataset(self.data[self.data['split'] == 'train'], transforms=default_transform()),
            shuffle=True,
            batch_size=self.config.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            TripletDataset(self.data[self.data['split'] == 'val'], transforms=default_transform()),
            shuffle=False,
            batch_size=self.config.batch_size
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            TripletDataset(self.data[self.data['split'] == 'test'], transforms=default_transform()),
            shuffle=False,
            batch_size=self.config.batch_size
        )
