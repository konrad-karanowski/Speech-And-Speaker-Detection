from typing import *

import os
import pytorch_lightning as pl
import pandas as pd
import tqdm
import librosa
import matplotlib.pyplot as plt

from utils.data_utils import create_df, get_classes, create_spectrogram, create_mel_spectrogram



class SpeechDataModule(pl.LightningDataModule):
    
    def __init__(self, config):
        super(SpeechDataModule, self).__init__()
        self.config = config
        self.data = None

    def prepare_data(self) -> None:
        """Base pipeline for preparing data:
        1. Download data if not exists TODO
        2. Create dataframe from wav dataset
        3. Create spectrograms using stft
        4. Create dataframe from image dataset
        """
        data_dir = os.path.join(self.config.root, self.config.data_dir)
        img_dir = os.path.join(self.config.root, self.config.img_dir)

        if not os.path.exists(os.path.join(data_dir)):
            # TODO download data
            print('Nie ma datasetu, pobiez go -.-')

        if not os.path.exists(os.path.join(img_dir)):
            self._create_spectrograms(data_dir, img_dir)


        self.data = create_df(
            classes=get_classes(img_dir),
            path=img_dir
        )
        print(self.data)

    def _create_spectrograms(self, data_dir: str, img_dir: str):
        
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
            # TODO add different spectrograms
            # spectrogram = create_spectrogram(
            #     audio=audio,
            #     frame_size=self.config.frame_size,
            #     hop_size=self.config.hop_size,
            #     window_function=self.config.window_function
            # )
            spectrogram = create_mel_spectrogram(
                audio=audio,
                sr=sr,
                frame_size=self.config.frame_size,
                hop_size=self.config.hop_size,
                window_function=self.config.window_function
            )
            new_path = path.replace(self.config.data_dir, self.config.img_dir).replace('.wav', '.jpg')
            plt.imsave(new_path, spectrogram)

    def setup(self) -> None:
        pass


    def train_dataloader(self):
        return super().train_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()
