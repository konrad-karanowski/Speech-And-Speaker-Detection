from pyclbr import Function
from typing import *

import librosa
import numpy as np

from utils.data_utils.fourier_transform import stft


def scale(x: np.ndarray, x_min: float = 0.0, x_max: float = 1.0) -> np.ndarray:
    x_std = (x - x.min()) / (x.max() - x.min())
    x_scaled = x_std * (x_max - x_min) + x_min
    return x_scaled


def create_spectrogram(
    audio: np.ndarray, 
    frame_size: int, 
    hop_size: int, 
    window_function: str = 'hann', 
    ref: Optional[Function] = None,
    *args,
    **kwargs) -> np.ndarray:
    """Creates spectrogram using Short-Time Fourier's Transform

    Args:
        audio (np.ndarray): signal of shape (N,)
        frame_size (int, optional): Size of frame. Must be power of 2. Defaults to 2048.
        hop_size (int, optional): Size of hop. Defaults to 512.
        window_function (str, optional): Function to convolve partial signal with. Defaults to 'hann'.
        ref (Optional[Function], optional): reference to librosa.power_to_db. Defaults to None.

    Raises:
        ValueError: If such window function is not supported.

    Returns:
        np.ndarray: spectrogram of size (freq_bins, win_size) = (frame_size // 2 + 1, (n_samples - frame_size) // hop_size + 1)
    """
    x_ft = stft(
        x=audio, 
        frame_size=frame_size, 
        hop_size=hop_size,
        window_function=window_function,
    )
    if ref is not None:
        spectrogram = librosa.power_to_db(np.abs(x_ft), ref=ref)
    else:
        spectrogram = librosa.power_to_db(np.abs(x_ft))
    return spectrogram


def create_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    frame_size: int, 
    hop_size: int, 
    window_function: str = 'hann', 
    num_mels: int = 128, 
    ref: Optional[Function] = None,
    *args,
    **kwargs) -> np.ndarray:
    """Creates mel spectrogram

    Args:
        audio (np.ndarray): _description_
        sr (int): _description_
        frame_size (int): _description_
        hop_size (int): _description_
        window_function (str, optional): _description_. Defaults to 'hann'.
        num_mels (int, optional): _description_. Defaults to 128.
        ref (Optional[Function], optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=frame_size,
        hop_length=hop_size,
        window=window_function,
        center=False
    )
    if ref is not None:
        spectrogram = librosa.power_to_db(np.abs(mel_spectrogram), ref=ref)
    else:
        spectrogram = librosa.power_to_db(np.abs(mel_spectrogram))
    return spectrogram


def _maybe_resample_signal(
        signal: np.ndarray, 
        sr: int, 
        target_sr: int, 
        res_type: str
    ) -> np.ndarray:
    if sr != target_sr:
        signal = librosa.resample(
            signal, 
            orig_sr=sr, 
            target_sr=target_sr, 
            res_type=res_type
        )
    return signal

def _maybe_cut_down_signal(
        signal: np.ndarray,
        target_num_samples
    ) -> np.ndarray:
    if signal.shape[0] > target_num_samples:
        signal = signal[:target_num_samples]
    return signal

def _maybe_pad_right_signal(
        signal: np.ndarray,
        target_num_samples
    ) -> np.ndarray:
    if signal.shape[0] < target_num_samples:
        zeros = np.zeros(target_num_samples)
        zeros[:signal.shape[0]] = signal
        signal = zeros
    return signal


def preprocess_signal(
        signal: np.ndarray, 
        sr: int,
        target_sr: int,
        target_num_samples: int,
        res_type: str,
    ) -> np.ndarray:
    signal = _maybe_resample_signal(signal, sr, target_sr, res_type)
    signal = _maybe_pad_right_signal(signal, target_num_samples)
    signal = _maybe_cut_down_signal(signal, target_num_samples)
    return signal
    
