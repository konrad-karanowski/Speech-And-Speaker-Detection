from typing import *

import librosa
import numpy as np

from utils.fourier_transform import stft


def create_spectrogram(
        audio: np.ndarray,
        frame_size: int,
        hop_size: int,
        window_function: str = 'hann',
        ref: Optional[Callable] = None,
        *args,
        **kwargs) -> np.ndarray:
    """Creates spectrogram using Short-Time Fourier's Transform

    Args:
        audio (np.ndarray): Signal of shape (N,).
        frame_size (int, optional): Size of frame. Must be power of 2. Defaults to 2048.
        hop_size (int, optional): Size of hop. Defaults to 512.
        window_function (str, optional): Function to convolve partial signal with. Defaults to 'hann'.
        ref (Optional[Function], optional): Reference to librosa.power_to_db. Defaults to None.

    Raises:
        ValueError: If such window function is not supported.

    Returns:
        np.ndarray: Spectrogram of size (freq_bins, win_size) = (frame_size // 2 + 1, (n_samples - frame_size) // hop_size + 1)
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
        ref: Optional[Callable] = None,
        *args,
        **kwargs) -> np.ndarray:
    """Creates mel spectrogram

    Args:
        audio (np.ndarray): Signal of shape (N,).
        sr (int): Sampling rate.
        frame_size (int): Size of frame.
        hop_size (int): Hop length.
        window_function (str, optional): Function to convolve partial signal with. Defaults to 'hann'.
        num_mels (int, optional): Number of mel bins. Defaults to 128.
        ref (Optional[Callable], optional): Reference to librosa.power_to_db. Defaults to None.

    Returns:
        np.ndarray: Mel spectrogram of size (..., num_mels).
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=frame_size,
        hop_length=hop_size,
        window=window_function,
        center=False,
        n_mels=num_mels
    )
    if ref is not None:
        spectrogram = librosa.power_to_db(np.abs(mel_spectrogram), ref=ref)
    else:
        spectrogram = librosa.power_to_db(np.abs(mel_spectrogram))
    return spectrogram


def create_mfcc_spectrogram(
    audio: np.ndarray,
    sr: int,
    frame_size: int,
    hop_size: int,
    window_function: str = 'hann',
    num_mels: int = 128,
    num_mfccs: int = 39,
    dct_type: int = 2,
    ref: Optional[Callable] = None,
    *args,
    **kwargs) -> np.ndarray:
    """Creates MFCC spectrogram

    Args:
        audio (np.ndarray): Signal of shape (N,).
        sr (int): Sampling rate.
        frame_size (int): Size of frame.
        hop_size (int): Hop length.
        window_function (str, optional): Function to convolve partial signal with. Defaults to 'hann'.
        num_mels (int, optional): Number of mel bins. Defaults to 128.
        num_mfccs (int, optional): Number of MFCC coefficients. Defaults to 39.
        dct_type (int, optional): Discrete cosine transform (DCT) type. Defaults to 2.
        ref (Optional[Callable], optional): Reference to librosa.power_to_db. Defaults to None.

    Returns:
        np.ndarray: MFCC spectrogram of size (num_mfccs, num_mels)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_fft=frame_size,
        hop_length=hop_size,
        window=window_function,
        center=False,
        n_mels=num_mels,
        n_mfcc=num_mfccs,
        dct_type=dct_type
    )
    if ref is not None:
        spectrogram = librosa.power_to_db(mfcc, ref=ref)
    else:
        spectrogram = librosa.power_to_db(mfcc)
    return spectrogram


def _maybe_resample_signal(
        signal: np.ndarray,
        sr: int,
        target_sr: int,
        res_type: str
) -> np.ndarray:
    """Resample signal if necessary

    Args:
        signal (np.ndarray): Original signal.
        sr (int): Original sample rate.
        target_sr (int): Desired sample rate.
        res_type (str): Strategy for sample rate.

    Returns:
        np.ndarray: Processed signal.
    """
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
    """Cut down signal if necessary

    Args:
        signal (np.ndarray): Original signal.
        target_num_samples (_type_): Target number of samples.

    Returns:
        np.ndarray: Processed signal.
    """
    if signal.shape[0] > target_num_samples:
        signal = signal[:target_num_samples]
    return signal


def _maybe_pad_right_signal(
        signal: np.ndarray,
        target_num_samples
) -> np.ndarray:
    """Pad signal with zeros if necessary

    Args:
        signal (np.ndarray): Original signal.
        target_num_samples (_type_): Target number of samples.

    Returns:
        np.ndarray: Processed signal.
    """
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
    """Process signal adjusting sample rate and length

    Args:
        signal (np.ndarray): Base signal.
        sr (int): original Sampling rate.
        target_sr (int): Target sampling rate.
        target_num_samples (int): Target number of samples.
        res_type (str): Strategy for resampling. 

    Returns:
        np.ndarray: Processed signal.
    """
    signal = _maybe_resample_signal(signal, sr, target_sr, res_type)
    signal = _maybe_pad_right_signal(signal, target_num_samples)
    signal = _maybe_cut_down_signal(signal, target_num_samples)
    return signal
