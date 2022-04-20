import torch
from torchvision.transforms import Compose, ToTensor

from audiomentations import SpecCompose, SpecChannelShuffle, SpecFrequencyMask


def spectrogram_transforms() -> SpecCompose:
    """Base transforms using spectrogram version of dataset

    Returns:
        SpecCompose: composition of transforms
    """
    return SpecCompose([
        SpecChannelShuffle(p=0.5),
        SpecFrequencyMask(p=0.5)
    ])


def default_transform() -> Compose:
    """Default transform for signal

    Returns:
        trf.Compose: composition of transforms
    """
    return Compose([
        ToTensor()
    ])
