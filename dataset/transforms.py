import torch
from torchvision import transforms as trf


def default_transforms() -> trf.Compose:
    return trf.Compose([
        trf.ToTensor()
    ])
