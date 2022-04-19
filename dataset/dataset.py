import torch
from torch.utils.data import Dataset



class TripletDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index) -> T_co:
        return super().__getitem__(index)
