from typing import *

import torch
import pytorch_lightning as pl



class MetricModel(pl.LightningModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


    def configure_optimizers(self):
        return super().configure_optimizers()
