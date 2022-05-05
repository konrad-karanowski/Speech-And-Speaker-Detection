from typing import *
import abc
import os
from datetime import datetime

import wandb
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import Callback

from common import PROJECT_ROOT


class AbstractSaveOutputsCallback(Callback):

    def __init__(self) -> None:
        super(AbstractSaveOutputsCallback, self).__init__()
        self.outputs = []

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.outputs.append(outputs)

    def _process_outputs(self, output):
        subpoutput = {}
        if 'speaker_logit' in output.keys():
            subpoutput['label_target'] = output['label_target']
            subpoutput['speaker_target'] = output['speaker_target']
            subpoutput['speaker_0'] = output['speaker_logit'][:, 0]
            subpoutput['speaker_1'] = output['speaker_logit'][:, 1]
            subpoutput['label_0'] = output['label_logit'][:, 0]
            subpoutput['label_1'] = output['label_logit'][:, 1]
            return subpoutput


    def _create_dataframe(self) -> pd.DataFrame:
        if not self.outputs:
            return None
        dfs = []
        for suboutput in self.outputs:
            suboutput = self._process_outputs(suboutput)
            df = pd.DataFrame(suboutput)
            dfs.append(df)
        cat_df = pd.concat(dfs, ignore_index=True)     
        return cat_df   

    @abc.abstractmethod
    def on_test_end(self, *args, **kwargs):
        pass

class SaveOutputsWandb(AbstractSaveOutputsCallback):

    def __init__(self, save_name: str = 'outputs.csv'):
        super(SaveOutputsWandb, self).__init__()
        self.save_name = save_name

    def on_test_end(self, *args, **kwargs) -> None:
        df = self._create_dataframe()
        if df is not None:
            df.to_csv(os.path.join(wandb.run.dir, self.save_name), index=False)


class SaveOutputsLocal(AbstractSaveOutputsCallback):
    
    def __init__(self, save_dir: str, **kwargs) -> None:

        super(SaveOutputsLocal, self).__init__()
        self.save_dir = str(PROJECT_ROOT / save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_name = f"{datetime.now().strftime('%m%d%Y_%h%m%s')}" + '_'.join(f'{key}={value}' for key, value in kwargs.items()) + ".csv"

    def on_test_end(self, *args, **kwargs):
        df = self._create_dataframe()
        if df is not None:
            save_path = os.path.join(self.save_dir, self.save_name)
            df.to_csv(save_path, index=False)
