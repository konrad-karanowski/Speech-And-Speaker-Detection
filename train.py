import logging
from typing import *

import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from common import PROJECT_ROOT

pylogger = logging.getLogger(__name__)


def build_callbacks(cfg: Any, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.
    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated
    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)
    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


@hydra.main(config_path=str(PROJECT_ROOT / 'config'), config_name='default')
def main(config):
    import os
    print(os.path.abspath(os.path.curdir))
    datamodule = hydra.utils.instantiate(config.datamodule.datamodule, _recursive_=False)
    datamodule.prepare_data()
    datamodule.setup()

    model = hydra.utils.instantiate(config.model.module, input_size=datamodule.input_size, _recursive_=False)

    logger = hydra.utils.instantiate(config.train.logger)

    callbacks = build_callbacks(config.train.callbacks)

    datamodule.setup()
    trainer = pl.Trainer(
            logger=logger, 
            callbacks=callbacks,
            **config.train.trainer
        )
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader)


if __name__ == '__main__':
    main()
