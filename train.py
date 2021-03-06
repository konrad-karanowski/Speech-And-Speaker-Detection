from typing import *
import logging

import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig

from common import PROJECT_ROOT
from models.siamese_model import SiameseModel

pylogger = logging.getLogger(__name__)


def build_callbacks(cfg: Any, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg (Any): List of callbacks instantiable configuration.

    Returns:
        List[Callback]: List of callbacks for training.
    """
    callbacks: List[Callback] = list(args)
    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


@hydra.main(config_path=str(PROJECT_ROOT / 'config'), config_name='default')
def main(config: DictConfig) -> None:
    """Creates datamodule, model, trains and tests it.

    Args:
        config (DictConfig): Configuration provided by Hydra.
    """
    datamodule = hydra.utils.instantiate(config.datamodule.datamodule, _recursive_=False)
    datamodule.prepare_data()

    if 'checkpoint' in config.model.module.keys():
        pretrained = SiameseModel.load_from_checkpoint(config.model.module.checkpoint)
        model = hydra.utils.instantiate(config.model.module, model=pretrained, _recursive_=False)
    else:
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
