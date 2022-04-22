import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


def train_test(
        model,
        datamodule,
        logger,
        callbacks,
        config
):
    datamodule.setup()
    trainer = pl.Trainer(
            gpus=config.gpus, 
            logger=logger, 
            max_epochs=config.max_epochs,
            min_epochs=config.min_epochs, 
            callbacks=callbacks
        )
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader)
