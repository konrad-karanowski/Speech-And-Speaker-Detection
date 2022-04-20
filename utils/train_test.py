import pytorch_lightning as pl


def train_test(
        model,
        datamodule,
        logger,
        config
):
    datamodule.setup()
    trainer = pl.Trainer(gpus=-1, logger=logger)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader)
