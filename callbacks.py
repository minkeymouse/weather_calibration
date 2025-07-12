from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

class WalkForward(Callback):
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dm = trainer.datamodule
        dm.advance_window()
        dm.setup(stage="fit")