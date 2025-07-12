# train_all.py

from dataset import ForecastingDataModule
import config as cfg
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from callbacks import WalkForward
from pytorch_forecasting.models import TemporalFusionTransformer, TiDEModel, TimeXer

import warnings

# sklearn 의 feature-name 경고 무시
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)

def _train_and_test(model, name, dm):
    """Helper to attach callbacks, train, then test on best ckpt."""
    # callbacks
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"{name}-{{epoch:02d}}-{{val_loss:.4f}}"
    )
    earlystop_cb = EarlyStopping(
        monitor="val_loss",
        patience=cfg.PATIENCE,
        mode="min"
    )
    wf_cb = WalkForward()

    trainer = Trainer(
        max_epochs=cfg.MAX_EPOCHS,
        accelerator="auto",
        callbacks=[checkpoint_cb, earlystop_cb, wf_cb],
        log_every_n_steps=cfg.LOG_INTERVAL,
        gradient_clip_val=cfg.GRAD_CLIP_VAL,
        reload_dataloaders_every_n_epochs=1,
    )

    # fit
    trainer.fit(model, datamodule=dm)

    # prepare test set and test best checkpoint
    dm.setup(stage="test")
    test_res = trainer.test(datamodule=dm, ckpt_path="best")
    print(f"\n=== {name.upper()} test results ===\n", test_res)


def main():
    # We'll train three models in turn, re-building the DataModule each time
    for name, ModelClass, params in [
        ("tft",    TemporalFusionTransformer, cfg.TFT_PARAMS),
        #("tide",   TiDEModel,                cfg.TIDE_PARAMS),
        #("txr",    TimeXer,                  cfg.TXR_PARAMS),
    ]:
        print(f"\n\n>>>>> TRAINING {name.upper()} <<<<<")
        # 1) build data
        dm = ForecastingDataModule()
        dm.prepare_data()          # loads CSVs & builds full_ds
        dm.setup(stage="fit")      # slices into train/val

        # 2) instantiate model from the shared full_ds
        model = ModelClass.from_dataset(dm.full_ds, **params)

        # 3) train & test
        _train_and_test(model, name, dm)


if __name__ == "__main__":
    main()
