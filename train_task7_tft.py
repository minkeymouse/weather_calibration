import pandas as pd
import numpy as np

import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

class WeatherDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 64, num_workers: int = 4, val_fraction: float = 0.1):
        super().__init__()
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.val_fraction = val_fraction

    def prepare_data(self) -> None:
        # 1) read raw train & test
        raw_train = pd.read_csv("data/weather_train.csv")
        self.test_df = pd.read_csv("data/weather_test.csv")

        # 2) convert to categories
        for df in (raw_train, self.test_df):
            for col in ["solar_term", "hod", "dow", "moy"]:
                df[col] = df[col].astype(str).astype("category")

        # 3) manual 90/10 time‐based split per series
        train_splits = []
        val_splits   = []
        for seg_id, group in raw_train.groupby("segment_id"):
            # ensure sorted by time_idx
            group = group.sort_values("time_idx")
            n      = len(group)
            split  = int((1 - self.val_fraction) * n)
            train_splits.append(group.iloc[:split])
            val_splits.append(  group.iloc[split:])
        self.train_df = pd.concat(train_splits).reset_index(drop=True)
        self.val_df   = pd.concat(val_splits).reset_index(drop=True)

    def setup(self, stage: str = None) -> None:
        # factory to build a dataset
        def make_dataset(df, is_train: bool):
            return TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target=["humid_obs", "degC_obs", "mmHg_obs"],
                group_ids=["segment_id"],
                max_encoder_length=336,
                min_encoder_length=72,
                min_prediction_idx=73,
                max_prediction_length=24,
                min_prediction_length=24,
                static_categoricals=None,
                static_reals=None,
                time_varying_known_categoricals=["solar_term","hod","dow","moy"],
                time_varying_known_reals=[
                    "sunlight_fcst","humid_fcst","abs_humid_fcst",
                    "degC_fcst","hPa_fcst",
                ],
                time_varying_unknown_categoricals=None,
                time_varying_unknown_reals=["humid_obs","degC_obs","mmHg_obs"],
                lags={"humid_obs":[1], "degC_obs":[1], "mmHg_obs":[1]},
                add_relative_time_idx=False,
                add_target_scales=False,
                add_encoder_length="auto",
                target_normalizer="auto",
                randomize_length=is_train,
                allow_missing_timesteps=False,
            )

        # build train/val/test datasets
        self.train_ds = make_dataset(self.train_df, is_train=True)
        # turn off randomness and use full windows for validation
        self.val_ds   = TimeSeriesDataSet.from_dataset(
            self.train_ds, self.val_df,
            predict=True, stop_randomization=True
        )
        self.test_ds  = make_dataset(self.test_df, is_train=False)

    def train_dataloader(self):
        return self.train_ds.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return self.val_ds.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return self.test_ds.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    dm = WeatherDataModule(batch_size=256, num_workers=4, val_fraction=0.1)
    dm.prepare_data()
    dm.setup()

    early_stop = EarlyStopping(
        monitor="val_loss",   # or whatever metric you care about
        patience=30,          # stop after 30 epochs with no improvement
        mode="min",           # smaller loss is better
        verbose=True
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        dirpath="checkpoints",
        filename="best-tft-{epoch:02d}-{val_loss:.4f}"
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    model = TemporalFusionTransformer.from_dataset(
        dm.train_ds,
        learning_rate=1e-3,
        hidden_size=256,
        lstm_layers=3,
        attention_head_size=16,
        dropout=0.1,
        hidden_continuous_size=128,
        loss=QuantileLoss(quantiles=(0.1, 0.25, 0.5, 0.9)),
        output_size=[4, 4, 4],
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
        max_epochs=200,
        min_epochs=30,
        callbacks=[early_stop, checkpoint_cb, lr_monitor],
        # overfit_batches=32,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
