from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet
import config as cfg

class ForecastingDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.batch_size  = cfg.BATCH_SIZE
        self.num_workers = cfg.NUM_WORKERS
        self.window_step = cfg.WINDOW_EPOCH_ADVANCE

    def prepare_data(self):
        # 1) seed + load CSV
        pl.seed_everything(cfg.SEED, workers=True)
        self.train_df = pd.read_csv(cfg.TRAIN_DF, parse_dates=["date"])
        self.test_df  = pd.read_csv(cfg.TEST_DF,  parse_dates=["date"])

        # 2) cast known-future categoricals consistently across train & test
        for col in cfg.TIME_VARYING_KNOWN_CATEGORICALS:
            if col in self.train_df and col in self.test_df:
                combined = pd.concat([self.train_df[col], self.test_df[col]]).astype(str).unique().tolist()
                dtype = pd.CategoricalDtype(categories=combined)
                self.train_df[col] = self.train_df[col].astype(str).astype(dtype)
                self.test_df[col]  = self.test_df[col].astype(str).astype(dtype)

        # 3) build one full TimeSeriesDataSet for encoding/scaling
        self.full_ds = TimeSeriesDataSet(
            data=self.train_df,
            time_idx=cfg.TIME_IDX,
            target=cfg.TARGET,
            group_ids=cfg.GROUP_IDS,
            time_varying_known_categoricals=cfg.TIME_VARYING_KNOWN_CATEGORICALS,
            time_varying_known_reals=cfg.TIME_VARYING_KNOWN_REALS,
            time_varying_unknown_categoricals=cfg.TIME_VARYING_UNKNOWN_CATEGORICALS,
            time_varying_unknown_reals=cfg.TIME_VARYING_UNKNOWN_REALS,
            allow_missing_timesteps=cfg.ALLOW_MISSING_TIMESTEPS,
            add_target_scales=cfg.ADD_TARGET_SCALES,
            add_relative_time_idx=cfg.ADD_RELATIVE_TIME_IDX,
            add_encoder_length=cfg.ADD_ENCODER_LENGTH,
            randomize_length=cfg.RANDOMIZE_LENGTH,
            target_normalizer=cfg.TARGET_NORMALIZER,
            max_encoder_length=cfg.MAX_ENCODER_LENGTH,
            min_encoder_length=cfg.MIN_ENCODER_LENGTH,
            max_prediction_length=cfg.MAX_PREDICTION_LENGTH,
            min_prediction_length=cfg.MIN_PREDICTION_LENGTH,
        )
        print(f"🔨 full_ds built, total windows: {len(self.full_ds)}")

    def setup(self, stage=None):
        enc    = cfg.MAX_ENCODER_LENGTH
        pred   = cfg.MAX_PREDICTION_LENGTH
        window = enc + pred
        k      = cfg.VALIDATION_WINDOW_COUNT

        # ── train+val split ────────────────────────────────────────────
        if stage in (None, "fit", "validate"):
            print("🛠 Building train_ds & validation_ds...")

            # 1) train_ds: windows ending at or before cutoff+pred
            cutoff_plus_pred = self.cutoff + pred
            self.training_ds = self.full_ds.filter(
                lambda idx: idx.time_idx_last <= cutoff_plus_pred
            )
            print(f"  train windows: {len(self.training_ds)}")

            # 2) validation_ds: windows whose FIRST time ≥ cutoff+1 and LAST ≤ cutoff+window+(k-1)
            val_start = self.cutoff + 1
            max_train_idx = self.train_df["time_idx"].max()
            val_end   = self.cutoff + window + (k - 1)
            val_end = min(val_end, max_train_idx - cfg.MAX_PREDICTION_LENGTH)
            self.validation_ds = self.full_ds.filter(
                lambda idx: (idx.time_idx_first >= val_start)
                         & (idx.time_idx_last  <= val_end)
            )
            print(f"  val windows:   {len(self.validation_ds)}")

        # ── test split ─────────────────────────────────────────────────
        if stage in (None, "test"):
            print("🛠 Building test_ds...")
            # build sliding windows over the held-out test_df
            self.test_ds = TimeSeriesDataSet.from_dataset(
                self.full_ds,     # reuse encoders & scalers
                self.test_df,     # only windows fully inside test_df
                predict=False,
                stop_randomization=True,
            )
            print(f"  test windows:  {len(self.test_ds)}")

    def train_dataloader(self):
        return self.training_ds.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return self.validation_ds.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return self.test_ds.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            persistent_workers=True,
        )

    def advance_window(self):
        """Move the global cutoff forward before the next epoch."""
        self.cutoff += self.window_step
        print(f"📈 cutoff advanced to {self.cutoff}")



class PredictionDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.batch_size  = cfg.BATCH_SIZE
        self.num_workers = cfg.NUM_WORKERS
        self.window_step = cfg.WINDOW_EPOCH_ADVANCE

    def prepare_data(self):
        # 1) seed + load CSV
        pl.seed_everything(cfg.SEED, workers=True)
        self.pred_df = pd.read_csv("input/base_data.csv", parse_dates=["date"])

        # 2) cast known-future categoricals consistently across train & test
        for col in cfg.TIME_VARYING_KNOWN_CATEGORICALS:
            if col in self.pred_df:
                combined = pd.concat([self.pred_df[col]]).astype(str).unique().tolist()
                dtype = pd.CategoricalDtype(categories=combined)
                self.pred_df[col] = self.pred_df[col].astype(str).astype(dtype)

        # 3) build one full TimeSeriesDataSet for encoding/scaling
        self.full_ds = TimeSeriesDataSet(
            data=self.pred_df,
            time_idx=cfg.TIME_IDX,
            target=cfg.TARGET,
            group_ids=cfg.GROUP_IDS,
            time_varying_known_categoricals=cfg.TIME_VARYING_KNOWN_CATEGORICALS,
            time_varying_known_reals=cfg.TIME_VARYING_KNOWN_REALS,
            time_varying_unknown_categoricals=cfg.TIME_VARYING_UNKNOWN_CATEGORICALS,
            time_varying_unknown_reals=cfg.TIME_VARYING_UNKNOWN_REALS,
            allow_missing_timesteps=cfg.ALLOW_MISSING_TIMESTEPS,
            add_target_scales=cfg.ADD_TARGET_SCALES,
            add_relative_time_idx=cfg.ADD_RELATIVE_TIME_IDX,
            add_encoder_length=cfg.ADD_ENCODER_LENGTH,
            randomize_length=cfg.RANDOMIZE_LENGTH,
            target_normalizer=cfg.TARGET_NORMALIZER,
            max_encoder_length=cfg.MAX_ENCODER_LENGTH,
            min_encoder_length=cfg.MIN_ENCODER_LENGTH,
            max_prediction_length=cfg.MAX_PREDICTION_LENGTH,
            min_prediction_length=cfg.MIN_PREDICTION_LENGTH,
        )
        print(f"🔨 full_ds built, total windows: {len(self.full_ds)}")

    def setup(self, stage=None):
        enc    = cfg.MAX_ENCODER_LENGTH
        pred   = cfg.MAX_PREDICTION_LENGTH
        window = enc + pred
        k      = cfg.VALIDATION_WINDOW_COUNT
        mask = self.pred_df[cfg.TARGET].isna().all(axis=1)
        self.cutoff = self.pred_df.loc[mask, 'time_idx'].min()

        # ── train+val split ────────────────────────────────────────────
        if stage in (None, "fit", "validate"):
            print("🛠 Building train_ds & validation_ds...")

            # 1) train_ds: windows ending at or before cutoff+pred
            cutoff_plus_pred = self.cutoff + pred
            self.training_ds = self.full_ds.filter(
                lambda idx: idx.time_idx_last <= cutoff_plus_pred
            )
            print(f"  train windows: {len(self.training_ds)}")

            # 2) validation_ds: windows whose FIRST time ≥ cutoff+1 and LAST ≤ cutoff+window+(k-1)
            val_start = self.cutoff + 1
            max_train_idx = self.pred_df["time_idx"].max()
            val_end   = self.cutoff + window + (k - 1)
            val_end = min(val_end, max_train_idx - cfg.MAX_PREDICTION_LENGTH)
            self.validation_ds = self.full_ds.filter(
                lambda idx: (idx.time_idx_first >= val_start)
                         & (idx.time_idx_last  <= val_end)
            )
            print(f"  val windows:   {len(self.validation_ds)}")

    def train_dataloader(self):
        return self.training_ds.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return self.validation_ds.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            persistent_workers=True,
        )
    
    def predict_dataloader(self):
        return self.validation_ds.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            persistent_workers=True,
        )

    def advance_window(self):
        """Move the global cutoff forward before the next epoch."""
        self.cutoff += self.window_step
        print(f"📈 cutoff advanced to {self.cutoff}")
