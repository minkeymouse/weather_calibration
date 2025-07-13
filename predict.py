#!/usr/bin/env python3
"""
predict.py – inference (and optional fine-tuning) without ground-truth metrics.
"""
from __future__ import annotations
import argparse, warnings, pickle
from pathlib import Path

import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer

import matplotlib.pyplot as plt  # still needed to satisfy imports elsewhere
from models.timexr_model import TimeXer
from dataset.dataset import PredictionDataModule
from callbacks import WalkForward
import config as cfg

warnings.filterwarnings("ignore", message="X does not have valid feature names")

TARGET_COLS = ["humid_obs", "degC_obs", "mmHg_obs"]

# ───────────────── helper ───────────────────────────────────────────────────
def to_df(batch_out, targets) -> pd.DataFrame:
    """Convert PF predict() output → DataFrame(horizon × targets)."""
    import torch
    if isinstance(batch_out, list):
        batch_out = torch.cat(
            [b["prediction"] if isinstance(b, dict) else b for b in batch_out], dim=0
        )
    if isinstance(batch_out, torch.Tensor):
        batch_out = batch_out.detach().cpu()
        if batch_out.ndim == 3:  # (batch, horizon, targets)
            batch_out = batch_out.reshape(-1, batch_out.shape[-1])
        elif batch_out.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected shape {batch_out.shape}")
        if batch_out.shape[1] != len(targets):
            batch_out = batch_out.T
    return pd.DataFrame(batch_out.numpy(), columns=targets)

# ───────────────── CLI ──────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("Fine-tune (optional) & predict – no metrics")
    p.add_argument("--train",  action="store_true",
                   help="online fine-tune both models before predicting")
    p.add_argument("--epochs", type=int, default=10,
                   help="epochs for fine-tune")
    return p.parse_args()

# ───────────────── main ─────────────────────────────────────────────────────
def main(online_train=False, epochs=10):
    pl.seed_everything(cfg.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    dm = PredictionDataModule(); dm.prepare_data()
    dm.full_ds = pickle.load(open("artifacts/full_ds.pkl", "rb"))
    dm.setup(stage="predict")
    if online_train:
        dm.setup(stage="fit")

    # Models
    tft = TemporalFusionTransformer.from_dataset(dm.full_ds, **cfg.TFT_PARAMS)
    txr = TimeXer.from_dataset(dm.full_ds, **cfg.TXR_PARAMS)

    # Optional fine-tune
    if online_train:
        logdir = Path("logs/predict")
        ckpt_tft = ModelCheckpoint(logdir/"ckpts/tft", monitor="val_loss",
                                   mode="min", save_top_k=1)
        ckpt_txr = ModelCheckpoint(logdir/"ckpts/txr", monitor="val_loss",
                                   mode="min", save_top_k=1)
        trainer = Trainer(
            accelerator=device, devices=1, max_epochs=epochs, precision=32,
            callbacks=[WalkForward(), ckpt_tft, ckpt_txr,
                       LearningRateMonitor(), EarlyStopping("val_loss", cfg.PATIENCE)],
            logger=[CSVLogger(logdir, name="tft"),
                    CSVLogger(logdir, name="txr")]
        )
        trainer.fit(tft, datamodule=dm)
        trainer.fit(txr, datamodule=dm)
        path_tft, path_txr = ckpt_tft.best_model_path, ckpt_txr.best_model_path
    else:
        path_tft = "base_ckpts/tft_base.ckpt"
        path_txr = "base_ckpts/txr_base.ckpt"

    # Load weights
    for mdl, path in ((tft, path_tft), (txr, path_txr)):
        mdl.load_state_dict(torch.load(path, map_location="cpu")["state_dict"],
                            strict=False)
        mdl.to(device).eval()

    # Predict
    loader   = dm.predict_dataloader()
    df_tft   = to_df(tft.predict(loader, mode="prediction"), TARGET_COLS)
    df_txr   = to_df(txr.predict(loader, mode="prediction"), TARGET_COLS)
    df_ens   = 0.5 * df_tft + 0.5 * df_txr

    # Append hourly date column (one hour after last observed date)
    base_df   = pd.read_csv("input/base_data.csv", parse_dates=["date"])
    last_date = base_df["date"].max()
    horizon   = len(df_ens)
    df_ens.insert(
        0, "date",
        pd.date_range(last_date + pd.Timedelta(hours=1), periods=horizon, freq="h")
    )

    # Save
    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    df_ens.to_csv(out_dir/"prediction.csv", index=False)
    print(f"✅ prediction.csv saved to {out_dir}")

# entry
if __name__ == "__main__":
    a = parse_args()
    main(online_train=a.train, epochs=a.epochs)
