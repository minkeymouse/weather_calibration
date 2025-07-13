#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import TemporalFusionTransformer
from models.timexr_model import TimeXer
from dataset.dataset import PredictionDataModule
from callbacks import WalkForward
import config as cfg

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
)

TARGET_COLS = ["humid_obs", "degC_obs", "mmHg_obs"]
METRIC_WEIGHTS = {"humid_obs": 0.3, "degC_obs": 0.5, "mmHg_obs": 0.2}

def parse_args():
    parser = argparse.ArgumentParser(description="Train (optional) & predict")
    parser.add_argument(
        "--train",
        action="store_true",
        help="If set, fine-tune both models before predicting",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for online training (if --train)",
    )
    return parser.parse_args()

def main(online_train: bool = False, epochs: int = 10) -> None:
    # ── reproducibility & device ─────────────────────────────────────────────
    pl.seed_everything(cfg.SEED, workers=True)
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    devices = 1

    # ── prepare data ──────────────────────────────────────────────────────────
    dm = PredictionDataModule()
    dm.prepare_data()

    # ── instantiate untrained models ──────────────────────────────────────────
    # do this after prepare_data so full_ds exists
    tft = TemporalFusionTransformer.from_dataset(dm.full_ds, **cfg.TFT_PARAMS)
    txr = TimeXer.from_dataset(dm.full_ds, **cfg.TXR_PARAMS)

    # ── set up loggers ────────────────────────────────────────────────────────
    logger_tft = CSVLogger(save_dir="logs/predict", name="tft")
    logger_txr = CSVLogger(save_dir="logs/predict", name="txr")

    # ── training phase (if requested) ────────────────────────────────────────
    if online_train:
        dm.setup(stage="fit")

        ckpt_tft = ModelCheckpoint(
            dirpath="ckpts/tft",
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        ckpt_txr = ModelCheckpoint(
            dirpath="ckpts/txr",
            filename="txr-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )

        trainer_tft = Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=32,
            default_root_dir="logs/predict",
            logger=logger_tft,
            callbacks=[WalkForward(), ckpt_tft, LearningRateMonitor(),
                       EarlyStopping(monitor="val_loss", patience=cfg.PATIENCE)],
            max_epochs=epochs,
        )
        trainer_txr = Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=32,
            default_root_dir="logs/predict",
            logger=logger_txr,
            callbacks=[WalkForward(), ckpt_txr, LearningRateMonitor(),
                       EarlyStopping(monitor="val_loss", patience=cfg.PATIENCE)],
            max_epochs=epochs,
        )

        trainer_tft.fit(tft, datamodule=dm)
        trainer_txr.fit(txr, datamodule=dm)
        trainer_tft.validate(tft, datamodule=dm)
        trainer_txr.validate(txr, datamodule=dm)

        ckpt_tft_path = ckpt_tft.best_model_path
        ckpt_txr_path = ckpt_txr.best_model_path
    else:
        ckpt_tft_path = "base_ckpts/tft_base.ckpt"
        ckpt_txr_path = "base_ckpts/txr_base.ckpt"

    # ── load checkpoint weights ───────────────────────────────────────────────
    for model, ckpt_path in [(tft, ckpt_tft_path), (txr, ckpt_txr_path)]:
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_data["state_dict"])
        model.to(accelerator).eval()

    preds_tft = tft.predict(dm.full_ds)
    preds_txr = trainer_txr.predict(txr, datamodule=dm)

    df_tft = pd.concat(preds_tft).reset_index(drop=True)
    df_txr = pd.concat(preds_txr).reset_index(drop=True)

    # simple average ensemble
    df_ens = pd.DataFrame({
        col: 0.5 * df_tft[col] + 0.5 * df_txr[col]
        for col in TARGET_COLS
    })

    # ── save final predictions ───────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    df_ens.to_csv("output/prediction.csv", index=False)

    # ── compute metrics against ground truth ─────────────────────────────────
    df_base = pd.read_csv("input/base_data.csv")
    y_true = df_base[TARGET_COLS].reset_index(drop=True)
    y_pred = df_ens

    rmse = {col: np.sqrt(mean_squared_error(y_true[col], y_pred[col]))
            for col in TARGET_COLS}
    mae  = {col: mean_absolute_error(y_true[col], y_pred[col])
            for col in TARGET_COLS}
    srmse = sum(METRIC_WEIGHTS[c] * rmse[c] for c in TARGET_COLS)
    smae  = sum(METRIC_WEIGHTS[c] * mae[c] for c in TARGET_COLS)
    saerror = float((srmse + smae) / 2)

    metrics = {"RMSE": rmse, "MAE": mae, "sRMSE": srmse, "sMAE": smae, "sAERROR": saerror}
    with open("output/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open("output/metrics.txt", "w") as f:
        for col in TARGET_COLS:
            f.write(f"{col} RMSE: {rmse[col]:.4f}, MAE: {mae[col]:.4f}\n")
        f.write(f"\nsRMSE: {srmse:.4f}\n")
        f.write(f"sMAE:  {smae:.4f}\n")
        f.write(f"sAERROR: {saerror:.4f}\n")

    # ── plot & save comparison ───────────────────────────────────────────────
    plots_dir = Path("output/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    for col in TARGET_COLS:
        plt.figure()
        plt.plot(y_true[col].values, label="True")
        plt.plot(y_pred[col].values, label="Pred")
        plt.title(f"{col} — Prediction vs True")
        plt.xlabel("Sample index")
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{col}_compare.png")
        plt.close()

    print("✅ Prediction complete. Files written to output/")

if __name__ == "__main__":
    args = parse_args()
    main(online_train=args.train, epochs=args.epochs)
