# train_tide.py
from __future__ import annotations
import torch
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from tide_model import TiDEModel
from dataset import ForecastingDataModule
from callbacks import WalkForward
import config as cfg

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
)

def average_checkpoints(ckpt_paths: list[str], out_path: str):
    """Average the 'state_dict' from a set of Lightning ckpts and save as a full checkpoint."""
    templates = [
        torch.load(p, map_location="cpu", weights_only=False) for p in ckpt_paths
    ]
    state_dicts = [tpl["state_dict"] for tpl in templates]

    avg_state: dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key] for sd in state_dicts], dim=0)
        avg_state[key] = stacked.mean(dim=0)

    templates[0]["state_dict"] = avg_state
    torch.save(templates[0], out_path)
    print(f"✅ Averaged into full Lightning checkpoint {out_path}")


def main() -> None:
    # reproducibility
    pl.seed_everything(cfg.SEED, workers=True)

    # ── data ───────────────────────────────────────────────────────────────
    dm = ForecastingDataModule()
    dm.prepare_data()
    dm.setup(stage="fit")

    # ── model ──────────────────────────────────────────────────────────────
    model = TiDEModel.from_dataset(dm.full_ds, **cfg.TIDE_PARAMS)

    # ── callbacks & logger ─────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints/tide",
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
    )
    logger = CSVLogger(save_dir="logs", name="tide")

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
        default_root_dir="logs/tide",
        strategy="ddp",
        logger=logger,
        callbacks=[
            WalkForward(),
            ckpt_cb,
            LearningRateMonitor(),
            EarlyStopping(monitor="val_loss", patience=cfg.PATIENCE),
        ],
        max_epochs=cfg.MAX_EPOCHS,
        log_every_n_steps=cfg.LOG_INTERVAL,
        gradient_clip_val=cfg.GRAD_CLIP_VAL,
        reload_dataloaders_every_n_epochs=1,
    )

    # ── train & validate ───────────────────────────────────────────────────
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)

    # ── average ckpts ──────────────────────────────────────────────────────
    best_ckpts = list(ckpt_cb.best_k_models.keys())
    last_ckpt = ckpt_cb.last_model_path
    to_avg = best_ckpts.copy()
    if last_ckpt and last_ckpt not in to_avg:
        to_avg.append(last_ckpt)

    avg_path = Path("checkpoints/tide") / "averaged_tide.ckpt"
    if len(to_avg) > 1:
        average_checkpoints(to_avg, str(avg_path))
    else:
        print("ℹ️ Not enough checkpoints to average (need ≥2)")

    # ── load & test ─────────────────────────────────────────────────────────
    ckpt_to_test = avg_path if avg_path.exists() else ckpt_cb.best_model_path
    print(f"🔍 Testing from checkpoint: {ckpt_to_test}")

    model = TiDEModel.from_dataset(dm.full_ds, **cfg.TIDE_PARAMS)
    ckpt_data = torch.load(
        str(ckpt_to_test),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(ckpt_data["state_dict"])

    model = model.to(cfg.DEVICE).eval()
    dm.setup(stage="test")
    metrics = trainer.test(model, datamodule=dm)
    print(f"✅ Test metrics: {metrics}")


if __name__ == "__main__":
    main()
