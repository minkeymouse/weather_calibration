import os
import pickle
import lightning.pytorch as pl
from dataset.dataset import ForecastingDataModule
import config as cfg

def main() -> None:
    pl.seed_everything(cfg.SEED, workers=True)

    # ── data ─────────────────────────────────────────────────────────────────
    dm = ForecastingDataModule()
    dm.prepare_data()

    # ── NEW: ensure artifacts dir exists and save full_ds ────────────────────
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/full_ds.pkl", "wb") as f:
        pickle.dump(dm.full_ds, f)
    print("✅ Saved full_ds to artifacts/full_ds.pkl")

    dm.setup(stage="fit")
    # … rest of your training logic …

if __name__ == "__main__":
    main()
