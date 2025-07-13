# test.py
import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer

from tide_model import TiDEModel
from dataset import ForecastingDataModule
import config as cfg

def main():
    # reproducibility
    pl.seed_everything(cfg.SEED, workers=True)

    # prepare test data
    dm = ForecastingDataModule()
    dm.prepare_data()
    dm.setup(stage="test")

    # instantiate model architecture
    model = TiDEModel.from_dataset(dm.full_ds, **cfg.TIDE_PARAMS)

    # load the averaged weights only
    ckpt_path = "checkpoints/tide/averaged_tide.ckpt"
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt_data["state_dict"])

    # move to device and set eval mode
    model = model.to(cfg.DEVICE).eval()

    # run test
    trainer = Trainer(
        accelerator="gpu" if "cuda" in str(cfg.DEVICE) else None,
        devices=1 if torch.cuda.is_available() else None,
    )
    metrics = trainer.test(model, datamodule=dm)
    print(f"✅ Test metrics: {metrics}")

if __name__ == "__main__":
    main()
