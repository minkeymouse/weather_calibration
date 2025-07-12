# test_debug_tide.py

import warnings
# suppress both sklearn’s “fitted with feature names” warning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import lightning.pytorch as pl
from pytorch_forecasting.models import TiDEModel
from dataset import ForecastingDataModule  # adjust to your actual import path
import config as cfg

def main(model, dm):
    # grab a single validation batch
    val_loader = dm.val_dataloader()
    x, y = next(iter(val_loader))  # x: dict of tensors, y: (target, weight)
    
    # encoder-side features
    enc_cont = x["encoder_cont"].shape[-1]
    enc_cat  = x["encoder_cat"].shape[-1]
    print("x['encoder_cont']:", x["encoder_cont"].shape)
    print("x['encoder_cat'] :", x["encoder_cat"].shape)
    print("  → total encoder feats =", enc_cont + enc_cat)
    print("  → model.encoder_covariate_size =", model.encoder_covariate_size)
    assert enc_cont + enc_cat == model.encoder_covariate_size
    
    # decoder-side features
    dec_cont = x["decoder_cont"].shape[-1]
    dec_cat  = x["decoder_cat"].shape[-1]
    print("x['decoder_cont']:", x["decoder_cont"].shape)
    print("x['decoder_cat'] :", x["decoder_cat"].shape)
    print("  → total decoder feats =", dec_cont + dec_cat)
    print("  → model.decoder_covariate_size =", model.decoder_covariate_size)
    assert dec_cont + dec_cat == model.decoder_covariate_size
    
    # one forward pass
    preds = model(x)
    print("Forward pass → preds.shape =", preds.shape)

if __name__ == "__main__":
    # 1) seed + prepare full dataset (for encoders/scalers)
    pl.seed_everything(cfg.SEED, workers=True)
    dm0 = ForecastingDataModule()
    dm0.prepare_data()
    full_ds = dm0.full_ds

    # 2) build TiDEModel from that dataset
    model = TiDEModel.from_dataset(
        full_ds,
        learning_rate=cfg.TIDE_PARAMS["learning_rate"],
        hidden_size=cfg.TIDE_PARAMS["hidden_size"],
        num_encoder_layers=cfg.TIDE_PARAMS["num_encoder_layers"],
        num_decoder_layers=cfg.TIDE_PARAMS["num_decoder_layers"],
        decoder_output_dim=cfg.TIDE_PARAMS["decoder_output_dim"],
        temporal_width_future=cfg.TIDE_PARAMS["temporal_width_future"],
        temporal_hidden_size_future=cfg.TIDE_PARAMS["temporal_hidden_size_future"],
        temporal_decoder_hidden=cfg.TIDE_PARAMS["temporal_decoder_hidden"],
        use_layer_norm=cfg.TIDE_PARAMS["use_layer_norm"],
        dropout=cfg.TIDE_PARAMS["dropout"],
        output_size=cfg.TIDE_PARAMS["output_size"],
        loss=cfg.TIDE_PARAMS["loss"],
        logging_metrics=cfg.TIDE_PARAMS["logging_metrics"],
        log_interval=cfg.TIDE_PARAMS["log_interval"],
    )
    model.eval()

    # 3) prepare a fresh datamodule for validation
    dm = ForecastingDataModule()
    dm.prepare_data()
    dm.setup(stage="validate")

    main(model, dm)
