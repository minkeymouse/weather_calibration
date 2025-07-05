import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, Trainer

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer, TiDEModel, TimeXer
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

class WeatherDataModule(LightningDataModule):
    def __init__(self):
        self.trainer = None

    def prepare_data(self) -> None:
        self.train_df = pd.read_csv("experiments/task7/weather_train.csv")
        self.test_df = pd.read_csv("experiments/task7/weather_test.csv")
        for col in ["solar_term","hod","dow","moy"]:
            df[col] = df[col].astype(str).astype("category")
        return super().prepare_data()
    
    
    def setup(self, stage: str) -> None:
        self.train_ds = TimeSeriesDataSet(
            self.train_df,
            time_idx="time_idx",
            target = ["humid_obs", "degC_obs", "mmHg_obs"],
            group_ids=["segment_id"],

            max_encoder_length=168,
            min_encoder_length=72,

            min_prediction_idx=73,
            max_prediction_length=24,
            min_prediction_length=24,

            static_categoricals=None,
            static_reals=None,
            time_varying_known_categoricals=["solar_term","hod","dow","moy"],
            time_varying_known_reals=["sunlight_fcst", "humid_fcst", "abs_humid_fcst", "degC_fcst", "hPa_fcst"],
            time_varying_unknown_categoricals=None,
            time_varying_unknown_reals=["humid_obs", "degC_obs", "mmHg_obs"],

            variable_groups=None,
            constant_fill_strategy=None,
            allow_missing_timesteps=False,

            lags={"humid_obs":[1], "degC_obs":[1], "mmHg_obs":[1]},

            add_relative_time_idx=False,
            add_target_scales=False,
            add_encoder_length="auto",

            target_normalizer="auto",

            randomize_length=True
        )
        self.test_ds = TimeSeriesDataSet(
            self.test_df,
            time_idx="time_idx",
            target = ["humid_obs", "degC_obs", "mmHg_obs"],
            group_ids=["segment_id"],

            max_encoder_length=168,
            min_encoder_length=72,

            min_prediction_idx=73,
            max_prediction_length=24,
            min_prediction_length=24,

            static_categoricals=None,
            static_reals=None,
            time_varying_known_categoricals=["solar_term","hod","dow","moy"],
            time_varying_known_reals=["sunlight_fcst", "humid_fcst", "abs_humid_fcst", "degC_fcst", "hPa_fcst"],
            time_varying_unknown_categoricals=None,
            time_varying_unknown_reals=["humid_obs", "degC_obs", "mmHg_obs"],

            variable_groups=None,
            constant_fill_strategy=None,
            allow_missing_timesteps=False,

            lags={"humid_obs":[1], "degC_obs":[1], "mmHg_obs":[1]},

            add_relative_time_idx=False,
            add_target_scales=False,
            add_encoder_length="auto",

            target_normalizer="auto",

            randomize_length=True
        )
        return super().setup(stage)
    
    def train_dataloader(self) -> np.Any:
        return super().train_dataloader()
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return super().val_dataloader()
    
    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return super().test_dataloader()

def Model:
    if model == "tft"
        return 

if __name__ == "__main__":
    model = mymodel("")


