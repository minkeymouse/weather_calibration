import torch
from torch.nn import ModuleList
from pytorch_forecasting.metrics import QuantileLoss, SMAPE, MAE, RMSE, MAPE, MASE

# Main params
TRAIN_DF = "data/train.csv"
TEST_DF = "data/test.csv"
BATCH_SIZE = 256
NUM_WORKERS = 2
SEED = 1234

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASET HYPERPARAMS
TIME_IDX = "time_idx"
TARGET = ["humid_obs", "degC_obs", "mmHg_obs"]
GROUP_IDS = ["group_id"]
TIME_VARYING_KNOWN_CATEGORICALS = [
    "solar_term",
    "hod",
    "dow",
    "moy",
    "sunlight_fcst_was_missing",
    "humid_fcst_was_missing",
    "abs_humid_fcst_was_missing",
    "degC_fcst_was_missing",
    "hPa_fcst_was_missing",
]
TIME_VARYING_KNOWN_REALS = [
    "humid_fcst",
    "abs_humid_fcst",
    "sunlight_fcst",
    "degC_fcst",
    "hPa_fcst",
    "degC_fcst_diff",
    "humid_fcst_diff",
    "abs_humid_fcst_diff",
    "hPa_fcst_diff",
    "sunlight_fcst_diff",
]
TIME_VARYING_UNKNOWN_CATEGORICALS = []
TIME_VARYING_UNKNOWN_REALS = [
    "humid_obs",
    "degC_obs",
    "mmHg_obs",
    "degC_obs_diff",
    "humid_obs_diff",
    "mmHg_obs_diff",
]
ALLOW_MISSING_TIMESTEPS = False
ADD_TARGET_SCALES = True
ADD_RELATIVE_TIME_IDX = False
ADD_ENCODER_LENGTH = False
RANDOMIZE_LENGTH = False
TARGET_NORMALIZER = "auto"
MAX_ENCODER_LENGTH = 168
MIN_ENCODER_LENGTH = 168
MAX_PREDICTION_LENGTH = 24
MIN_PREDICTION_LENGTH = 24

# Datamodule params
WINDOW_EPOCH_ADVANCE = 24
VALIDATION_WINDOW_COUNT = 100
INITIAL_CUTOFF = MAX_ENCODER_LENGTH + 100

# Trainer params
MAX_EPOCHS      = 520
PATIENCE        = 120
LOG_INTERVAL    = 1
GRAD_CLIP_VAL   = 0.1

TFT_PARAMS = dict(
    # learning
    learning_rate=1e-3,
    reduce_on_plateau_patience=4,
    hidden_size=256,
    lstm_layers=2,
    attention_head_size=128,
    dropout=0.05,
    hidden_continuous_size=256,
    output_size=[1, 1, 1],
    loss=QuantileLoss(quantiles=[0.5]),
    logging_metrics=ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]),
    log_interval=1,
)

TIDE_PARAMS = dict(
    output_chunk_length = MAX_PREDICTION_LENGTH,
    input_chunk_length = MAX_ENCODER_LENGTH,
    num_encoder_layers = 10,
    num_decoder_layers = 10,
    decoder_output_dim = 64,
    hidden_size = 1024,
    temporal_width_future = 16,
    temporal_hidden_size_future = 64,
    temporal_decoder_hidden = 64,
    use_layer_norm = True,
    dropout = 0.1,
    learning_rate = 1e-3,
    logging_metrics = ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]),
)

TXR_PARAMS = dict(
        context_length = MAX_ENCODER_LENGTH,
        prediction_length = MAX_PREDICTION_LENGTH,
        task_name = "long_term_forecast",
        features = "M",
        hidden_size = 512,
        n_heads = 8,
        e_layers = 4,
        d_ff = 1024,
        dropout = 0.2,
        patch_length = 6,
        learning_rate  = 3e-4,    # lower LR for stability on larger nets
        weight_decay  = 1e-5,    # L2 regularization
        reduce_on_plateau_patience = 5,       # drop LR if no val gain
        logging_metrics = ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]),
)