from torch.nn import ModuleList
from pytorch_forecasting.metrics import QuantileLoss, SMAPE, MAE, RMSE, MAPE, MASE

# Main params
TRAIN_DF = "data/train.csv"
TEST_DF = "data/test.csv"
BATCH_SIZE = 512
NUM_WORKERS = 4
SEED = 1234

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
WINDOW_EPOCH_ADVANCE = 12
VALIDATION_WINDOW_COUNT = 100
INITIAL_CUTOFF = MAX_ENCODER_LENGTH + 100

# Trainer params
MAX_EPOCHS      = 15000
PATIENCE        = 150
LOG_INTERVAL    = 10
GRAD_CLIP_VAL   = 0.1

TFT_PARAMS = dict(
    # learning
    learning_rate=1e-3,
    reduce_on_plateau_patience=4,
    # architecture
    hidden_size=256,
    lstm_layers=2,
    attention_head_size=64,
    dropout=0.05,
    hidden_continuous_size=128,
    output_size=[1, 1, 1],
    loss=QuantileLoss(quantiles=[0.5]),
    logging_metrics=ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]),
    log_interval=10,
)

TIDE_PARAMS = dict(
    # optimizer
    learning_rate=1e-3,
    # encoder / decoder depths
    num_encoder_layers=2,
    num_decoder_layers=2,
    # widths
    hidden_size=128,
    decoder_output_dim=16,
    # future‐covariate projections
    temporal_width_future=4,
    temporal_hidden_size_future=32,
    # temporal decoder
    temporal_decoder_hidden=32,
    # regularization
    use_layer_norm=False,
    dropout=0.1,
    # multi‐target output
    output_size=[1, 1, 1],
    # loss & logging
    loss=QuantileLoss(quantiles=[0.5]),
    logging_metrics=ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]),
    # how often to log internally
    log_interval=10,
)

TXR_PARAMS = dict(
    # optimizer
    learning_rate=1e-3,
    # transformer backbone
    hidden_size=256,
    n_heads=4,
    e_layers=2,           # number of encoder layers
    d_ff=1024,            # feed-forward inner dimension
    dropout=0.2,
    activation="relu",
    # patching/exogenous settings
    patch_length=16,
    factor=5,
    embed_type="fixed",   # fixed time embeddings
    freq="h",             # hourly data
    # output
    output_size=[1, 1, 1],               # one forecast per target
    # loss & metrics
    loss=QuantileLoss(quantiles=[0.5]),
    logging_metrics=ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]),
    # how often to log internally
    log_interval=10,
)