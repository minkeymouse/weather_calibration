"""
Implements the TiDE (Time-series Dense Encoder-decoder) model, which is designed for
long-term time-series forecasting.
"""

from copy import copy
from typing import Optional, Union

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn

from pytorch_forecasting.metrics.base_metrics import MultiLoss
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    RMSE,
    SMAPE,
    MultiHorizonMetric,
)
from pytorch_forecasting.models.base import BaseModelWithCovariates
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

MixedCovariatesTrainTensorType = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

class _ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        dropout: float,
        use_layer_norm: bool,
    ):
        """Pytorch module implementing the Residual Block from the TiDE paper."""
        super().__init__()

        # dense layer with ReLU activation with dropout
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout),
        )

        # linear skip connection from input to output of self.dense
        self.skip = nn.Linear(input_dim, output_dim)

        # layer normalization as output
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual connection
        x = self.dense(x) + self.skip(x)

        # layer normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x


class _TideModule(nn.Module):
    def __init__(
        self,
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        output_chunk_length: int,
        input_chunk_length: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        decoder_output_dim: int,
        hidden_size: int,
        temporal_decoder_hidden: int,
        temporal_width_future: int,
        use_layer_norm: bool,
        dropout: float,
        temporal_hidden_size_future: int,
    ):
        """PyTorch module implementing the TiDE architecture.

        Parameters
        ----------
        input_dim
            The total number of input features, including the target
            and optional covariates.
        output_dim
            The number of output features in the target.
        future_cov_dim
            The number of covariates available for the future time steps.
        static_cov_dim
            The number of covariates that remain constant across time steps.
        num_encoder_layers
            The number of stacked Residual Blocks used in the encoder.
        num_decoder_layers
            The number of stacked Residual Blocks used in the decoder.
        decoder_output_dim
            The dimensionality of the decoder's output.
        hidden_size
            The size of the hidden layers within the encoder/decoder Residual Blocks.
        temporal_decoder_hidden
            The size of the hidden layers in the temporal decoder.
        temporal_width_future
            The dimensionality of the embedding space for future covariates.
        temporal_hidden_size_future
            The size of the hidden layers in the Residual Block projecting
            future covariates.
        use_layer_norm
            Indicates whether to apply layer normalization in the Residual Blocks.
        dropout
            The dropout rate.

        Inputs
        ------
        x
            A tuple of Tensors (x_past, x_future, x_static)
            where x_past represents the input/past sequence,
            and x_future represents the output/future sequence. The input dimensions are
            (batch_size, time_steps, components).
        Outputs
        -------
        y
            A Tensor with dimensions (batch_size, output_chunk_length, output_dim)
            epresenting the model's output.
        """
        super().__init__()

        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.output_chunk_length = output_chunk_length
        self.input_chunk_length = input_chunk_length
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.temporal_width_future = temporal_width_future
        self.temporal_hidden_size_future = temporal_hidden_size_future or hidden_size

        # future covariates handling: either feature projection,
        # raw features, or no features
        self.future_cov_projection = None
        if future_cov_dim > 0 and self.temporal_width_future:
            # residual block for future covariates feature projection
            self.future_cov_projection = _ResidualBlock(
                input_dim=future_cov_dim,
                output_dim=temporal_width_future,
                hidden_size=temporal_hidden_size_future,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            historical_future_covariates_flat_dim = (
                self.input_chunk_length + self.output_chunk_length
            ) * temporal_width_future
        elif future_cov_dim > 0:
            # skip projection and use raw features
            historical_future_covariates_flat_dim = (
                self.input_chunk_length + self.output_chunk_length
            ) * future_cov_dim
        else:
            historical_future_covariates_flat_dim = 0

        encoder_dim = (
            self.input_chunk_length * output_dim
            + historical_future_covariates_flat_dim
            + static_cov_dim
        )

        self.encoders = nn.Sequential(
            _ResidualBlock(
                input_dim=encoder_dim,
                output_dim=hidden_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers - 1)
            ],
        )

        self.decoders = nn.Sequential(
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers - 1)
            ],
            # add decoder output layer
            _ResidualBlock(
                input_dim=hidden_size,
                output_dim=decoder_output_dim * self.output_chunk_length,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
        )

        decoder_input_dim = decoder_output_dim
        if temporal_width_future and future_cov_dim:
            decoder_input_dim += temporal_width_future
        elif future_cov_dim:
            decoder_input_dim += future_cov_dim

        self.temporal_decoder = _ResidualBlock(
            input_dim=decoder_input_dim,
            output_dim=output_dim,
            hidden_size=temporal_decoder_hidden,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        self.lookback_skip = nn.Linear(
            self.input_chunk_length, self.output_chunk_length
        )

    def forward(
        self, x_in: tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """TiDE model forward pass.

        Parameters
        ----------
        x_in
            comes as tuple (x_past, x_future, x_static)
            where x_past is the input/past chunk and x_future
            is the output/future chunk. Input dimensions are
            (batch_size, time_steps, components)
        Returns
        -------
        torch.Tensor
            The output Tensor of shape (batch_size, output_chunk_length, output_dim)
        """

        # x has shape (batch_size, input_chunk_length, input_dim)
        # x_future_covariates has shape (batch_size, input_chunk_length, future_cov_dim)
        # x_static_covariates has shape (batch_size, static_cov_dim)
        x, x_future_covariates, x_static_covariates = x_in

        x_lookback = x[:, :, : self.output_dim]

        # future covariates: feature projection or raw features
        # historical future covariates need to be extracted from x and
        # stacked with part of future covariates
        if x_future_covariates is not None:
            # slice out exactly the same last‐dim that x_future_covariates has,
            # so encoder‐and‐decoder covariates line up
            cov_dim = x_future_covariates.size(-1)
            x_dynamic_future_covariates = torch.cat(
                [
                    x[:, :, -cov_dim:],         # now matches decoder covariate dim
                    x_future_covariates,
                ],
                dim=1,
            )
            if self.temporal_width_future:
                # project input features across all input and output time steps
                x_dynamic_future_covariates = self.future_cov_projection(
                    x_dynamic_future_covariates
                )
        else:
            x_dynamic_future_covariates = None

        # setup input to encoder
        encoded = [
            x_lookback,
            x_dynamic_future_covariates,
            x_static_covariates,
        ]
        encoded = [t.flatten(start_dim=1) for t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=1)

        # encoder, decode, reshape
        encoded = self.encoders(encoded)
        decoded = self.decoders(encoded)

        # get view that is batch size x output chunk length x self.decoder_output_dim
        decoded = decoded.view(x.shape[0], self.output_chunk_length, -1)

        # stack and temporally decode with future covariate last output steps
        temporal_decoder_input = [
            decoded,
            (
                x_dynamic_future_covariates[:, -self.output_chunk_length :, :]
                if self.future_cov_dim > 0
                else None
            ),
        ]
        temporal_decoder_input = [t for t in temporal_decoder_input if t is not None]

        temporal_decoder_input = torch.cat(temporal_decoder_input, dim=2)
        temporal_decoded = self.temporal_decoder(temporal_decoder_input)

        # pass x_lookback through self.lookback_skip but swap the last two dimensions
        # this is needed because the skip connection is applied across
        # the input time steps and not across the output time steps
        skip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)

        # add skip connection
        y = temporal_decoded + skip.reshape_as(
            temporal_decoded
        )  # skip.view(temporal_decoded.shape)

        y = y.view(-1, self.output_chunk_length, self.output_dim)
        return y



class TiDEModel(BaseModelWithCovariates):
    """TiDE model for long-term time-series forecasting."""

    def __init__(
        self,
        output_chunk_length: int,
        input_chunk_length: int,
        features: str = "S",
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        decoder_output_dim: int = 16,
        hidden_size: int = 128,
        temporal_width_future: int = 4,
        temporal_hidden_size_future: int = 32,
        temporal_decoder_hidden: int = 32,
        use_layer_norm: bool = False,
        dropout: float = 0.1,
        output_size: Union[int, list[int]] = 1,
        static_categoricals: Optional[list[str]] = None,
        static_reals: Optional[list[str]] = None,
        time_varying_categoricals_encoder: Optional[list[str]] = None,
        time_varying_categoricals_decoder: Optional[list[str]] = None,
        categorical_groups: Optional[dict[str, list[str]]] = None,
        time_varying_reals_encoder: Optional[list[str]] = None,
        time_varying_reals_decoder: Optional[list[str]] = None,
        embedding_sizes: Optional[dict[str, tuple[int, int]]] = None,
        embedding_paddings: Optional[list[str]] = None,
        embedding_labels: Optional[list[str]] = None,
        x_reals: Optional[list[str]] = None,
        x_categoricals: Optional[list[str]] = None,
        loss: MultiHorizonMetric = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """An implementation of the TiDE model.

        TiDE shares similarities with Transformers
        (implemented in :class:TransformerModel), but aims to deliver better performance
        with reduced computational requirements by utilizing MLP-based encoder-decoder
        architectures without attention mechanisms.

        This model supports future covariates (known for output_chunk_length points
        after the prediction time) andstatic covariates.

        The encoder and decoder are constructed using residual blocks. The number of
        residual blocks in the encoder and decoder can be specified with
        `num_encoder_layers` and `num_decoder_layers` respectively. The layer width in
        the residual blocks can be adjusted using `hidden_size`, while the layer width
        in the temporal decoder can be controlled via `temporal_decoder_hidden`.

        Parameters
        ----------
        input_chunk_length :int
            Number of past time steps to use as input for themodel (per chunk).
            This applies to the target series and future covariates
            (if supported by the model).
        output_chunk_length : int
            Number of time steps the internal model predicts simultaneously (per chunk).
            This also determines how many future values from future covariates
            are used as input (if supported by the model).
        num_encoder_layers : int, default=2
            Number of residual blocks in the encoder
        num_decoder_layers : int, default=2
            Number of residual blocks in the decoder
        decoder_output_dim : int, default=16
            Dimensionality of the decoder's output
        hidden_size : int, default=128
            Size of hidden layers in the encoder and decoder.
            Typically ranges from 32 to 128 when no covariates are used.
        temporal_width_future (int): Width of the output layer in the residual block for future covariate projections.
            If set to 0, bypasses feature projection and uses raw feature data. Defaults to 4.
        temporal_hidden_size_future (int): Width of the hidden layer in the residual block for future covariate
            projections. Defaults to 32.
        temporal_decoder_hidden (int): Width of the layers in the temporal decoder. Defaults to 32.
        use_layer_norm (bool): Whether to apply layer normalization in residual blocks. Defaults to False.
        dropout (float): Dropout probability for fully connected layers. Defaults to 0.1.
        output_size: Union[int, List[int]]: included as its required by deduce_default_output_parameters in
            from_dataset function. Defaults to 1.
        static_categoricals (List[str]): names of static categorical variables
        static_reals (List[str]): names of static continuous variables
        time_varying_categoricals_encoder (List[str]): names of categorical variables for encoder
        time_varying_categoricals_decoder (List[str]): names of categorical variables for decoder
        time_varying_reals_encoder (List[str]): names of continuous variables for encoder
        time_varying_reals_decoder (List[str]): names of continuous variables for decoder
        x_reals (List[str]): order of continuous variables in tensor passed to forward function
        x_categoricals (List[str]): order of categorical variables in tensor passed to forward function
        embedding_sizes (Dict[str, Tuple[int, int]]): dictionary mapping categorical variables to tuple of integers
            where the first integer denotes the number of categorical classes and the second the embedding size
        embedding_labels (Dict[str, List[str]]): dictionary mapping (string) indices to list of categorical labels
        embedding_paddings (List[str]): names of categorical variables for which label 0 is always mapped to an
            embedding vector filled with zeros
        categorical_groups (Dict[str, List[str]]): dictionary of categorical variables that are grouped together and
            can also take multiple values simultaneously (e.g. holiday during octoberfest). They should be implemented
            as bag of embeddings
        logging_metrics (nn.ModuleList[MultiHorizonMetric]): list of metrics that are logged during training.
            Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        **kwargs
            Allows optional arguments to configure pytorch_lightning.Module, pytorch_lightning.Trainer, and
            pytorch-forecasting's :class:BaseModelWithCovariates.

        Note:
            The model supports future covariates and static covariates.
        """  # noqa: E501
        if static_categoricals is None:
            static_categoricals = []
        if static_reals is None:
            static_reals = []
        if time_varying_categoricals_encoder is None:
            time_varying_categoricals_encoder = []
        if time_varying_categoricals_decoder is None:
            time_varying_categoricals_decoder = []
        if categorical_groups is None:
            categorical_groups = {}
        if time_varying_reals_encoder is None:
            time_varying_reals_encoder = []
        if time_varying_reals_decoder is None:
            time_varying_reals_decoder = []
        if embedding_sizes is None:
            embedding_sizes = {}
        if embedding_paddings is None:
            embedding_paddings = []
        if embedding_labels is None:
            embedding_labels = {}
        if x_reals is None:
            x_reals = []
        if x_categoricals is None:
            x_categoricals = []
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])
        if loss is None:
            if features == "M":
                loss = MultiLoss([RMSE()] * 3, weights=[0.3,0.5,0.2])
            else:
                loss = MAE()

        # loss and logging_metrics are ignored as they are modules
        # and stored before calling save_hyperparameters
        self.save_hyperparameters(ignore=["loss", "logging_metrics"])
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)
        self.output_dim = len(self.target_names)

        self.embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
        )

        self.model = _TideModule(
            output_dim=self.output_dim,
            future_cov_dim=self.decoder_covariate_size,
            static_cov_dim=self.static_size,
            output_chunk_length=output_chunk_length,
            input_chunk_length=input_chunk_length,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            decoder_output_dim=decoder_output_dim,
            hidden_size=hidden_size,
            temporal_decoder_hidden=temporal_decoder_hidden,
            temporal_width_future=temporal_width_future,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            temporal_hidden_size_future=temporal_hidden_size_future,
        )

    @property
    def decoder_covariate_size(self) -> int:
        """Decoder covariates size.

        Returns:
            int: size of time-dependent covariates used by the decoder
        """
        return len(
            set(self.hparams.time_varying_reals_decoder) - set(self.target_names)
        ) + sum(
            self.embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_decoder
        )

    @property
    def encoder_covariate_size(self) -> int:
        """Encoder covariate size.

        Returns:
            int: size of time-dependent covariates used by the encoder
        """
        return len(
            set(self.hparams.time_varying_reals_encoder) - set(self.target_names)
        ) + sum(
            self.embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_encoder
        )

    @property
    def static_size(self) -> int:
        """Static covariate size.

        Returns:
            int: size of static covariates
        """
        return len(self.hparams.static_reals) + sum(
            self.embeddings.output_size[name]
            for name in self.hparams.static_categoricals
        )

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        """
        Convenience function to create network from
        :py:class`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Args:
            dataset (TimeSeriesDataSet): dataset where sole predictor is the target.
            **kwargs: additional arguments to be passed to `__init__` method.

        Returns:
            TiDE
        """

        # validate arguments
        assert not isinstance(
            dataset.target_normalizer, NaNLabelEncoder
        ), "only regression tasks are supported - target must not be categorical"

        assert dataset.min_encoder_length == dataset.max_encoder_length, (
            "only fixed encoder length is allowed,"
            " but min_encoder_length != max_encoder_length"
        )

        assert dataset.max_prediction_length == dataset.min_prediction_length, (
            "only fixed prediction length is allowed,"
            " but max_prediction_length != min_prediction_length"
        )

        assert (
            dataset.randomize_length is None
        ), "length has to be fixed, but randomize_length is not None"
        assert (
            not dataset.add_relative_time_idx
        ), "add_relative_time_idx has to be False"

        new_kwargs = copy(kwargs)
        new_kwargs.update(
            {
                "output_chunk_length": dataset.max_prediction_length,
                "input_chunk_length": dataset.max_encoder_length,
            }
        )
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, new_kwargs, MAE()))
        # initialize class
        return super().from_dataset(dataset, **new_kwargs)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Pass forward of network.

        Args:
            x (Dict[str, torch.Tensor]): input from dataloader generated from
                :py:class:~pytorch_forecasting.data.timeseries.TimeSeriesDataSet.

        Returns:
            Dict[str, torch.Tensor]: output of model
        """

        # target
        encoder_y = x["encoder_cont"][..., self.target_positions]
        # covariates
        encoder_features = self.extract_features(x, self.embeddings, period="encoder")

        if self.encoder_covariate_size > 0:
            # encoder_features = self.extract_features(
            #                   x, self.embeddings, period="encoder")
            encoder_x_t = torch.concat(
                [
                    encoder_features[name]
                    for name in self.encoder_variables
                    if name not in self.target_names
                ],
                dim=2,
            )
            input_vector = torch.concat((encoder_y, encoder_x_t), dim=2)

        else:
            encoder_x_t = None
            input_vector = encoder_y

        if self.decoder_covariate_size > 0:
            decoder_features = self.extract_features(
                x, self.embeddings, period="decoder"
            )
            decoder_x_t = torch.concat(
                [decoder_features[name] for name in self.decoder_variables], dim=2
            )
        else:
            decoder_x_t = None

        # statics
        if self.static_size > 0:
            x_s = torch.concat(
                [encoder_features[name][:, 0] for name in self.static_variables], dim=1
            )
        else:
            x_s = None

        x_in = (input_vector, decoder_x_t, x_s)
        prediction = self.model(x_in)

        if self.output_dim > 1:  # for multivariate targets
            # adjust prefictions dimensions according
            # to format required for consequent processes
            # from (batch size, seq len, output_dim) to
            # (output_dim, batch size, seq len)
            prediction = prediction.permute(2, 0, 1)
            prediction = [i.clone().detach().requires_grad_(True) for i in prediction]

        # rescale predictions into target space
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])
        # transform output to format processed by other functions
        return self.to_network_output(prediction=prediction)

def configure_optimizers(self):
    return torch.optim.AdamW(
        self.parameters(),
        lr=self.hparams.learning_rate,
        weight_decay=self.hparams.get("weight_decay", 0.0),
    )
