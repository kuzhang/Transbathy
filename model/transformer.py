import torch
from torch import Tensor, nn
from .encoder import TransformerEncoder
# from .decoder import TransformerDecoder


def fcnn(dim_input: int = 3, dim_output: int = 1, kernel=3) -> nn.Module:
    # return nn.Conv2d(dim_input, dim_output, kernel_size=kernel)
    # return nn.Linear(dim_input, dim_output)
    return nn.Sequential(
        nn.AvgPool1d(kernel),
        nn.Linear(dim_input, dim_output)
    )


def avgpool(window=3) -> nn.Module:
    return nn.AvgPool1d(window)


def linear(dim_input: int = 3, dim_output: int = 1) -> nn.Module:
    return nn.Linear(dim_input, dim_output)


class Transformer(nn.Module):
    def __init__(
            self,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            dim_out: int = 1,
            kernel: int = 3
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.linear = linear(kernel, dim_out)
        self.pool = avgpool(dim_model)
        self.channel = dim_model

    def forward(self, src: Tensor) -> Tensor:
        out = self.encoder(src)
        out = self.pool(out)
        out = torch.squeeze(out, -1)
        out = self.linear(out)
        return out
