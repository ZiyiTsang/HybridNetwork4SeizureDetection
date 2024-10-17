import comet_ml
import torch
from torch import nn
from torchinfo import summary
from positional_encodings.torch_encodings import PositionalEncoding1D
from einops import rearrange, repeat
from torchvision.ops import sigmoid_focal_loss
import transformers
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from .CommonBlock import Classification_block
import lightning as L


class TransformerBlock(nn.Module):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--d_in", type=int, default=1430)
        parser.add_argument("--d_model", type=int, default=768)
        parser.add_argument("--n_heads", type=int, default=6)
        parser.add_argument("--n_layers", type=int, default=3)
        return parent_parser

    def __init__(self, d_in=1430, d_model=768, n_heads=8, num_layers=3):
        super(TransformerBlock, self).__init__()
        self.linear_projection = nn.Linear(d_in, 768)
        self.positionEncoding = PositionalEncoding1D(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classification_block = Classification_block()
        self.cls_token = nn.Parameter(torch.randn(1, d_model))

        self.encoder_name = "transformer"

    def forward(self, x):
        x = rearrange(x, 'a b c d -> a d (b c)')
        x = self.linear_projection(x)
        cls_tokens = repeat(self.cls_token, 'n d -> b n d', b=x.size(0))
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positionEncoding(x)
        x = self.transformer_encoder(x)
        return x[:, 0, :]



