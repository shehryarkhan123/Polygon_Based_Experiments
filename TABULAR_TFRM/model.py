import torch
import torch.nn as nn

from config import D_MODEL, DROPOUT, FF_DIM, N_CLASSES, N_FEATURES, N_HEADS, N_LAYERS


class FeatureTokenizer(nn.Module):
    """Numeric FT-Transformer tokenizer: feature j -> x_j * W_j + b_j."""

    def __init__(self, n_features: int = N_FEATURES, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias


class TabularTransformer(nn.Module):
    """Small FT-Transformer: 7 feature tokens + CLS, 2 layers, d=32."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_classes: int = N_CLASSES,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        out = self.encoder(seq)
        return self.head(out[:, 0])
