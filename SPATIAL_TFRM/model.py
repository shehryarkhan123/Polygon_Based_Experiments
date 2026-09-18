import torch
import torch.nn as nn

from config import D_MODEL, DROPOUT, FF_DIM, N_CLASSES, N_FEATURES, N_HEADS, N_LAYERS


class SpatialContextTransformer(nn.Module):
    """Neighbor transformer: one token per pixel, Linear 4→64, one encoder layer, mean pool."""

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        num_classes: int = N_CLASSES,
        d_model: int = D_MODEL,
        nhead: int = N_HEADS,
        num_layers: int = N_LAYERS,
        dim_feedforward: int = FF_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)


class FlattenMLP(nn.Module):
    """3x3 spatial MLP: flatten 9x4 then 36 → 64 → 5."""

    def __init__(
        self,
        n_tokens: int = 9,
        n_features: int = N_FEATURES,
        hidden: int = D_MODEL,
        n_classes: int = N_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_tokens * n_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchCNN2D(nn.Module):
    """Small 2D CNN on a (C, H, W) neighborhood. Not the pixel 1D CNN."""

    def __init__(
        self,
        in_ch: int = N_FEATURES,
        n_classes: int = N_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
