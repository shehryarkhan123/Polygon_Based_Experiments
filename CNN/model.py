import torch
import torch.nn as nn

from config import DROPOUT, N_CLASSES, N_FEATURES


class SpectralCNN1D(nn.Module):
    """Small 1D CNN over a 7-feature pixel spectrum."""

    def __init__(self, n_classes: int = N_CLASSES, seq_len: int = N_FEATURES) -> None:
        super().__init__()
        conv_len = seq_len - 2
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * conv_len, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
