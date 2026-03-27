import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self,
        embedDim: int,
        ffDim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedDim, ffDim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffDim, embedDim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
