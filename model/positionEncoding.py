import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embedDim: int,
        encodingBase: float,
        contextWindow: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(contextWindow, embedDim)
        position = torch.arange(0, contextWindow).unsqueeze(1).float()
        divTerm = torch.exp(
            torch.arange(0, embedDim, 2).float() * (-math.log(encodingBase) / embedDim)
        )

        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
