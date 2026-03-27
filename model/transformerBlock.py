import torch
import torch.nn as nn

from model.attention   import MultiHeadSelfAttention
from model.feedForward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedDim: int,
        numHeads: int,
        ffDim: int,
        contextWindow: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention   = MultiHeadSelfAttention(embedDim, numHeads, contextWindow, dropout)
        self.feedForward = FeedForward(embedDim, ffDim, dropout)

        self.normOne = nn.LayerNorm(embedDim)
        self.normTwo = nn.LayerNorm(embedDim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normOne(x + self.attention(x))
        x = self.normTwo(x + self.feedForward(x))
        return x
