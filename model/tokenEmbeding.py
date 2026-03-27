import torch
import torch.nn as nn

from model.positionEncoding import PositionalEncoding


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocabSize: int,
        embedDim: int,
        encodingBase: float,
        contextWindow: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenEmb = nn.Embedding(vocabSize, embedDim)
        self.posEnc = PositionalEncoding(embedDim, encodingBase, contextWindow, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokenEmbeddings = self.tokenEmb(x)
        return self.posEnc(tokenEmbeddings)
