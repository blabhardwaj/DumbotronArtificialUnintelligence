import torch
import torch.nn as nn

from model.tokenEmbeding   import TokenEmbedding
from model.transformerBlock import TransformerBlock


class Transformer(nn.Module):
    def __init__(
        self,
        vocabSize: int,
        embedDim: int,
        numHeads: int,
        ffDim: int,
        numLayers: int,
        encodingBase: float,
        contextWindow: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = TokenEmbedding(
            vocabSize, embedDim, encodingBase, contextWindow, dropout
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(embedDim, numHeads, ffDim, contextWindow, dropout)
            for _ in range(numLayers)
        ])

        self.norm     = nn.LayerNorm(embedDim)
        self.outputHead = nn.Linear(embedDim, vocabSize, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)           # (batch, seq, embedDim)

        for block in self.blocks:
            x = block(x)                # (batch, seq, embedDim)

        x = self.norm(x)                # final layer norm
        logits = self.outputHead(x)     # (batch, seq, vocabSize)
        return logits
