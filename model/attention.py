import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embedDim: int,
        numHeads: int,
        contextWindow: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedDim,
            num_heads=numHeads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)

        # Causal mask moved here since attention owns it
        self.register_buffer(
            "causalMask",
            torch.triu(torch.ones(contextWindow, contextWindow), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seqLen = x.size(1)
        mask   = self.causalMask[:seqLen, :seqLen]

        attnOutput, _ = self.attention(x, x, x, attn_mask=mask)
        return self.dropout(attnOutput)
