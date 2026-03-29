from torch import nn
from transformer.causalSelfAttention import CausalSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, dModel, numHeads, dff, dropoutRate=0.1):
        super().__init__()
        
        self.layernorm1 = nn.LayerNorm(dModel)
        self.attnLayer = CausalSelfAttention(dModel, numHeads, dropoutRate)
        
        self.layernorm2 = nn.LayerNorm(dModel)
        self.ffnLayer = nn.Sequential(
            nn.Linear(dModel, dff),
            nn.GELU(),
            nn.Linear(dff, dModel),
            nn.Dropout(dropoutRate)
        )

    def forward(self, x):
        x = x + self.attnLayer(self.layernorm1(x))
        x = x + self.ffnLayer(self.layernorm2(x))
        return x
