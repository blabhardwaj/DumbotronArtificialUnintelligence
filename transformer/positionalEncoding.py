import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dModel, maxContextLength=1024):
        super().__init__()
        pe = torch.zeros(maxContextLength, dModel)
        position = torch.arange(0, maxContextLength, dtype=torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, dModel, 2).float() * (-math.log(10000.0) / dModel))
        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
