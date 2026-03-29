import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.positionalEncoding import PositionalEncoding
from transformer.transformerBlock import TransformerBlock

class PytorchModel(nn.Module):
    def __init__(self, vocabSize, dModel, numHeads, dff, maxContextLength, numLayers, dropoutRate=0.1):
        super().__init__()
        self.dModel = dModel
        self.tokenEmbedding = nn.Embedding(vocabSize, dModel)
        nn.init.normal_(self.tokenEmbedding.weight, mean=0.0, std=0.02)
        self.posEncoding = PositionalEncoding(dModel, maxContextLength)
        
        self.dropout = nn.Dropout(dropoutRate)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dModel, numHeads, dff, dropoutRate)
            for _ in range(numLayers)
        ])
        
        self.finalLayernorm = nn.LayerNorm(dModel)
        
        self.lmHead = nn.Linear(dModel, vocabSize, bias=False)
        self.lmHead.weight = self.tokenEmbedding.weight
        self.maxContextLength = maxContextLength

    def forward(self, x):
        x = self.tokenEmbedding(x) * math.sqrt(self.dModel)
        x = self.posEncoding(x)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.finalLayernorm(x)
        logits = self.lmHead(x)
        
        return logits

def BuildPytorchModel(vocabSize, dModel, numHeads, dff, maxContextLength, numLayers, dropoutRate=0.1):
    return PytorchModel(vocabSize, dModel, numHeads, dff, maxContextLength, numLayers, dropoutRate)
