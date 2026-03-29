import torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, dModel, numHeads, dropoutRate=0.1):
        super().__init__()
        self.numHeads = numHeads
        self.headDim = dModel // numHeads
        
        self.qkvProjection = nn.Linear(dModel, 3 * dModel)
        self.outputProjection = nn.Linear(dModel, dModel)
        
        self.dropoutRate = dropoutRate

    def forward(self, x):
        batchSize, seqLen, dModel = x.shape
        
        qkv = self.qkvProjection(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batchSize, seqLen, self.numHeads, self.headDim).transpose(1, 2)
        k = k.view(batchSize, seqLen, self.numHeads, self.headDim).transpose(1, 2)
        v = v.view(batchSize, seqLen, self.numHeads, self.headDim).transpose(1, 2)
        
        attnOutput = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropoutRate if self.training else 0.0,
            is_causal=True
        )
        
        attnOutput = attnOutput.transpose(1, 2).contiguous().view(batchSize, seqLen, dModel)
        
        return self.outputProjection(attnOutput)
