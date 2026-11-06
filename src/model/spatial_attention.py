import math

import torch
import torch.nn as nn


class SpatialBlock(nn.Module):
    """Parallel heads, post-norm residuals; export pre-mask S and pre-dropout A."""

    def __init__(self, d_model=64, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, z):
        b, n, d = z.shape
        if n < 2:
            raise ValueError("At least two nodes are required to mask self-attention")
        q, k, v = [p(z).reshape(b, n, self.n_heads, self.head_dim).transpose(1, 2)
                   for p in (self.q, self.k, self.v)]
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        mask = torch.eye(n, dtype=torch.bool, device=z.device)
        attention = logits.masked_fill(mask, -torch.inf).softmax(dim=-1)
        merged = (self.attention_dropout(attention) @ v).transpose(1, 2).reshape(b, n, d)
        z = self.norm1(z + self.dropout(self.output(merged)))
        z = self.norm2(z + self.dropout(self.ffn(z)))
        return z, logits, attention
