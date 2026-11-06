import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_attention import SpatialBlock
from .tcn import SharedTCN


class AttentionProbe(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config["model"]
        d = cfg["d_model"]
        self.observed = config["data"]["observed"]
        self.future = config["data"]["future"]
        self.tcn = SharedTCN(d, cfg["kernel_size"], cfg["dilations"], cfg["dropout"])
        self.layers = nn.ModuleList([SpatialBlock(d, cfg["n_heads"],
            cfg["d_ff"], cfg["dropout"]) for _ in range(cfg["n_layers"])])
        self.forecast_head = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, self.future*2))

    def forward(self, x_obs):
        if x_obs.ndim != 4 or x_obs.shape[2:] != (self.observed, 2):
            raise ValueError(f"Expected [B,N,{self.observed},2] historical observations")
        h = self.tcn(x_obs)
        tokens = h.mean(dim=2)
        ctx = tokens
        logits, weights = [], []
        for layer in self.layers:
            ctx, s, a = layer(ctx)
            logits.append(s); weights.append(a)
        forecast = self.forecast_head(ctx).reshape(*ctx.shape[:2], self.future, 2)
        forecast = F.normalize(forecast, dim=-1, eps=1e-6)
        return dict(forecast=forecast, attention_logits=torch.stack(logits, dim=1) if logits else None,
                    attention_weights=torch.stack(weights, dim=1) if weights else None,
                    temporal_features=h, node_tokens=tokens, context_tokens=ctx)
