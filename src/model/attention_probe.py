import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_attention import SpatialBlock
from .tcn import SharedTCN


class AttentionProbe(nn.Module):
    def __init__(self, config, node_only=False):
        super().__init__()
        cfg = config["model"]
        d = cfg["d_model"]
        self.observed = config["data"]["observed"]
        self.future = config["data"]["future"]
        self.node_only = node_only
        self.tcn = SharedTCN(d, cfg["kernel_size"], cfg["dilations"], cfg["dropout"])
        self.layers = nn.ModuleList([] if node_only else [SpatialBlock(d, cfg["n_heads"],
            cfg["d_ff"], cfg["dropout"]) for _ in range(cfg["n_layers"])])
        # Replace spatial blocks with parameter-matched shared per-node residual FFNs.
        self.node_blocks = nn.ModuleList([nn.Sequential(nn.Linear(d, 4*d), nn.GELU(),
            nn.Dropout(cfg["dropout"]), nn.Linear(4*d, d), nn.Dropout(cfg["dropout"]))
            for _ in range(cfg["n_layers"])] if node_only else [])
        self.node_norms = nn.ModuleList([nn.LayerNorm(d) for _ in self.node_blocks])
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
        for block, norm in zip(self.node_blocks, self.node_norms):
            ctx = norm(ctx + block(ctx))
        forecast = self.forecast_head(ctx).reshape(*ctx.shape[:2], self.future, 2)
        forecast = F.normalize(forecast, dim=-1, eps=1e-6)
        return dict(forecast=forecast, attention_logits=torch.stack(logits, dim=1) if logits else None,
                    attention_weights=torch.stack(weights, dim=1) if weights else None,
                    node_tokens=tokens, context_tokens=ctx)
