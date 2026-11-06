import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ForecastDataset(Dataset):
    """Materialize ONLY historical inputs and future targets; never open label arrays."""

    def __init__(self, root, split):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        observed = manifest["config"]["data"]["observed"]
        xs, ys = [], []
        for topo in manifest["topologies"]:
            if topo["split"] != split:
                continue
            with np.load(root / topo["file"], allow_pickle=False) as f:
                x = f["x_all"]
            xs.append(x[:, :, :observed]); ys.append(x[:, :, observed:])
        if not xs:
            raise ValueError(f"Empty split: {split}")
        self.x_obs = torch.from_numpy(np.concatenate(xs))
        self.y_future = torch.from_numpy(np.concatenate(ys))

    def __len__(self):
        return len(self.x_obs)

    def __getitem__(self, index):
        return {"x_obs": self.x_obs[index], "y_future": self.y_future[index]}


def forecast_batch(batch, device):
    if set(batch) != {"x_obs", "y_future"}:
        raise ValueError("Training accepts exactly x_obs/y_future; labels and metadata are forbidden")
    return batch["x_obs"].to(device), batch["y_future"].to(device)
