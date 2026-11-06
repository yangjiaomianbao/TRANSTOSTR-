import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from .data.dataset import ForecastDataset, forecast_batch
from .model.attention_probe import AttentionProbe
from .utils import (environment, fresh_directory, get_device, read_config, run_id,
                    seed_all, sha256_file, source_snapshot, write_csv, write_json)


def loss_for_batch(model, batch, device):
    """The entire training boundary: exactly two fields, no structure or metadata."""
    x, target = forecast_batch(batch, device)
    return F.mse_loss(model(x)["forecast"], target)


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    total, count = 0., 0
    for batch in loader:
        loss = loss_for_batch(model, batch, device)
        total += loss.item()*len(batch["x_obs"])
        count += len(batch["x_obs"])
    return total/count


def train_model(config, data, out, device, seed, node_only=False, identity=""):
    seed_all(seed, config["training"]["num_threads"])
    model = AttentionProbe(config, node_only=node_only).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    train_set, val_set = ForecastDataset(data, "train"), ForecastDataset(data, "val")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"],
                              shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=config["training"]["batch_size"])
    hp = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    best, stalled, history = float("inf"), 0, []
    kind = "node-only" if node_only else "forecast-trained"
    def save(filename, epoch, val):
        torch.save(dict(run_id=identity, config=config, seed=seed, node_only=node_only,
            model_state=model.state_dict(), initial_state=initial_state, optimizer_state=optimizer.state_dict(),
            shuffle_generator_state=generator.get_state(), torch_rng_state=torch.get_rng_state(),
            epoch=epoch, val_mse=val, data_manifest_sha256=sha256_file(Path(data)/"manifest.json")),
            out / filename)
    save("untrained.pt", 0, validation_loss(model, val_loader, device))
    for epoch in range(1, hp["epochs"]+1):
        model.train()
        total, count = 0., 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, batch, device)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite forecast loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"], error_if_nonfinite=True)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError("Invalid or zero forecast gradient")
            optimizer.step()
            size = len(batch["x_obs"])
            total += loss.item()*size; count += size
        val = validation_loss(model, val_loader, device)
        history.append(dict(run_id=identity, model=kind, epoch=epoch, train_mse=total/count, val_mse=val))
        if val < best:
            best, stalled = val, 0
            save("best.pt", epoch, val)
        else:
            stalled += 1
        save("last.pt", epoch, val)
        write_csv(out / "train_history.csv", history)
        print(f"{kind} epoch={epoch} train_mse={total/count:.6f} val_mse={val:.6f}", flush=True)
        if stalled >= hp["patience"]:
            break
    return history


@torch.no_grad()
def forecast_metrics(model, dataset, device, batch_size):
    if model is not None:
        model.eval()
    squared, circular, per_sample, per_angle = [], [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size):
        x, y = forecast_batch(batch, device)
        pred = model(x)["forecast"] if model is not None else x[:, :, -1:, :].expand_as(y)
        err = (pred-y).square()
        angle_pred, angle_y = torch.atan2(pred[..., 0], pred[..., 1]), torch.atan2(y[..., 0], y[..., 1])
        diff = angle_pred-angle_y
        angle = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()
        squared.append(err.mean(dim=(1, 3)).cpu().numpy())
        circular.append(angle.mean(dim=1).cpu().numpy())
        per_sample.extend(err.mean(dim=(1, 2, 3)).cpu().tolist())
        per_angle.extend(angle.mean(dim=(1, 2)).cpu().tolist())
    squared, circular = np.concatenate(squared), np.concatenate(circular)
    return dict(mse=float(squared.mean()), circular_mae=float(circular.mean()),
                mse_per_horizon=squared.mean(0).tolist(), circular_mae_per_horizon=circular.mean(0).tolist(),
                mse_per_trajectory=per_sample, circular_mae_per_trajectory=per_angle)


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "data", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = read_config(args.config)
    device = get_device(args.device)
    manifest = json.loads((Path(args.data)/"manifest.json").read_text())
    if config["data"] != manifest["config"]["data"]:
        raise ValueError("Requested configuration differs from generated data")
    out = fresh_directory(args.out)
    identity = run_id(out)
    config["training"]["seed"] = args.seed
    config["run_id"] = identity
    config["device"] = str(device)
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(out / "environment.json", dict(run_id=identity, **environment()))
    source_snapshot(out)
    train_model(config, args.data, out, device, args.seed, identity=identity)
    node_out = fresh_directory(out / "node_only")
    train_model(config, args.data, node_out, device, args.seed, node_only=True, identity=identity)
    dataset = ForecastDataset(args.data, "test")
    metrics = dict(run_id=identity, data_run_id=manifest["run_id"], models={}, parameter_counts={})
    for name, path, node in (("forecast-trained", out/"best.pt", False),
                              ("node-only", node_out/"best.pt", True),
                              ("untrained", out/"untrained.pt", False)):
        model = AttentionProbe(config, node_only=node).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        metrics["models"][name] = forecast_metrics(model, dataset, device, config["training"]["batch_size"])
        metrics["models"][name]["checkpoint_epoch"] = ckpt["epoch"]
        metrics["parameter_counts"][name] = sum(p.numel() for p in model.parameters())
    metrics["models"]["persistence"] = forecast_metrics(None, dataset, device, config["training"]["batch_size"])
    # Validation persistence baseline is recorded separately from checkpoint selection.
    metrics["validation_persistence"] = forecast_metrics(None, ForecastDataset(args.data, "val"),
                                                         device, config["training"]["batch_size"])
    metadata = []
    for t in manifest["topologies"]:
        if t["split"] == "test":
            for idx in range(config["data"]["trajectories"]):
                metadata.append(dict(family_id=t["family_id"], topology_id=t["topology_id"],
                                     variant=t["variant"], trajectory_index=idx))
    metrics["test_trajectory_order"] = metadata
    write_json(out / "metrics_forecast.json", metrics)
    print(json.dumps({k:v["mse"] for k,v in metrics["models"].items()}, indent=2), flush=True)
    print(f"RUN_PATH={out.resolve()} RUN_ID={identity}", flush=True)


if __name__ == "__main__":
    main()
