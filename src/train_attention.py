"""Train only the attention model; prediction provides the optimization signal."""
import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from .data.dataset import ForecastDataset, forecast_batch
from .model.attention_probe import AttentionProbe
from .utils import (environment, fresh_directory, get_device, read_config, run_id,
                    seed_all, sha256_file, source_snapshot, write_csv, write_json)


def loss_for_batch(model, batch, device):
    """Only history and future targets cross the training boundary."""
    x, target = forecast_batch(batch, device)
    return F.mse_loss(model(x)["forecast"], target)


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        loss = loss_for_batch(model, batch, device)
        total += loss.item() * len(batch["x_obs"])
        count += len(batch["x_obs"])
    return total / count


def train_model(config, data, out, device, seed, identity=""):
    seed_all(seed, config["training"]["num_threads"])
    model = AttentionProbe(config).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    train_set, val_set = ForecastDataset(data, "train"), ForecastDataset(data, "val")
    generator = torch.Generator().manual_seed(seed)
    hp = config["training"]
    train_loader = DataLoader(train_set, batch_size=hp["batch_size"], shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=hp["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    best, stalled, history = float("inf"), 0, []

    def save(filename, epoch, val):
        torch.save(dict(run_id=identity, config=config, seed=seed,
            model_state=model.state_dict(), initial_state=initial_state, optimizer_state=optimizer.state_dict(),
            shuffle_generator_state=generator.get_state(), torch_rng_state=torch.get_rng_state(),
            epoch=epoch, val_mse=val, data_manifest_sha256=sha256_file(Path(data) / "manifest.json")),
            out / filename)

    save("untrained.pt", 0, validation_loss(model, val_loader, device))
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, batch, device)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite forecast loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"], error_if_nonfinite=True)
            if norm <= 0:
                raise RuntimeError("Zero forecast gradient")
            optimizer.step()
            size = len(batch["x_obs"])
            total += loss.item() * size
            count += size
        val = validation_loss(model, val_loader, device)
        history.append(dict(run_id=identity, epoch=epoch, train_mse=total / count, val_mse=val))
        if val < best:
            best, stalled = val, 0
            save("best.pt", epoch, val)
        else:
            stalled += 1
        save("last.pt", epoch, val)
        write_csv(out / "train_history.csv", history)
        print(f"epoch={epoch} train_mse={total/count:.6f} val_mse={val:.6f}", flush=True)
        if stalled >= hp["patience"]:
            break
    return history


def train_attention(config_path, data, out, device="cpu", seed=42):
    config = read_config(config_path)
    manifest_path = Path(data) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Generate data first:\n"
            f"python -m src.data.generate --config {config_path} --out {data}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config["data"] != manifest["config"]["data"]:
        raise ValueError("Configuration differs from the generated data")
    device = get_device(device)
    if config["training"]["epochs"] < 1:
        raise ValueError("At least one epoch is required")
    out = fresh_directory(out)
    identity = run_id(out)
    config["training"]["seed"] = seed
    config["run_id"] = identity
    config["device"] = str(device)
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(out / "environment.json", dict(run_id=identity, **environment()))
    source_snapshot(out)
    # Reuse the existing forecast-only optimization and validation selection.
    history = train_model(config, data, out, device, seed, identity=identity)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for field in ("train_mse", "val_mse"):
        ax.plot([r["epoch"] for r in history], [r[field] for r in history], label=field)
    ax.set(xlabel="epoch", ylabel="forecast MSE (optimization signal only)", title=identity)
    ax.legend()
    fig.savefig(out / "training_loss.png", dpi=150)
    plt.close(fig)
    write_json(out / "training_status.json", dict(run_id=identity, status="complete",
        epochs=len(history), best_epoch=min(history, key=lambda r: r["val_mse"])["epoch"],
        checkpoint_selection="validation forecast MSE only",
        purpose="data-driven attention feature extraction; no prediction benchmarking"))
    print(f"RUN_PATH={out.resolve()} RUN_ID={identity}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        train_attention(args.config, args.data, args.out, args.device, args.seed)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
