import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def read_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fresh_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_id(path):
    return f"{Path(path).name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"


def seed_all(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def get_device(name):
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {name}, but CUDA is unavailable; no silent CPU fallback")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index unavailable: {name}")
    return device


def environment():
    packages = {}
    for name in ["numpy", "scipy", "torch", "PyYAML", "matplotlib", "pytest"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages=packages, cuda_available=torch.cuda.is_available(),
                cuda_version=torch.version.cuda, command=[sys.executable, *sys.argv])


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot(out):
    """Archive only this project's source; never consume any other project."""
    import shutil
    root = Path(__file__).resolve().parents[1]
    dest = Path(out) / "source"
    dest.mkdir()
    hashes = {}
    for folder in ("src", "configs", "tests"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(rel)] = sha256_file(path)
    for name in ("README.md", "requirements.txt", "pytest.ini", ".gitignore"):
        path = root / name
        if path.exists():
            shutil.copy2(path, dest / name)
            hashes[name] = sha256_file(path)
    write_json(Path(out) / "source_hashes.json", hashes)
