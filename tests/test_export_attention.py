import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.export_attention import (attention_diagnostics, export_attention,
                                  sample_limit, symmetrize)
from src.model.attention_probe import AttentionProbe
from src.train_attention import train_attention
from src.utils import sha256_file


@pytest.fixture
def export_inputs(tmp_path, config):
    root = tmp_path / "data"
    root.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    theta = np.random.default_rng(0).normal(size=(2, 30, 128))
    x = np.stack((np.sin(theta), np.cos(theta)), axis=-1).astype(np.float32)
    # Deliberately omit all structural arrays: the exporter must not need them.
    np.savez_compressed(root / "t.npz", x_all=x, permutation=np.tile(np.arange(30), (2, 1)))
    records = [dict(file="t.npz", index=i, trajectory_id=f"t{i}", topology_id="topo", split="test")
               for i in range(2)]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(dict(run_id="fixture", config=config, trajectories=records)))
    model = AttentionProbe(config)
    checkpoint = tmp_path / "best.pt"
    torch.save(dict(run_id="fixture_train", config=config, seed=42, epoch=1, node_only=False,
                    model_state=model.state_dict(), data_manifest_sha256=sha256_file(manifest_path)), checkpoint)
    return config_path, root, checkpoint, x


@pytest.mark.parametrize("limit,expected", [(1, 1), (8, 2), (None, 2)])
def test_export_real_features_without_labels(export_inputs, tmp_path, monkeypatch, limit, expected):
    config_path, root, checkpoint, x = export_inputs
    original = np.lib.npyio.NpzFile.__getitem__
    requested = []

    def guarded(self, key):
        requested.append(key)
        if key not in ("x_all", "permutation"):
            raise AssertionError(f"Unexpected dataset array read: {key}")
        return original(self, key)

    with monkeypatch.context() as patch:
        patch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded)
        output = tmp_path / f"export_{expected}"
        metadata = export_attention(config_path, root, checkpoint, output,
                                    max_samples=limit, batch_size=1)
    assert set(requested) == {"x_all", "permutation"}
    with np.load(output / "attention_features.npz", allow_pickle=False) as f:
        assert f["attention_logits"].shape == (expected, 2, 4, 30, 30)
        assert f["temporal_features"].shape == (expected, 30, 96, 64)
        np.testing.assert_array_equal(f["x_obs"], x[:expected, :, :96])
        # PyTorch and NumPy accumulate float32 means in different orders.
        np.testing.assert_allclose(f["node_tokens"], f["temporal_features"].mean(2), atol=1e-6, rtol=1e-5)
        assert f["trajectory_id"].tolist() == [f"t{i}" for i in range(expected)]
        assert not {"y_pair", "y_tri", "omega"}.intersection(f.files)
        np.testing.assert_allclose(f["attention_weights"].sum(-1), 1, atol=2e-7)
        for name in ("symmetric_logits", "symmetric_attention"):
            np.testing.assert_array_equal(f[name], f[name].swapaxes(-1, -2))
            assert not np.diagonal(f[name], axis1=-2, axis2=-1).any()
    assert metadata["new_training_during_export"] is False
    diag = json.loads((output / "attention_diagnostics.json").read_text())
    assert len(diag["per_head"]) == 8
    assert np.array(diag["head_similarity"]["attention"]["cosine"]["mean"]).shape == (8, 8)
    assert all((output / name).is_file() for name in
               ("heads_layer1.png", "heads_layer2.png", "symmetric_heads.png", "README_attention_results.md"))
    assert not (output / "training_loss.png").exists()
    with pytest.raises(FileExistsError):
        export_attention(config_path, root, checkpoint, output, max_samples=1)


def test_missing_inputs_have_repair_commands(export_inputs, tmp_path):
    config_path, root, checkpoint, _ = export_inputs
    with pytest.raises(FileNotFoundError, match="src.data.generate"):
        export_attention(config_path, tmp_path / "missing_data", checkpoint, tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="src.train_attention"):
        export_attention(config_path, root, tmp_path / "missing.pt", tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert sample_limit("all") is None and sample_limit("8") == 8


def test_uniform_entropy_and_undefined_constant_head_similarity():
    a = np.ones((1, 2, 4, 30, 30), dtype=np.float64)
    a[..., np.arange(30), np.arange(30)] = 0
    a /= 29
    s = np.zeros_like(a)
    features = dict(attention_logits=s, attention_weights=a,
                    symmetric_logits=symmetrize(s), symmetric_attention=symmetrize(a))
    diag = attention_diagnostics(features)
    for head in diag["per_head"]:
        np.testing.assert_allclose(head["entropy_nats"]["mean"], np.log(29), atol=1e-12)
    assert np.isnan(diag["head_similarity"]["logits"]["cosine"]["mean"]).all()
    assert np.isnan(diag["head_similarity"]["attention"]["pearson"]["mean"]).all()


def test_main_only_training_entry(small_data, tmp_path):
    root, manifest = small_data
    cfg = copy.deepcopy(manifest["config"])
    cfg["training"]["epochs"] = 1
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "train"
    train_attention(config_path, root, output)
    assert (output / "best.pt").is_file()
    assert (output / "last.pt").is_file()
    assert (output / "training_loss.png").is_file()
    assert not (output / "node_only").exists()
    assert not (output / "metrics_forecast.json").exists()
    status = json.loads((output / "training_status.json").read_text())
    assert status["epochs"] == 1 and status["status"] == "complete"
