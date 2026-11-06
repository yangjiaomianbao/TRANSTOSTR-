import inspect

import numpy as np
import pytest
import torch

from src.data.dataset import ForecastDataset, forecast_batch
from src.model.attention_probe import AttentionProbe
from src.train_attention import loss_for_batch, train_model


def test_training_dataset_and_guard(small_data, config):
    root, _ = small_data
    dataset = ForecastDataset(root, "train")
    assert set(dataset[0]) == {"x_obs", "y_future"}
    with pytest.raises(ValueError, match="forbidden"):
        forecast_batch({**dataset[0], "y_pair": torch.ones(15)}, "cpu")
    assert list(inspect.signature(AttentionProbe.forward).parameters) == ["self", "x_obs"]
    source = inspect.getsource(train_model)
    assert "y_pair" not in source and "y_tri" not in source
    assert "ap" not in inspect.signature(loss_for_batch).parameters


def test_tcn_causality(config):
    model = AttentionProbe(config).eval()
    x = torch.randn(1, 4, 96, 2)
    changed = x.clone(); changed[:,:,48:] = torch.randn_like(changed[:,:,48:])*5
    with torch.no_grad():
        h, perturbed = model.tcn(x), model.tcn(changed)
    torch.testing.assert_close(h[:,:,:48], perturbed[:,:,:48])
    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 128, 2))


def test_future_and_label_arrays_cannot_change_training_inputs(small_data, tmp_path):
    import json
    import shutil
    root, manifest = small_data
    clone = tmp_path/"clone"
    shutil.copytree(root, clone)
    before = ForecastDataset(root,"train")
    for t in manifest["topologies"]:
        if t["split"] != "train":
            continue
        f = clone/t["file"]
        arrays = dict(np.load(f))
        arrays["y_pair"] = 1-arrays["y_pair"]
        arrays["y_tri"] = 1-arrays["y_tri"]
        observed = manifest["config"]["data"]["observed"]
        arrays["x_all"][:,:,observed:] = 0
        np.savez_compressed(f,**arrays)
    after = ForecastDataset(clone,"train")
    torch.testing.assert_close(before.x_obs, after.x_obs)
    assert torch.count_nonzero(after.y_future) == 0
