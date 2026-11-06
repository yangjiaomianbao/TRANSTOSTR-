import copy

import pytest

from src.data.generate import generate
from src.utils import read_config, seed_all


@pytest.fixture(scope="session", autouse=True)
def deterministic():
    seed_all(42, threads=2)


@pytest.fixture
def config():
    return read_config("configs/smoke.yaml")


@pytest.fixture(scope="session")
def small_data(tmp_path_factory):
    cfg = copy.deepcopy(read_config("configs/smoke.yaml"))
    cfg["data"].update(n_nodes=6, families={"train": 2, "val": 1, "test": 1},
                       expected_pairs=4, expected_triads=2, observed=8, future=4)
    path = tmp_path_factory.mktemp("datasets") / "tiny"
    manifest = generate(cfg, path)
    return path, manifest
