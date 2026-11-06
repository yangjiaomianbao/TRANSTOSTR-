import json

import numpy as np
import pytest

from scripts import compare_attention_pair_and_hypergraph as comparison
from src.data.kuramoto import (candidates, labels_for_permutation, pair_adjacency,
                               triad_incidence)
from src.utils import sha256_file


def test_true_pair_adjacency_and_hyperedge_columns():
    pairs, tris = candidates(5)
    edges = np.array([[0, 2], [3, 4]])
    triads = np.array([[0, 1, 3], [1, 2, 4]])
    perm = np.array([4, 2, 0, 3, 1])
    yp, yt = labels_for_permutation(edges, triads, perm, pairs, tris)
    a2 = pair_adjacency(yp, pairs, n=5)
    h3, selected = triad_incidence(yt, tris, n=5)
    expected_a2 = np.zeros((5, 5), dtype=np.uint8)
    expected_a2[0, 2] = expected_a2[2, 0] = 1
    expected_a2[3, 4] = expected_a2[4, 3] = 1
    np.testing.assert_array_equal(a2, expected_a2[perm][:, perm])
    assert a2.sum() == 4 and not a2.diagonal().any()
    assert h3.shape == (5, 2)
    np.testing.assert_array_equal(h3.sum(0), [3, 3])
    for col, nodes in enumerate(selected):
        np.testing.assert_array_equal(np.flatnonzero(h3[:, col]), nodes)
    canonical_triads = {tuple(sorted(perm[t])) for t in selected}
    assert canonical_triads == set(map(tuple, triads))
    empty, nodes = triad_incidence(np.zeros(len(tris)), tris, n=5)
    assert empty.shape == (5, 0) and nodes.shape == (0, 3)
    with pytest.raises(ValueError, match="binary"):
        pair_adjacency(np.full(len(pairs), .5), pairs, n=5)


@pytest.fixture
def comparison_source(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    export = tmp_path / "export"
    export.mkdir()
    pairs, tris = candidates(30)
    candidate_path = root / "candidates.npz"
    np.savez_compressed(candidate_path, pair_index=pairs, tri_index=tris)
    rng = np.random.default_rng(1)
    observations, perms, records, hashes = [], [], [], {}
    for i, variant in enumerate(("pair_only", "mixed_a")):
        perm = rng.permutation(30)
        x = rng.normal(size=(1, 30, 128, 2)).astype(np.float32)
        yp, yt = labels_for_permutation(np.array([[0, 1], [2, 3]]),
            np.array([[0, 1, 2], [2, 3, 4]]) if i else np.empty((0, 3), dtype=int), perm, pairs, tris)
        path = root / f"{variant}.npz"
        np.savez_compressed(path, x_all=x, permutation=perm[None], y_pair=yp[None], y_tri=yt[None])
        hashes[path.name] = sha256_file(path)
        records.append(dict(trajectory_id=f"f0_{variant}_t0", topology_id=f"f0_{variant}", file=path.name, index=0))
        observations.append(x[0, :, :96]); perms.append(perm)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(dict(config=dict(data=dict(n_nodes=30)),
        trajectories=records, candidate_sha256=sha256_file(candidate_path))))
    metadata_path = export / "metadata.json"
    metadata_path.write_text(json.dumps(dict(run_id="test-export", data=str(root), samples=records,
        source_data_sha256=hashes, manifest_sha256=sha256_file(manifest_path), checkpoint_epoch=3)))
    s = rng.normal(size=(2, 2, 4, 30, 30)).astype(np.float32)
    a = np.exp(s - s.max(-1, keepdims=True))
    a[..., np.arange(30), np.arange(30)] = 0
    a /= a.sum(-1, keepdims=True)
    rs, ra = (s + s.swapaxes(-1, -2)) / 2, (a + a.swapaxes(-1, -2)) / 2
    rs[..., np.arange(30), np.arange(30)] = 0
    ra[..., np.arange(30), np.arange(30)] = 0
    features = export / "attention_features.npz"
    np.savez_compressed(features, attention_logits=s, attention_weights=a, symmetric_logits=rs,
        symmetric_attention=ra, permutation=np.stack(perms), x_obs=np.stack(observations),
        trajectory_id=[r["trajectory_id"] for r in records], topology_id=[r["topology_id"] for r in records])
    return metadata_path, features, records


def test_default_all_and_single_sample_keep_all_four_matrices(comparison_source, tmp_path, monkeypatch):
    metadata, features, records = comparison_source
    plotted = []
    monkeypatch.setattr(comparison, "plot_pairwise", lambda entry, *args: plotted.append(entry["trajectory_id"]))
    monkeypatch.setattr(comparison, "plot_incidence", lambda *args: None)
    output = comparison.compare(metadata, features, out=tmp_path / "all")
    assert plotted == [r["trajectory_id"] for r in records]
    for record, m in zip(records, (0, 2)):
        folder = output / record["trajectory_id"]
        a2, h3 = np.load(folder / "A2_true.npy"), np.load(folder / "H3_true.npy")
        assert a2.shape == (30, 30) and h3.shape == (30, m)
        triples = json.loads((folder / "triads_true.json").read_text())
        assert len(triples) == m
        for col, nodes in enumerate(triples):
            assert np.flatnonzero(h3[:, col]).tolist() == nodes
        with np.load(features) as original, np.load(folder / "attention_matrices.npz") as actual:
            idx = records.index(record)
            for key in comparison.MATRIX_KEYS:
                np.testing.assert_array_equal(actual[key], original[key][idx])
    plotted.clear()
    single = comparison.compare(metadata, features, out=tmp_path / "single", sample=records[1]["trajectory_id"])
    assert plotted == [records[1]["trajectory_id"]]
    assert not (single / records[0]["trajectory_id"]).exists()
    with pytest.raises(FileExistsError):
        comparison.compare(metadata, features, out=single)


def test_reject_unknown_id_and_misaligned_node_permutation(comparison_source, tmp_path):
    metadata, features, _ = comparison_source
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="Unknown --sample"):
        comparison.compare(metadata, features, out=output, sample="unknown")
    with np.load(features) as f:
        arrays = dict(f)
    arrays["permutation"][0] = arrays["permutation"][0, ::-1]
    np.savez_compressed(features, **arrays)
    with pytest.raises(ValueError, match="Node permutation mismatch"):
        comparison.compare(metadata, features, out=output)
    assert not output.exists()


def test_empty_incidence_plot_is_explicit(tmp_path):
    output = tmp_path / "empty.png"
    comparison.plot_incidence(np.zeros((30, 0), dtype=np.uint8), output, "empty-test", "unit")
    assert output.is_file() and output.stat().st_size > 1000
