import numpy as np
import pytest

from src.data.kuramoto import candidates, label_matrices, make_rhs, labels_for_permutation
from src.data.generate import generate


def test_rhs_all_three_targets_and_degrees():
    theta = np.array([.2, .8, -.5, .7])
    edges = np.array([[0, 1], [0, 2]])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    omega = np.array([.1, -.2, .3, .4])
    result = make_rhs(omega, edges, tris, .8, .4)(0, theta)
    expected = omega.copy()
    for i in range(4):
        neighbors = [e[1] if e[0] == i else e[0] for e in edges if i in e]
        others = [[j for j in t if j != i] for t in tris if i in t]
        expected[i] += .8*sum(np.sin(theta[j]-theta[i]) for j in neighbors)/max(1,len(neighbors))
        expected[i] += .4*sum(np.sin(theta[j]+theta[k]-2*theta[i]) for j,k in others)/max(1,len(others))
    np.testing.assert_allclose(result, expected, atol=1e-14)
    np.testing.assert_array_equal(make_rhs(omega, [], [], .8, .4)(0, theta), omega)


def test_data_and_family_controls(small_data):
    root, manifest = small_data
    cfg = manifest["config"]["data"]
    splits = {s: {t["family_id"] for t in manifest["topologies"] if t["split"] == s}
              for s in ("train", "val", "test")}
    assert not splits["train"] & splits["test"]
    assert not splits["train"] & splits["val"]
    assert not splits["val"] & splits["test"]
    pairs, tris = candidates(cfg["n_nodes"])
    for family in set.union(*splits.values()):
        topologies = [t for t in manifest["topologies"] if t["family_id"] == family]
        data = [dict(np.load(root/t["file"])) for t in topologies]
        for d in data:
            assert np.isfinite(d["x_all"]).all()
            np.testing.assert_allclose((d["x_all"]**2).sum(-1), 1, atol=2e-7)
            assert d["times"][cfg["observed"]] > d["times"][cfg["observed"]-1]
            assert all(len(set(t)) == 3 for t in d["canonical_e3"])
            for j in range(cfg["trajectories"]):
                yp, yt = labels_for_permutation(d["canonical_e2"], d["canonical_e3"],
                                               d["permutation"][j], pairs, tris)
                np.testing.assert_array_equal(yp, d["y_pair"][j])
                np.testing.assert_array_equal(yt, d["y_tri"][j])
                a2, a3 = label_matrices(yp, yt, pairs, tris, cfg["n_nodes"])
                for a in (a2, a3):
                    np.testing.assert_array_equal(a, a.T)
                    assert not a.diagonal().any()
        for d in data[1:]:
            for key in ("canonical_e2", "canonical_omega", "theta0", "omega", "permutation", "seed"):
                np.testing.assert_array_equal(d[key], data[0][key])
        assert not len(data[0]["canonical_e3"])
        assert not np.array_equal(data[1]["canonical_e3"], data[2]["canonical_e3"])


def test_no_silent_overwrite(small_data):
    root, manifest = small_data
    with pytest.raises(FileExistsError):
        generate(manifest["config"], root)


def test_production_candidate_order():
    pairs, tris = candidates(30)
    assert pairs.shape == (435, 2) and tris.shape == (4060, 3)
    assert pairs.tolist() == sorted(pairs.tolist())
    assert tris.tolist() == sorted(tris.tolist())
