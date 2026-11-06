import numpy as np
import pytest

from src.metrics import (association_baselines, binary_metrics, family_ci,
                         score_matrices)


def test_probe_symmetry_and_zero_variance():
    rng = np.random.default_rng(17)
    for s in (rng.normal(size=(2,4,30,30)), np.ones((2,4,30,30))):
        a = np.exp(s-s.max(-1,keepdims=True))
        a[...,np.arange(30),np.arange(30)] = 0
        a /= a.sum(-1,keepdims=True)
        scores = score_matrices(s,a)
        assert len(scores) == 22
        for matrix in scores.values():
            assert np.isfinite(matrix).all()
            np.testing.assert_allclose(matrix, matrix.T,atol=1e-7)
            assert not matrix.diagonal().any()
        if np.all(s == 1):
            assert not scores["fused_logit"].any()


def test_metrics_single_class_and_known_ranking():
    assert np.isnan(binary_metrics([0,0],[.1,.2])["ap"])
    assert np.isnan(binary_metrics([1,1],[.1,.2])["auroc"])
    assert binary_metrics([0,1,0,1],[0,1,.1,.9])["precision_at_k"] == 1
    assert binary_metrics([0,0],[.1,.2])["status"] == "no_positive"
    with pytest.raises(ValueError):
        binary_metrics([0,1],[.2,np.nan])


def test_random_ap_matches_density():
    y = np.zeros(435); y[:60]=1
    scores = [binary_metrics(y, np.random.default_rng(seed).random(435))["ap"] for seed in range(150)]
    assert abs(np.mean(scores)-y.mean()) < .025


def test_family_bootstrap_and_baselines():
    ci = family_ci([1,1,3,3],[0,0,1,1],samples=300)
    assert ci["mean"] == 2 and ci["n_families"] == 2
    assert ci["ci95"] == [1.,3.]
    theta = np.linspace(0,3,96)[None,:] + np.array([0,.3,.5])[:,None]
    x = np.stack((np.sin(theta),np.cos(theta)),axis=-1)
    baselines = association_baselines(x,42)
    np.testing.assert_allclose(baselines["PLV"][np.triu_indices(3,1)],1,atol=1e-12)
    for matrix in baselines.values():
        assert np.isfinite(matrix).all()
        assert not matrix.diagonal().any()
