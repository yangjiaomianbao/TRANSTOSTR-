"""Offline evaluation only. Training does not import this module."""
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def symmetric(x):
    result = (x + np.swapaxes(x, -1, -2)) / 2
    idx = np.arange(result.shape[-1])
    result[..., idx, idx] = 0
    return result


def score_matrices(logits, attention):
    """Input [layers,heads,N,N]; no truth or trainable weights are accepted."""
    s, a = symmetric(logits), symmetric(attention)
    n = s.shape[-1]
    upper = np.triu_indices(n, 1)
    vectors = s[..., upper[0], upper[1]]
    centered = vectors - vectors.mean(-1, keepdims=True)
    std = vectors.std(-1, keepdims=True)
    standardized = np.divide(centered, std, out=np.zeros_like(centered), where=std > 1e-12)
    fused = np.zeros((n, n), dtype=s.dtype)
    fused[upper] = standardized.mean(axis=(0, 1))
    fused += fused.T
    result = {}
    for l in range(len(s)):
        for h in range(s.shape[1]):
            result[f"L{l+1}H{h+1}_logit_sym"] = s[l, h]
            result[f"L{l+1}H{h+1}_attn_sym"] = a[l, h]
        result[f"L{l+1}_mean_logit_z"] = symmetric_from_upper(standardized[l].mean(0), n)
        result[f"L{l+1}_mean_attention"] = a[l].mean(0)
    result["fused_logit"] = fused
    result["fused_attention"] = a.mean(axis=(0, 1))
    return result


def symmetric_from_upper(vector, n):
    x = np.zeros((n, n), dtype=vector.dtype)
    x[np.triu_indices(n, 1)] = vector
    return x + x.T


def association_baselines(x_obs, seed):
    phase = x_obs[..., 1].astype(np.float64) + 1j*x_obs[..., 0].astype(np.float64)
    plv = np.abs(phase @ phase.conj().T / phase.shape[1])
    delta = np.angle(phase[:, 1:] * phase[:, :-1].conj())
    centered = delta - delta.mean(axis=-1, keepdims=True)
    denom = np.linalg.norm(centered, axis=-1)
    corr = centered @ centered.T
    norm = denom[:, None] * denom[None, :]
    corr = np.divide(corr, norm, out=np.zeros_like(corr), where=norm > 1e-15)
    n = len(phase)
    random = symmetric_from_upper(np.random.default_rng(seed).random(n*(n-1)//2), n)
    return {"PLV": symmetric(plv), "abs_pearson_delta_phase": symmetric(np.abs(corr)), "random": random}


def binary_metrics(y, score):
    y, score = np.asarray(y, dtype=bool), np.asarray(score, dtype=np.float64)
    if y.shape != score.shape or not np.isfinite(score).all():
        raise ValueError("Labels/scores must have matching shape and finite scores")
    positives, negatives = int(y.sum()), int((~y).sum())
    status = "ok" if positives and negatives else "no_positive" if not positives else "no_negative"
    return dict(ap=float(average_precision_score(y, score)) if positives else np.nan,
        auroc=float(roc_auc_score(y, score)) if positives and negatives else np.nan,
        precision_at_k=float(y[np.argsort(-score, kind="stable")[:positives]].mean()) if positives else np.nan,
        positives=positives, negatives=negatives,
        density=float(y.mean()) if len(y) else np.nan,
        positive_score_mean=float(score[y].mean()) if positives else np.nan,
        negative_score_mean=float(score[~y].mean()) if negatives else np.nan, status=status)


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else np.nan


def family_ci(values, families, seed=42, samples=1000):
    """Topology means as observations, resampling whole families (cluster bootstrap)."""
    values, families = np.asarray(values, dtype=float), np.asarray(families)
    good = np.isfinite(values)
    values, families = values[good], families[good]
    unique = np.unique(families)
    if not len(unique):
        return dict(mean=np.nan, ci95=[np.nan, np.nan], n_topologies=0, n_families=0)
    sums = np.array([values[families == f].sum() for f in unique])
    counts = np.array([(families == f).sum() for f in unique])
    draws = np.random.default_rng(seed).integers(0, len(unique), size=(samples, len(unique)))
    boot = sums[draws].sum(1)/counts[draws].sum(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                n_topologies=len(values), n_families=len(unique))


def matrix_comparison(a, b):
    upper = np.triu_indices(len(a), 1)
    va, vb = a[upper], b[upper]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float(spearmanr(va, vb).statistic)
    return dict(mean_abs_difference=float(np.abs(va-vb).mean()), spearman=rho)
