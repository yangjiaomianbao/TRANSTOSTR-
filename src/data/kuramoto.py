from itertools import combinations

import numpy as np
from scipy.integrate import solve_ivp


def candidates(n):
    return (np.array(list(combinations(range(n), 2)), dtype=np.int64),
            np.array(list(combinations(range(n), 3)), dtype=np.int64))


def make_rhs(omega, edges, triads, k2, k3):
    omega = np.asarray(omega, dtype=np.float64)
    n = len(omega)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    triads = np.asarray(triads, dtype=np.int64).reshape(-1, 3)
    degree2 = np.maximum(1, np.bincount(edges.ravel(), minlength=n))
    degree3 = np.maximum(1, np.bincount(triads.ravel(), minlength=n))
    # Cache directed contributions over present interactions, never all candidates.
    target2 = np.concatenate((edges[:, 0], edges[:, 1]))
    source2 = np.concatenate((edges[:, 1], edges[:, 0]))
    target3 = triads.T.ravel()
    others1 = triads[:, [1, 0, 0]].T.ravel()
    others2 = triads[:, [2, 2, 1]].T.ravel()

    def rhs(_t, theta):
        pair = np.bincount(target2, weights=np.sin(theta[source2] - theta[target2]), minlength=n)
        tri = np.bincount(target3, weights=np.sin(theta[others1] + theta[others2]
                                                - 2 * theta[target3]), minlength=n)
        return omega + k2 * pair / degree2 + k3 * tri / degree3

    return rhs


def integrate(omega, edges, triads, theta0, cfg):
    times = np.arange(cfg["observed"] + cfg["future"], dtype=np.float64) * cfg["dt"]
    result = solve_ivp(make_rhs(omega, edges, triads, cfg["k2"], cfg["k3"]),
                       (times[0], times[-1]), np.asarray(theta0, dtype=np.float64),
                       t_eval=times, method="RK45", rtol=cfg["rtol"], atol=cfg["atol"])
    if not result.success or result.y.shape[1] != len(times):
        raise RuntimeError(result.message)
    x = np.stack((np.sin(result.y), np.cos(result.y)), axis=-1).astype(np.float32)
    assert np.isfinite(x).all()
    assert np.allclose((x * x).sum(-1), 1, atol=2e-7)
    assert times[cfg["observed"]] > times[cfg["observed"] - 1]
    return x, times, result.nfev


def labels_for_permutation(edges, triads, perm, pair_index, tri_index):
    """new_x = old_x[perm]; map canonical node indices using inverse permutation."""
    inverse = np.argsort(perm)
    e2 = {tuple(sorted(inverse[e])) for e in edges}
    e3 = {tuple(sorted(inverse[e])) for e in triads}
    return (np.array([tuple(p) in e2 for p in pair_index], dtype=np.uint8),
            np.array([tuple(t) in e3 for t in tri_index], dtype=np.uint8))


def label_matrices(y_pair, y_tri, pair_index, tri_index, n):
    a2 = np.zeros((n, n), dtype=np.uint8)
    a3 = np.zeros_like(a2)
    e = pair_index[np.asarray(y_pair, dtype=bool)]
    a2[e[:, 0], e[:, 1]] = a2[e[:, 1], e[:, 0]] = 1
    for i, j, k in tri_index[np.asarray(y_tri, dtype=bool)]:
        for u, v in ((i, j), (i, k), (j, k)):
            a3[u, v] = a3[v, u] = 1
    return a2, a3
