"""Read-only audit of a generated dataset; labels never enter training."""
import argparse
import json
from pathlib import Path

import numpy as np

from .kuramoto import candidates, label_matrices, labels_for_permutation
from ..utils import sha256_file


def validate(root):
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    cfg = manifest["config"]["data"]
    n, observed, future = cfg["n_nodes"], cfg["observed"], cfg["future"]
    pairs, tris = candidates(n)
    assert sha256_file(root/"candidates.npz") == manifest["candidate_sha256"]
    with np.load(root/"candidates.npz") as c:
        np.testing.assert_array_equal(c["pair_index"], pairs)
        np.testing.assert_array_equal(c["tri_index"], tris)
    families, counts = {}, {}
    max_circle_error = 0.
    for topo in manifest["topologies"]:
        assert sha256_file(root/topo["file"]) == topo["sha256"]
        d = dict(np.load(root/topo["file"]))
        assert d["x_all"].shape == (cfg["trajectories"], n, observed+future, 2)
        assert np.isfinite(d["x_all"]).all()
        error = np.abs((d["x_all"]**2).sum(-1)-1).max()
        max_circle_error = max(max_circle_error,float(error))
        assert error < 2e-7
        assert np.all(np.diff(d["times"])>0)
        assert d["times"][observed] > d["times"][observed-1]
        for key, width in (("canonical_e2",2),("canonical_e3",3)):
            assert all(len(set(e))==width for e in d[key])
            assert len(set(map(tuple,d[key]))) == len(d[key])
        for i in range(cfg["trajectories"]):
            perm = d["permutation"][i]
            assert sorted(perm.tolist()) == list(range(n))
            np.testing.assert_array_equal(d["omega"][i],d["canonical_omega"][perm])
            np.testing.assert_allclose(d["x_all"][i,:,0],
                np.stack((np.sin(d["theta0"][i]),np.cos(d["theta0"][i])),axis=-1),atol=4e-8)
            yp, yt = labels_for_permutation(d["canonical_e2"],d["canonical_e3"],perm,pairs,tris)
            np.testing.assert_array_equal(yp,d["y_pair"][i]); np.testing.assert_array_equal(yt,d["y_tri"][i])
            a2,a3=label_matrices(yp,yt,pairs,tris,n)
            for matrix in (a2,a3):
                np.testing.assert_array_equal(matrix,matrix.T)
                assert not matrix.diagonal().any()
        families.setdefault(topo["family_id"],[]).append((topo,d))
        counts[topo["split"]] = counts.get(topo["split"],0)+cfg["trajectories"]
    for family, entries in families.items():
        assert len(entries)==3 and len({t["split"] for t,_ in entries})==1
        by_variant={t["variant"]:d for t,d in entries}
        reference=by_variant["pair_only"]
        assert not len(reference["canonical_e3"])
        assert not np.array_equal(by_variant["mixed_a"]["canonical_e3"],by_variant["mixed_b"]["canonical_e3"])
        for d in by_variant.values():
            for key in ("canonical_e2","canonical_omega","theta0","omega","permutation","seed"):
                np.testing.assert_array_equal(d[key],reference[key])
    return dict(run_id=manifest["run_id"],status="passed",families=len(families),
                topologies=len(manifest["topologies"]), trajectories=counts,
                max_unit_circle_error=max_circle_error)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    args=parser.parse_args()
    print(json.dumps(validate(args.data),indent=2))
