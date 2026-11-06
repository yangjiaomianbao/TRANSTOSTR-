import argparse
import hashlib
import json

import numpy as np

from .kuramoto import candidates, integrate, labels_for_permutation
from ..utils import environment, fresh_directory, read_config, run_id, sha256_file, write_json

VARIANTS = ("pair_only", "mixed_a", "mixed_b")


def rng_for(seed, family, purpose, trajectory=0):
    return np.random.default_rng(np.random.SeedSequence([seed, family, purpose, trajectory]))


def generate(config, out):
    out = fresh_directory(out)
    cfg = config["data"]
    n, seed = cfg["n_nodes"], cfg["seed"]
    pairs, tris = candidates(n)
    p2, p3 = cfg["expected_pairs"] / len(pairs), cfg["expected_triads"] / len(tris)
    if not (0 <= p2 <= 1 and 0 < p3 < 1):
        raise ValueError("Invalid Bernoulli probabilities; p3 must allow distinct variants")
    identity = run_id(out)
    np.savez_compressed(out / "candidates.npz", pair_index=pairs, tri_index=tris, run_id=identity)
    manifest = dict(run_id=identity, config=config, probabilities=dict(p2=p2, p3=p3),
                    generator="independent Bernoulli; degree-normalized RK45",
                    config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                    environment=environment(), topologies=[], trajectories=[])
    family = 0
    for split, count in cfg["families"].items():
        for _ in range(count):
            rng = rng_for(seed, family, 0)
            e2 = pairs[rng.random(len(pairs)) < p2]
            e3a = tris[rng.random(len(tris)) < p3]
            e3b = tris[rng.random(len(tris)) < p3]
            while np.array_equal(e3a, e3b):
                e3b = tris[rng.random(len(tris)) < p3]
            omega = rng.normal(0, 1, n)
            for variant, e3 in zip(VARIANTS, (tris[:0], e3a, e3b)):
                tid = f"f{family:04d}_{variant}"
                filename = tid + ".npz"
                records, pair_labels, tri_labels, permutations, initial, omegas, seeds, nfevs = ([] for _ in range(8))
                for trajectory in range(cfg["trajectories"]):
                    init_seed = [seed, family, 1, trajectory]
                    theta0 = rng_for(seed, family, 1, trajectory).uniform(-np.pi, np.pi, n)
                    perm = rng_for(seed, family, 2, trajectory).permutation(n)
                    x, times, nfev = integrate(omega, e2, e3, theta0, cfg)
                    yp, yt = labels_for_permutation(e2, e3, perm, pairs, tris)
                    records.append(x[perm]); pair_labels.append(yp); tri_labels.append(yt)
                    permutations.append(perm); initial.append(theta0[perm]); omegas.append(omega[perm])
                    seeds.append(init_seed); nfevs.append(nfev)
                    manifest["trajectories"].append(dict(file=filename, index=trajectory, split=split,
                        family_id=family, variant=variant, topology_id=tid,
                        trajectory_id=f"{tid}_t{trajectory:03d}", seed=init_seed))
                np.savez_compressed(out / filename, x_all=np.stack(records), y_pair=np.stack(pair_labels),
                    y_tri=np.stack(tri_labels), permutation=np.stack(permutations),
                    omega=np.stack(omegas), theta0=np.stack(initial), times=times,
                    canonical_e2=e2, canonical_e3=e3, canonical_omega=omega,
                    seed=np.array(seeds), nfev=nfevs, run_id=identity)
                manifest["topologies"].append(dict(file=filename, split=split, family_id=family,
                    variant=variant, topology_id=tid, n_pairs=len(e2), n_triads=len(e3),
                    sha256=sha256_file(out / filename)))
            print(f"generated family={family} split={split} E2={len(e2)} E3a={len(e3a)} E3b={len(e3b)}", flush=True)
            family += 1
    manifest["candidate_sha256"] = sha256_file(out / "candidates.npz")
    write_json(out / "manifest.json", manifest)
    print(f"DATA_PATH={out.resolve()} RUN_ID={identity}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    generate(read_config(args.config), args.out)


if __name__ == "__main__":
    main()
