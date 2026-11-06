"""Compare saved attention heads to A2; export true triadic incidence H3."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.kuramoto import candidates, pair_adjacency, triad_incidence
from src.utils import fresh_directory, run_id, sha256_file, write_json

MATRIX_KEYS = ("attention_logits", "attention_weights", "symmetric_logits", "symmetric_attention")


def load_samples(metadata_path, features_path, data_path=None, sample=None):
    """Align saved features with per-trajectory labels before creating output."""
    metadata_path, features_path = Path(metadata_path).resolve(), Path(features_path).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    root = Path(data_path).resolve() if data_path else Path(metadata["data"])
    if not root.is_dir() and data_path is None:
        root = ROOT / "data" / root.name
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}; use --data <actual data directory>")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(manifest_path) != metadata["manifest_sha256"]:
        raise ValueError("Data manifest differs from the attention export source")
    if manifest["config"]["data"]["n_nodes"] != 30:
        raise ValueError("Expected 30 nodes")
    candidate_path = root / "candidates.npz"
    if sha256_file(candidate_path) != manifest["candidate_sha256"]:
        raise ValueError("Candidate index file hash differs from the data manifest")
    with np.load(candidate_path, allow_pickle=False) as f:
        pairs, tris = f["pair_index"], f["tri_index"]
    expected = candidates(30)
    if not np.array_equal(pairs, expected[0]) or not np.array_equal(tris, expected[1]):
        raise ValueError("Candidate indices must be the common lexicographic pair/tri tables")
    with np.load(features_path, allow_pickle=False) as f:
        saved = {key: f[key] for key in (*MATRIX_KEYS, "trajectory_id", "topology_id", "permutation", "x_obs")}
    ids = saved["trajectory_id"].tolist()
    metadata_ids = [r["trajectory_id"] for r in metadata["samples"]]
    if len(set(ids)) != len(ids) or len(set(metadata_ids)) != len(metadata_ids) or set(ids) != set(metadata_ids):
        raise ValueError("Metadata and exported trajectory IDs must match uniquely")
    for key in MATRIX_KEYS:
        matrix = saved[key]
        if matrix.shape != (len(ids), 2, 4, 30, 30) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid saved attention matrices: {key}")
    s, a = saved["attention_logits"], saved["attention_weights"]
    for raw, name in ((s, "symmetric_logits"), (a, "symmetric_attention")):
        expected_sym = (raw + raw.swapaxes(-1, -2)) / 2
        expected_sym[..., np.arange(30), np.arange(30)] = 0
        if not np.allclose(saved[name], expected_sym, atol=1e-7, rtol=1e-6):
            raise ValueError(f"Saved {name} is inconsistent with the raw matrices")
    if not np.allclose(a.sum(-1), 1, atol=2e-6) or np.any(np.diagonal(a, axis1=-2, axis2=-1)):
        raise ValueError("Attention must have zero diagonal and row sums of one")
    chosen = metadata["samples"]
    if sample is not None:
        chosen = [r for r in chosen if r["trajectory_id"] == sample]
        if not chosen:
            raise ValueError(f"Unknown --sample {sample}; available: {', '.join(metadata_ids)}")
    if not chosen:
        raise ValueError("No samples in export metadata")
    source_records = {r["trajectory_id"]: r for r in manifest["trajectories"]}
    result, cache = [], {}
    for record in chosen:
        tid = record["trajectory_id"]
        if Path(tid).name != tid or tid in (".", ".."):
            raise ValueError("trajectory_id must be a safe directory name")
        original = source_records.get(tid)
        if original is None or any(record[k] != original[k] for k in ("file", "index", "topology_id")):
            raise ValueError(f"Trajectory source metadata mismatch: {tid}")
        filename = record["file"]
        if filename not in cache:
            path = root / filename
            if sha256_file(path) != metadata["source_data_sha256"][filename]:
                raise ValueError(f"Source trajectory file hash mismatch: {filename}")
            with np.load(path, allow_pickle=False) as f:
                # Evaluation labels enter only this separate offline comparison.
                cache[filename] = {key: f[key] for key in ("y_pair", "y_tri", "permutation", "x_all")}
        truth = cache[filename]
        source_index, export_index = record["index"], ids.index(tid)
        perm = truth["permutation"][source_index]
        if sorted(perm.tolist()) != list(range(30)) or not np.array_equal(perm, saved["permutation"][export_index]):
            raise ValueError(f"Node permutation mismatch: {tid}")
        if not np.array_equal(truth["x_all"][source_index, :, :96], saved["x_obs"][export_index]):
            raise ValueError(f"Historical observations mismatch: {tid}")
        if saved["topology_id"][export_index] != record["topology_id"]:
            raise ValueError(f"Topology ID mismatch: {tid}")
        a2 = pair_adjacency(truth["y_pair"][source_index], pairs)
        h3, triads = triad_incidence(truth["y_tri"][source_index], tris)
        result.append(dict(**record, export_index=export_index, A2=a2, H3=h3, triads=triads,
                           permutation=perm, **{key: saved[key][export_index] for key in MATRIX_KEYS}))
    return metadata, root.resolve(), result


def _draw_pair_row(axes, entry, layer, head, logit_limit, attention_limit, colorbars=True):
    matrices = (entry["A2"], entry["symmetric_logits"][layer, head], entry["symmetric_attention"][layer, head])
    titles = ("True pairwise A2", "symmetric_logits", "symmetric_attention")
    for idx, (ax, matrix, title) in enumerate(zip(axes, matrices, titles)):
        options = (dict(cmap="Greys", vmin=0, vmax=1),
                   dict(cmap="coolwarm", vmin=-logit_limit, vmax=logit_limit),
                   dict(cmap="viridis", vmin=0, vmax=attention_limit))[idx]
        im = ax.imshow(matrix, interpolation="nearest", **options)
        ax.set_title(f"L{layer+1} H{head+1} | {title}", fontsize=9)
        ax.set(xlabel="node j (export order)", ylabel="node i (export order)")
        if colorbars:
            ax.figure.colorbar(im, ax=ax, shrink=.7)


def plot_pairwise(entry, out, identity, stage):
    logit_limit = max(float(np.abs(entry["symmetric_logits"]).max()), 1e-12)
    attention_limit = max(float(entry["symmetric_attention"].max()), 1e-12)
    caption = f"{entry['trajectory_id']} | {stage} | pairwise comparison only\n{identity}; association features, not inferred physical edges"
    for layer in range(2):
        for head in range(4):
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
            _draw_pair_row(axes, entry, layer, head, logit_limit, attention_limit)
            fig.suptitle(caption, fontsize=8)
            fig.savefig(out / f"compare_layer{layer+1}_head{head+1}.png", dpi=120)
            plt.close(fig)
    fig, axes = plt.subplots(8, 3, figsize=(11, 25), layout="constrained")
    for layer in range(2):
        for head in range(4):
            _draw_pair_row(axes[layer*4+head], entry, layer, head, logit_limit, attention_limit)
    fig.suptitle(caption + "\nAll eight heads; each score type uses a shared scale within this sample", fontsize=9)
    fig.savefig(out / "summary_pairwise.png", dpi=110)
    plt.close(fig)


def plot_incidence(h3, out, trajectory_id, identity):
    fig, ax = plt.subplots(figsize=(max(6, min(12, h3.shape[1] * .25)), 7), layout="constrained")
    if h3.shape[1]:
        im = ax.imshow(h3, cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        fig.colorbar(im, ax=ax, ticks=[0, 1], shrink=.7, label="node belongs to hyperedge")
        if h3.shape[1] <= 30:
            ax.set_xticks(range(h3.shape[1]))
    else:
        ax.text(.5, .5, "No true triadic hyperedges\nH3 has shape [30, 0]", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set(xlabel="hyperedge column m (same order as triads_true.json)", ylabel="node i (export order)")
    ax.set_title(f"True triadic node-hyperedge incidence H3 {list(h3.shape)}\n{trajectory_id}\n{identity}", fontsize=9)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def compare(metadata_path, features_path=None, data_path=None, out=None, sample=None):
    metadata_path = Path(metadata_path).resolve()
    features_path = Path(features_path).resolve() if features_path else metadata_path.parent / "attention_features.npz"
    metadata, data_root, entries = load_samples(metadata_path, features_path, data_path, sample)
    out = fresh_directory(out or ROOT / "attention_outputs/pair_and_hypergraph_compare")
    identity = run_id(out)
    stage = "forecast-trained" if metadata["checkpoint_epoch"] > 0 else "untrained"
    records = []
    for entry in entries:
        path = fresh_directory(out / entry["trajectory_id"])
        np.save(path / "A2_true.npy", entry["A2"], allow_pickle=False)
        np.save(path / "H3_true.npy", entry["H3"], allow_pickle=False)
        write_json(path / "triads_true.json", entry["triads"].tolist())
        np.savez_compressed(path / "attention_matrices.npz", run_id=identity,
            trajectory_id=entry["trajectory_id"], topology_id=entry["topology_id"],
            permutation=entry["permutation"], **{key: entry[key] for key in MATRIX_KEYS})
        info = dict(run_id=identity, trajectory_id=entry["trajectory_id"], topology_id=entry["topology_id"],
            source_file=entry["file"], source_index=entry["index"], export_index=entry["export_index"],
            permutation=entry["permutation"].tolist(), n_pairs=int(entry["A2"].sum() // 2),
            n_triads=len(entry["triads"]), A2_shape=list(entry["A2"].shape), H3_shape=list(entry["H3"].shape),
            node_order="0-based exported/permuted node IDs; canonical IDs = permutation[exported ID]",
            hyperedge_order="positive y_tri entries in tri_index order; column m equals triads_true.json[m]")
        write_json(path / "sample_metadata.json", info)
        plot_pairwise(entry, path, identity, stage)
        plot_incidence(entry["H3"], path / "hypergraph_incidence.png", entry["trajectory_id"], identity)
        records.append(info)
        print(f"{entry['trajectory_id']}: A2={entry['A2'].shape}, H3={entry['H3'].shape}, pairs={info['n_pairs']}", flush=True)
    write_json(out / "comparison_manifest.json", dict(run_id=identity, source_export_run_id=metadata["run_id"],
        metadata_path=str(metadata_path), metadata_sha256=sha256_file(metadata_path),
        features_path=str(features_path), features_sha256=sha256_file(features_path), data_path=str(data_root),
        script_sha256=sha256_file(__file__), samples=records,
        purpose="TCN + Transformer attention extracts node interaction features; visual analysis of structural information, not final structural inference"))
    command = "python scripts/compare_attention_pair_and_hypergraph.py"
    (out / "README.md").write_text(f"""# 注意力与真实二元结构、三元关联矩阵的可视化对比

run_id：`{identity}`。本次实际处理 {len(entries)} 条已有导出轨迹，没有新增模型或训练。
自注意力利用 TCN + Transformer 从时序数据提取节点交互关联；当前只分析其包含的结构信息，不做最终结构推断结论。

每个轨迹子目录包含：

- A2_true.npy：真实 y_pair + pair_index 构造的二元邻接 [30,30]，对称且主对角为0。
- H3_true.npy：真实 y_tri + tri_index 构造的三元节点-边关联 [30,M]，每列恰有三个1；无超边时为 [30,0]。
- triads_true.json：0起始节点编号的三元组列表，与H3各列一一对应，不是节点对投影。
- attention_matrices.npz：原样保留四种矩阵 attention_logits、attention_weights、symmetric_logits、symmetric_attention，均为 [2,4,30,30]；另有节点置换和样本身份。
- compare_layerL_headH.png：左 A2，中 symmetric_logits，右 symmetric_attention；共8张，不阈值化、不排序挑头。
- summary_pairwise.png：8行×3列汇总所有头，同一样本同类型分数共用色标；不同类型矩阵使用各自的数值范围。
- hypergraph_incidence.png：H3热图；M=0时明确显示空三元超边集，不补造列。
- sample_metadata.json：轨迹来源、源索引、导出索引、节点置换、二元边数与三元超边数。

所有节点编号均沿用原导出的置换顺序。运行前核对manifest、候选表、源NPZ哈希、轨迹ID、节点置换和历史输入；未改动原始特征、checkpoint或训练代码。不输出三元节点对投影，也不进行预测基线、结构评分或消融。

默认运行 `{command}` 处理源 metadata.json 中全部样本；单样本用 `--sample trajectory_id --out attention_outputs/new_single_compare`。输出必须为新目录，拒绝覆盖已有结果。
""", encoding="utf-8")
    print("A2 是真实二元结构矩阵。", flush=True)
    print("H3 是真实三元超边节点-边关联矩阵。", flush=True)
    print("注意力矩阵用于从 TCN + Transformer 时序表示提取节点间交互关联特征，不直接等同于最终推断结构。", flush=True)
    print(f"OUTPUT_PATH={out.resolve()} RUN_ID={identity}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=ROOT / "attention_outputs/extract_8/metadata.json", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", default=ROOT / "attention_outputs/pair_and_hypergraph_compare", type=Path)
    parser.add_argument("--sample", help="Process only this trajectory_id from metadata")
    args = parser.parse_args()
    try:
        compare(args.metadata, args.features, args.data, args.out, args.sample)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError, IndexError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
