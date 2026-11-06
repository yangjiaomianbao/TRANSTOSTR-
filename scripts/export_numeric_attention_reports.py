"""Export numeric matrices and descriptive edge/triad/node comparisons only."""
import argparse
import hashlib
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
from src.utils import fresh_directory, run_id, sha256_file, write_json

SCORE_KEYS = ("symmetric_logits", "symmetric_attention")
RULES = """节点编号沿用输入导出的0起始节点顺序；所有统计排除主对角。
节点显著性是整行/整列普遍偏大，不是统计检验的显著性。
R为对称矩阵，所以同一节点的行均值=列均值，不是两份独立证据。
广泛偏高节点（预先固定的描述规则）：行均值>=全局离对角均值+各节点行均值的1个总体标准差，且该行至少75%的非对角值严格大于全局非对角中位数。常数矩阵不标记节点。
节点加性分解（纯代数描述，不训练模型）：mu=无向非对角均值；a_i=(N-1)/(N-2)*(row_mean_i-mu)；node_component_ij=mu+a_i+a_j；residual_ij=R_ij-node_component_ij。报告其解释的离对角变异比例，不代表结构恢复能力。
三元表“高”固定定义为R(i,j)严格大于该矩阵全部无向非对角元素的75%分位数，不使用真值选择阈值；只表示相对较高，不是边判定。
三对都高：all_three_pairs_high；仅一对高：one_pair_high，明确高的一对及第三个节点；恰两对高：two_pairs_high，单独列出共享中心节点，不能误写成三对都高；没有高对：no_pairs_high。
若原始A行和为1且对角为0，则对称R_A的row_mean_i=(1+sum_j A[j,i])/(2*(N-1))。因此对称注意力的高行均值也可能只是某节点被普遍关注，不能据此证明具体节点对关联。
所有CSV保留9位有效数字；数值图仅用于阅读，2/3位小数可能掩盖微小差异。统计、分组和判断使用未舍入原值。未计算AP、未训练、未做消融，不给出最终结构推断结论。
"""


def read_sample(folder):
    folder = Path(folder)
    a2 = np.load(folder / "A2_true.npy", allow_pickle=False)
    h3 = np.load(folder / "H3_true.npy", allow_pickle=False)
    triads = np.asarray(json.loads((folder / "triads_true.json").read_text(encoding="utf-8")), dtype=np.int64)
    if triads.size == 0:
        triads = triads.reshape(0, 3)
    if a2.shape != (30, 30) or not np.isin(a2, [0, 1]).all() or not np.array_equal(a2, a2.T) or a2.diagonal().any():
        raise ValueError(f"Invalid binary, symmetric, zero-diagonal A2: {folder.name}")
    if triads.ndim != 2 or triads.shape[1] != 3 or (triads < 0).any() or (triads >= 30).any():
        raise ValueError(f"Invalid triads: {folder.name}")
    if (np.diff(triads, axis=1) <= 0).any() or len(np.unique(triads, axis=0)) != len(triads):
        raise ValueError("Triads must be unique ascending triples")
    expected = np.zeros((30, len(triads)), dtype=np.uint8)
    expected[triads.ravel(), np.repeat(np.arange(len(triads)), 3)] = 1
    if not np.array_equal(h3, expected):
        raise ValueError(f"H3 columns and triads_true.json disagree: {folder.name}")
    with np.load(folder / "attention_matrices.npz", allow_pickle=False) as f:
        matrices = {key: f[key] for key in SCORE_KEYS}
        if "trajectory_id" in f and str(f["trajectory_id"]) != folder.name:
            raise ValueError("NPZ trajectory_id differs from its sample directory")
    for key, r in matrices.items():
        if r.shape != (2, 4, 30, 30) or not np.isfinite(r).all():
            raise ValueError(f"Invalid {key} shape/values: {folder.name}")
        if not np.allclose(r, r.swapaxes(-1, -2), atol=1e-7, rtol=1e-6) or np.diagonal(r, axis1=-2, axis2=-1).any():
            raise ValueError(f"Expected symmetric, zero-diagonal {key}: {folder.name}")
    return dict(trajectory_id=folder.name, A2=a2, H3=h3, triads=triads, **matrices)


def node_description(r):
    """Separate row/column level variation from residual pair variation algebraically."""
    r = np.asarray(r, dtype=np.float64)
    n = len(r)
    upper = np.triu_indices(n, 1)
    values = r[upper]
    mu, median = float(values.mean()), float(np.median(values))
    row_mean = r.sum(1) / (n - 1)
    col_mean = r.sum(0) / (n - 1)
    row_std = float(row_mean.std())
    tol = 1e-12 * max(1.0, float(np.abs(values).max()))
    z = (row_mean - mu) / row_std if row_std > tol else np.zeros(n)
    mask = ~np.eye(n, dtype=bool)
    coverage = ((r > median) & mask).sum(1) / (n - 1)
    broad = (z >= 1 - 1e-12) & (coverage >= .75) if row_std > tol else np.zeros(n, dtype=bool)
    effect = (n - 1) / (n - 2) * (row_mean - mu)
    component = mu + effect[:, None] + effect[None, :]
    residual = r - component
    np.fill_diagonal(residual, 0)
    total = float(np.square(values - mu).sum())
    fraction = float(np.clip(1 - np.square(residual[upper]).sum() / total, 0, 1)) if total > tol**2 else np.nan
    rows = [dict(node=i, row_mean=float(row_mean[i]), column_mean=float(col_mean[i]),
                 row_mean_z=float(z[i]), row_median=float(np.median(r[i, mask[i]])),
                 fraction_above_global_median=float(coverage[i]), additive_effect=float(effect[i]),
                 broadly_high=bool(broad[i])) for i in range(n)]
    if not np.isfinite(fraction):
        statement = "近似常数矩阵：没有足够变异判断节点效应或节点对特异性。"
    elif fraction >= .5:
        statement = f"节点加性项解释 {fraction:.1%} 的离对角变异，超过一半；当前矩阵主要体现节点水平变化。剩余节点对差异是否对应真实结构不能由此确定。"
    else:
        statement = f"节点加性项解释 {fraction:.1%} 的离对角变异；剩余节点对变化更多，但不能据此断言它对应真实结构。"
    return dict(global_mean=mu, global_median=median, high_pair_q75=float(np.quantile(values, .75)),
                row_mean_std=row_std, broadly_high_nodes=np.flatnonzero(broad).tolist(),
                node_component_variation_fraction=fraction, assessment=statement, nodes=rows), component, residual


def triad_values(r, triads, threshold):
    result = []
    for column, (i, j, k) in enumerate(triads):
        i, j, k = int(i), int(j), int(k)
        pairs = [(i, j), (i, k), (j, k)]
        values = np.array([r[u, v] for u, v in pairs], dtype=np.float64)
        high_pairs = [p for p, value in zip(pairs, values) if value > threshold]
        count = len(high_pairs)
        category = {0: "no_pairs_high", 1: "one_pair_high", 2: "two_pairs_high", 3: "all_three_pairs_high"}[count]
        third = next(x for x in (i, j, k) if x not in high_pairs[0]) if count == 1 else None
        center = next(iter(set(high_pairs[0]) & set(high_pairs[1]))) if count == 2 else None
        result.append(dict(hyperedge_column=column, nodes=[i, j, k], R_ij=float(values[0]),
            R_ik=float(values[1]), R_jk=float(values[2]), min=float(values.min()), mean=float(values.mean()),
            max=float(values.max()), high_pair_count=count, high_pairs=high_pairs, category=category,
            third_node=third, shared_center_node=center))
    return result


def nonedge_sample(a2, trajectory_id, seed, count):
    upper = np.transpose(np.triu_indices(len(a2), 1))
    available = upper[a2[upper[:, 0], upper[:, 1]] == 0]
    stable_id = int(hashlib.sha256(trajectory_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(np.random.SeedSequence([seed, stable_id]))
    selection = rng.choice(len(available), min(count, len(available)), replace=False)
    return available[np.sort(selection)], [seed, stable_id]


def focus_nodes(a2, size=10):
    """Deterministic greedy dense subset using only true A2, never attention values."""
    degree = a2.sum(1)
    selected = [int(np.argmax(degree))]
    while len(selected) < min(size, len(a2)):
        remaining = [i for i in range(len(a2)) if i not in selected]
        chosen = max(remaining, key=lambda i: (int(a2[i, selected].sum()), int(degree[i]), -i))
        selected.append(chosen)
    return sorted(selected)


def draw_numeric(ax, matrix, nodes, title, decimals=3, integer=False, font_size=6):
    """White background, black numbers and grey grid; no colored heatmap."""
    n = len(nodes)
    ax.set_facecolor("white")
    ax.set_xlim(-.5, n - .5)
    ax.set_ylim(n - .5, -.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n), [str(i) for i in nodes], fontsize=8)
    ax.set_yticks(range(n), [str(i) for i in nodes], fontsize=8)
    ax.set_xticks(np.arange(n + 1) - .5, minor=True)
    ax.set_yticks(np.arange(n + 1) - .5, minor=True)
    ax.grid(which="minor", color="0.7", linewidth=.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.xaxis.tick_top()
    ax.set_xlabel("node j (export order)")
    ax.set_ylabel("node i (export order)")
    ax.set_title(title, fontsize=10, pad=25)
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            text = str(int(value)) if integer else f"{value:.{decimals}f}"
            ax.text(j, i, text, color="black", ha="center", va="center", fontsize=font_size,
                    fontfamily="DejaVu Sans Mono")


def numeric_image(matrix, out, title, decimals=3, integer=False):
    fig, ax = plt.subplots(figsize=(16, 16), facecolor="white")
    fig.subplots_adjust(left=.06, right=.98, bottom=.05, top=.92)
    draw_numeric(ax, matrix, list(range(len(matrix))), title, decimals, integer, 8 if integer else 6)
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)


def focus_image(entry, nodes, out, title, decimals):
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), facecolor="white")
    fig.subplots_adjust(left=.035, right=.99, bottom=.1, top=.82, wspace=.22)
    fig.suptitle(title + "\nFixed L1H1; subset selection does not use attention", fontsize=11)
    index = np.ix_(nodes, nodes)
    draw_numeric(axes[0], entry["A2"][index], nodes, "A2 true", integer=True, font_size=10)
    draw_numeric(axes[1], entry["symmetric_logits"][0, 0][index], nodes, "L1H1 symmetric_logits", decimals, font_size=9)
    draw_numeric(axes[2], entry["symmetric_attention"][0, 0][index], nodes, "L1H1 symmetric_attention", decimals, font_size=9)
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)


def fmt(value):
    return f"{value:.9g}" if np.isfinite(value) else "N/A"


def reports(entry, out, identity, seed, nonedges, decimals):
    a2, tid = entry["A2"], entry["trajectory_id"]
    upper = np.transpose(np.triu_indices(30, 1))
    edges = upper[a2[upper[:, 0], upper[:, 1]] == 1]
    negative, sample_seed = nonedge_sample(a2, tid, seed, nonedges)
    dense = focus_nodes(a2)
    pair_lines = [f"节点对数值报告 | {tid} | run_id={identity}", RULES,
        f"真实边数={len(edges)}；随机非边样本={len(negative)}；SeedSequence={sample_seed}。全部头使用同一组非边。", ""]
    tri_lines = [f"真实三元超边数值报告 | {tid} | run_id={identity}", RULES,
        f"真实超边数={len(entry['triads'])}；超边列顺序严格对应H3与输入triads_true.json。", ""]
    if not len(entry["triads"]):
        tri_lines.append("该样本没有真实三元超边，H3为[30,0]；不虚构三元组。")
    np.savetxt(out / "A2_true.csv", a2, delimiter=",", fmt="%d")
    numeric_image(a2, out / "A2_true_numeric.png", f"{tid} | true A2 (0/1)", integer=True)
    summary = dict(run_id=identity, trajectory_id=tid, n_true_pairs=len(edges), n_true_triads=len(entry["triads"]),
                   nonedge_seed=sample_seed, sampled_nonedges=negative.tolist(),
                   edge_focus_nodes=dense, edge_focus_true_edges=int(a2[np.ix_(dense, dense)].sum() // 2), heads=[])
    for layer in range(2):
        for head in range(4):
            for key in SCORE_KEYS:
                r = entry[key][layer, head]
                descriptor, component, residual = node_description(r)
                label = f"L{layer+1}H{head+1} {key}"
                np.savetxt(out / f"layer{layer+1}_head{head+1}_{key}.csv", r, delimiter=",", fmt="%.9g")
                suffix = "logits" if key == "symmetric_logits" else "attention"
                numeric_image(r, out / f"layer{layer+1}_head{head+1}_numeric_{suffix}.png",
                              f"{tid} | {label} | numeric values only", decimals)
                pair_lines += [f"=== {label} ===",
                    f"mu={fmt(descriptor['global_mean'])}；median={fmt(descriptor['global_median'])}；q75={fmt(descriptor['high_pair_q75'])}",
                    f"广泛偏高节点={descriptor['broadly_high_nodes']}。{descriptor['assessment']}",
                    "未标记节点不等于不存在节点效应；此规则只是固定描述筛选。", "",
                    "所有节点的行/列描述（排除主对角）：",
                    "node | row_mean | column_mean | row_mean_z | row_median | fraction_above_global_median | a_i | broadly_high"]
                for row in descriptor["nodes"]:
                    pair_lines.append(f"{row['node']} | " + " | ".join(fmt(row[k]) for k in
                        ("row_mean", "column_mean", "row_mean_z", "row_median", "fraction_above_global_median", "additive_effect"))
                        + f" | {row['broadly_high']}")
                for title, pairs in (("所有真实二元边", edges), ("固定随机抽取的非边", negative)):
                    pair_lines += ["", title, "i | j | R(i,j) | node_component(i,j) | residual(i,j)"]
                    for i, j in pairs:
                        pair_lines.append(f"{i} | {j} | {fmt(r[i,j])} | {fmt(component[i,j])} | {fmt(residual[i,j])}")
                values = triad_values(r, entry["triads"], descriptor["high_pair_q75"])
                tri_lines += ["", f"=== {label} ===", f"相对高值阈值（严格大于）q75={fmt(descriptor['high_pair_q75'])}",
                    "m | (i,j,k) | R(i,j) | R(i,k) | R(j,k) | min | mean | max | category | high_pairs | third_node | shared_center"]
                for row in values:
                    tri_lines.append(f"{row['hyperedge_column']} | {tuple(row['nodes'])} | " +
                        " | ".join(fmt(row[k]) for k in ("R_ij", "R_ik", "R_jk", "min", "mean", "max")) +
                        f" | {row['category']} | {row['high_pairs']} | {row['third_node']} | {row['shared_center_node']}")
                categories = {category: [row["nodes"] for row in values if row["category"] == category] for category in
                              ("all_three_pairs_high", "one_pair_high", "two_pairs_high", "no_pairs_high")}
                tri_lines += ["", "本头汇总（只描述已知真实超边内部数值，不进行预测）："]
                tri_lines += [f"{category}: {nodes}" for category, nodes in categories.items()]
                summary["heads"].append(dict(layer=layer+1, head=head+1, score=key, **descriptor,
                                             triad_values=values, triad_categories=categories))
    pair_lines += ["", f"edge_focus 节点={dense}；子图真实边数={summary['edge_focus_true_edges']}。",
        "子集只按真实A2贪心选择内部边较多的10个节点；放大图固定L1H1，不按注意力挑选。"]
    (out / "pair_value_report.txt").write_text("\n".join(pair_lines) + "\n", encoding="utf-8")
    (out / "triad_value_report.txt").write_text("\n".join(tri_lines) + "\n", encoding="utf-8")
    write_json(out / "numeric_summary.json", summary)
    focus_image(entry, list(range(10)), out / "top_left_10x10_numeric.png", f"{tid} | first 10 exported nodes", decimals)
    focus_image(entry, dense, out / "edge_focus_numeric.png", f"{tid} | true-edge dense subset", decimals)
    return summary


def export_numeric(input_root=ROOT / "attention_outputs/pair_and_hypergraph_compare", out=None,
                   sample=None, seed=42, nonedges=20, decimals=3):
    root = Path(input_root).resolve()
    if seed < 0 or nonedges < 1 or decimals not in (2, 3):
        raise ValueError("seed>=0, nonedges>=1, decimals in {2,3} required")
    manifest_path = root / "comparison_manifest.json"
    if manifest_path.is_file():
        ids = [row["trajectory_id"] for row in json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]]
        folders = [root / tid for tid in ids]
    else:
        folders = sorted(p for p in root.iterdir() if p.is_dir() and (p / "A2_true.npy").is_file())
    if sample is not None:
        folders = [p for p in folders if p.name == sample]
    if not folders:
        raise ValueError(f"No matching input samples (sample={sample}) in {root}")
    if len({p.name for p in folders}) != len(folders) or any(p.resolve().parent != root for p in folders):
        raise ValueError("Input sample names must be unique child directories")
    entries = [read_sample(folder) for folder in folders]
    out = fresh_directory(out or ROOT / "attention_outputs/numeric_matrix_compare")
    identity = run_id(out)
    samples = []
    for folder, entry in zip(folders, entries):
        target = fresh_directory(out / entry["trajectory_id"])
        report = reports(entry, target, identity, seed, nonedges, decimals)
        samples.append(dict(trajectory_id=entry["trajectory_id"], n_true_pairs=report["n_true_pairs"],
            n_true_triads=report["n_true_triads"],
            node_component_fractions=[h["node_component_variation_fraction"] for h in report["heads"]],
            broad_node_counts=[len(h["broadly_high_nodes"]) for h in report["heads"]],
            input_sha256={name: sha256_file(folder / name) for name in
                ("A2_true.npy", "H3_true.npy", "triads_true.json", "attention_matrices.npz")}))
        print(f"Exported {entry['trajectory_id']}: 17 CSV, 19 numeric figures, pair/triad reports", flush=True)
    write_json(out / "numeric_manifest.json", dict(run_id=identity, input_root=str(root), seed=seed,
        nonedges_per_sample=nonedges, decimals_in_figures=decimals, rules=RULES,
        source_sha256=sha256_file(__file__), samples=samples))
    (out / "README.md").write_text(f"""# 注意力与真实结构：逐元素数值对比

run_id：`{identity}`。实际处理 {len(samples)} 条已有样本；没有训练、没有AP或消融。

## 固定规则

{RULES}

## 每条样本的输出

- A2_true.csv：30×30真实二阶矩阵，无表头，行列均按节点0–29编号。
- layerL_headH_symmetric_logits.csv / symmetric_attention.csv：每头两个30×30矩阵，9位有效数字，可直接np.loadtxt(delimiter=',')。
- pair_value_report.txt：各头/分数类型的全部真实边、同一组固定随机非边、全部30个节点的行/列描述、广泛偏高节点及代数分解说明。
- triad_value_report.txt：每头每条真实三元超边的三个两两值及min/mean/max；按预先固定的q75规则汇总三对都高、仅一对高、两对高、无高对。
- numeric_summary.json：相同数值的机器可读版本，包含非边随机种子、节点子集、节点描述和三元组分组。
- A2_true_numeric.png：格子内直接标0/1。
- layerL_headH_numeric_logits.png / numeric_attention.png：全部16张30×30黑白数字图，无彩色底图，保留{decimals}位小数。
- top_left_10x10_numeric.png：固定L1H1，前10节点的A2与两种分数并列。
- edge_focus_numeric.png：仅按真实A2贪心选10个内部边较多的节点，固定L1H1，不按注意力择优。

注意：对称矩阵行/列均值相同。高行均值可以来自节点受普遍关注，而非少数真实边位置独有高值。节点加性分解仅帮助描述该差别，残差也不等同于真实结构。

默认命令：`python scripts/export_numeric_attention_reports.py`。支持 `--sample trajectory_id`、`--input-root`、`--out`、`--seed`、`--nonedges`、`--decimals 2|3`。输出须为新目录，不覆盖现有结果。输入文件均未修改。
""", encoding="utf-8")
    print("节点显著性：某节点整行/整列普遍偏大，而不是仅少数真实边位置偏大；本报告仅描述数值，不做最终结构推断。", flush=True)
    print(f"OUTPUT_PATH={out.resolve()} RUN_ID={identity}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "attention_outputs/pair_and_hypergraph_compare")
    parser.add_argument("--out", type=Path, default=ROOT / "attention_outputs/numeric_matrix_compare")
    parser.add_argument("--sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nonedges", type=int, default=20)
    parser.add_argument("--decimals", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    try:
        export_numeric(args.input_root, args.out, args.sample, args.seed, args.nonedges, args.decimals)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
