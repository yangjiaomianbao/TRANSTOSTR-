"""Export TCN features and all eight attention heads without structure evaluation."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model.attention_probe import AttentionProbe
from .utils import (environment, fresh_directory, get_device, read_config, run_id,
                    seed_all, sha256_file, write_json)


FEATURE_KEYS = ("temporal_features", "node_tokens", "context_tokens",
                "attention_logits", "attention_weights")


def sample_limit(value):
    if str(value).lower() == "all":
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("--max-samples must be a positive integer or 'all'") from None
    if limit < 1:
        raise argparse.ArgumentTypeError("--max-samples must be positive or 'all'")
    return limit


def symmetrize(matrix):
    result = (matrix + matrix.swapaxes(-1, -2)) / 2
    idx = np.arange(matrix.shape[-1])
    result[..., idx, idx] = 0
    return result


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(min=float(values.min()), max=float(values.max()), mean=float(values.mean()),
                std=float(values.std()), quantiles_05_50_95=np.quantile(values, [.05, .5, .95]).tolist())


def similarity(vectors, center=False):
    """Per-sample similarities of eight flattened heads; zero norms are undefined."""
    vectors = vectors.astype(np.float64)
    if center:
        vectors = vectors - vectors.mean(-1, keepdims=True)
    norm = np.linalg.norm(vectors, axis=-1)
    denominator = norm[..., :, None] * norm[..., None, :]
    products = vectors @ vectors.swapaxes(-1, -2)
    valid = denominator > 1e-15
    values = np.divide(products, denominator, out=np.full_like(products, np.nan), where=valid)
    counts = valid.sum(0)
    mean = np.divide(np.where(valid, values, 0).sum(0), counts,
                     out=np.full_like(products[0], np.nan), where=counts > 0)
    return dict(mean=np.clip(mean, -1, 1).tolist(), valid_sample_counts=counts.tolist())


def attention_diagnostics(features):
    s, a = features["attention_logits"], features["attention_weights"]
    rs, ra = features["symmetric_logits"], features["symmetric_attention"]
    b, layers, heads, n, _ = a.shape
    offdiag = ~np.eye(n, dtype=bool)
    diagnostics = []
    for layer in range(layers):
        for head in range(heads):
            sl, al = s[:, layer, head], a[:, layer, head]
            rsl, ral = rs[:, layer, head], ra[:, layer, head]
            row_sums = al.sum(-1)
            entropy = -(al * np.log(np.clip(al, np.finfo(al.dtype).tiny, None))).sum(-1)
            diagnostics.append(dict(layer=layer + 1, head=head + 1,
                all_logits_finite=bool(np.isfinite(sl).all()),
                row_sums=dict(min=float(row_sums.min()), max=float(row_sums.max()),
                    max_abs_error_from_one=float(np.abs(row_sums - 1).max()),
                    per_sample_min=row_sums.min(-1).tolist(), per_sample_max=row_sums.max(-1).tolist()),
                diagonal_abs_max=float(np.abs(np.diagonal(al, axis1=-2, axis2=-1)).max()),
                raw_logits_asymmetry_max=float(np.abs(sl - sl.swapaxes(-1, -2)).max()),
                raw_attention_asymmetry_max=float(np.abs(al - al.swapaxes(-1, -2)).max()),
                symmetric_logits_error_max=float(np.abs(rsl - rsl.swapaxes(-1, -2)).max()),
                symmetric_attention_error_max=float(np.abs(ral - ral.swapaxes(-1, -2)).max()),
                symmetric_diagonal_abs_max=float(max(np.abs(np.diagonal(rsl, axis1=-2, axis2=-1)).max(),
                                                     np.abs(np.diagonal(ral, axis1=-2, axis2=-1)).max())),
                off_diagonal_distributions={"logits": distribution(sl[:, offdiag]),
                    "attention": distribution(al[:, offdiag]),
                    "symmetric_logits": distribution(rsl[:, offdiag]),
                    "symmetric_attention": distribution(ral[:, offdiag])},
                entropy_nats=distribution(entropy), entropy_per_sample_mean=entropy.mean(-1).tolist(),
                entropy_uniform_reference=float(np.log(n - 1))))
    similarities = {}
    for name, raw in (("logits", s), ("attention", a)):
        vectors = raw.reshape(b, layers * heads, n, n)[..., offdiag]
        similarities[name] = dict(cosine=similarity(vectors), pearson=similarity(vectors, center=True))
    return dict(n_samples=b, per_head=diagnostics,
        head_similarity=dict(head_order=[f"L{l+1}H{h+1}" for l in range(layers) for h in range(heads)],
            definition="Flatten directed off-diagonal entries; compute within each sample then average across samples. Undefined zero-norm comparisons are null. Descriptive only, including between layers.",
            **similarities))


def validate_features(features):
    for key, value in features.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite exported feature: {key}")
    s, a = features["attention_logits"], features["attention_weights"]
    if s.shape[1:] != (2, 4, 30, 30) or s.shape != a.shape:
        raise ValueError(f"Expected [B,2,4,30,30], got {s.shape} and {a.shape}")
    np.testing.assert_allclose(a.sum(-1), 1, atol=2e-6)
    if np.any(np.diagonal(a, axis1=-2, axis2=-1)):
        raise ValueError("Self-attention diagonal must be zero")
    for key in ("symmetric_logits", "symmetric_attention"):
        value = features[key]
        np.testing.assert_array_equal(value, value.swapaxes(-1, -2))
        if np.any(np.diagonal(value, axis1=-2, axis2=-1)):
            raise ValueError(f"Nonzero diagonal: {key}")


def plot_heads(features, out, trajectory_id, identity, stage):
    a = features["attention_weights"][0]
    # One scale for all raw heads, so both layer figures can be compared directly.
    vmax = float(a.max())
    for layer in range(2):
        fig, axes = plt.subplots(1, 4, figsize=(13, 4), layout="constrained")
        for head, ax in enumerate(axes):
            im = ax.imshow(a[layer, head], vmin=0, vmax=vmax, cmap="viridis")
            ax.set(title=f"Layer {layer+1}, head {head+1}", xlabel="source node j", ylabel="query node i")
        fig.colorbar(im, ax=axes.tolist(), shrink=.7, label="A[i,j] (pre-dropout)")
        fig.suptitle(f"{stage} | raw attention | {trajectory_id}\n{identity}", fontsize=9)
        fig.savefig(out / f"heads_layer{layer+1}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), layout="constrained")
    ra = features["symmetric_attention"][0]
    for layer in range(2):
        for head in range(4):
            im = axes[layer, head].imshow(ra[layer, head], vmin=0, vmax=float(ra.max()), cmap="viridis")
            axes[layer, head].set(title=f"Layer {layer+1}, head {head+1}", xlabel="node j", ylabel="node i")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7, label="(A + A.T) / 2, diagonal zero")
    fig.suptitle(f"{stage} | symmetric attention (analysis only) | {trajectory_id}\n{identity}; not physical edge probabilities", fontsize=9)
    fig.savefig(out / "symmetric_heads.png", dpi=150)
    plt.close(fig)


def plot_training_loss(checkpoint, out):
    path = Path(checkpoint).parent / "train_history.csv"
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"epoch", "train_mse", "val_mse"}.issubset(rows[0]):
        return False
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for field in ("train_mse", "val_mse"):
        ax.plot([int(r["epoch"]) for r in rows], [float(r[field]) for r in rows], label=field)
    ax.set(xlabel="epoch", ylabel="forecast MSE (training signal only)",
           title="Existing checkpoint training log; no new training during export")
    ax.legend()
    fig.savefig(out / "training_loss.png", dpi=150)
    plt.close(fig)
    return True


def readme(out, features, metadata, has_curve):
    first = metadata["samples"][0]
    s, a = features["attention_logits"], features["attention_weights"]
    shapes = "\n".join(f"| `{key}` | `{list(value.shape)}` |" for key, value in features.items())
    text = f"""# 多头注意力特征提取：实际导出结果

run_id：`{metadata['run_id']}`；模式：`eval()` + `torch.inference_mode()`。
checkpoint：`{metadata['checkpoint']}`，epoch={metadata['checkpoint_epoch']}。
本次仅导出既有 checkpoint；没有新增训练或结构评估。该 checkpoint 的来源与原训练配置见 metadata.json，不能把历史 smoke 训练当作当前正式结果。

## 固定样例及读取方式

按 manifest 中 split=`{metadata['split']}` 的轨迹顺序选取前 {len(metadata['samples'])} 条；绘图固定为第一条 `{first['trajectory_id']}`（拓扑 `{first['topology_id']}`）。未依据矩阵数值或结构指标选择样例。
通道顺序为 sin(theta)、cos(theta)，历史取 x_all 的前96步。置换约定：new_x = canonical_x[permutation]。

```python
import numpy as np
with np.load({str((out / 'attention_features.npz').resolve())!r}, allow_pickle=False) as f:
    x = f['x_obs'][0]                    # [30,96,2]
    H = f['temporal_features'][0]        # [30,96,64]
    Z = f['node_tokens'][0]              # [30,64], H.mean(axis=1)
    S = f['attention_logits'][0,0,0]     # [30,30], 第一条轨迹/第一层/第一头
    A = f['attention_weights'][0,0,0]    # [30,30]
    print(f['trajectory_id'], f['permutation'])
```

| NPZ key | 实际形状 |
|---|---|
{shapes}

另存 trajectory_id、topology_id、permutation、checkpoint_path、config_json、run_id；均可 allow_pickle=False 读取。完整来源、配置和输入文件哈希见 metadata.json。

## 如何解释一张矩阵

取样例 `[sample=0, layer=0, head=0]`：`S[0,1]={float(s[0,0,0,0,1]):.8f}`，`A[0,1]={float(a[0,0,0,0,1]):.8f}`。
S[i,j] 是该层该头 query_i 与 key_j 的点积除以 sqrt(16)，保留有限的 mask 前原始分数；A[i,j] 是先排除 j=i，再沿源节点 j 做 Softmax 后的相对权重。eval 下通过 sum_j A[i,j] V[j] 聚合，头并行后拼接，完整第一层输出再传入第二层。
`symmetric_logits=(S+S.T)/2`、`symmetric_attention=(A+A.T)/2`，对角设0，仅为分析派生输出，不替代 Transformer 内部注意力。对称后的行和不要求为1。分数和权重由表示、Softmax竞争及训练目标共同决定，未经物理结构标定；不能直接解释为真实边概率或真实结构。

## 图像与描述性诊断

- heads_layer1.png：固定样例第一层四头原始 A；heads_layer2.png：第二层四头原始 A。两张图共用同一数值范围，行是查询节点 i，列是源节点 j。
- symmetric_heads.png：同一样例全部8头的对称 attention，统一色条；不是实际聚合矩阵。
- attention_diagnostics.json：各头行和/对角、原始非对称性、派生矩阵对称性误差、非对角分布、逐节点注意力熵（自然对数，均匀参照 log(29)）、8头间 cosine/Pearson 相似度。相似度只描述当前样本的表示，零方差/零范数为 null。没有计算结构恢复指标。
- training_loss.png：{'来自原 checkpoint 同目录的历史主模型训练日志，只用于查看优化过程；本次没有重新训练。' if has_curve else '未生成：checkpoint 同目录没有可用的训练日志，不能凭空补画。'}

## 对应代码

- src/model/tcn.py：各节点共享、仅沿时间卷积，4个单卷积残差块，感受野31。
- src/model/attention_probe.py：保留 H，沿时间池化得到 Z，再进入两层串行 Transformer。
- src/model/spatial_attention.py：每层4头并行；导出未 mask S、mask 后 dropout 前 A。
- src/export_attention.py：仅读取历史与置换，导出特征、画图和计算描述性诊断。
- src/train_attention.py：可选的主模型单次训练入口，预测损失只作为注意力训练信号。

复现当前导出的命令见 metadata.json / environment.json。本目录是实际特征提取结果，不是新的正式结构实验结论。
"""
    (out / "README_attention_results.md").write_text(text, encoding="utf-8")


def export_attention(config_path, data, checkpoint, out, device="cpu", split="test", max_samples=8, batch_size=8):
    config = read_config(config_path)
    root, checkpoint = Path(data), Path(checkpoint)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}. Generate data first:\n"
            f"python -m src.data.generate --config {config_path} --out {root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint}. Train the main model first:\n"
            f"python -m src.train_attention --config {config_path} --data {root} --out runs/attention_train_new --device {device}")
    if batch_size < 1 or (max_samples is not None and max_samples < 1):
        raise ValueError("batch_size/max_samples must be positive; use 'all' for all samples")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("data",):
        if config[key] != manifest["config"][key]:
            raise ValueError("Configuration differs from the generated data")
    device = get_device(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("node_only", False):
        raise ValueError("Checkpoint must contain the spatial attention model")
    for key in ("data", "model"):
        if ckpt["config"][key] != config[key]:
            raise ValueError(f"Checkpoint/config mismatch: {key}")
    if ckpt.get("data_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Checkpoint and data manifest hashes differ")
    if (config["data"]["n_nodes"], config["data"]["observed"], config["model"]["d_model"],
            config["model"]["n_layers"], config["model"]["n_heads"]) != (30, 96, 64, 2, 4):
        raise ValueError("This export requires N=30, history=96, d_model=64, layers=2, heads=4")
    selected = [r for r in manifest["trajectories"] if r["split"] == split]
    if max_samples is not None:
        selected = selected[:max_samples]
    if not selected:
        raise ValueError(f"No trajectories in split={split}")
    # Preflight input existence before reserving a fresh output directory.
    for record in selected:
        if not (root / record["file"]).is_file():
            raise FileNotFoundError(f"Missing input {root / record['file']}. Regenerate into a new directory:\n"
                f"python -m src.data.generate --config {config_path} --out data/attention_data_new")
    seed_all(ckpt["seed"], config["training"]["num_threads"])
    model = AttentionProbe(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    out = fresh_directory(out)
    identity = run_id(out)
    parts, permutations, samples, source_hashes = defaultdict(list), [], [], {}
    # Read only observations and their node ordering. No evaluation module is imported.
    with torch.inference_mode():
        for offset in range(0, len(selected), batch_size):
            records = selected[offset:offset + batch_size]
            observations = []
            for record in records:
                path = root / record["file"]
                with np.load(path, allow_pickle=False) as arrays:
                    observations.append(arrays["x_all"][record["index"], :, :96, :])
                    permutations.append(arrays["permutation"][record["index"]])
                samples.append({k: record[k] for k in ("trajectory_id", "topology_id", "file", "index")})
                if record["file"] not in source_hashes:
                    source_hashes[record["file"]] = sha256_file(path)
            x = np.stack(observations)
            if x.shape[1:] != (30, 96, 2) or not np.isfinite(x).all():
                raise ValueError(f"Invalid historical observations: {x.shape}")
            result = model(torch.from_numpy(x).to(device))
            parts["x_obs"].append(x)
            for key in FEATURE_KEYS:
                parts[key].append(result[key].cpu().numpy())
    features = {key: np.concatenate(values) for key, values in parts.items()}
    features["symmetric_logits"] = symmetrize(features["attention_logits"])
    features["symmetric_attention"] = symmetrize(features["attention_weights"])
    validate_features(features)
    metadata = dict(run_id=identity, training_run_id=ckpt.get("run_id"), checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha256_file(checkpoint), checkpoint_epoch=ckpt["epoch"], model_seed=ckpt["seed"],
        data=str(root.resolve()), data_run_id=manifest["run_id"], manifest_sha256=sha256_file(manifest_path),
        split=split, selection="manifest trajectory order, first max_samples or all; plots use sample 0",
        config=config, checkpoint_training_config=ckpt["config"], source_data_sha256=source_hashes, samples=samples,
        interpretation="Node-relation features only; neither physical edge probabilities nor reconstructed structure",
        new_training_during_export=False, mode="eval + inference_mode", batch_size=batch_size,
        shapes={key: list(value.shape) for key, value in features.items()})
    source_root = Path(__file__).resolve().parents[1]
    metadata["source_code_sha256"] = {str(path.relative_to(source_root)): sha256_file(path) for path in
        [Path(__file__), source_root / "src/model/tcn.py", source_root / "src/model/spatial_attention.py",
         source_root / "src/model/attention_probe.py", source_root / "src/utils.py"]}
    np.savez_compressed(out / "attention_features.npz", **features,
        trajectory_id=np.array([r["trajectory_id"] for r in samples]),
        topology_id=np.array([r["topology_id"] for r in samples]), permutation=np.stack(permutations),
        checkpoint_path=str(checkpoint.resolve()), config_json=json.dumps(config, ensure_ascii=False), run_id=identity)
    write_json(out / "metadata.json", metadata)
    write_json(out / "environment.json", dict(run_id=identity, **environment()))
    write_json(out / "attention_diagnostics.json", dict(run_id=identity, **attention_diagnostics(features)))
    stage = "forecast-trained" if ckpt["epoch"] > 0 else "untrained"
    plot_heads(features, out, samples[0]["trajectory_id"], identity, stage)
    has_curve = plot_training_loss(checkpoint, out) if ckpt["epoch"] > 0 else False
    readme(out, features, metadata, has_curve)
    print(f"EXPORTED_SAMPLES={len(samples)} OUTPUT_PATH={out.resolve()} RUN_ID={identity}", flush=True)
    for key, value in features.items():
        print(f"{key}: {list(value.shape)}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "checkpoint", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-samples", type=sample_limit, default=8, help="positive integer or all")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    try:
        export_attention(args.config, args.data, args.checkpoint, args.out, args.device,
                         args.split, args.max_samples, args.batch_size)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
