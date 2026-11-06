"""Generate a Chinese report strictly from completed run artifacts."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .utils import fresh_directory


def number(x):
    return "N/A" if x is None or not np.isfinite(x) else f"{x:.5f}"


def interval(metric):
    return f"{number(metric['mean'])} [{number(metric['ci95'][0])}, {number(metric['ci95'][1])}]"


def make_report(probe, data, output):
    probe, data, output=Path(probe).resolve(),Path(data).resolve(),Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Report already exists; use a new filename: {output}")
    result=json.loads((probe/"metrics_summary.json").read_text())
    manifest=json.loads((data/"manifest.json").read_text())
    checkpoint=Path(result["checkpoint"])
    train=checkpoint.parent
    env=json.loads((train/"environment.json").read_text())
    cfg=manifest["config"]; dc=cfg["data"]
    forecast=result["forecast"]
    def summary(stage,method,target,variant="all"):
        return next(r for r in result["summary"] if (r["stage"],r["method"],r["target"],r["variant"])==
                    (stage,method,target,variant))["metrics"]
    def link(path):
        return os.path.relpath(path,output.parent)
    lines=["# Kuramoto 多头注意力结构信息探针：实际运行报告", "",
        f"生成时间：{datetime.now(ZoneInfo('Asia/Shanghai')):%Y-%m-%d %H:%M:%S %Z}。",
        f"实验档位：**{cfg['name']}**；训练 run_id：`{result['training_run_id']}`；探针 run_id：`{result['run_id']}`；数据 run_id：`{manifest['run_id']}`。", "",
        "## 结论与实验边界", "",
        "**本次只完成 smoke 数据流验证，不能得出科学结论，也没有统计证据支持从注意力可靠恢复结构。**"
        if cfg["name"]=="smoke" else "结果以独立测试 family 为统计单位；以下判断只适用于本配置和实际执行的种子。",
        "模型只用未来动力学 MSE 训练。结构标签仅用于离线指标与图像；未实现二元/三元预测头、Gumbel、HGNN 或结构监督。三元指标针对节点对投影，不代表恢复三元超边。", "",
        "## 数据、物理与可复现性", "",
        f"N={dc['n_nodes']}，K2={dc['k2']}，K3={dc['k3']}，dt={dc['dt']}；共 {dc['observed']+dc['future']} 个观测，输入前 {dc['observed']} 个，预测后 {dc['future']} 个。",
        f"历史窗口 0–{(dc['observed']-1)*dc['dt']:.2f}，未来窗口 {dc['observed']*dc['dt']:.2f}–{(dc['observed']+dc['future']-1)*dc['dt']:.2f}；保留瞬态。",
        f"float64 RK45，rtol={dc['rtol']}、atol={dc['atol']}，存储 float32；按实际二元度与三元参与度分别归一化，分母最小为1。",
        f"独立 Bernoulli：p2={manifest['probabilities']['p2']:.8f}，p3={manifest['probabilities']['p3']:.8f}；没有固定边数，也未使用 XGI。",
        f"family 数 train/val/test={dc['families']['train']}/{dc['families']['val']}/{dc['families']['test']}，每个变体 {dc['trajectories']} 条轨迹；每个 family 含 pair_only/mixed_a/mixed_b。",
        f"实际测试 {result['n_test_families']} 个 family / {result['n_test_topologies']} 个拓扑 / {result['n_test_trajectories']} 条轨迹。",
        "同 family 共享 E2、omega 和编号初值；不同变体 E3 独立且 mixed_a≠mixed_b。所有节点观测、标签、omega、初值一致置换。family 严格不跨 split。", "",
        "| split / variant | 拓扑数 | 实际 E2 均值 (范围) | 实际 E3 均值 (范围) |", "|---|---:|---:|---:|"]
    for split in ("train","val","test"):
        for variant in ("pair_only","mixed_a","mixed_b"):
            ts=[t for t in manifest["topologies"] if t["split"]==split and t["variant"]==variant]
            e2=[t["n_pairs"] for t in ts]; e3=[t["n_triads"] for t in ts]
            lines.append(f"| {split}/{variant} | {len(ts)} | {np.mean(e2):.2f} ({min(e2)}–{max(e2)}) | {np.mean(e3):.2f} ({min(e3)}–{max(e3)}) |")
    lines += ["",f"数据 seed={dc['seed']}；实际模型 seed={result['model_seed']}。43/44 尚未运行，因此没有模型种子稳定性证据。",
        "仅以验证预测 MSE 选 best.pt，主模型与 node-only 各自选择；保存了同初始化 untrained、best 和 last。",
        "TCN 每 block 单个 causal Conv1d，4 层 dilation=[1,2,4,8]、k=3，感受野31；不做跨时间归一化。Transformer LayerNorm 仅沿每个节点的64维通道；两层串行、每层4头并行。",
        "node-only 使用共享逐节点残差 MLP 替代空间层，保留相近参数量；不访问其他节点。两者使用相同优化、seed 和 MSE 选 checkpoint。", "",
        f"环境：Python `{env['python'].splitlines()[0]}`；解释器 `{env['executable']}`；PyTorch {env['packages']['torch']}；CUDA available={env['cuda_available']}。",
        f"包版本、命令见 [environment.json]({link(train/'environment.json')})；完整源码快照与 SHA256 见 [source_hashes.json]({link(train/'source_hashes.json')})。",
        f"选中 checkpoint epoch={result['checkpoint_epoch']}，SHA256=`{result['checkpoint_sha256']}`。", "",
        "## 预测诊断（不使用结构标签）", "",
        "各拓扑先对多条轨迹求均值，再跨拓扑汇总；方括号为按完整 family bootstrap 的95%区间。smoke 只有4个测试 family，区间只用于验证统计流程，不能用于确认科学假设。", "",
        "| 模型 | 测试 MSE [95% CI] | 圆周绝对角误差/弧度 [95% CI] | 参数量 |", "|---|---:|---:|---:|"]
    if forecast:
        for name,m in forecast["models"].items():
            lines.append(f"| {name} | {interval(m['mse'])} | {interval(m['circular_mae'])} | {forecast['parameter_counts'].get(name,0)} |")
        for d in forecast["differences"]:
            lines.append(f"\n配对差 {d['comparison']}：MSE {interval(d['mse'])}；负值有利于主模型。")
        full=forecast["models"]["forecast-trained"]["mse"]["mean"]
        baseline=min(forecast["models"][k]["mse"]["mean"] for k in ("node-only","persistence"))
        lines += ["", "主模型未同时超过 node-only 和 persistence；当前预测诊断不支持其已学到有用的跨节点结构。"
                  if full>=baseline else "主模型点估计超过两种预测基线，但需结合 family 区间和更大样本，不能据此将注意力视为真实结构。"]
    lines += ["", "## 结构一致性：固定融合与输入关联基线", "",
        "所有分数均不使用标签构造。fused_logit 将每个样本的8个对称 logits 在435个无向节点对上各自 z-score 后等权平均；零方差返回0。fused_attention 对8头对称化 attention 等权平均。不选择测试最优头，不学融合权重。", "",
        "| 状态 / 分数 | 目标 | AP [95% CI] | AUROC [95% CI] | Precision@真实正例数 | 正例密度 |", "|---|---|---:|---:|---:|---:|"]
    for target in TARGET_ORDER:
        for stage,method in [(s,m) for s in ("untrained","forecast-trained") for m in ("fused_logit","fused_attention")]+[("input-baseline",m) for m in ("PLV","abs_pearson_delta_phase","random")]:
            try:
                m=summary(stage,method,target)
            except StopIteration:
                continue
            lines.append(f"| {stage}/{method} | {target} | {interval(m['ap'])} | {interval(m['auroc'])} | {number(m['precision_at_k']['mean'])} | {number(m['density']['mean'])} |")
    lines += ["", "AP/AUPRC 使用 average precision 定义。AUROC 在单类时为 N/A；无正例的 AP 和 Precision@K 为 N/A。pair_only 的 projected-triad 与 triad-only 全部无正例，不进入有效拓扑平均；未将其记为0。表中正例密度也只在 AP 有效的拓扑上平均，避免空三元拓扑稀释密度后夸大 AP。triad-only 仅在 y_pair=0 的节点对上评价三元投影。", "",
        "逐轨迹各层各头、两层均值、两种融合、四类目标和分组结果见 "
        f"[metrics_per_head.csv]({link(probe/'metrics_per_head.csv')})、[metrics_per_topology.csv]({link(probe/'metrics_per_topology.csv')})、[metrics_summary.json]({link(probe/'metrics_summary.json')})。正负分数均值也保存于这些表格。", "",
        "### 各头及各层 AP（不按性能挑头）", "", "| 分数 | pairwise 未训练→训练 | projected-triad 未训练→训练 |", "|---|---:|---:|"]
    methods=[f"L{l}H{h}_{kind}" for l in (1,2) for h in range(1,5) for kind in ("logit_sym","attn_sym")]
    methods += [f"L{l}_mean_{kind}" for l in (1,2) for kind in ("logit_z","attention")]
    for method in methods:
        cells=[]
        for target in ("pairwise","projected-triad"):
            cells.append("→".join(number(summary(stage,method,target)["ap"]["mean"]) for stage in ("untrained","forecast-trained")))
        lines.append(f"| {method} | {cells[0]} | {cells[1]} |")
    lines += ["", "### 按变体分组", "", "| 变体 | 目标 | 训练后 fused_logit AP | 训练后 fused_attention AP | PLV AP |", "|---|---|---:|---:|---:|"]
    for variant in ("pair_only","mixed_a","mixed_b"):
        for target in ("pairwise","projected-triad"):
            vals=[summary(stage,method,target,variant)["ap"]["mean"] for stage,method in
                  (("forecast-trained","fused_logit"),("forecast-trained","fused_attention"),("input-baseline","PLV"))]
            lines.append(f"| {variant} | {target} | {' | '.join(map(number,vals))} |")
    lines += ["", "### 配对 AP 变化", "", "| 目标 | 训练后分数 | 参照 | AP 差 [95% CI] |", "|---|---|---|---:|"]
    for d in result["paired_ap_differences"]:
        if d["target"] in ("pairwise","projected-triad"):
            lines.append(f"| {d['target']} | {d['method']} | {d['reference_stage']}/{d['reference_method']} | {interval(d['ap_difference'])} |")
    lines += ["", "## 同二元骨架、不同三元结构与初值稳定性", "",
        "配对比较使用相同 family、omega 和编号初值；独立初值稳定性比较前将矩阵逆置换回规范节点编号。不会把不同排列导致的矩阵差异当成动力学差异。", "",
        "| 状态 / 分数 | 比较 | 上三角平均绝对差 | Spearman | 三元投影变化率 |", "|---|---|---:|---:|---:|"]
    for row in result["paired_controls"]:
        lines.append(f"| {row['stage']}/{row['method']} | {row['comparison']} | {interval(row['mean_abs_difference'])} | {interval(row['spearman'])} | {number(row['triad_projection_change']['mean'])} |")
    lines += ["", "对 mixed_a/mixed_b，二元骨架变化严格为0。三元投影变化引起的权重变化只能说明模型响应动力学变化，不能说明其识别了三元超边。初值变化也可能产生相近或更大权重变化。没有运行43/44模型种子，不能报告跨模型种子稳定。", "",
        "## Q1–Q5 与下一步判断", "",
        "1. **Q1**：上表列出同初始化训练前后全部头和固定融合的 AP/AUROC。smoke 的训练短、测试 family 少；观察到的升降均不能作为可靠结构恢复证据。",
        "2. **Q2**：头间和层间模式确有数值差异，逐头表与热图可审计。融合为事先固定的等权规则；未使用测试最优头构造泛化结论。是否优于单头需要 pilot 与多种子证据。",
        "3. **Q3**：配对表量化了固定 E2、更改 E3 后的 fused 矩阵 MAD/Spearman，并与三元投影变化并列。权重变化不等于三元识别。",
        "4. **Q4**：PLV、相位增量绝对 Pearson、随机分数均只用历史，配对 AP 差已保存。当前 smoke 不足以证实超过简单关联基线。",
        "5. **Q5**：以参数量相近的 node-only 和 persistence 检查预测是否需要其他节点。即使主模型点估计略好，也须结合区间、训练充分性及多种子，不能直接证明注意力可解释。", "",
        "下一步值得完成预注册 pilot（120/30/30 family、每变体4条轨迹、至多100轮、patience15），再按预测 MSE 选模，固定融合并重复42/43/44。只有在独立新拓扑上稳定超过预测和关联基线，才值得继续研究结构解释。不要调整测试阈值、按AP调参或筛除不利测试样本。", "",
        "## 图像来源与选择规则", "",
        "固定图例：manifest 中第一个测试 family 的 mixed_a、轨迹0，在计算指标前确定。示例 NPZ 另保留同 family 三变体轨迹0。所有测试轨迹的 S/A/对称分数与融合矩阵均按拓扑保存于 matrices/；不是只保存最好样例。", ""]
    captions=[("attention_heads_example.png","forecast-trained / head / attention；两层四头，统一色条。"),
              ("attention_heads_untrained.png","untrained / head / attention；同初始化、同一固定轨迹。"),
              ("fused_vs_ground_truth.png","forecast-trained / fused / logits；对照 pairwise 与 projected-triad 真值，标签仅用于评估。"),
              ("head_ap_comparison.png","untrained 与 forecast-trained / head+fused / logits+attention；pairwise 与 projected-triad AP。"),
              ("score_distributions.png","forecast-trained / fused / logits+attention；固定样例 pairwise 正负分数分布。"),
              ("forecast_learning_curves.png","未来预测训练/验证曲线、persistence 验证线与测试逐预测时刻误差。"),
              ("paired_topology_control.png","forecast-trained / fused / logits+attention；固定二元骨架，三元投影变化并列。")]
    for name,caption in captions:
        if (probe/"figures"/name).exists():
            lines += [f"{caption}", f"![{caption}]({link(probe/'figures'/name)})", ""]
    exe=env["executable"]
    lines += ["## 实际命令、质量门禁与输出", "", "本报告对应的生成、训练、评估命令（各 environment.json 另存实际 argv）：", "", "```bash",
        f"cd {Path(__file__).resolve().parents[1]}",
        f"{exe} -m src.data.generate --config configs/{cfg['name']}.yaml --out {data}",
        f"{exe} -m src.data.validate --data {data}",
        f"{exe} -m src.train --config configs/{cfg['name']}.yaml --data {data} --out {train} --device cpu --seed {result['model_seed']}",
        f"{exe} -m src.probe --config configs/{cfg['name']}.yaml --data {data} --checkpoint {checkpoint} --out {probe} --device cpu",
        f"{exe} -m pytest -q", "```", "",
        f"数据 manifest：[manifest.json]({link(data/'manifest.json')})；训练：[config_resolved.yaml]({link(train/'config_resolved.yaml')})、[train_history.csv]({link(train/'train_history.csv')})、[metrics_forecast.json]({link(train/'metrics_forecast.json')})；探针：[attention_examples.npz]({link(probe/'attention_examples.npz')})。",
        "具体门禁结果以本 run 下 validation.json 与 pytest.log 为准。测试覆盖形状B=1、因果性、物理贡献、无标签接口、节点置换、非零有限梯度、单步更新、注意力归一化、单类指标、随机AP密度、family bootstrap、禁止覆盖。", "",
        "## 失败条件与限制", "",
        "运行机器的 NVIDIA 驱动不可通信且 torch.cuda.is_available() 为 False，因此未执行 GPU pilot。CPU smoke 不替代 pilot。",
        "只运行数据/模型 seed42；3轮训练不能证明收敛，4个测试 family 不能支持科学结论。没有按结构AP选模型、头、融合权重或超参数。",
        "AP只是排序一致性，logits 或 sigmoid(logits) 都不是已校准连接概率；投影无法唯一确定三元超边。预测性能、头间差异和配对矩阵变化均不能单独证明结构恢复。", ""]
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("\n".join(lines),encoding="utf-8")
    print(f"REPORT_PATH={output}",flush=True)


TARGET_ORDER=("pairwise","projected-triad","any-interaction","triad-only-subset")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("probe","data","out"):
        parser.add_argument(f"--{name}",required=True)
    args=parser.parse_args()
    make_report(args.probe,args.data,args.out)
