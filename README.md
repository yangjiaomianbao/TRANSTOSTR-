# Kuramoto TCN–Spatial Transformer 结构信息探针

从零生成混合二元/三元 Kuramoto 动力学，仅用未来动力学预测训练 TCN + Spatial Transformer，随后离线检验节点关联分数的结构一致性。标签不参与训练、选模或融合。三元诊断是节点对投影，并非三元超边恢复。

## 环境与快速执行

Python 3.10+；Linux/WSL2。依赖见 `requirements.txt`。无需下载模型或连接私有服务。建议独立虚拟环境；若依赖已齐全可直接使用。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/run_smoke.sh
```

本机已验证的解释器为 `/home/yg/miniconda3/envs/nd3/bin/python`，可以指定：

```bash
PYTHON=/home/yg/miniconda3/envs/nd3/bin/python bash scripts/run_smoke.sh data/smoke runs/smoke_seed42
```

脚本先测试，然后生成/审计数据，训练主模型与参数量相近的 node-only，对同一初始化的 untrained 与 forecast-trained 运行完整探针，最后测试并生成中文报告。失败立即退出。默认数据和运行目录含唯一时间戳；已有运行/探针/报告目录禁止覆盖。显式指定已有数据时会提示复用，并完整校验哈希、物理和配对关系；配置不匹配会在训练前报错。

## 单独执行

```bash
python -m src.data.generate --config configs/smoke.yaml --out data/smoke
python -m src.data.validate --data data/smoke
python -m src.train --config configs/smoke.yaml --data data/smoke --out runs/smoke_seed42 --device cpu --seed 42
python -m src.probe --config configs/smoke.yaml --data data/smoke --checkpoint runs/smoke_seed42/best.pt --out runs/smoke_seed42/probe --device cpu
python -m pytest -q
python -m src.report --probe runs/smoke_seed42/probe --data data/smoke --out reports/attention_probe_report.md
```

`src.probe` 默认 `--stage both`，从 checkpoint 保存的 `initial_state` 重建同初始化未训练模型，与所选预测 checkpoint 对照。独立导出未训练模型：

```bash
python -m src.probe --config configs/smoke.yaml --data data/smoke --checkpoint runs/smoke_seed42/untrained.pt --stage untrained --out runs/smoke_seed42/probe_untrained
```

可用 GPU 时，在完成 smoke 后执行 pilot；CUDA 不可用会明确失败，不静默回退或假装完成。支持显式选择 CPU 进行 pilot。pilot 最多100轮，验证 MSE patience15；模型种子42/43/44需分别运行，新 run 复用同一份 pilot 数据。

```bash
PYTHON=python bash scripts/run_pilot.sh runs/smoke_seed42 data/pilot runs/pilot_seed42 cuda:0 42
PYTHON=python bash scripts/run_pilot.sh runs/smoke_seed42 data/pilot runs/pilot_seed43 cuda:0 43
PYTHON=python bash scripts/run_pilot.sh runs/smoke_seed42 data/pilot runs/pilot_seed44 cuda:0 44
```

## 数据和模型约定

- N=30，独立 Bernoulli 期望60条二元边、20条三元超边；每条真实超边贡献给其三个目标节点。动力学对二元度和三元参与度分别归一化。float64 RK45 rtol=1e-8/atol=1e-10；128个观测、dt=0.05，前96步历史、后32步目标；保留瞬态。
- 同 family 的 pair_only/mixed_a/mixed_b 共享 E2/omega/编号初值，mixed_a 和 mixed_b 的 E3 保证不同。按 family 拆分；时序、omega、E2/E3标签使用同一节点置换。单个 topology NPZ 包含多条轨迹，各条标签随置换而变化；统一 pair/tri 字典序表在 `candidates.npz`。
- 训练数据集只读取 `x_all` 并返回 `x_obs/y_future`，不打开结构标签数组；训练 batch 的额外字段会被拒绝。模型推理只接受 `x_obs`；不使用节点ID embedding、omega、拓扑身份或未来观测。
- TCN 每节点共享4个单卷积残差 causal block，k=3、dilation=1/2/4/8，感受野31；沿时间平均；两层空间 Transformer 各4头，d=64、FFN128。TCN不做归一化；Transformer LayerNorm 仅沿每节点通道。头并行、层串行。
- 保存有限未mask logits S；attention A 先mask对角再softmax，保存的是 dropout 前概率。内部聚合可使用 attention dropout。预测头共享逐节点 MLP，输出稳定归一化的 sin/cos。node-only 使用两层逐节点共享残差 MLP 替代空间层，使参数量相近。
- `R_attn/R_logit` 为离线对称分数、对角清零。fused_logit 每样本每头仅按上三角自身数值z-score再等权平均；fused_attention 为8头等权平均。无标签融合，无训练权重；不是已校准连接概率。

## 评估与产物

测试指标先按 topology 合并轨迹，再按完整 family 重采样 bootstrap 95% CI。分别报告 pairwise、projected-triad、any-interaction、非二元边上的 triad-only；无正例 AP/P@K/AUROC 为 NaN（JSON null），全正例 AUROC 为 NaN。不根据测试AP挑头或选图。结构基线只有历史PLV、相位增量绝对Pearson、随机对称分数。预测基线为 node-only、persistence。

每次运行保存 resolved config、环境/命令、源文件快照/SHA256、untrained/best/last checkpoint、训练CSV、预测指标与逐时刻误差。探针保存：

- `metrics_per_head.csv`：逐轨迹/逐头/融合/基线/四类目标；包含AP、AUROC、P@K、正负分数均值与无正例状态。
- `metrics_per_topology.csv`、`metrics_summary.json`：拓扑均值、family CI、变体分组、训练前后与基线配对差、预测比较。
- `matrices/<topology>.npz`：**全部**测试轨迹训练前后 S/A、所有对称分数和融合矩阵，附分数名称、节点置换和身份；原始矩阵轴 `[trajectory,layer,head,node,node]`。
- `attention_examples.npz`：首个测试 family 三变体的轨迹0；默认标准key为 trained，另有 untrained 前缀；S/A `[B,2,4,30,30]`、R、fused、历史与标签。
- `paired_controls.csv`：固定E2、同初值不同三元结构的MAD/Spearman，以及独立初值稳定性；跨初值比较先逆置换到规范节点顺序。
- `figures/*.png`：固定首个测试 family 的 mixed_a 轨迹0；不按AP选择。矩阵变化不代表三元识别。
- `reports/*.md`：从真实结果生成的中文报告。初次完整执行的报告见 `reports/attention_probe_report.md`。

smoke 为 train/val/test=12/4/4 family、每变体2轨迹、最多3轮，**仅验证完整流程**。pilot 为120/30/30 family、每变体4轨迹。只有完成实际 pilot 和多种子后才能检验科学假设；预测或结构基线未超过时必须明确报告不支持可靠结构恢复。
