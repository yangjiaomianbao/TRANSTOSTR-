# Codex 任务书：从零实现 Kuramoto 时序的 TCN–Spatial Transformer 多头注意力结构信息探针

> **执行方式**：将本文件作为 Codex 的完整开发指令。当前仓库视为**空仓库/新工程**。不依赖、不假定、不恢复任何旧仓库的代码、缓存、配置、论文草稿或训练检查点。可在当前仓库根目录直接创建项目。
>
> **实验边界**：只做到 **TCN → 时间池化 → 多头 Spatial Self-Attention → 注意力矩阵导出与结构信息评估**。**不要实现二元/三元结构预测头、Gumbel-Softmax、离散结构采样、HGNN、结构 BCE 或结构监督训练**。
>
> **核心问题**：在**不以真实结构标签训练模型**的情况下，仅通过未来动力学预测训练的多头注意力，其单头和融合后的节点对关联分数与真实二元边、三元超边的节点对投影有多大一致性？

## 0. 工作原则和完成标准

- 先检查仓库现状。仓库即使有少量文件，也不要假定存在旧版 `src/`、`data/`、`runs/`。新建独立、清晰的模块；不要覆盖用户现有非本实验文件。
- 不调用外部私有服务、不下载模型、不连接 W&B。仅使用 Python、NumPy、SciPy、PyTorch、PyYAML、scikit-learn、matplotlib、pytest（按需列入 requirements）。优先支持 Linux / Windows WSL2。
- **必须真正编写、运行、验证代码**，先完成 CPU smoke test；有 CUDA 时再执行 pilot。不能只输出一篇设计文档或伪代码。不能将未实际运行的结果写成已完成。
- 以可复现、可审计为首要标准：固定 seed，保存配置、环境、运行命令、数据生成参数、训练曲线、检查点、评估结果和图像。所有输出包含 `run_id`，禁止无提示覆盖。
- **不要用结构标签 `y_pair/y_tri` 训练编码器或融合权重**。它们只能进入隔离的评估模块。验证集可以用于选择预测 MSE 最低的 checkpoint，但不允许根据结构 AP 选择模型、头、融合权重或超参数。
- 所有评估图必须标明是否为：`untrained / forecast-trained`、`head / fused`、`logits / attention`、`pairwise / projected-triad`；不能把关联热力图直接命名为“恢复的真实网络”。

## 1. 研究问题与最终交付

### 1.1 研究问题

- Q1：预测训练前后，各个注意力头的节点对关系分数与真实二元边的 AP/AUROC 是否改善？
- Q2：不同头、不同 Transformer 层捕捉的关系模式是否有差异？多头平均融合是否优于单头？
- Q3：在固定二元骨架而改变三元超边时，注意力矩阵是否发生可检测变化？
- Q4：注意力关联与简单的相位同步基线（PLV）或变化率相关基线相比有无优势？
- Q5：模型是否真的需要跨节点信息才能预测未来？若预测性能主要靠节点自身历史完成，注意力矩阵不具结构可解释性也属合理结果。

### 1.2 结果

- 对每条测试轨迹输出每层每头 **原始 attention logits** `S`、**Softmax attention weights** `A`、**对称化关系分数**、**融合矩阵**，并输出 `N×N` 热力图。
- 使用真实结构**离线评估**，展示每头、每层和融合后的 pairwise AP、AUROC、Precision@真实边数；另做三元超边的节点对投影诊断。
- 输出跨 topology 的统计汇总，避免把同一拓扑的多条轨迹当作独立拓扑样本。
- 写一份中文 `reports/attention_probe_report.md`，明确结论是否有统计证据、哪些结果不支持结构解释、下一步是否值得继续。

## 2. 从零构建数据：混合二元/三元 Kuramoto

### 2.1 动力学

节点数 `N=30`，无向二元边 `E2`、无向三元超边 `E3` 分别随机采样，所有候选内节点唯一、自环不存在。初版使用**彼此独立的 Bernoulli 采样**，不要声称是 XGI 特定生成器。二元、三元关系在动力学上分别作用：

\[
\dot\theta_i(t)=\omega_i
+\frac{K_2}{\max(1,d_i^{(2)})}\sum_{j:\{i,j\}\in E_2}\sin(\theta_j-\theta_i)
+\frac{K_3}{\max(1,d_i^{(3)})}\sum_{\{i,j,k\}\in E_3}\sin(\theta_j+\theta_k-2\theta_i).
\]

对三元超边中的三个目标节点都计算对应的耦合贡献；遍历时注意各节点的其他两个节点组合。`d_i^(2)` 是节点 i 参与的二元边数，`d_i^(3)` 是参与的三元超边数。此归一化是本实验的明确选择，贯穿数据生成与记录。

- `N=30, K2=0.8, K3=0.4, dt=0.05`；自然频率 `omega_i ~ Normal(0,1)`，每个拓扑固定自然频率；每条轨迹随机初始相位 `theta0_i ~ Uniform(-pi,pi)`。
- 每张二元骨架期望约 **60 条二元边**，每张三元网络期望约 **20 条三元超边**：`p2=60/C(30,2)`, `p3=20/C(30,3)`。不要人为固定精确边数，记录每张网络实际边数。
- 使用 `scipy.integrate.solve_ivp`、`method=RK45`、`rtol=1e-8`、`atol=1e-10`；生成 **128 个观测时刻**，输入前 96 步、预测后 32 步；保留瞬态，不基于测试标签筛选数据。
- 每个观测存 `[sin(theta), cos(theta)]`，得到单轨迹 `x_all [30,128,2]`、`x_obs [30,96,2]`、`y_future [30,32,2]`。
- 对速度与耦合计算使用 float64 积分，存储可为 float32；核验 sin²+cos²≈1、数值有限、真实邻接对称无自环、结构正确、预测时刻严格晚于输入时刻。
- **效率**：ODE 导数只遍历真实存在的二元边和三元超边，不在每次 RK45 评估中遍历全部 4060 个候选三元组。

### 2.2 关键控制：相同 pairwise skeleton、不同 triads

每个 `topology_family` 包含三个相互配对的拓扑变体：

1. `pair_only`: 相同的 E2，E3 为空；
2. `mixed_a`: 相同的 E2，独立采样的 E3(a)；
3. `mixed_b`: 相同的 E2，另一组独立采样的 E3(b)，且必须与 E3(a) 不同（若碰巧相同则重采样）。

**同一 family 的三个变体使用相同 `omega` 和相同编号的 `theta0` 种子**，从而构成可解释的配对对照：二元结构不变，三元结构变了。`K3` 在 pair_only 中自然没有作用。三种变体要分别保存标签和轨迹。请记录 `family_id`、`variant`、`topology_id`、`trajectory_id`、`seed`。

- Train/Val/Test 依据 `family_id` 划分，而不是 trajectory：同一家族绝不跨拆分，防止同一二元骨架泄漏。
- 实验时随机重排节点编号，必须**对时序、E2、E3、omega 做一致置换**；建议采用基于 family/trajectory seed 的一致随机置换，训练和测试都要有。不要使用可学习的固定节点 ID Embedding。
- 不要把 `omega`、标签、真实耦合、拓扑 ID 或未来观测作为模型输入；它们仅用于生成数据和评估。

### 2.3 两档数据规模（配置化）

- `smoke`：family 数量 train=12 / val=4 / test=4，每个变体 2 条轨迹，训练最多 3 epochs，仅用于检查完整数据流；此档不得得出科学结论。
- `pilot`：family 数量 train=120 / val=30 / test=30，每个变体 4 条轨迹。优先跑通主配置；必要时扩展多种数据和模型种子。
- 保留 `data/` 结构，NPZ 与 JSON manifest；数据文件必须携带 `y_pair [435]`（固定字典序 i<j）、`y_tri [4060]`（固定字典序 i<j<k）。建立统一的 pair/tri 候选索引表；评估从中重建 `A2_true [30,30]`、`A3_projected [30,30]`。
- **三元投影定义**：`A3_projected[i,j]=1` 当存在至少一个真实三元超边包含 i,j。它不是三元关联张量，更不是等同于真实 pairwise 边；必须单独报告。

## 3. 模型：TCN → Pooling → Spatial Transformer（不要结构预测头）

### 3.1 输入输出

- 输入 `x_obs [B,30,96,2]`。
- 共享 TCN（所有节点共享参数，节点维上独立编码）：输出 `H [B,30,96,64]`。
- Global Average Pooling（沿时间轴）：`Z [B,30,64]`。
- **2 层** Spatial Transformer Encoder，**每层 4 个头**，`d_model=64`, `head_dim=16`, `d_ff=128`, `dropout=0.1`；输出 `Z_ctx [B,30,64]`。
- 同时捕获 `S [B,2,4,30,30]`（Softmax 之前的 scaled dot-product logits）和 `A [B,2,4,30,30]`（Softmax 后 attention）。`S` 要保存 **未掩码的可微分有限分数**，以便做离线对称化；`A` 在对角元素 mask 为 `-inf` 后 Softmax，使自注意力不依赖自身，残差仍保留自身状态。必须正确排除对角线并处理所有维度。
- 仅将 `Z_ctx` 输入**轻量预测头**，预测 `y_future [B,30,32,2]`。可采用逐节点共享 MLP：`Linear(64,128)→GELU→Linear(128,64)` 再 reshape `[32,2]`，最后通过 `v / clamp(norm(v),min=eps)` 限制单位圆（或采用稳定的单位圆损失与数值保护）。预测头必须为 permutation-equivariant，不能含 node-specific weights。
- 输出统一接口：`forecast`, `attention_logits`, `attention_weights`, `node_tokens`, `context_tokens`。模型推理 API 只传 `x_obs`。

### 3.2 TCN 配置

- 4 个残差 causal Conv1d Block，卷积核 `k=3`，dilation `[1,2,4,8]`，每 block **一层** causal Conv1d + GELU + dropout + residual（必要时 1×1 投影到64通道）。每个节点独立进行，共享权重，不能在节点轴卷积。
- 正确实现左侧 causal padding，输出长度保持 96；单层/单卷积 block 的总理论感受野 `1 + (k-1)*(1+2+4+8)=31` 个时间点。若实现两卷积 block 必须在配置和文档中如实更新感受野，不要混淆。
- TCN 不引入结构标签，不在全数据上拟合依赖未来的归一化。可以使用 LayerNorm 或 GroupNorm，但说明归一化范围。

### 3.3 标准多头注意力：各头并行，层间串行

对于第 `l` 层、第 `r` 个头，使用该层输入 `Z^(l)` 分别计算：

\[
Q^{(l,r)}=Z^{(l)}W_Q^{(l,r)},\quad
K^{(l,r)}=Z^{(l)}W_K^{(l,r)},\quad
V^{(l,r)}=Z^{(l)}W_V^{(l,r)}.
\]

\[
S^{(l,r)}=Q^{(l,r)}K^{(l,r)\top}/\sqrt{16},\qquad
A^{(l,r)}=\operatorname{softmax}_{j}(\operatorname{maskdiag}(S^{(l,r)})).
\]

\[
\operatorname{head}^{(l,r)}=A^{(l,r)}V^{(l,r)},\qquad
Z^{(l+1)}=\operatorname{TransformerBlock}_l(Z^{(l)}).
\]

Head 并行计算后 Concat+W_O，再加残差、LayerNorm、FFN；Layer1 的完整输出作为 Layer2 的输入。注意保存每层每头矩阵，而不是误把一头输出作为下一头输入。

### 3.4 注意力矩阵 → 无向节点关联分数（**仅分析，不训练监督**）

为了不改变原始 Transformer 的节点特征聚合机制，内部仍使用标准 Softmax attention。另设离线 `probe` 分支：

**单头两个测量方式**：

1. `attn_sym`: `R_attn^(l,r)=(A^(l,r)+A^(l,r).T)/2`；对角清零。它不一定行和为1，这是正常的。
2. `logit_sym`: `R_logit^(l,r)=(S^(l,r)+S^(l,r).T)/2`；对角忽略。可以再经 sigmoid 获得 `[0,1]` 分数，但**不得称其为已校准的连接概率**；AP/AUROC 不受严格单调 sigmoid 影响。

**融合**：优先保留所有 8 个 `[2层×4头]` 分数；同时生成无标签融合矩阵：

- `fused_logit`：每条样本、每个头的 `R_logit` 在上三角非对角位置上做 z-score（均值、标准差均只用该矩阵数值，不用真实标签；零方差安全处理），再对 8 个头等权平均，恢复对称、对角清零。
- `fused_attention`：对 8 个头的 `R_attn` 等权平均，恢复对称、对角清零。
- 禁止使用真实 `y_pair/y_tri` 学习融合权重。可以额外报告各层4头均值以比较层级，但不要把基于测试 AP 最优挑头的分数当成无监督泛化结果。
- 输出浮点 score 矩阵，不强制 Top-k、阈值二值化，不输出声称为真实边的邻接矩阵。

## 4. 训练方案：仅未来轨迹预测，不使用结构标签

### 4.1 Loss

基础训练损失：

\[
\mathcal L_{\mathrm{forecast}}=\frac{1}{B\cdot N\cdot32\cdot2}
\sum_{b,i,t,c}(\hat Y_{bitc}-Y_{bitc})^2.
\]

也可补充相位角误差，但主损失保持简单、统一。禁止结构 BCE、pair/tri AP 反馈训练、结构 regularizer 和借真值生成 Attention mask。

训练实现应确保训练 batch 只解包 `x_obs` 和 `y_future`，**数据集加载标签可以存在，但训练函数不能接收标签字段**；在训练代码中加入接口/断言防止误用。

### 4.2 必须设置的对照

1. **untrained**：同初始化的随机模型，直接导出注意力/评估（不给结构监督）。
2. **forecast-trained**：经过未来预测训练的完整 TCN + 2 层 4 头 Transformer，评估同一初始 seed 的训练前后差异。
3. **node-only baseline**：同一个 TCN + pooling + **相近规模的逐节点预测 MLP**，不含空间 Transformer，比较测试预测 MSE；不能要求它输出注意力。
4. **persistence baseline**：重复历史最后一步 sin/cos 作为未来32步预测；用于识别平凡预测任务。
5. **关联基线**：随机对称分数、PLV `abs(mean_t(exp(i*(theta_i(t)-theta_j(t)))))`（直接由 sin/cos 复数相位计算）、每节点相位增量/角速度的绝对 Pearson 相关。所有基线只使用输入历史96步，不接触真实标签计算分数；标签仅进入指标函数。

### 4.3 超参数

- `AdamW(lr=1e-3, weight_decay=1e-4)`, `batch_size=32`, `grad_clip=1.0`。
- pilot 最多 `100 epochs`，验证集预测 MSE 早停 patience=15；只按**预测**指标选 checkpoint，记录 `best.pt` 和 `last.pt`。
- smoke 最多3 epochs，只做代码与梯度检查。
- 初始 `seed=42`，pilot 如资源允许再用 `43,44` 两个模型 seed 重复。多种子平均不够时在报告中明确局限。
- 不强制 CUDA；使用 `--device cpu/cuda:0`，自动检查 CUDA 是否可用，禁止设备不可用时静默假装训练完成。
- 训练日志要记录训练/验证 loss、node-only 对照、persistence baseline，最好保存每轮 CSV/JSON。

## 5. 评估：注意力包含多少结构信息？

### 5.1 结构与注意力的对照指标（评价阶段独立执行）

对每条测试轨迹，每个 score 矩阵只取无向上三角 `i<j` 共 `C(30,2)=435` 个元素，严格排除对角线。与如下标签分别比较：

- **pairwise label**：真实 `y_pair`，二元边存在性。
- **projected-triad label**：每对节点是否共同属于至少一条真实三元超边；显式声明这是**三元投影**，并非恢复三元超边。
- **any-interaction projection**：`pairwise OR projected-triad`。
- **triad-only subset**：在 `y_pair=0` 的节点对中评估 projected-triad，降低直接二元边混杂；样本没有正类时返回 NaN 并记录，不能默认 AP=0。

每个目标报告：AP/AUPRC、AUROC（单类标签时 NaN）、`Precision@|E_target|`（按该标签实际正数设置 K）、正例/负例分数分布、可选 Mann–Whitney/AUC。不要把 F1 阈值经过测试集调到最优再汇报为无监督性能。

主结果以**每个 topology 的多轨迹指标均值**为单位，再跨拓扑平均或给 bootstrap 95% CI（**按 family 重采样**，不可逐节点对或逐轨迹当作独立样本）。训练、验证、测试 split 依旧严格隔离。pair_only 上 projected-triad 没有正例，应报告 `N/A`；不要污染平均值。

至少产出：

- 每层每头 `attn_sym` 和 `logit_sym` 指标；
- `fused_attention` 和 `fused_logit` 指标；
- untrained 与 trained 在同一测试集上的对比；
- PLV、abs Pearson Δphase 和随机对照；
- 按 `pair_only/mixed_a/mixed_b` 分组；
- paired-control 对照：同一 family、同编号初始条件下，比较 `mixed_a`/`mixed_b` 的 `fused_*` 矩阵差异（上三角平均绝对差、Spearman）和真实三元投影的变化；再加 seed/独立初值稳定性分析。谨慎表述：权重变化≠识别三元结构。

### 5.2 预测指标（诊断训练信号是否有效）

报告 `forecast-trained`、`node-only` 和 `persistence` 的测试 MSE、相位圆周误差（由 sin/cos 算 wrap angle），以及预测时间步上的误差曲线。若 full model 未超过简单基线，不能声称 Attention 从预测任务中学习到了结构信息。

### 5.3 图形

- `attention_heads_example.png`：选**事先固定**的一条测试轨迹，显示第一层4头、第二层4头的 `A` 或 `R_attn` 热力图，统一合理的色条；
- `fused_vs_ground_truth.png`：并排 `fused_logit`、真实二元邻接矩阵、真实三元投影、`pair OR tri projection`，注明真实标签仅用于可视化/评估；
- `head_ap_comparison.png`：各层各头 + fused + PLV 等，在 pairwise 及 projected-triad（适用时）上的 AP；
- `score_distributions.png`：真实 pair 正负样本的分数分布；
- `forecast_learning_curves.png`：预测训练/验证曲线；
- `paired_topology_control.png`：同 pair skeleton 不同 triad 时注意力矩阵差异，与 triad 投影差异并列展示；
- 清晰写入 `report` 中图的来源和选择规则，不允许挑选结构 AP 最好的测试样例当成典型例子。

## 6. 代码文件结构（可按需微调，但必须完整）

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── smoke.yaml
│   └── pilot.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── kuramoto.py          # RK45、真实二元/三元相互作用
│   │   ├── generate.py          # family、split、manifest、缓存
│   │   └── dataset.py           # 仅输入 x_obs/y_future 的训练接口
│   ├── model/
│   │   ├── __init__.py
│   │   ├── tcn.py
│   │   ├── spatial_attention.py  # 可返回 S 和 A 的 MHA
│   │   └── attention_probe.py   # TCN/pooling/Transformer/forecast
│   ├── train.py
│   ├── probe.py                 # 离线矩阵导出与结构评估
│   ├── metrics.py
│   └── utils.py
├── scripts/
│   ├── run_smoke.sh
│   └── run_pilot.sh
├── tests/
│   ├── test_data.py
│   ├── test_shapes.py
│   ├── test_symmetry.py
│   ├── test_no_label_leakage.py
│   ├── test_permutation.py
│   └── test_gradients.py
├── data/                        # 自动生成，gitignore
├── runs/                        # 自动生成，gitignore
└── reports/
    └── attention_probe_report.md
```

## 7. 命令行约定与实验输出

必须提供可以从仓库根目录执行的命令（参数名称可调整，但文档要同步）：

```bash
python -m src.data.generate --config configs/smoke.yaml --out data/smoke
python -m src.train --config configs/smoke.yaml --data data/smoke --out runs/smoke_seed42 --device cpu --seed 42
python -m src.probe --config configs/smoke.yaml --data data/smoke --checkpoint runs/smoke_seed42/best.pt --out runs/smoke_seed42/probe
python -m pytest -q

# 通过 smoke 后，再进行 pilot
python -m src.data.generate --config configs/pilot.yaml --out data/pilot
python -m src.train --config configs/pilot.yaml --data data/pilot --out runs/pilot_seed42 --device cuda:0 --seed 42
python -m src.probe --config configs/pilot.yaml --data data/pilot --checkpoint runs/pilot_seed42/best.pt --out runs/pilot_seed42/probe
```

允许根据生成器的数据哈希采用 `data/pilot/<hash>/` 的目录形式，前提是脚本在命令行中清晰输出真实路径并让调用者不必猜测。要实现从 `untrained`/`best` 导出注意力的命令。`run_smoke.sh` 自动生成、训练、探针、测试，失败即停止；pilot 不能在 smoke 失败时声称结果有效。

保存：

- `runs/<run>/config_resolved.yaml`, `environment.json`, `train_history.csv`, `best.pt`, `last.pt`, `metrics_forecast.json`；
- `runs/<run>/probe/attention_examples.npz`（至少包含 `attention_logits [B,2,4,30,30]`, `attention_weights`, `fused_logit`, `fused_attention`, `x_obs`, `y_pair`, `y_tri`, `family_id`, `variant`；只保存少量固定 example，避免文件爆炸）；
- `runs/<run>/probe/metrics_per_head.csv`, `metrics_per_topology.csv`, `metrics_summary.json`, `baselines.json`；
- `runs/<run>/probe/figures/*.png`；
- `reports/attention_probe_report.md`：报告真实运行的命令、样本规模、各类基线、效果、失败条件、科学解释。

## 8. 必须通过的测试与质量门禁

1. **Shape**：`[B,30,96,2]→[B,30,96,64]→[B,30,64]→[B,2,4,30,30]`，预测 `[B,30,32,2]`；含 `B=1` 测试。
2. **因果/不泄漏**：输入模型只有历史96步；后32步仅进入 forecast loss，不能在时间池化前与历史拼接；训练 API 不传 `y_pair/y_tri`。
3. **图结构**：`A2_true` 对称无自环；三元超边三节点唯一；paired variants 的 E2 相同、E3 不同；family split 完全隔离。
4. **注意力**：每层每头 `A` 的非对角各行和≈1、对角为0；`S` 有限；`R_attn`、`R_logit` 与融合矩阵数值有限、对称、对角为0；没有拿对称 logits 再行 Softmax 却声称保持对称的错误。
5. **反向传播**：未来预测损失到 TCN、Q/K/V、FFN 的参数有非零有限梯度（如存在退化样本需记录并处理），单步 optimizer 更新成功。
6. **置换等变性**：同时对节点轴置换后，forecast 应按同样置换；注意力矩阵应满足 `A_perm≈Π A Πᵀ`；测试 `eval()` 模式且 dropout 关闭。
7. **评估真实性**：`y_pair/y_tri` 不进入训练；结构标签不决定 checkpoint、头选择或融合权重；单类 AUROC/AP 合理处理；随机基线 AP 与标签密度合理接近（多次随机种子）。
8. **负面对照**：`untrained` 导出可运行；`node-only` 与 persistence 可运行。若预测 MSE 不优于简单基线或结构 AP 约等于基线，报告明确为“当前实验不支持从注意力可靠恢复结构”，不能调整阈值或过滤样本制造成功。

## 9. Codex 执行顺序与最终回复

请依序完成：

1. 新建项目骨架、配置和生成器；先生成 `smoke` 数据并完成基础物理/分割验证。
2. 实现独立共享 TCN、时间池化、Spatial Transformer、轻量预测头，完成单测。
3. 实现预测训练、untrained 导出、node-only/persistence 对照，运行 smoke 并核验梯度。
4. 实现注意力矩阵保留、对称分数、无标签融合、结构评估及图形。
5. 在可用 GPU 上运行 pilot（若运行预算/环境不足，至少保证端到端 smoke 运行成功，并明确真实阻碍，不能伪造 pilot）。
6. 生成中文报告，说明所用时间窗口、pair/tri 数量、不同 seed、所有主要指标，回答 Q1–Q5。
7. 最终回复用户：实际创建的文件、运行的命令、通过的测试、真实保存的结果路径、最重要的实验发现、失败/限制。不要扩展到结构预测阶段。

**最终判断标准**：我们不是让注意力“预测出正确邻接矩阵”，而是检验“预测训练得到的多头节点关联分数，是否在独立新拓扑上包含超出简单相关基线的真实结构信息”。如果发现信号很弱或只对二元边敏感，仍是有价值的结论。
