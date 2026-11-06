# Kuramoto 多头注意力结构信息探针：实际运行报告

生成时间：2026-11-05 22:47:07 CST。
实验档位：**smoke**；训练 run_id：`smoke_seed42_final_20261105T144426057520Z`；探针 run_id：`probe_20261105T144434705070Z`；数据 run_id：`smoke_20261105T143228550286Z`。

## 结论与实验边界

**本次只完成 smoke 数据流验证，不能得出科学结论，也没有统计证据支持从注意力可靠恢复结构。**
模型只用未来动力学 MSE 训练。结构标签仅用于离线指标与图像；未实现二元/三元预测头、Gumbel、HGNN 或结构监督。三元指标针对节点对投影，不代表恢复三元超边。

## 数据、物理与可复现性

N=30，K2=0.8，K3=0.4，dt=0.05；共 128 个观测，输入前 96 个，预测后 32 个。
历史窗口 0–4.75，未来窗口 4.80–6.35；保留瞬态。
float64 RK45，rtol=1e-08、atol=1e-10，存储 float32；按实际二元度与三元参与度分别归一化，分母最小为1。
独立 Bernoulli：p2=0.13793103，p3=0.00492611；没有固定边数，也未使用 XGI。
family 数 train/val/test=12/4/4，每个变体 2 条轨迹；每个 family 含 pair_only/mixed_a/mixed_b。
实际测试 4 个 family / 12 个拓扑 / 24 条轨迹。
同 family 共享 E2、omega 和编号初值；不同变体 E3 独立且 mixed_a≠mixed_b。所有节点观测、标签、omega、初值一致置换。family 严格不跨 split。

| split / variant | 拓扑数 | 实际 E2 均值 (范围) | 实际 E3 均值 (范围) |
|---|---:|---:|---:|
| train/pair_only | 12 | 61.00 (51–74) | 0.00 (0–0) |
| train/mixed_a | 12 | 61.00 (51–74) | 20.42 (11–28) |
| train/mixed_b | 12 | 61.00 (51–74) | 21.08 (17–27) |
| val/pair_only | 4 | 63.00 (50–68) | 0.00 (0–0) |
| val/mixed_a | 4 | 63.00 (50–68) | 19.50 (16–23) |
| val/mixed_b | 4 | 63.00 (50–68) | 18.50 (17–20) |
| test/pair_only | 4 | 64.00 (57–73) | 0.00 (0–0) |
| test/mixed_a | 4 | 64.00 (57–73) | 19.75 (17–22) |
| test/mixed_b | 4 | 64.00 (57–73) | 23.00 (20–26) |

数据 seed=42；实际模型 seed=42。43/44 尚未运行，因此没有模型种子稳定性证据。
仅以验证预测 MSE 选 best.pt，主模型与 node-only 各自选择；保存了同初始化 untrained、best 和 last。
TCN 每 block 单个 causal Conv1d，4 层 dilation=[1,2,4,8]、k=3，感受野31；不做跨时间归一化。Transformer LayerNorm 仅沿每个节点的64维通道；两层串行、每层4头并行。
node-only 使用共享逐节点残差 MLP 替代空间层，保留相近参数量；不访问其他节点。两者使用相同优化、seed 和 MSE 选 checkpoint。

环境：Python `3.11.6 | packaged by conda-forge | (main, Oct  3 2023, 10:40:35) [GCC 12.3.0]`；解释器 `/home/yg/miniconda3/envs/nd3/bin/python`；PyTorch 2.5.1；CUDA available=False。
包版本、命令见 [environment.json](../runs/smoke_seed42_final/environment.json)；完整源码快照与 SHA256 见 [source_hashes.json](../runs/smoke_seed42_final/source_hashes.json)。
选中 checkpoint epoch=3，SHA256=`a40cf8e23a062f8dbd7213288e8565f97679cc1e793ba0959938556c2fc6fa85`。

## 预测诊断（不使用结构标签）

各拓扑先对多条轨迹求均值，再跨拓扑汇总；方括号为按完整 family bootstrap 的95%区间。smoke 只有4个测试 family，区间只用于验证统计流程，不能用于确认科学假设。

| 模型 | 测试 MSE [95% CI] | 圆周绝对角误差/弧度 [95% CI] | 参数量 |
|---|---:|---:|---:|
| forecast-trained | 0.83228 [0.79010, 0.86973] | 1.35996 [1.30705, 1.40742] | 121216 |
| node-only | 0.90211 [0.88498, 0.92548] | 1.44642 [1.42412, 1.47707] | 120704 |
| untrained | 0.96334 [0.94283, 0.98650] | 1.52357 [1.49671, 1.55318] | 121216 |
| persistence | 0.29336 [0.22734, 0.36687] | 0.61820 [0.51736, 0.72797] | 0 |

配对差 forecast-trained minus node-only：MSE -0.06984 [-0.11052, -0.03866]；负值有利于主模型。

配对差 forecast-trained minus persistence：MSE 0.53891 [0.48966, 0.60934]；负值有利于主模型。

主模型未同时超过 node-only 和 persistence；当前预测诊断不支持其已学到有用的跨节点结构。

## 结构一致性：固定融合与输入关联基线

所有分数均不使用标签构造。fused_logit 将每个样本的8个对称 logits 在435个无向节点对上各自 z-score 后等权平均；零方差返回0。fused_attention 对8头对称化 attention 等权平均。不选择测试最优头，不学融合权重。

| 状态 / 分数 | 目标 | AP [95% CI] | AUROC [95% CI] | Precision@真实正例数 | 正例密度 |
|---|---|---:|---:|---:|---:|
| untrained/fused_logit | pairwise | 0.16744 [0.14197, 0.19291] | 0.50958 [0.50316, 0.51426] | 0.16896 | 0.14713 |
| untrained/fused_attention | pairwise | 0.16881 [0.14191, 0.19570] | 0.50868 [0.50218, 0.51518] | 0.16713 | 0.14713 |
| forecast-trained/fused_logit | pairwise | 0.16896 [0.14221, 0.19572] | 0.50925 [0.48894, 0.52956] | 0.16236 | 0.14713 |
| forecast-trained/fused_attention | pairwise | 0.17038 [0.14406, 0.19669] | 0.50901 [0.48338, 0.53465] | 0.17480 | 0.14713 |
| input-baseline/PLV | pairwise | 0.17671 [0.14821, 0.21610] | 0.53331 [0.50840, 0.56613] | 0.17015 | 0.14713 |
| input-baseline/abs_pearson_delta_phase | pairwise | 0.17333 [0.15590, 0.18750] | 0.53052 [0.51154, 0.55278] | 0.16413 | 0.14713 |
| input-baseline/random | pairwise | 0.16465 [0.14653, 0.17962] | 0.52201 [0.49012, 0.55370] | 0.15076 | 0.14713 |
| untrained/fused_logit | projected-triad | 0.14438 [0.13508, 0.15968] | 0.50457 [0.47943, 0.52972] | 0.11797 | 0.13649 |
| untrained/fused_attention | projected-triad | 0.14289 [0.13287, 0.16009] | 0.49924 [0.47128, 0.52362] | 0.11820 | 0.13649 |
| forecast-trained/fused_logit | projected-triad | 0.14178 [0.12974, 0.15835] | 0.49898 [0.48503, 0.51293] | 0.11003 | 0.13649 |
| forecast-trained/fused_attention | projected-triad | 0.14078 [0.12804, 0.15787] | 0.49169 [0.47953, 0.50360] | 0.11475 | 0.13649 |
| input-baseline/PLV | projected-triad | 0.13756 [0.12757, 0.14718] | 0.47067 [0.45811, 0.48091] | 0.12394 | 0.13649 |
| input-baseline/abs_pearson_delta_phase | projected-triad | 0.14571 [0.14122, 0.15079] | 0.50397 [0.50125, 0.50737] | 0.13659 | 0.13649 |
| input-baseline/random | projected-triad | 0.15640 [0.14395, 0.16885] | 0.51298 [0.48675, 0.53243] | 0.15761 | 0.13649 |
| untrained/fused_logit | any-interaction | 0.24088 [0.22343, 0.25833] | 0.51100 [0.49902, 0.51974] | 0.23550 | 0.22280 |
| untrained/fused_attention | any-interaction | 0.24157 [0.22222, 0.26055] | 0.50926 [0.49657, 0.51711] | 0.22814 | 0.22280 |
| forecast-trained/fused_logit | any-interaction | 0.24036 [0.22314, 0.26106] | 0.50813 [0.49595, 0.52268] | 0.23435 | 0.22280 |
| forecast-trained/fused_attention | any-interaction | 0.24178 [0.22463, 0.26346] | 0.50692 [0.49003, 0.52381] | 0.24154 | 0.22280 |
| input-baseline/PLV | any-interaction | 0.24441 [0.22468, 0.27931] | 0.51789 [0.50232, 0.54348] | 0.24336 | 0.22280 |
| input-baseline/abs_pearson_delta_phase | any-interaction | 0.24648 [0.22862, 0.25918] | 0.52814 [0.51034, 0.54863] | 0.23969 | 0.22280 |
| input-baseline/random | any-interaction | 0.24365 [0.23143, 0.25587] | 0.51891 [0.49835, 0.53288] | 0.24264 | 0.22280 |
| untrained/fused_logit | triad-only-subset | 0.14430 [0.13692, 0.15527] | 0.51146 [0.49362, 0.52847] | 0.11887 | 0.13295 |
| untrained/fused_attention | triad-only-subset | 0.14451 [0.13409, 0.15719] | 0.50868 [0.48959, 0.52560] | 0.12301 | 0.13295 |
| forecast-trained/fused_logit | triad-only-subset | 0.14166 [0.12946, 0.15567] | 0.50393 [0.49155, 0.51302] | 0.10867 | 0.13295 |
| forecast-trained/fused_attention | triad-only-subset | 0.14204 [0.13024, 0.15665] | 0.50171 [0.49420, 0.50611] | 0.12115 | 0.13295 |
| input-baseline/PLV | triad-only-subset | 0.13722 [0.12692, 0.14767] | 0.47961 [0.46732, 0.48803] | 0.12109 | 0.13295 |
| input-baseline/abs_pearson_delta_phase | triad-only-subset | 0.14839 [0.14315, 0.15362] | 0.50871 [0.50528, 0.51214] | 0.14138 | 0.13295 |
| input-baseline/random | triad-only-subset | 0.15411 [0.14353, 0.16665] | 0.51050 [0.48841, 0.53032] | 0.16056 | 0.13295 |

AP/AUPRC 使用 average precision 定义。AUROC 在单类时为 N/A；无正例的 AP 和 Precision@K 为 N/A。pair_only 的 projected-triad 与 triad-only 全部无正例，不进入有效拓扑平均；未将其记为0。表中正例密度也只在 AP 有效的拓扑上平均，避免空三元拓扑稀释密度后夸大 AP。triad-only 仅在 y_pair=0 的节点对上评价三元投影。

逐轨迹各层各头、两层均值、两种融合、四类目标和分组结果见 [metrics_per_head.csv](../runs/smoke_seed42_final/probe/metrics_per_head.csv)、[metrics_per_topology.csv](../runs/smoke_seed42_final/probe/metrics_per_topology.csv)、[metrics_summary.json](../runs/smoke_seed42_final/probe/metrics_summary.json)。正负分数均值也保存于这些表格。

### 各头及各层 AP（不按性能挑头）

| 分数 | pairwise 未训练→训练 | projected-triad 未训练→训练 |
|---|---:|---:|
| L1H1_logit_sym | 0.17168→0.15664 | 0.14919→0.14876 |
| L1H1_attn_sym | 0.17631→0.15590 | 0.14855→0.14946 |
| L1H2_logit_sym | 0.17089→0.16252 | 0.14782→0.14826 |
| L1H2_attn_sym | 0.16137→0.16113 | 0.14299→0.14556 |
| L1H3_logit_sym | 0.15327→0.17049 | 0.14696→0.14179 |
| L1H3_attn_sym | 0.16230→0.17163 | 0.14608→0.14318 |
| L1H4_logit_sym | 0.15490→0.14059 | 0.14142→0.14748 |
| L1H4_attn_sym | 0.14303→0.13996 | 0.14443→0.14371 |
| L2H1_logit_sym | 0.16296→0.17085 | 0.13789→0.13937 |
| L2H1_attn_sym | 0.16806→0.16862 | 0.14205→0.13738 |
| L2H2_logit_sym | 0.16721→0.16540 | 0.14870→0.14405 |
| L2H2_attn_sym | 0.16642→0.16115 | 0.13932→0.14098 |
| L2H3_logit_sym | 0.17789→0.16950 | 0.17248→0.14211 |
| L2H3_attn_sym | 0.18818→0.17227 | 0.15956→0.14121 |
| L2H4_logit_sym | 0.16757→0.17740 | 0.14233→0.14252 |
| L2H4_attn_sym | 0.17008→0.18019 | 0.14093→0.14409 |
| L1_mean_logit_z | 0.16061→0.15035 | 0.14283→0.14564 |
| L1_mean_attention | 0.16039→0.15178 | 0.14249→0.14004 |
| L2_mean_logit_z | 0.17024→0.17006 | 0.14698→0.14294 |
| L2_mean_attention | 0.17039→0.17138 | 0.14262→0.14174 |

### 按变体分组

| 变体 | 目标 | 训练后 fused_logit AP | 训练后 fused_attention AP | PLV AP |
|---|---|---:|---:|---:|
| pair_only | pairwise | 0.17036 | 0.17334 | 0.17749 |
| pair_only | projected-triad | N/A | N/A | N/A |
| mixed_a | pairwise | 0.16804 | 0.16794 | 0.17933 |
| mixed_a | projected-triad | 0.12749 | 0.12653 | 0.13213 |
| mixed_b | pairwise | 0.16849 | 0.16986 | 0.17331 |
| mixed_b | projected-triad | 0.15607 | 0.15502 | 0.14300 |

### 配对 AP 变化

| 目标 | 训练后分数 | 参照 | AP 差 [95% CI] |
|---|---|---|---:|
| pairwise | fused_logit | untrained/fused_logit | 0.00152 [-0.00345, 0.00705] |
| pairwise | fused_logit | input-baseline/PLV | -0.00775 [-0.02611, 0.00887] |
| pairwise | fused_logit | input-baseline/abs_pearson_delta_phase | -0.00437 [-0.02072, 0.00825] |
| pairwise | fused_logit | input-baseline/random | 0.00432 [-0.01086, 0.01610] |
| pairwise | fused_attention | untrained/fused_attention | 0.00157 [-0.00473, 0.00787] |
| pairwise | fused_attention | input-baseline/PLV | -0.00633 [-0.02268, 0.00782] |
| pairwise | fused_attention | input-baseline/abs_pearson_delta_phase | -0.00295 [-0.01933, 0.00922] |
| pairwise | fused_attention | input-baseline/random | 0.00573 [-0.01037, 0.01817] |
| projected-triad | fused_logit | untrained/fused_logit | -0.00260 [-0.00781, 0.00045] |
| projected-triad | fused_logit | input-baseline/PLV | 0.00422 [-0.00231, 0.01150] |
| projected-triad | fused_logit | input-baseline/abs_pearson_delta_phase | -0.00392 [-0.01206, 0.00756] |
| projected-triad | fused_logit | input-baseline/random | -0.01462 [-0.02667, -0.00257] |
| projected-triad | fused_attention | untrained/fused_attention | -0.00211 [-0.00601, 0.00170] |
| projected-triad | fused_attention | input-baseline/PLV | 0.00322 [-0.00297, 0.01054] |
| projected-triad | fused_attention | input-baseline/abs_pearson_delta_phase | -0.00493 [-0.01375, 0.00708] |
| projected-triad | fused_attention | input-baseline/random | -0.01562 [-0.02732, -0.00392] |

## 同二元骨架、不同三元结构与初值稳定性

配对比较使用相同 family、omega 和编号初值；独立初值稳定性比较前将矩阵逆置换回规范节点编号。不会把不同排列导致的矩阵差异当成动力学差异。

| 状态 / 分数 | 比较 | 上三角平均绝对差 | Spearman | 三元投影变化率 |
|---|---|---:|---:|---:|
| untrained/fused_logit | mixed_a_vs_mixed_b_same_initial_condition | 0.20118 [0.18624, 0.21766] | 0.85674 [0.83810, 0.87539] | 32.19713 |
| untrained/fused_attention | mixed_a_vs_mixed_b_same_initial_condition | 0.00064 [0.00053, 0.00073] | 0.84563 [0.81355, 0.87771] | 32.19713 |
| forecast-trained/fused_logit | mixed_a_vs_mixed_b_same_initial_condition | 0.19681 [0.18073, 0.21119] | 0.82568 [0.80314, 0.85666] | 32.19713 |
| forecast-trained/fused_attention | mixed_a_vs_mixed_b_same_initial_condition | 0.00102 [0.00081, 0.00120] | 0.81015 [0.77821, 0.85373] | 32.19713 |
| untrained/fused_logit | pair_only_independent_initial_conditions | 0.61518 [0.53407, 0.71517] | -0.02570 [-0.17797, 0.12658] | 0.00000 |
| untrained/fused_attention | pair_only_independent_initial_conditions | 0.00198 [0.00158, 0.00236] | -0.03710 [-0.19669, 0.07337] | 0.00000 |
| forecast-trained/fused_logit | pair_only_independent_initial_conditions | 0.54953 [0.48424, 0.61482] | -0.03120 [-0.11181, 0.04941] | 0.00000 |
| forecast-trained/fused_attention | pair_only_independent_initial_conditions | 0.00287 [0.00230, 0.00343] | -0.03045 [-0.04976, -0.01297] | 0.00000 |
| untrained/fused_logit | mixed_a_independent_initial_conditions | 0.62375 [0.52131, 0.73345] | -0.04374 [-0.20666, 0.12130] | 0.00000 |
| untrained/fused_attention | mixed_a_independent_initial_conditions | 0.00207 [0.00166, 0.00249] | -0.06171 [-0.25397, 0.06533] | 0.00000 |
| forecast-trained/fused_logit | mixed_a_independent_initial_conditions | 0.55030 [0.47783, 0.62276] | -0.02804 [-0.09324, 0.03717] | 0.00000 |
| forecast-trained/fused_attention | mixed_a_independent_initial_conditions | 0.00292 [0.00227, 0.00356] | -0.02503 [-0.04647, -0.00488] | 0.00000 |
| untrained/fused_logit | mixed_b_independent_initial_conditions | 0.59974 [0.54413, 0.67259] | 0.02273 [-0.10844, 0.15391] | 0.00000 |
| untrained/fused_attention | mixed_b_independent_initial_conditions | 0.00198 [0.00164, 0.00222] | 0.00114 [-0.10875, 0.10467] | 0.00000 |
| forecast-trained/fused_logit | mixed_b_independent_initial_conditions | 0.53636 [0.47559, 0.59738] | -0.02037 [-0.11214, 0.07141] | 0.00000 |
| forecast-trained/fused_attention | mixed_b_independent_initial_conditions | 0.00276 [0.00230, 0.00318] | -0.03377 [-0.08494, 0.01002] | 0.00000 |

对 mixed_a/mixed_b，二元骨架变化严格为0。三元投影变化引起的权重变化只能说明模型响应动力学变化，不能说明其识别了三元超边。初值变化也可能产生相近或更大权重变化。没有运行43/44模型种子，不能报告跨模型种子稳定。

## Q1–Q5 与下一步判断

1. **Q1**：上表列出同初始化训练前后全部头和固定融合的 AP/AUROC。smoke 的训练短、测试 family 少；观察到的升降均不能作为可靠结构恢复证据。
2. **Q2**：头间和层间模式确有数值差异，逐头表与热图可审计。融合为事先固定的等权规则；未使用测试最优头构造泛化结论。是否优于单头需要 pilot 与多种子证据。
3. **Q3**：配对表量化了固定 E2、更改 E3 后的 fused 矩阵 MAD/Spearman，并与三元投影变化并列。权重变化不等于三元识别。
4. **Q4**：PLV、相位增量绝对 Pearson、随机分数均只用历史，配对 AP 差已保存。当前 smoke 不足以证实超过简单关联基线。
5. **Q5**：以参数量相近的 node-only 和 persistence 检查预测是否需要其他节点。即使主模型点估计略好，也须结合区间、训练充分性及多种子，不能直接证明注意力可解释。

下一步值得完成预注册 pilot（120/30/30 family、每变体4条轨迹、至多100轮、patience15），再按预测 MSE 选模，固定融合并重复42/43/44。只有在独立新拓扑上稳定超过预测和关联基线，才值得继续研究结构解释。不要调整测试阈值、按AP调参或筛除不利测试样本。

## 图像来源与选择规则

固定图例：manifest 中第一个测试 family 的 mixed_a、轨迹0，在计算指标前确定。示例 NPZ 另保留同 family 三变体轨迹0。所有测试轨迹的 S/A/对称分数与融合矩阵均按拓扑保存于 matrices/；不是只保存最好样例。

forecast-trained / head / attention；两层四头，统一色条。
![forecast-trained / head / attention；两层四头，统一色条。](../runs/smoke_seed42_final/probe/figures/attention_heads_example.png)

untrained / head / attention；同初始化、同一固定轨迹。
![untrained / head / attention；同初始化、同一固定轨迹。](../runs/smoke_seed42_final/probe/figures/attention_heads_untrained.png)

forecast-trained / fused / logits；对照 pairwise 与 projected-triad 真值，标签仅用于评估。
![forecast-trained / fused / logits；对照 pairwise 与 projected-triad 真值，标签仅用于评估。](../runs/smoke_seed42_final/probe/figures/fused_vs_ground_truth.png)

untrained 与 forecast-trained / head+fused / logits+attention；pairwise 与 projected-triad AP。
![untrained 与 forecast-trained / head+fused / logits+attention；pairwise 与 projected-triad AP。](../runs/smoke_seed42_final/probe/figures/head_ap_comparison.png)

forecast-trained / fused / logits+attention；固定样例 pairwise 正负分数分布。
![forecast-trained / fused / logits+attention；固定样例 pairwise 正负分数分布。](../runs/smoke_seed42_final/probe/figures/score_distributions.png)

未来预测训练/验证曲线、persistence 验证线与测试逐预测时刻误差。
![未来预测训练/验证曲线、persistence 验证线与测试逐预测时刻误差。](../runs/smoke_seed42_final/probe/figures/forecast_learning_curves.png)

forecast-trained / fused / logits+attention；固定二元骨架，三元投影变化并列。
![forecast-trained / fused / logits+attention；固定二元骨架，三元投影变化并列。](../runs/smoke_seed42_final/probe/figures/paired_topology_control.png)

## 实际命令、质量门禁与输出

本报告对应的生成、训练、评估命令（各 environment.json 另存实际 argv）：

```bash
cd /home/yg/TRANSTSR
/home/yg/miniconda3/envs/nd3/bin/python -m src.data.generate --config configs/smoke.yaml --out /home/yg/TRANSTSR/data/smoke
/home/yg/miniconda3/envs/nd3/bin/python -m src.data.validate --data /home/yg/TRANSTSR/data/smoke
/home/yg/miniconda3/envs/nd3/bin/python -m src.train --config configs/smoke.yaml --data /home/yg/TRANSTSR/data/smoke --out /home/yg/TRANSTSR/runs/smoke_seed42_final --device cpu --seed 42
/home/yg/miniconda3/envs/nd3/bin/python -m src.probe --config configs/smoke.yaml --data /home/yg/TRANSTSR/data/smoke --checkpoint /home/yg/TRANSTSR/runs/smoke_seed42_final/best.pt --out /home/yg/TRANSTSR/runs/smoke_seed42_final/probe --device cpu
/home/yg/miniconda3/envs/nd3/bin/python -m pytest -q
```

数据 manifest：[manifest.json](../data/smoke/manifest.json)；训练：[config_resolved.yaml](../runs/smoke_seed42_final/config_resolved.yaml)、[train_history.csv](../runs/smoke_seed42_final/train_history.csv)、[metrics_forecast.json](../runs/smoke_seed42_final/metrics_forecast.json)；探针：[attention_examples.npz](../runs/smoke_seed42_final/probe/attention_examples.npz)。
具体门禁结果以本 run 下 validation.json 与 pytest.log 为准。测试覆盖形状B=1、因果性、物理贡献、无标签接口、节点置换、非零有限梯度、单步更新、注意力归一化、单类指标、随机AP密度、family bootstrap、禁止覆盖。

## 失败条件与限制

运行机器的 NVIDIA 驱动不可通信且 torch.cuda.is_available() 为 False，因此未执行 GPU pilot。CPU smoke 不替代 pilot。
只运行数据/模型 seed42；3轮训练不能证明收敛，4个测试 family 不能支持科学结论。没有按结构AP选模型、头、融合权重或超参数。
AP只是排序一致性，logits 或 sigmoid(logits) 都不是已校准连接概率；投影无法唯一确定三元超边。预测性能、头间差异和配对矩阵变化均不能单独证明结构恢复。


## 最终验收补充（本次实际执行）

最终有效运行为 `runs/smoke_seed42_final/`。此前 `smoke_seed42` 的脚本末尾曾误收集源码快照测试而失败；已用 pytest.ini 限定收集目录并完整重跑。`smoke_seed42_verified` 为中间版本，最终版本进一步统一了 AP 与参照密度的有效拓扑口径；中间目录保留审计标记，均不作为最终报告数据源。

- 完整 smoke 脚本退出码0；**17 项测试通过**。11条 warning 来自 matplotlib 3.7.3 与 pyparsing 的弃用接口，不影响测试结果。
- 产物复核通过：12个测试拓扑、24条轨迹、4,512行逐轨迹指标、7张主评估图；全部原始 S/A 与对称分数、融合矩阵完整且有限。AP按拓扑汇总，CI按family重采样。
- 同初始化未训练 checkpoint 与训练 checkpoint 内 initial_state 完全相同；best epoch 与验证预测 MSE 最小值匹配。独立 untrained CLI 的矩阵与双阶段导出逐元素相同。
- 重复执行的 CPU 训练预测指标完全一致。这是同 seed 确定性复现，不代表跨模型种子稳定。
- 最终测试 MSE：full **0.832278**，node-only **0.902115**，persistence **0.293365**。full优于node-only但明显弱于persistence。
- pairwise fused-logit AP：未训练 **0.167442** → 训练后 **0.168963**；PLV **0.176713**，相位增量Pearson **0.173329**，随机 **0.164646**。
- projected-triad fused-logit AP：未训练 **0.144381** → 训练后 **0.141781**；随机 **0.156399**。只有非空三元拓扑参与这些AP，匹配的正例密度为 **0.136494**。这些smoke结果不支持可靠结构恢复。

实际执行的总入口及补充检查：

```bash
PYTHON=/home/yg/miniconda3/envs/nd3/bin/python bash scripts/run_smoke.sh data/smoke runs/smoke_seed42_final
/home/yg/miniconda3/envs/nd3/bin/python -m src.probe --config configs/smoke.yaml --data data/smoke --checkpoint runs/smoke_seed42_final/untrained.pt --stage untrained --out runs/smoke_seed42_final/probe_untrained --device cpu
/home/yg/miniconda3/envs/nd3/bin/python runs/smoke_seed42_final/artifact_audit.py
PYTHON=/home/yg/miniconda3/envs/nd3/bin/python bash scripts/run_pilot.sh runs/smoke_seed42_final data/pilot runs/pilot_seed42 cuda:0 42
```

最后一条pilot命令退出码1，在设备检查时报 `Requested cuda:0, but CUDA is unavailable; no silent CPU fallback`。未生成pilot数据、未开始pilot训练；`nvidia-smi` 实测退出码9、无法通信驱动。没有将未运行的pilot或seed43/44结果写为已完成。

验收证据：[artifact_audit.json](../runs/smoke_seed42_final/artifact_audit.json)、[可重跑的产物核验脚本](../runs/smoke_seed42_final/artifact_audit.py)、[hardware_status.json](../runs/smoke_seed42_final/hardware_status.json)、[execution_status.json](../runs/smoke_seed42_final/execution_status.json)、[pytest.log](../runs/smoke_seed42_final/pytest.log)。
