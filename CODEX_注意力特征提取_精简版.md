# Codex 任务书：TRANSTOSTR 精简为多头注意力特征提取实验

## 0. 背景与边界（必须遵守）

项目仓库：https://github.com/yangjiaomianbao/TRANSTOSTR

**当前目标不是网络结构恢复，也不是评估预测模型的 SOTA 性能。** 当前阶段唯一目标：把混合 Kuramoto 的节点时序输入 TCN + Temporal Pooling + Spatial Transformer，完整导出并解释 **每层每个头** 的节点间关系矩阵。最终结构推断器将来另行设计，本任务不实施。

**不做**：Node-only、Persistence、PLV、Pearson、随机评分基线；不做 AP、AUROC、F1、Precision@K、真实结构恢复优劣排名；不做独立头数/层数/网络规模/种子大规模消融；不做三元超边预测、Gumbel、HGNN；不做 family bootstrap、配对骨架统计检验，不按真实边挑最优头。

已知仓库中的 `reports/` 是先前 Smoke Test 的历史报告，不能把过去的 smoke 结果宣称为当前正式结果，也不要因本次任务重构全部数据流程。优先复用现有生成数据与模型代码，最小化变更。

## 1. 固定的数据和网络

- 数据：`X ∈ R[B,30,96,2]`，通道依次 `sin(theta),cos(theta)`；若数据来自现有 NPZ，历史窗口取 `x_all[:,:,:96,:]`。
- 保留现有 `src/data/generate.py`、`src/data/kuramoto.py`、`src/data/dataset.py` 及已有配置；如本地数据不存在，按现有脚本生成少量可运行数据；不要新增复杂生成方案。
- 共享 TCN：4 个因果残差卷积块，`d_model=64`、`kernel_size=3`、`dilations=[1,2,4,8]`；输出 `H ∈ R[B,30,96,64]`。
- 沿时间维度平均池化，`Z=H.mean(dim=2) ∈ R[B,30,64]`。保留 `H` 与 `Z` 供检查。
- Spatial Transformer：**2 层串行，每层 4 头并行**，总共 8 张矩阵；单头维度 16。每层输出更新后的节点特征供下一层输入。不要误将 8 个头串联。
- 每层每头保留：(a) 未经过 Softmax 的关系 logits `S^(l,r)`；(b) 对角 mask 后、Softmax 得到的注意力 `A^(l,r)`；(c) `R_S=(S+S.T)/2`（去对角），`R_A=(A+A.T)/2`（去对角）。对称矩阵是**供分析用的派生输出**，不取代 Transformer 内部用于 Value 聚合的注意力。
- 原模型的轻量未来状态预测头**仅保留作为训练信号**，以使注意力经过数据驱动训练；可沿用当前 `src/train.py` 的预测损失与 checkpoint 选择逻辑，**不新增预测基线，也不做预测性能评比**。不得使用 `y_pair/y_tri/omega/拓扑标签` 参与训练。

## 2. 实现范围：最小改动

优先保留：
- `src/model/tcn.py`
- `src/model/spatial_attention.py`
- `src/model/attention_probe.py`
- `src/train.py`

修改或新增一个精简导出脚本，例如 `src/export_attention.py`，**不要沿用旧 `src/probe.py` 的整套结构评估作为本次主任务**。

推荐命令接口：

```bash
python -m src.export_attention \
  --config configs/smoke.yaml \
  --data <实际本地数据目录> \
  --checkpoint <实际本地best.pt> \
  --out <新的空输出目录> \
  --device cpu \
  --split test \
  --max-samples 8
```

`--max-samples` 只限制导出示例数量，可配置 1、8、全部；选择方式固定（清单顺序），不要基于结构指标筛选。

如果不已有 checkpoint：用**当前最简 train 命令**训练主模型一次；删除或跳过 `node_only` 模型训练、预测 baseline 和复杂评估；留下损失曲线供训练是否正常的基本检查即可。不要强行在 smoke 三轮内讨论结构是否有效。如果原 `src.train` 混杂辅助任务，可新增精简 `src/train_attention.py`，不改坏现有代码。

## 3. 必须保存的产物

对所选每条轨迹，输出到 `attention_outputs/`（或命令指定目录），建议 `.npz` 键：

| key | shape | 含义 |
|---|---|---|
| `x_obs` | `[B,30,96,2]` | 模型输入 |
| `temporal_features` | `[B,30,96,64]` | TCN 输出 |
| `node_tokens` | `[B,30,64]` | 时间池化后的节点特征 |
| `context_tokens` | `[B,30,64]` | 最后一层 Spatial Transformer 输出 |
| `attention_logits` | `[B,2,4,30,30]` | 每层每头 Softmax 前的关系评分 |
| `attention_weights` | `[B,2,4,30,30]` | 每层每头实际使用的原始注意力权重（dropout 前） |
| `symmetric_logits` | `[B,2,4,30,30]` | 离线对称化 logits，对角设 0 |
| `symmetric_attention` | `[B,2,4,30,30]` | 离线对称化 attention，对角设 0 |

保存对应 `trajectory_id`、`topology_id`、模型 checkpoint 路径、配置摘要、节点置换等必要元数据；明确 **没有** 把矩阵解释为真实物理边概率。

至少生成：
1. `heads_layer1.png`：第一层四个头的 30×30 注意力热图，统一色条；
2. `heads_layer2.png`：第二层四个头的热图，统一色条；
3. `symmetric_heads.png`：对应8张对称化的关系热图；
4. `attention_diagnostics.json`：每头注意力的行和（非对角 softmax 权重应为1）、对称性误差、非对角分数分布、注意力熵和头间相似度（描述性统计，不做任何结构恢复评价）；
5. `README_attention_results.md`：简述如何读 NPZ、对应代码位置、样例输入及各张图的含义。

按需输出 `training_loss.png` 或训练日志，只用于确定模型确实得到梯度和优化，不需要预测基线图或性能榜单。

## 4. 基本正确性检查

- `TCN` 只在时间维度卷积，各节点共享权重；`Pooling` 沿 `dim=2`；Spatial Attention 沿节点维度；
- 两层×四头的输出形状 `[B,2,4,30,30]`；层串行、头并行；
- 训练和 eval 模式区分：导出时 `model.eval()`、`torch.inference_mode()`；
- 对角 mask 后注意力对角为0、行和约为1；所有 logits 有限；对称矩阵严格对称且对角为0；
- `src/export_attention.py` 不得加载任何 `y_pair`、`y_tri` 参与计算或样本选择；
- 有 CPU 环境也可完成端到端导出；不存在 checkpoint/data 时给出明确错误及最短补救命令；
- 所有产物写入新输出目录，不覆盖既有实验结果；不可凭空报告未运行的实验。

## 5. 完成标准与最终回复给用户的内容

Codex 完成后只回答：
- 修改了哪些文件（及每个文件一句职责）；
- **一条真实轨迹**从 `[30,96,2]` → `[30,96,64]` → `[30,64]` → `[2,4,30,30]` 的实际形状检查；
- 注意力矩阵与 3~4 张图在本地的**实际存在的文件路径**；
- 提取其中一张矩阵，说明 `S[i,j]`、`A[i,j]` 的含义，以及为何对称化矩阵不能直接解释为真实结构；
- 如有未完成的运行环节，如实说明。

**绝不扩展到结构推断器、基线预测竞赛、或复杂消融实验。**
