# 多头注意力特征提取：实际导出结果

run_id：`extract_8_20261106T062925999637Z`；模式：`eval()` + `torch.inference_mode()`。
checkpoint：`/home/yg/TRANSTSR/checkpoints/attention_seed42/best.pt`，epoch=3。
本次仅导出既有 checkpoint；没有新增训练或结构评估。该 checkpoint 的来源与原训练配置见 metadata.json，不能把历史 smoke 训练当作当前正式结果。

## 固定样例及读取方式

按 manifest 中 split=`test` 的轨迹顺序选取前 8 条；绘图固定为第一条 `f0016_pair_only_t000`（拓扑 `f0016_pair_only`）。未依据矩阵数值或结构指标选择样例。
通道顺序为 sin(theta)、cos(theta)，历史取 x_all 的前96步。置换约定：new_x = canonical_x[permutation]。

```python
import numpy as np
with np.load('/home/yg/TRANSTSR/attention_outputs/extract_8/attention_features.npz', allow_pickle=False) as f:
    x = f['x_obs'][0]                    # [30,96,2]
    H = f['temporal_features'][0]        # [30,96,64]
    Z = f['node_tokens'][0]              # [30,64], H.mean(axis=1)
    S = f['attention_logits'][0,0,0]     # [30,30], 第一条轨迹/第一层/第一头
    A = f['attention_weights'][0,0,0]    # [30,30]
    print(f['trajectory_id'], f['permutation'])
```

| NPZ key | 实际形状 |
|---|---|
| `x_obs` | `[8, 30, 96, 2]` |
| `temporal_features` | `[8, 30, 96, 64]` |
| `node_tokens` | `[8, 30, 64]` |
| `context_tokens` | `[8, 30, 64]` |
| `attention_logits` | `[8, 2, 4, 30, 30]` |
| `attention_weights` | `[8, 2, 4, 30, 30]` |
| `symmetric_logits` | `[8, 2, 4, 30, 30]` |
| `symmetric_attention` | `[8, 2, 4, 30, 30]` |

另存 trajectory_id、topology_id、permutation、checkpoint_path、config_json、run_id；均可 allow_pickle=False 读取。完整来源、配置和输入文件哈希见 metadata.json。

## 如何解释一张矩阵

取样例 `[sample=0, layer=0, head=0]`：`S[0,1]=-0.09523244`，`A[0,1]=0.03504439`。
S[i,j] 是该层该头 query_i 与 key_j 的点积除以 sqrt(16)，保留有限的 mask 前原始分数；A[i,j] 是先排除 j=i，再沿源节点 j 做 Softmax 后的相对权重。eval 下通过 sum_j A[i,j] V[j] 聚合，头并行后拼接，完整第一层输出再传入第二层。
`symmetric_logits=(S+S.T)/2`、`symmetric_attention=(A+A.T)/2`，对角设0，仅为分析派生输出，不替代 Transformer 内部注意力。对称后的行和不要求为1。分数和权重由表示、Softmax竞争及训练目标共同决定，未经物理结构标定；不能直接解释为真实边概率或真实结构。

## 图像与描述性诊断

- heads_layer1.png：固定样例第一层四头原始 A；heads_layer2.png：第二层四头原始 A。两张图共用同一数值范围，行是查询节点 i，列是源节点 j。
- symmetric_heads.png：同一样例全部8头的对称 attention，统一色条；不是实际聚合矩阵。
- attention_diagnostics.json：各头行和/对角、原始非对称性、派生矩阵对称性误差、非对角分布、逐节点注意力熵（自然对数，均匀参照 log(29)）、8头间 cosine/Pearson 相似度。相似度只描述当前样本的表示，零方差/零范数为 null。没有计算结构恢复指标。
- training_loss.png：来自原 checkpoint 同目录的历史主模型训练日志，只用于查看优化过程；本次没有重新训练。

## 对应代码

- src/model/tcn.py：各节点共享、仅沿时间卷积，4个单卷积残差块，感受野31。
- src/model/attention_probe.py：保留 H，沿时间池化得到 Z，再进入两层串行 Transformer。
- src/model/spatial_attention.py：每层4头并行；导出未 mask S、mask 后 dropout 前 A。
- src/export_attention.py：仅读取历史与置换，导出特征、画图和计算描述性诊断。
- src/train_attention.py：可选的主模型单次训练入口，预测损失只作为注意力训练信号。

复现当前导出的命令见 metadata.json / environment.json。本目录是实际特征提取结果，不是新的正式结构实验结论。

## 仓库清理后的复现

旧评估代码和历史结果已删除。checkpoint迁移到 `checkpoints/attention_seed42/best.pt`，文件哈希保持不变。科学数组与图片均保留；原始运行命令仍记录在 environment.json，复现请使用以下新路径及新的输出目录。

```bash
python -m src.export_attention --config configs/smoke.yaml --data data/smoke --checkpoint checkpoints/attention_seed42/best.pt --out attention_outputs/new_export --device cpu --split test --max-samples 8
```
