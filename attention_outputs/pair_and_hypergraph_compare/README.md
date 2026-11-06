# 注意力与真实二元结构、三元关联矩阵的可视化对比

run_id：`pair_and_hypergraph_compare_20261106T080006271303Z`。本次实际处理 8 条已有导出轨迹，没有新增模型或训练。
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

默认运行 `python scripts/compare_attention_pair_and_hypergraph.py` 处理源 metadata.json 中全部样本；单样本用 `--sample trajectory_id --out attention_outputs/new_single_compare`。输出必须为新目录，拒绝覆盖已有结果。
