# 注意力与真实结构：逐元素数值对比

run_id：`numeric_matrix_compare_20261106T084808563165Z`。实际处理 8 条已有样本；没有训练、没有AP或消融。

## 固定规则

节点编号沿用输入导出的0起始节点顺序；所有统计排除主对角。
节点显著性是整行/整列普遍偏大，不是统计检验的显著性。
R为对称矩阵，所以同一节点的行均值=列均值，不是两份独立证据。
广泛偏高节点（预先固定的描述规则）：行均值>=全局离对角均值+各节点行均值的1个总体标准差，且该行至少75%的非对角值严格大于全局非对角中位数。常数矩阵不标记节点。
节点加性分解（纯代数描述，不训练模型）：mu=无向非对角均值；a_i=(N-1)/(N-2)*(row_mean_i-mu)；node_component_ij=mu+a_i+a_j；residual_ij=R_ij-node_component_ij。报告其解释的离对角变异比例，不代表结构恢复能力。
三元表“高”固定定义为R(i,j)严格大于该矩阵全部无向非对角元素的75%分位数，不使用真值选择阈值；只表示相对较高，不是边判定。
三对都高：all_three_pairs_high；仅一对高：one_pair_high，明确高的一对及第三个节点；恰两对高：two_pairs_high，单独列出共享中心节点，不能误写成三对都高；没有高对：no_pairs_high。
若原始A行和为1且对角为0，则对称R_A的row_mean_i=(1+sum_j A[j,i])/(2*(N-1))。因此对称注意力的高行均值也可能只是某节点被普遍关注，不能据此证明具体节点对关联。
所有CSV保留9位有效数字；数值图仅用于阅读，2/3位小数可能掩盖微小差异。统计、分组和判断使用未舍入原值。未计算AP、未训练、未做消融，不给出最终结构推断结论。


## 每条样本的输出

- A2_true.csv：30×30真实二阶矩阵，无表头，行列均按节点0–29编号。
- layerL_headH_symmetric_logits.csv / symmetric_attention.csv：每头两个30×30矩阵，9位有效数字，可直接np.loadtxt(delimiter=',')。
- pair_value_report.txt：各头/分数类型的全部真实边、同一组固定随机非边、全部30个节点的行/列描述、广泛偏高节点及代数分解说明。
- triad_value_report.txt：每头每条真实三元超边的三个两两值及min/mean/max；按预先固定的q75规则汇总三对都高、仅一对高、两对高、无高对。
- numeric_summary.json：相同数值的机器可读版本，包含非边随机种子、节点子集、节点描述和三元组分组。
- A2_true_numeric.png：格子内直接标0/1。
- layerL_headH_numeric_logits.png / numeric_attention.png：全部16张30×30黑白数字图，无彩色底图，保留3位小数。
- top_left_10x10_numeric.png：固定L1H1，前10节点的A2与两种分数并列。
- edge_focus_numeric.png：仅按真实A2贪心选10个内部边较多的节点，固定L1H1，不按注意力择优。

注意：对称矩阵行/列均值相同。高行均值可以来自节点受普遍关注，而非少数真实边位置独有高值。节点加性分解仅帮助描述该差别，残差也不等同于真实结构。

默认命令：`python scripts/export_numeric_attention_reports.py`。支持 `--sample trajectory_id`、`--input-root`、`--out`、`--seed`、`--nonedges`、`--decimals 2|3`。输出须为新目录，不覆盖现有结果。输入文件均未修改。
