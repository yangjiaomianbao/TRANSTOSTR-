# TCN + Spatial Transformer 注意力特征提取

仓库仅保留共享 TCN、时间平均池化、两层四头 Spatial Transformer 的训练与特征导出流程。预测头只提供优化信号，不进行结构恢复评价或预测基线比较。

## 现有文件与结果

- `src/data/`：Kuramoto 生成器、历史/未来数据接口与数据校验。
- `src/model/`：TCN、Spatial Transformer 和特征返回接口。
- `src/train_attention.py`：仅主模型的预测损失训练与验证损失选 checkpoint。
- `src/export_attention.py`：导出时序特征、每层每头 S/A、对称矩阵及描述性诊断。
- `configs/smoke.yaml`、`tests/`：小规模配置与正确性测试。
- `data/smoke/`：可复用的本地数据。
- `checkpoints/attention_seed42/`：当前结果所用的已有3轮训练 checkpoint、训练日志、配置和环境记录；移动时校验了文件哈希。
- `attention_outputs/extract_8/`：第二次任务实际导出的8条轨迹及4张图，见其中 `README_attention_results.md`。

旧的结构评估/基线代码、旧脚本、旧报告、旧运行目录和旧任务书已删除。Git历史未修改。

## 复用现有 checkpoint 导出

本机解释器为 `/home/yg/miniconda3/envs/nd3/bin/python`。已有结果目录不会被覆盖，重复导出请指定新的目录：

```bash
cd /home/yg/TRANSTSR
/home/yg/miniconda3/envs/nd3/bin/python -m src.export_attention \
  --config configs/smoke.yaml \
  --data data/smoke \
  --checkpoint checkpoints/attention_seed42/best.pt \
  --out attention_outputs/new_export \
  --device cpu --split test --max-samples 8
```

`--max-samples 1` 导出一条，`--max-samples all` 导出所选 split 全部轨迹；`--batch-size` 默认8。按 manifest 顺序选择，图像固定为所选第一条轨迹。导出使用 `eval()` 和 `inference_mode()`，只从数据NPZ读取 x_all 的前96步及 permutation，不读取结构标签。

## 从数据生成到主模型训练

Python 3.10+，依赖见 requirements.txt。数据、checkpoint和输出目录默认不提交Git；全新克隆可运行：

```bash
python -m src.data.generate --config configs/smoke.yaml --out data/smoke
python -m src.train_attention --config configs/smoke.yaml --data data/smoke --out runs/attention_train --device cpu --seed 42
python -m src.export_attention --config configs/smoke.yaml --data data/smoke --checkpoint runs/attention_train/best.pt --out attention_outputs/new_export --device cpu --max-samples 8
python -m pytest -q
```

已有数据时跳过生成命令。每个输出路径必须为新目录。训练只保存主模型 untrained/best/last、损失CSV和 training_loss.png，checkpoint只按验证预测MSE选择。smoke的3轮仅检查优化和数据流，不能据此判断物理结构。

## NPZ 与矩阵含义

输入通道依次为 sin(theta)、cos(theta)。共享 TCN 为4个因果单卷积残差块，kernel=3、dilation=[1,2,4,8]，感受野31。仅沿时间 dim=2 平均得到Z。Spatial Transformer两层串行、每层四头并行，单头维度16。

| key | shape |
|---|---|
| x_obs | [B,30,96,2] |
| temporal_features | [B,30,96,64] |
| node_tokens | [B,30,64] |
| context_tokens | [B,30,64] |
| attention_logits | [B,2,4,30,30] |
| attention_weights | [B,2,4,30,30] |
| symmetric_logits | [B,2,4,30,30] |
| symmetric_attention | [B,2,4,30,30] |

S[i,j]为 query_i 与 key_j 的缩放点积，保存对角mask前的有限值。A[i,j]是对角mask后沿j做Softmax的dropout前权重。A每行和为1，对角为0。对称矩阵为 `(M+M.T)/2` 并清零对角，仅用于分析，不替代内部用于Value聚合的A，也不代表标定后的真实物理边概率。

NPZ另含轨迹/拓扑ID、节点置换、checkpoint路径、配置JSON和run_id，可用 `np.load(path, allow_pickle=False)` 读取。

## 结果目录

`attention_outputs/extract_8/` 包含：

- `attention_features.npz`：8条真实轨迹的全部特征与矩阵。
- `heads_layer1.png`、`heads_layer2.png`：第一条轨迹两层各四头的原始A，两图共用数值范围。
- `symmetric_heads.png`：同一轨迹的八头对称attention，统一色条。
- `training_loss.png`：该checkpoint已有训练日志的损失曲线。
- `attention_diagnostics.json`：行和、对角、对称误差、非对角分布、熵与头间相似度，仅为描述统计。
- `metadata.json`、`environment.json`：原始导出来源和运行记录；checkpoint迁移信息另有记录。
- `README_attention_results.md`：实际形状、读取方式和矩阵解释。
- `cleanup_record.json`：清理前后的文件/数组哈希及checkpoint迁移信息，用于确认第二次结果未损坏。
