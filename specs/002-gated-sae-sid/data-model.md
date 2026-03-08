# Data Model: GatedSAE SID 生成

**Date**: 2026-03-08
**Feature**: 002-gated-sae-sid

## Entities

### GatedSAEConfig

训练和推理的超参数配置。

| Field               | Type       | Default              | Description                        |
|---------------------|------------|----------------------|------------------------------------|
| embedding_path      | str        | ""                   | 输入 .npy 文件路径                 |
| d_in                | int        | 384                  | 输入 embedding 维度（自动检测）    |
| expansion_factor    | int        | 4                    | 字典大小 = d_in × expansion_factor |
| k                   | int        | 8                    | top-K 特征数（SID token 数）       |
| l1_coefficient      | float      | 1.0                  | L1 稀疏性惩罚权重                  |
| lr                  | float      | 3e-4                 | 学习率                             |
| total_training_samples | int     | 1000000              | 总训练样本数（含多 epoch 重复）    |
| train_batch_size    | int        | 4096                 | 训练批次大小                       |
| output_dir          | str        | "models/gated_sae"   | 输出目录                           |
| device              | str        | "cuda:0"             | 训练设备                           |
| seed                | int        | 42                   | 随机种子                           |
| max_dedup_iters     | int        | 20                   | 最大去碰撞迭代次数                 |

### GatedSAE Checkpoint

保存到磁盘的模型状态。

| Component     | Format          | Description                                   |
|---------------|-----------------|-----------------------------------------------|
| sae_weights.safetensors | safetensors | SAELens 标准权重格式（W_enc, W_dec, b_gate, b_mag, r_mag, b_dec） |
| cfg.json      | JSON            | SAELens SAEConfig 序列化                      |
| training_config.json | JSON     | GatedSAEConfig 完整配置（含 k, expansion_factor 等项目特定参数） |

### SID Index (.index.json)

与 RQ-VAE 输出完全一致的格式。

```json
{
  "item_id_1": ["[a_42]", "[b_1028]", "[c_7]", "[d_15]", "[e_892]", "[f_3]", "[g_401]", "[h_67]"],
  "item_id_2": ["[a_100]", "[b_523]", "[c_201]", "[d_88]", "[e_1200]", "[f_45]", "[g_77]", "[h_310]"],
  ...
}
```

- Key: item_id（字符串）
- Value: K 个 token 字符串的列表（默认 K=8），按激活值降序排列
- Token 格式: `[{prefix}_{feature_index}]`
- prefix 来自字母表 `a, b, c, ..., z`（第 i 个位置用第 i 个字母）
- feature_index 范围: `[0, d_sae - 1]`

### 稀疏特征激活

中间计算结果，不持久化。

| Field               | Shape              | Description                        |
|---------------------|--------------------|------------------------------------|
| feature_acts        | (batch, d_sae)     | GatedSAE 编码输出，大部分为零      |
| top_k_indices       | (batch, k)         | 激活值最大的 K 个特征索引          |
| top_k_values        | (batch, k)         | 对应的激活值（降序排列）           |

## Data Flow

```
输入: {dataset}.emb-{model}-td.npy (n_items, d_in)
  │
  ├── [训练] NpyDataProvider → SAETrainer → GatedSAE checkpoint
  │
  └── [推理] GatedSAE.encode() → top-K → dedup → .index.json
                                                      │
                                                      └── [下游] convert_dataset → CSV + info TXT
```

## Validation Rules

- `d_in` 必须与 .npy 文件的最后一维匹配
- `k` 必须 ≤ 26（字母表长度限制）
- `expansion_factor` 必须 ≥ 1
- 输出 .index.json 中每个 item 必须有恰好 K 个 token
- SID 唯一性：去碰撞后不应存在重复 SID（允许少量残余碰撞并警告）
