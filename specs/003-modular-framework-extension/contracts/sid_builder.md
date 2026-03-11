# CLI Contract: sid_builder

**Module**: `python -m SAEGenRec.sid_builder`

## Commands

### build_sid

统一 SID 生成入口（训练 + 生成 .index.json）。

```bash
python -m SAEGenRec.sid_builder build_sid \
    --method=rqvae \
    --category=Beauty \
    [--emb_path=data/interim/Beauty.emb-all-MiniLM-L6-v2-text.npy] \
    [--k=3] \
    [--token_format=auto] \
    [--output_dir=models/sid/Beauty] \
    [--index_output=data/interim/Beauty.index.json] \
    [--device=cuda:0]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| method | str | **required** | SID 方法：`rqvae`, `rqkmeans`, `gated_sae` |
| category | str | **required** | 商品类别名 |
| emb_path | str | auto-detect | 输入嵌入 .npy 路径（默认自动查找文本嵌入） |
| k | int | method-dependent | SID token 数（rqvae: 3, gated_sae: 8） |
| token_format | str | `auto` | token 前缀格式（`auto`, `a-h`, `f`, 或自定义字符） |
| output_dir | str | `models/sid/{category}` | 模型输出目录 |
| index_output | str | `data/interim/{category}.index.json` | .index.json 输出路径 |
| device | str | `cuda:0` | 计算设备 |

**Behavior**:
1. 从注册表查找 method 对应的 `SIDMethod` 实现
2. 调用 `method.train(emb_path, output_dir, **config)`
3. 调用 `method.generate(checkpoint, emb_path, index_output, k, token_format)`
4. `token_format=auto`: rqvae/rqkmeans → 位置前缀 `a/b/c/...`，gated_sae → 统一前缀 `f`
5. `emb_path` 未指定时自动查找 `data/interim/{category}.emb-*-text.npy`

**Output**: `.index.json` 格式
```json
{
  "item_asin_1": ["[a_42]", "[b_128]", "[c_7]"],
  "item_asin_2": ["[f_10]", "[f_23]", "[f_45]", ...]
}
```

---

### train_sid

仅训练 SID 模型（不生成 .index.json）。

```bash
python -m SAEGenRec.sid_builder train_sid \
    --method=rqvae \
    --category=Beauty \
    [--emb_path=...] \
    [--output_dir=models/sid/Beauty]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| method | str | **required** | SID 方法 |
| category | str | **required** | 商品类别名 |
| emb_path | str | auto-detect | 输入嵌入 .npy 路径 |
| output_dir | str | `models/sid/{category}` | checkpoint 输出目录 |

**Output**: 模型 checkpoint 目录

---

### generate_sid

从已训练模型生成 .index.json。

```bash
python -m SAEGenRec.sid_builder generate_sid \
    --method=rqvae \
    --category=Beauty \
    [--checkpoint=models/sid/Beauty] \
    [--emb_path=...] \
    [--k=3] \
    [--token_format=auto] \
    [--index_output=data/interim/Beauty.index.json]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| method | str | **required** | SID 方法 |
| category | str | **required** | 商品类别名 |
| checkpoint | str | **required** | 模型 checkpoint 路径 |
| emb_path | str | auto-detect | 输入嵌入 .npy 路径 |
| k | int | method-dependent | SID token 数 |
| token_format | str | `auto` | token 前缀格式 |
| index_output | str | `data/interim/{category}.index.json` | 输出路径 |

---

### 保留命令（向后兼容）

以下现有命令保持不变：
- `text2emb` → 重命名为 `embed_text`（移至 data_process 模块），原名保留为别名
- `rqvae_train` → 内部由 `RQVAEMethod.train()` 调用
- `generate_indices` → 内部由 `RQVAEMethod.generate()` 调用
- `gated_sae_train` → 内部由 `GatedSAEMethod.train()` 调用
- `generate_sae_indices` → 内部由 `GatedSAEMethod.generate()` 调用
- `rqkmeans_*` → 内部由 `RQKMeansMethod.train()` 调用
