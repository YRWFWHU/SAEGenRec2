# CLI Contracts: GatedSAE SID 生成

## 新增 CLI 命令

所有命令通过 `python -m SAEGenRec.sid_builder <command>` 调用。

### `gated_sae_train`

训练 GatedSAE 模型。

```bash
python -m SAEGenRec.sid_builder gated_sae_train \
  --embedding_path=data/interim/Beauty.emb-all-MiniLM-L6-v2-td.npy \
  --expansion_factor=4 \
  --l1_coefficient=1.0 \
  --lr=3e-4 \
  --total_training_samples=1000000 \
  --train_batch_size=4096 \
  --output_dir=models/gated_sae/Beauty \
  --device=cuda:0
```

**输入**: .npy embedding 文件
**输出**: GatedSAE checkpoint 目录（`sae_weights.safetensors` + `cfg.json` + `training_config.json`）

### `generate_sae_indices`

从 GatedSAE checkpoint 生成 SID。

```bash
python -m SAEGenRec.sid_builder generate_sae_indices \
  --checkpoint=models/gated_sae/Beauty \
  --embedding_path=data/interim/Beauty.emb-all-MiniLM-L6-v2-td.npy \
  --k=8 \
  --output_path=data/interim/Beauty.index.json \
  --max_dedup_iters=20 \
  --device=cuda:0
```

**输入**: GatedSAE checkpoint + .npy embedding 文件
**输出**: .index.json（格式与 RQ-VAE `generate_indices` 一致）

## Makefile 快捷命令

```makefile
# GatedSAE SID 构建
build_sae_sid:
	python -m SAEGenRec.sid_builder gated_sae_train \
		--embedding_path=data/interim/$(CATEGORY).emb-all-MiniLM-L6-v2-td.npy \
		--output_dir=models/gated_sae/$(CATEGORY)
	python -m SAEGenRec.sid_builder generate_sae_indices \
		--checkpoint=models/gated_sae/$(CATEGORY) \
		--embedding_path=data/interim/$(CATEGORY).emb-all-MiniLM-L6-v2-td.npy \
		--output_path=data/interim/$(CATEGORY).index.json
```

## 输出文件格式契约

### .index.json

与 RQ-VAE `generate_indices` 输出格式完全一致：

```json
{
  "<item_id>": ["[a_<idx>]", "[b_<idx>]", "[c_<idx>]", "[d_<idx>]", "[e_<idx>]", "[f_<idx>]", "[g_<idx>]", "[h_<idx>]"],
  ...
}
```

- Key: 字符串 item_id
- Value: 长度为 K 的字符串列表（默认 K=8）
- Token 格式: `[{chr(97+pos)}_{feature_index}]`
