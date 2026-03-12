# Experiment 001: LOO Strong Baseline

**Date**: 2026-03-11
**Branch**: 004-sae-beauty-baseline
**目标**: Beauty 数据集上 HR@10 = 0.8（拉伸目标），记录迭代过程

---

## 参数配置

| 参数 | 值 |
|------|-----|
| Dataset | Beauty (Amazon 2015) |
| Split | LOO (Leave-One-Out) |
| max_history_len | 20 |
| SAE K | 6 |
| SAE expansion_factor | 4 |
| SAE l1_coefficient | 0.2 |
| Embedding model | Qwen3-Embedding-0.6B (dim=1024) |
| LLM | Qwen3.5-0.8B |
| beam_size | 30 |
| Eval frequency | 每 epoch (on_epoch_end) |
| micro_batch_size | 4 |
| gradient_accumulation_steps | 8 |
| effective_batch_size | 32 |
| num_epochs | 5 |

---

## 实验步骤记录

### Step 0: 环境检查

- Beauty.emb-Qwen3-Embedding-0.6B-text.npy: (11196, 1024) ✓（已存在，可复用）
- Qwen3.5-0.8B: 已缓存本地 (1.8G) ✓
- Qwen3-Embedding-0.6B: 已缓存本地 (1.2G) ✓
- 当前 inter 文件为 TO 划分，需要用 LOO + max_history=20 重新预处理

### Step 1: 数据预处理 (LOO + max_history=20)

**改动**: split_method=TO → LOO，max_history_len=50 → 20，st_year=0，ed_year=9999（无时间过滤）
**原因**: LOO 是推荐系统标准评估协议；时间过滤原范围 2017-2018 会排除所有 Beauty 数据

- [x] 完成
- train: 98393 行 (98379 样本 + 1 header + 13 others)，valid: 20479 样本，test: 20479 样本
- items: 11196 (与原 embedding 文件一致)

### Step 2: GatedSAE 训练

**配置**: K=6, ef=4, d_sae=4096 (1024×4), l1=0.2

- [x] 完成
- 0 dead neurons
- 碰撞率: 初始 8.62%，post-dedup 0.0%（11196 唯一 SID）
- 模型保存在: models/gated_sae/Beauty-k6-ef4-l1_02/

### Step 3: SID 生成

- [x] 完成
- 唯一 SID 数: 11196 / 11196 物品 (100%，post-dedup)
- 索引文件: data/interim/Beauty.sae-k6.index.json

### Step 4: 数据集转换 + JSONL 准备

- [x] 完成
- CSV → JSONL (sid_seq task): data/processed/gated_sae/Amazon/Beauty/sid_seq/
- train.jsonl: 98379 行，valid.jsonl: 20479 行，test.jsonl: 20479 行

### Step 5: SFT 训练 (Qwen3.5-0.8B)

**配置**:
- Qwen3.5-0.8B (SSM混合架构，无 causal-conv1d → 纯 Python 实现，~22s/step)
- eval every epoch (on_epoch_end callback)
- beam_size=30，eval_samples=200
- 总步数: 3845 steps / 5 epochs

**Bug 修复 (2026-03-11)**:
- `logit_processor.py`: `base_model.lower()` 在 `base_model=None` 时崩溃 → 改为 `if base_model and base_model.lower()`
- `training_evaluator.py`: 添加 `on_epoch_end` 回调实现 per-epoch eval
- `sft.py` eval dataset: 限制为 min(1000, eval_rec_samples×5) 避免 tokenization 过慢

**运行中** (PID: 2853562, 启动于 17:37 UTC+8，预计 ~24 小时):
```
python -m SAEGenRec.training sft \
  --sft_data_dir=data/processed/gated_sae/Amazon/Beauty/sid_seq \
  --category=Beauty \
  --model_path=~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06... \
  --num_epochs=5 --eval_rec=True --eval_rec_steps=999999 \
  --eval_rec_beams=30 --eval_rec_samples=200 \
  --output_dir=models/sft/Beauty-k6-baseline \
  --micro_batch_size=4 --gradient_accumulation_steps=8
```

| Epoch | train_loss | eval_loss | HR@1 | HR@5 | HR@10 | NDCG@10 | 时间 |
|-------|-----------|----------|------|------|-------|---------|------|
| 1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 2 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 3 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 4 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 5 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

### Step 6: 评估

- [ ] HR@1:
- [ ] HR@5:
- [ ] HR@10:
- [ ] HR@20:
- [ ] NDCG@10:

---

## 改动历史

| 时间 | 改动 | 原因 | 效果 |
|------|------|------|------|
| 2026-03-11 | 初始设置 LOO+K=6 | 目标配置 | 待测 |
| 2026-03-11 | 修复 base_model.lower() NoneType 错误 | eval callback 崩溃 | 修复 |
| 2026-03-11 | 添加 on_epoch_end 回调 | 支持 per-epoch rec eval | 修复 |
| 2026-03-11 | LOO preprocess 移除时间过滤 | Beauty 数据在 2017-2018 范围外 | 获得全部 11196 items |

---

## 下一步改进方向（待填）

- [ ] SID-MSL loss（见 roadmap/strong_baseline_design.md）
- [ ] 调整 SAE 参数
- [ ] 调整 LLM LoRA rank/alpha
- [ ] 若 Qwen3.5-0.8B 过慢，对比 Qwen2.5-0.5B
