# Quickstart: GatedSAE SID 生成

## 前置条件

1. 已完成 text2emb 步骤，生成 `.npy` embedding 文件
2. 已安装 SAELens 依赖：`pip install sae-lens`

## 快速使用

### 1. 训练 GatedSAE

```bash
python -m SAEGenRec.sid_builder gated_sae_train \
  --embedding_path=data/interim/Beauty.emb-all-MiniLM-L6-v2-td.npy \
  --output_dir=models/gated_sae/Beauty
```

### 2. 生成 SID

```bash
python -m SAEGenRec.sid_builder generate_sae_indices \
  --checkpoint=models/gated_sae/Beauty \
  --embedding_path=data/interim/Beauty.emb-all-MiniLM-L6-v2-td.npy \
  --output_path=data/interim/Beauty.index.json
```

### 3. 后续流程（与 RQ-VAE 相同）

```bash
# 转换数据集
python -m SAEGenRec.data_process convert_dataset \
  --dataset=Beauty --data_dir=data/processed \
  --index_file=data/interim/Beauty.index.json

# SFT 训练
python -m SAEGenRec.training sft \
  --model_path=models/llm --train_csv=data/processed/Beauty.train.csv \
  --info_file=data/processed/info/Beauty.txt

# 评估
python -m SAEGenRec.evaluation evaluate \
  --model_path=models/sft/Beauty/final_checkpoint \
  --test_csv=data/processed/Beauty.test.csv \
  --info_file=data/processed/info/Beauty.txt
```

### 一键流水线（Makefile）

```bash
make build_sae_sid CATEGORY=Beauty
make convert CATEGORY=Beauty
make sft MODEL_PATH=models/llm
make evaluate RL_MODEL_PATH=models/sft/Beauty/final_checkpoint
```

## 与 RQ-VAE 对比实验

```bash
# RQ-VAE 基线
make build_sid CATEGORY=Beauty
make convert CATEGORY=Beauty
make sft MODEL_PATH=models/llm
make evaluate RL_MODEL_PATH=models/sft/Beauty/final_checkpoint
# 结果保存在 results/Beauty_sft/

# GatedSAE 实验
make build_sae_sid CATEGORY=Beauty
make convert CATEGORY=Beauty
make sft MODEL_PATH=models/llm
make evaluate RL_MODEL_PATH=models/sft/Beauty/final_checkpoint
# 结果保存在 results/Beauty_sft/（需分别保存）
```
