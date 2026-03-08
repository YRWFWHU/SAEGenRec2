# SAEGenRec 多卡服务器训练指南

> 覆盖范围：raw data → 数据预处理 → SID 生成 → SFT 训练 → RL 训练 → 评估
>
> 适用场景：多 GPU 单机（DDP）、多 GPU + DeepSpeed ZeRO

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [SID 生成](#3-sid-生成)
4. [数据集转换](#4-数据集转换)
5. [SFT 训练（多卡）](#5-sft-训练多卡)
6. [RL 训练（多卡）](#6-rl-训练多卡)
7. [评估](#7-评估)
8. [完整流水线脚本](#8-完整流水线脚本)
9. [常用参数参考](#9-常用参数参考)
10. [已知问题与解决方案](#10-已知问题与解决方案)

---

## 1. 环境准备

### 1.1 安装依赖

```bash
conda create -n saegenrec python=3.11 -y
conda activate saegenrec

# 安装项目
pip install -e .

# 多卡训练额外依赖
pip install deepspeed          # DeepSpeed ZeRO（可选，显存不足时使用）
pip install flash-attn         # Flash Attention 2（标准 Transformer 模型加速）
pip install flash-linear-attention  # Qwen3.5 等 SSM 混合架构模型加速
```

### 1.2 确认多卡环境

```bash
# 查看可用 GPU
nvidia-smi

# 确认 NCCL 可用
python -c "import torch; print(torch.cuda.device_count(), 'GPUs available')"
python -c "import torch.distributed; print('NCCL available:', torch.distributed.is_nccl_available())"
```

### 1.3 目录结构

```
SAEGenRec/
├── data/
│   ├── raw/           # 原始 Amazon 数据（下载后放这里）
│   ├── interim/       # 中间数据（embeddings, index.json）
│   └── processed/     # 处理后数据（CSV, info TXT）
├── models/
│   ├── gated_sae/     # GatedSAE checkpoint
│   ├── sft/           # SFT checkpoint
│   └── rl/            # RL checkpoint
└── results/           # 评估结果
```

---

## 2. 数据准备

### 2.1 下载原始数据

将 Amazon Review 数据集（2014/2018/2023 版本）放入 `data/raw/`：

```
data/raw/
├── reviews_Beauty_5.json.gz     # 评论文件
└── meta_Beauty.json.gz          # 商品元数据
```

### 2.2 数据预处理（单卡即可）

```bash
python -m SAEGenRec.data_process preprocess \
    --data_dir=data/raw \
    --output_dir=data/processed \
    --category=Beauty \
    --k_core=5              # k-core 过滤（默认 5）
```

输出：
- `data/interim/Beauty.inter` — 用户交互记录（TSV）
- `data/interim/Beauty.item.json` — 商品元数据
- `data/interim/Beauty.review.json` — 评论数据

### 2.3 生成文本 Embedding（单卡）

```bash
python -m SAEGenRec.sid_builder text2emb \
    --dataset=Beauty \
    --data_dir=data/processed \
    --output_dir=data/processed/emb \
    --model_name=all-MiniLM-L6-v2    # sentence-transformers 模型
```

输出：
- `data/interim/Beauty.emb-all-MiniLM-L6-v2-td.npy` — (n_items, 384) embedding 矩阵

---

## 3. SID 生成

### 方案 A：RQ-VAE SID（K=3，原始方案）

```bash
# 训练 RQ-VAE
python -m SAEGenRec.sid_builder rqvae_train \
    --dataset=Beauty \
    --emb_dir=data/processed/emb \
    --output_dir=data/processed

# 生成索引
python -m SAEGenRec.sid_builder generate_indices \
    --dataset=Beauty \
    --emb_dir=data/processed/emb \
    --checkpoint_dir=data/processed \
    --output_dir=data/processed
```

输出：`data/processed/Beauty.index.json`，格式 `{"item_id": "[a_42][b_128][c_7]", ...}`

### 方案 B：GatedSAE SID（K=8，改进方案）

```bash
# 训练 GatedSAE
python -m SAEGenRec.sid_builder gated_sae_train \
    --embedding_path=data/interim/Beauty.emb-all-MiniLM-L6-v2-td.npy \
    --output_dir=models/gated_sae/Beauty \
    --expansion_factor=4 \
    --k=8 \
    --l1_coefficient=1.0 \
    --total_training_samples=1000000 \
    --device=cuda:0

# 生成索引
python -m SAEGenRec.sid_builder generate_sae_indices \
    --checkpoint=models/gated_sae/Beauty \
    --embedding_path=data/interim/Beauty.emb-all-MiniLM-L6-v2-td.npy \
    --output_path=data/interim/Beauty.sae.index.json \
    --k=8
```

输出：`data/interim/Beauty.sae.index.json`，格式 `{"item_id": ["[a_x]", "[b_y]", ..., "[h_z]"], ...}`

### Makefile 快捷命令

```bash
make embed CATEGORY=Beauty
make build_sid CATEGORY=Beauty        # RQ-VAE 方案
make build_sae_sid CATEGORY=Beauty    # GatedSAE 方案
```

---

## 4. 数据集转换

```bash
python -m SAEGenRec.data_process convert_dataset \
    --dataset=Beauty \
    --data_dir=data/processed \
    --index_json=data/processed/Beauty.index.json \   # 或 Beauty.sae.index.json
    --inter_dir=data/interim \
    --output_dir=data/processed
```

输出：
- `data/processed/Beauty.{train,valid,test}.csv`
- `data/processed/info/Beauty.txt` — `semantic_id\ttitle\titem_id`

```bash
# Makefile
make convert CATEGORY=Beauty
```

---

## 5. SFT 训练（多卡）

### 5.1 方式一：torchrun DDP（推荐，N 卡线性加速）

```bash
# 4 卡训练示例
torchrun --nproc_per_node=4 \
    -m SAEGenRec.training sft \
    --model_name=Qwen/Qwen2.5-1.5B \
    --train_csv=data/processed/Beauty.train.csv \
    --valid_csv=data/processed/Beauty.valid.csv \
    --sid_index_path=data/processed/Beauty.index.json \
    --item_meta_path=data/interim/Beauty.item.json \
    --output_dir=models/sft/Beauty \
    --batch_size=512 \          # 全局 batch size（所有卡合计）
    --micro_batch_size=8 \      # 每卡每步 batch size
    --num_epochs=3 \
    --learning_rate=3e-4 \
    --freeze_llm=False \        # 全参数训练
    --category=Beauty
```

> **batch_size 计算**：`gradient_accumulation = batch_size / micro_batch_size / num_gpus`
>
> 示例：`512 / 8 / 4 = 16`步累积

### 5.2 方式二：DeepSpeed ZeRO（显存不足时）

**创建 DeepSpeed 配置**：

```bash
mkdir -p configs
```

**ZeRO-2（梯度+优化器状态分片，适合 7B 以下模型）**：

```json
// configs/ds_zero2.json
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8
  },
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

**ZeRO-3（参数+梯度+优化器全分片，适合超大模型）**：

```json
// configs/ds_zero3.json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "gather_16bit_weights_on_model_save": true
  },
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

**启动训练**：

```bash
deepspeed --num_gpus=4 \
    -m SAEGenRec.training sft \
    --model_name=Qwen/Qwen2.5-1.5B \
    --train_csv=data/processed/Beauty.train.csv \
    --valid_csv=data/processed/Beauty.valid.csv \
    --sid_index_path=data/processed/Beauty.index.json \
    --item_meta_path=data/interim/Beauty.item.json \
    --output_dir=models/sft/Beauty \
    --batch_size=512 \
    --micro_batch_size=8 \
    --num_epochs=3 \
    --deepspeed=configs/ds_zero2.json \
    --freeze_llm=False \
    --category=Beauty
```

### 5.3 方式三：多节点训练

```bash
# 节点 0（主节点，IP: 192.168.1.100）
torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    --nproc_per_node=4 \
    -m SAEGenRec.training sft \
    --model_name=Qwen/Qwen2.5-7B \
    --train_csv=data/processed/Beauty.train.csv \
    --valid_csv=data/processed/Beauty.valid.csv \
    --output_dir=models/sft/Beauty \
    --batch_size=1024 \
    --micro_batch_size=4 \
    --num_epochs=3 \
    --category=Beauty

# 节点 1（同样命令，仅 node_rank 不同）
torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    --nproc_per_node=4 \
    -m SAEGenRec.training sft \
    [同上参数]
```

### 5.4 SFT 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | — | HuggingFace 模型名或本地路径 |
| `batch_size` | 128 | 全局 batch size（所有卡合计） |
| `micro_batch_size` | 4 | 每卡每步 batch size |
| `num_epochs` | 10 | 训练轮数 |
| `learning_rate` | 3e-4 | 学习率 |
| `freeze_llm` | False | True=只训练新增 token embedding |
| `sample` | -1 | -1=全量；正整数=采样 N 条 |
| `cutoff_len` | 512 | 最大序列长度 |
| `deepspeed` | None | DeepSpeed 配置文件路径 |
| `sid_index_path` | — | .index.json 路径（TokenExtender 使用） |
| `item_meta_path` | — | .item.json 路径（辅助训练数据） |

SFT 输出：
- `models/sft/Beauty/final_checkpoint/` — 最终模型
- `models/sft/Beauty/checkpoint-{step}/` — 中间 checkpoint

---

## 6. RL 训练（多卡）

RL 训练基于 GRPO，需在 SFT checkpoint 基础上进行。

### 6.1 DDP 启动

```bash
torchrun --nproc_per_node=4 \
    -m SAEGenRec.training rl \
    --model_path=models/sft/Beauty/final_checkpoint \
    --train_csv=data/processed/Beauty.train.csv \
    --info_file=data/processed/info/Beauty.txt \
    --output_dir=models/rl/Beauty \
    --category=Beauty \
    --train_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --num_generations=8 \       # 每个 prompt 生成的样本数（越大越稳定，越慢）
    --num_train_epochs=1 \
    --learning_rate=1e-6 \
    --reward_type=rule          # rule / ranking / semantic / sasrec
```

### 6.2 DeepSpeed 启动

```bash
deepspeed --num_gpus=4 \
    -m SAEGenRec.training rl \
    --model_path=models/sft/Beauty/final_checkpoint \
    --train_csv=data/processed/Beauty.train.csv \
    --info_file=data/processed/info/Beauty.txt \
    --output_dir=models/rl/Beauty \
    --category=Beauty \
    --train_batch_size=32 \
    --num_generations=8 \
    --deepspeed=configs/ds_zero2.json
```

### 6.3 RL 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | — | SFT checkpoint 路径 |
| `train_batch_size` | 32 | 训练 batch size |
| `num_generations` | 16 | 每条 prompt 的采样数（GRPO） |
| `num_train_epochs` | 1 | 训练轮数 |
| `learning_rate` | 1e-6 | 学习率（比 SFT 小 2-3 个数量级） |
| `beta` | 0.04 | KL 散度惩罚系数 |
| `reward_type` | rule | 奖励函数类型 |
| `sample` | -1 | 快速验证用（如 500） |
| `beam_search` | False | 是否用 beam search 生成 |

> ⚠️ **注意**：全量 RL 训练（129k samples × num_generations=16）约需 27 小时（单卡）。
> 快速验证：`--sample=500 --num_generations=4`（约 35 分钟）

RL 输出：
- `models/rl/Beauty/checkpoint-{step}/` — 中间 checkpoint

---

## 7. 评估

评估使用约束波束搜索（Constrained Beam Search），**只需单卡**：

```bash
python -m SAEGenRec.evaluation evaluate \
    --model_path=models/rl/Beauty/checkpoint-{最优step} \
    --test_csv=data/processed/Beauty.test.csv \
    --info_file=data/processed/info/Beauty.txt \
    --category=Beauty \
    --output_dir=results/Beauty \
    --batch_size=2 \            # 建议 2（CUDA 稳定性）
    --num_beams=50 \            # 波束宽度
    --k_values=1,3,5,10,20
```

输出：
- `results/Beauty/metrics.json` — HR@K 和 NDCG@K
- `results/Beauty/predictions.json` — 每条样本的预测结果

```json
// metrics.json 示例
{
  "HR@1": 0.0039, "HR@5": 0.0166, "HR@10": 0.0240, "HR@20": 0.0486,
  "NDCG@1": 0.0039, "NDCG@5": 0.0101, "NDCG@10": 0.0124, "NDCG@20": 0.0186
}
```

---

## 8. 完整流水线脚本

### 8.1 RQ-VAE 方案（标准流水线）

```bash
#!/bin/bash
set -e

CATEGORY=Beauty
NUM_GPUS=4
MODEL_NAME=Qwen/Qwen2.5-1.5B
BASE_DIR=$(pwd)

echo "=== Step 1: Preprocess ==="
python -m SAEGenRec.data_process preprocess \
    --data_dir=data/raw --output_dir=data/processed --category=$CATEGORY

echo "=== Step 2: Embed ==="
python -m SAEGenRec.sid_builder text2emb \
    --dataset=$CATEGORY --data_dir=data/processed --output_dir=data/processed/emb

echo "=== Step 3: Build SID (RQ-VAE) ==="
python -m SAEGenRec.sid_builder rqvae_train \
    --dataset=$CATEGORY --emb_dir=data/processed/emb --output_dir=data/processed
python -m SAEGenRec.sid_builder generate_indices \
    --dataset=$CATEGORY --emb_dir=data/processed/emb \
    --checkpoint_dir=data/processed --output_dir=data/processed

echo "=== Step 4: Convert Dataset ==="
python -m SAEGenRec.data_process convert_dataset \
    --dataset=$CATEGORY --data_dir=data/processed \
    --index_json=data/processed/$CATEGORY.index.json \
    --inter_dir=data/interim --output_dir=data/processed

echo "=== Step 5: SFT Training (${NUM_GPUS} GPUs) ==="
torchrun --nproc_per_node=$NUM_GPUS \
    -m SAEGenRec.training sft \
    --model_name=$MODEL_NAME \
    --train_csv=data/processed/$CATEGORY.train.csv \
    --valid_csv=data/processed/$CATEGORY.valid.csv \
    --sid_index_path=data/processed/$CATEGORY.index.json \
    --item_meta_path=data/interim/$CATEGORY.item.json \
    --output_dir=models/sft/$CATEGORY \
    --batch_size=512 --micro_batch_size=8 --num_epochs=3 \
    --freeze_llm=False --category=$CATEGORY

echo "=== Step 6: RL Training (${NUM_GPUS} GPUs) ==="
torchrun --nproc_per_node=$NUM_GPUS \
    -m SAEGenRec.training rl \
    --model_path=models/sft/$CATEGORY/final_checkpoint \
    --train_csv=data/processed/$CATEGORY.train.csv \
    --info_file=data/processed/info/$CATEGORY.txt \
    --output_dir=models/rl/$CATEGORY \
    --category=$CATEGORY \
    --train_batch_size=32 --num_generations=8

echo "=== Step 7: Evaluate ==="
# 找到最新 checkpoint
CKPT=$(ls -d models/rl/$CATEGORY/checkpoint-* | sort -V | tail -1)
python -m SAEGenRec.evaluation evaluate \
    --model_path=$CKPT \
    --test_csv=data/processed/$CATEGORY.test.csv \
    --info_file=data/processed/info/$CATEGORY.txt \
    --output_dir=results/$CATEGORY \
    --batch_size=2 --num_beams=50

echo "=== Done! Results in results/$CATEGORY/metrics.json ==="
cat results/$CATEGORY/metrics.json
```

### 8.2 GatedSAE 方案

上述脚本 Step 3-4 替换为：

```bash
echo "=== Step 3: Build SID (GatedSAE K=8) ==="
python -m SAEGenRec.sid_builder gated_sae_train \
    --embedding_path=data/interim/$CATEGORY.emb-all-MiniLM-L6-v2-td.npy \
    --output_dir=models/gated_sae/$CATEGORY \
    --expansion_factor=4 --k=8 --total_training_samples=1000000

python -m SAEGenRec.sid_builder generate_sae_indices \
    --checkpoint=models/gated_sae/$CATEGORY \
    --embedding_path=data/interim/$CATEGORY.emb-all-MiniLM-L6-v2-td.npy \
    --output_path=data/interim/$CATEGORY.sae.index.json --k=8

echo "=== Step 4: Convert Dataset (GatedSAE) ==="
python -m SAEGenRec.data_process convert_dataset \
    --dataset=$CATEGORY --data_dir=data/processed \
    --index_json=data/interim/$CATEGORY.sae.index.json \
    --inter_dir=data/interim --output_dir=data/processed
```

---

## 9. 常用参数参考

### 模型选择建议

| 模型 | 参数量 | 训练速度（4×A100） | 推荐场景 |
|------|--------|-------------------|---------|
| Qwen2.5-0.5B | 494M | 快（~1s/step） | 快速实验 |
| Qwen2.5-1.5B | 1.5B | 中（~2s/step） | 平衡性能 |
| Qwen2.5-7B | 7B | 慢（~6s/step） | 高精度 |
| Qwen3.5-0.8B | 0.8B | 较慢（SSM架构） | 需要 causal-conv1d |

> ⚠️ **Qwen3.5 系列注意**：包含 SSM 线性注意力层，需要 `causal-conv1d` CUDA 内核（RTX 5080/SM12.0 暂不支持）。当前回退到 PyTorch 实现，速度约 8s/step。

### batch_size 选择

```
全局 batch_size = micro_batch_size × gradient_accumulation × num_gpus

推荐配置（4×A100 80G）：
  Qwen2.5-1.5B: micro_batch=8, accumulation=16, 4卡 → 全局512
  Qwen2.5-7B:   micro_batch=2, accumulation=16, 4卡 → 全局128

推荐配置（8×A100 40G）：
  Qwen2.5-1.5B: micro_batch=4, accumulation=16, 8卡 → 全局512
  Qwen2.5-7B:   micro_batch=1, accumulation=16, 8卡 → 全局128 （需ZeRO-3）
```

---

## 10. 已知问题与解决方案

### SFT 训练

| 问题 | 解决方案 |
|------|---------|
| OOM（显存不足） | 减小 `micro_batch_size`；使用 `deepspeed=configs/ds_zero2.json` |
| `group_by_length` 报错 | 已移除，无需处理 |
| `report_to=None` 无效 | 已改为 `report_to="none"` |
| Qwen3.5 SSM 层慢 | 安装 `causal-conv1d`（SM 12.0 暂不支持） |

### 评估

| 问题 | 解决方案 |
|------|---------|
| CUDA illegal memory access | 过滤越界 token ID（已修复） |
| batch 内序列长度不一 | 已手动 left-pad |
| k_values 解析错误 | 已添加 isinstance 检查 |
| 建议 batch_size | 使用 `--batch_size=2`（最稳定） |

### 数据处理

| 问题 | 解决方案 |
|------|---------|
| Amazon 2015 meta 格式 | 已添加 `ast.literal_eval` fallback |
| inter 文件路径 | inter 文件在 `data/interim/`，不是 `data/processed/` |
| convert_dataset 参数名 | 使用 `--index_json`（不是 `--index_file`） |

---

*文档生成时间：2026-03-08*
*基于 SAEGenRec 项目，spec 001-modularize-framework + 002-gated-sae-sid*
