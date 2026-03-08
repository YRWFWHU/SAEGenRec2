# Quickstart: SAEGenRec Pipeline

## 安装

```bash
# 创建环境
make create_environment
conda activate SAEGenRec

# 安装依赖（可编辑模式）
make requirements
# 或
pip install -e .
```

## 完整流水线（以 Beauty 数据集为例）

### Step 1: 数据预处理

```bash
# 下载原始 Amazon 2015 数据到 data/raw/
# 需要: meta_All_Beauty.json, All_Beauty_5.json

# 运行预处理（TO 划分, k-core=5）
python -m SAEGenRec.data_process preprocess \
  --category All_Beauty \
  --k_core 5 \
  --split_method TO \
  --max_history_len 50 \
  --output_dir data/interim/

# 输出:
#   data/interim/All_Beauty.train.inter
#   data/interim/All_Beauty.valid.inter
#   data/interim/All_Beauty.test.inter
#   data/interim/All_Beauty.item.json
#   data/interim/All_Beauty.review.json
```

### Step 2: 文本嵌入

```bash
python -m SAEGenRec.sid_builder text2emb \
  --item_json data/interim/All_Beauty.item.json \
  --review_json data/interim/All_Beauty.review.json \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --output_dir data/interim/

# 输出:
#   data/interim/All_Beauty.emb-all-MiniLM-L6-v2-td.npy
#   data/interim/All_Beauty.emb-all-MiniLM-L6-v2-review.npy
```

### Step 3: SID 构建 (RQ-VAE)

```bash
# 训练 RQ-VAE
python -m SAEGenRec.sid_builder rqvae_train \
  --embedding_path data/interim/All_Beauty.emb-all-MiniLM-L6-v2-td.npy \
  --num_levels 3 \
  --codebook_size 256 \
  --output_dir models/rqvae/

# 生成 SID 索引
python -m SAEGenRec.sid_builder generate_indices \
  --checkpoint models/rqvae/best.pt \
  --embedding_path data/interim/All_Beauty.emb-all-MiniLM-L6-v2-td.npy \
  --output_path data/interim/All_Beauty.index.json
```

### Step 4: 数据集转换

```bash
python -m SAEGenRec.data_process convert_dataset \
  --inter_dir data/interim/ \
  --item_json data/interim/All_Beauty.item.json \
  --index_json data/interim/All_Beauty.index.json \
  --dataset All_Beauty \
  --output_dir data/processed/

# 输出:
#   data/processed/All_Beauty.train.csv
#   data/processed/All_Beauty.valid.csv
#   data/processed/All_Beauty.test.csv
#   data/processed/info/All_Beauty.txt
```

### Step 5: SFT 训练

```bash
python -m SAEGenRec.training sft \
  --model_name Qwen/Qwen2.5-0.5B \
  --train_csv data/processed/All_Beauty.train.csv \
  --valid_csv data/processed/All_Beauty.valid.csv \
  --info_file data/processed/info/All_Beauty.txt \
  --output_dir models/sft/ \
  --epochs 5 \
  --batch_size 8

# 可选: 使用 DeepSpeed
# deepspeed --num_gpus 4 -m SAEGenRec.training sft ... --deepspeed ds_config.json
```

### Step 6: RL 训练 (可选)

```bash
python -m SAEGenRec.training rl \
  --model_path models/sft/best/ \
  --train_csv data/processed/All_Beauty.train.csv \
  --info_file data/processed/info/All_Beauty.txt \
  --reward_type rule \
  --output_dir models/rl/
```

### Step 7: 评估

```bash
python -m SAEGenRec.evaluation evaluate \
  --model_path models/sft/best/ \
  --test_csv data/processed/All_Beauty.test.csv \
  --info_file data/processed/info/All_Beauty.txt \
  --num_beams 50

# 输出: HR@{1,3,5,10,20}, NDCG@{1,3,5,10,20}
```

## Make 命令

```bash
make preprocess CATEGORY=All_Beauty    # Step 1
make embed CATEGORY=All_Beauty         # Step 2
make build_sid CATEGORY=All_Beauty     # Step 3
make convert CATEGORY=All_Beauty       # Step 4
make sft CATEGORY=All_Beauty           # Step 5
make rl CATEGORY=All_Beauty            # Step 6
make evaluate CATEGORY=All_Beauty      # Step 7
make pipeline CATEGORY=All_Beauty      # Steps 1-7 一键执行
```

## 开发与测试

```bash
make test          # 运行所有测试
make lint          # ruff 代码检查
make format        # ruff 自动格式化

# 运行单个测试
python -m pytest tests/test_data_process.py -v
```
