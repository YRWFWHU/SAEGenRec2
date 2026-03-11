# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAEGenRec 是一个基于 Cookiecutter Data Science 模板的 Python 数据科学/机器学习项目。项目名称暗示与稀疏自编码器（SAE）和生成式推荐（GenRec）相关。

## Environment Setup

```bash
# 创建 conda 环境
make create_environment
conda activate SAEGenRec

# 安装依赖
make requirements
```

需要 Python 3.12。使用 `python-dotenv` 加载 `.env` 中的环境变量（不提交到版本控制）。

## Common Commands

```bash
make lint      # 用 ruff 检查代码格式和风格
make format    # 用 ruff 自动格式化代码
make test      # 运行测试 (python -m pytest tests)
make clean     # 删除编译的 Python 文件
```

运行单个测试：
```bash
python -m pytest tests/test_data.py::test_function_name
```

## Code Style

使用 `ruff` 进行 lint 和格式化，行长限制 99 字符，并强制 import 排序（isort 规则）。`SAEGenRec` 包被视为第一方模块。

## Package Structure

```
SAEGenRec/
├── config.py                  # 全局配置：CATEGORY_MAP, set_seed(), dataclass configs
├── data_process/              # 数据预处理 (python -m SAEGenRec.data_process)
│   ├── preprocess.py          # k-core 过滤 + TO/LOO 分割 → .inter/.item.json/.review.json
│   └── convert_dataset.py     # inter → CSV + info TXT (semantic_id \t title \t item_id)
├── sid_builder/               # SID 构建 (python -m SAEGenRec.sid_builder)
│   ├── text2emb.py            # sentence-transformers → .npy embeddings
│   ├── rqvae.py               # RQ-VAE 训练
│   ├── rqkmeans.py            # RQ-Kmeans (FAISS / constrained) 变体
│   ├── generate_indices.py    # RQVAE checkpoint → .index.json ([a_42][b_128][c_7] format)
│   └── models/                # 模型组件 (layers, vq, rq, rqvae)
├── datasets/                  # PyTorch 数据集
│   ├── sft_datasets.py        # SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset
│   ├── rl_datasets.py         # SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset
│   └── eval_datasets.py       # EvalSidDataset
├── training/                  # 训练入口 (python -m SAEGenRec.training)
│   ├── sft.py                 # SFT 训练 (HuggingFace Trainer + TokenExtender)
│   ├── rl.py                  # GRPO RL 训练
│   ├── trainer.py             # ReReTrainer (扩展 trl.GRPOTrainer)
│   └── rewards.py             # rule / ranking / semantic / sasrec reward 函数
├── evaluation/                # 评估 (python -m SAEGenRec.evaluation)
│   ├── evaluate.py            # 约束波束搜索 + HR@K/NDCG@K
│   ├── logit_processor.py     # build_prefix_tree, ConstrainedLogitsProcessor
│   └── metrics.py             # compute_hr_ndcg, hr_at_k, ndcg_at_k
└── models/                    # CF 模型 (python -m SAEGenRec.models)
    └── sasrec.py              # SASRec, GRU, Caser + sasrec_train()
```

## CLI Commands

```bash
# 数据预处理
python -m SAEGenRec.data_process preprocess --data_dir=data/raw --category=Beauty
python -m SAEGenRec.data_process convert_dataset --dataset=Beauty --data_dir=data/processed --index_file=...

# SID 构建
python -m SAEGenRec.sid_builder text2emb --dataset=Beauty --data_dir=data/processed
python -m SAEGenRec.sid_builder rqvae_train --dataset=Beauty --emb_dir=data/processed/emb
python -m SAEGenRec.sid_builder generate_indices --dataset=Beauty

# 训练
python -m SAEGenRec.training sft --model_path=... --train_csv=... --info_file=...
python -m SAEGenRec.training rl  --model_path=... --train_csv=... --info_file=...

# 评估
python -m SAEGenRec.evaluation evaluate --model_path=... --test_csv=... --info_file=...

# CF 模型 (SASRec reward)
python -m SAEGenRec.models sasrec_train --model_type=SASRec --train_csv=...

# Makefile 快捷命令
make preprocess CATEGORY=Beauty
make embed CATEGORY=Beauty
make build_sid CATEGORY=Beauty
make build_sae_sid CATEGORY=Beauty   # GatedSAE 替代方案
make convert CATEGORY=Beauty
make sft MODEL_PATH=models/llm
make rl MODEL_PATH=models/sft
make evaluate RL_MODEL_PATH=models/rl
make pipeline CATEGORY=Beauty MODEL_PATH=models/llm   # 全流程
```

## Data Flow

```
raw Amazon JSON → preprocess → .inter + .item.json + .review.json
                            → text2emb → .npy embeddings
                            → rqvae_train → RQVAE checkpoint
                            → generate_indices → .index.json
                            → convert_dataset → {split}.csv + info/{dataset}.txt
                            → sft → SFT checkpoint
                            → rl  → RL checkpoint
                            → evaluate → predictions.json + metrics.json
```

## SID Token Format

`[a_42][b_128][c_7]` — 3 levels × 256 codebook entries per level.

## Key File Schemas

- `.inter`: TSV — `user_id\titem_id\titem_asin\ttimestamp\trating` (no header)
- `{split}.csv`: `user_id, history_item_sid, target_item_sid, history_item_id, item_id, item_asin`
- `info/{dataset}.txt`: `semantic_id\ttitle\titem_id` (no header)
- `.index.json`: `{"item_asin": "[a_42][b_128][c_7]", ...}`

## Other Directories

- `data/` — 分层数据目录（raw → interim → processed）
- `notebooks/` — Jupyter 笔记本，命名约定：`序号-作者缩写-描述`
- `models/` — 训练好的模型文件
- `reports/figures/` — 生成的图表
- `tests/` — pytest 测试
              
## Build System

使用 `flit` 作为构建后端（`flit_core >=3.2,<4`），通过 `pip install -e .` 以可编辑模式安装。

## Active Technologies
- Python 3.12 + PyTorch, HuggingFace Transformers, TRL (GRPO), DeepSpeed, FAISS, fire, loguru, sentence-transformers (001-modularize-framework)
- SAELens v6.37.6 (GatedTrainingSAE, SAETrainer, SAE.load_from_disk) (002-gated-sae-sid)
- 文件系统（TSV `.inter`, JSON `.item.json`/`.review.json`/`.index.json`, NPY `.npy`, CSV, TXT） (001-modularize-framework)
- Python 3.12 + SAELens (sae-lens), PyTorch, NumPy, loguru, fire (002-gated-sae-sid)
- .npy（输入 embedding）、safetensors + JSON（GatedSAE checkpoint）、JSON（.index.json 输出） (002-gated-sae-sid)
- Python 3.12 + PyTorch, HuggingFace Transformers, TRL (GRPO), sentence-transformers, SAELens, FAISS, fire, loguru, Pillow, requests (图像下载), transformers (视觉模型) (003-modular-framework-extension)
- 文件系统（NPY、JSONL、JSON、CSV、TXT、JPEG） (003-modular-framework-extension)

## Recent Changes
- 001-modularize-framework: Added Python 3.12 + PyTorch, HuggingFace Transformers, TRL (GRPO), DeepSpeed, FAISS, fire, loguru, sentence-transformers
