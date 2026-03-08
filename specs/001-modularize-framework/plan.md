# Implementation Plan: Modularize MiniOneRec Framework

**Branch**: `001-modularize-framework` | **Date**: 2026-03-07 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-modularize-framework/spec.md`

## Summary

将 MiniOneRec 参考实现模块化为 `SAEGenRec` Python 包，覆盖完整的生成式推荐流水线：数据预处理 → 文本嵌入 → SID 构建（RQ-VAE / RQ-Kmeans） → 数据集转换 → SFT 训练 → RL 训练（GRPO） → 约束波束搜索评估。保持与参考实现完全一致的行为，同时实现模块间解耦、配置驱动、独立可执行。

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: PyTorch, HuggingFace Transformers, TRL (GRPO), DeepSpeed, FAISS, fire, loguru, sentence-transformers
**Storage**: 文件系统（TSV `.inter`, JSON `.item.json`/`.review.json`/`.index.json`, NPY `.npy`, CSV, TXT）
**Testing**: pytest (`python -m pytest tests`)
**Target Platform**: Linux (GPU server, CUDA)
**Project Type**: library + CLI
**Performance Goals**: 与 MiniOneRec 参考实现等价（相同硬件上相同运行时间量级）
**Constraints**: 行为对等（SC-001: 相同输入+种子 → 相同输出, 1e-6 容差）
**Scale/Scope**: Amazon Beauty 全品类 ~300K 交互，Industrial ~5K items；支持单 GPU 和多 GPU (DeepSpeed)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Modularity-First | PASS | 每个 pipeline 阶段为独立子模块（`data_process/`, `sid_builder/`, `datasets/`, `training/`, `evaluation/`, `models/`），模块间仅通过文件格式通信 |
| II. Pipeline Composability | PASS | 阶段边界为持久化文件（.inter, .item.json, .review.json, .npy, .index.json, CSV, checkpoint），任意阶段可替换 |
| III. Config-Driven | PASS | FR-014 要求所有参数通过 dataclass 暴露，FR-012 集中类别映射 |
| IV. Behavioral Parity | PASS | SC-001 要求等价输出，SC-006 禁止修改参考目录 |
| V. Incremental Migration | PASS | 6 阶段渐进推进，每步独立验证 |

**Gate Result**: ALL PASS — proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/001-modularize-framework/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── inter-format.md
│   ├── item-json.md
│   ├── review-json.md
│   ├── index-json.md
│   ├── training-csv.md
│   └── info-txt.md
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
SAEGenRec/
├── __init__.py
├── config.py                  # FR-012/013/014: 集中配置、类别映射、set_seed()
├── data_process/
│   ├── __init__.py
│   ├── __main__.py            # fire.Fire() CLI 入口
│   ├── preprocess.py          # FR-001: k-core 过滤、TO/LOO 划分、.inter/.item.json/.review.json
│   └── convert_dataset.py     # FR-006: .inter + .item.json + .index.json → CSV + info TXT
├── sid_builder/
│   ├── __init__.py
│   ├── __main__.py            # fire.Fire() CLI 入口
│   ├── text2emb.py            # FR-002: 文本 → .npy 嵌入
│   ├── rqvae.py               # FR-003: RQ-VAE 训练入口
│   ├── rqkmeans.py            # FR-004: RQ-Kmeans 变体
│   ├── generate_indices.py    # FR-005: 量化器 → .index.json
│   └── models/
│       ├── __init__.py
│       ├── rqvae.py           # RQVAE 模型架构
│       ├── rq.py              # ResidualQuantizer
│       ├── vq.py              # VectorQuantizer
│       └── layers.py          # kmeans, sinkhorn_algorithm
├── datasets/
│   ├── __init__.py
│   ├── base.py                # BaseDataset, CSVBaseDataset
│   ├── sft_datasets.py        # FR-007: SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset
│   ├── rl_datasets.py         # FR-007: SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset
│   └── eval_datasets.py       # FR-007: EvalSidDataset
├── training/
│   ├── __init__.py
│   ├── __main__.py            # fire.Fire() CLI 入口
│   ├── sft.py                 # FR-008: SFT 训练 + TokenExtender + DeepSpeed
│   ├── rl.py                  # FR-009: GRPO RL 训练
│   ├── trainer.py             # ReReTrainer (extends GRPOTrainer)
│   └── rewards.py             # 4 种 reward 函数 (rule, ranking, semantic, sasrec)
├── evaluation/
│   ├── __init__.py
│   ├── __main__.py            # fire.Fire() CLI 入口
│   ├── evaluate.py            # FR-010: 约束波束搜索评估
│   ├── logit_processor.py     # ConstrainedLogitsProcessor
│   └── metrics.py             # HR@K, NDCG@K 计算
└── models/
    ├── __init__.py
    └── sasrec.py              # FR-011: SASRec, GRU, Caser

tests/
├── test_config.py
├── test_data_process.py
├── test_sid_builder.py
├── test_datasets.py
├── test_training.py
└── test_evaluation.py
```

**Structure Decision**: 采用单包多子模块结构，每个 pipeline 阶段为一个子包（子目录），内含 `__main__.py` 支持 `python -m SAEGenRec.{subpackage}` 调用。模型定义（RQ-VAE 架构、SASRec）与训练逻辑分离。

## Complexity Tracking

> No constitution violations detected. Table intentionally left empty.

## Post-Phase 1 Constitution Re-check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Modularity-First | PASS | 6 个独立子包，各含 `__main__.py`，无循环依赖 |
| II. Pipeline Composability | PASS | 6 个文件格式 contract 已定义（contracts/），阶段边界明确 |
| III. Config-Driven | PASS | `config.py` 集中所有 dataclass + CATEGORY_MAP |
| IV. Behavioral Parity | PASS | 模型架构完整迁移，prompt 模板/tokenization 逻辑一致 |
| V. Incremental Migration | PASS | 项目结构按 6 阶段组织，支持渐进实现 |

**Gate Result**: ALL PASS — ready for `/speckit.tasks`.
