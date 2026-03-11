# Implementation Plan: 模块化框架扩展

**Branch**: `003-modular-framework-extension` | **Date**: 2026-03-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-modular-framework-extension/spec.md`

## Summary

扩展 SAEGenRec 框架，增加五个核心能力：(1) 多模态数据处理（图像下载 + 视觉特征提取）；(2) SID 生成统一接口（RQ-VAE/RQ-KMeans/GatedSAE 通过 `--method` 切换）；(3) 可自定义 SFT 任务（JSONL 持久化 + prompt 模板 + 多阶段渐进式训练）；(4) 可自定义 RL 奖励函数（装饰器注册 + 组合奖励）；(5) 训练期间推荐指标评估。所有新增模块遵循现有的 fire CLI + Makefile 命令模式，通过文件接口（.npy、.jsonl、.index.json）保持模块间解耦。

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: PyTorch, HuggingFace Transformers, TRL (GRPO), sentence-transformers, SAELens, FAISS, fire, loguru, Pillow, requests (图像下载), transformers (视觉模型)
**Storage**: 文件系统（NPY、JSONL、JSON、CSV、TXT、JPEG）
**Testing**: pytest
**Target Platform**: Linux (GPU servers, RTX 5080 / multi-GPU)
**Project Type**: CLI/library（Python 包 + Makefile 命令）
**Performance Goals**: 图像下载 ≤ 8 并发、视觉特征提取 GPU 批量处理、SFT 训练与现有速度持平
**Constraints**: 单机 GPU 显存 ≤ 16GB（RTX 5080），磁盘空间需足够存储图像（约 500MB-5GB/类别）
**Scale/Scope**: 数千-数万 items/类别，3 种 SID 方法，4 种内置 SFT 任务，5 种内置 RL 奖励

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Phase 0 Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Modularity-First | ✅ PASS | 每个新模块（image_downloader、visual_embed、prepare_sft、reward registry）独立实现，可单独 import 和测试。模块间仅通过文件格式通信。 |
| II. Pipeline Composability | ✅ PASS | 新增阶段边界均为持久化文件：图像→.jpg、视觉特征→.npy、SFT数据→.jsonl、SID→.index.json。任意阶段可替换。 |
| III. Config-Driven | ✅ PASS | 所有新参数（VISION_MODEL、SID_TYPE、TASK、PROMPT_TEMPLATE、REWARD_TYPE）通过 CLI/dataclass 管理，无硬编码。 |
| IV. Behavioral Parity | ⚠️ JUSTIFIED | 新功能（多模态、JSONL SFT、sid_to_title 任务）在 MiniOneRec 中不存在，属于功能扩展而非重构。现有 RQ-VAE/SFT/RL 流程保持等价行为（FR-026、SC-007）。 |
| V. Incremental Migration | ✅ PASS | 用户故事按 P1→P2→P3 优先级排列，依赖关系清晰：US1(数据)→US2(SID)→US3(SFT)→US4(RL)→US5(评估)。每步独立验证。 |

## Project Structure

### Documentation (this feature)

```text
specs/003-modular-framework-extension/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0: technology research
├── data-model.md        # Phase 1: entity definitions
├── quickstart.md        # Phase 1: integration scenarios
├── contracts/           # Phase 1: CLI interface contracts
│   ├── data_process.md  # download_images, embed_text, extract_visual
│   ├── sid_builder.md   # build_sid (unified), train_sid, generate_sid
│   ├── sft.md           # prepare_sft, sft (JSONL-based)
│   ├── rl.md            # rl (reward registry, prompt template)
│   └── evaluation.md    # evaluate (training-time metrics)
├── checklists/
│   └── requirements.md  # Quality checklist
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
SAEGenRec/
├── config.py                          # 扩展：VisualEmbConfig, PrepSFTConfig dataclass
├── data_process/
│   ├── __main__.py                    # 扩展：注册 download_images, embed_text
│   ├── preprocess.py                  # 现有（不变）
│   ├── convert_dataset.py             # 现有（不变）
│   ├── image_downloader.py            # 新增：并发图像下载
│   └── visual_embed.py                # 新增：视觉特征提取
├── sid_builder/
│   ├── __main__.py                    # 重构：统一 build_sid 入口
│   ├── registry.py                    # 新增：SID 方法注册表
│   ├── base.py                        # 新增：SIDMethod 基类（train + generate 接口）
│   ├── rqvae.py                       # 现有（适配统一接口）
│   ├── rqkmeans.py                    # 现有（适配统一接口）
│   ├── gated_sae.py                   # 现有（适配统一接口）
│   ├── generate_indices.py            # 现有（适配 token_format 参数）
│   ├── generate_sae_indices.py        # 现有（适配 token_format 参数）
│   ├── text2emb.py                    # 现有（重命名为 embed_text 入口，保留别名）
│   └── models/                        # 现有（不变）
├── datasets/
│   ├── sft_datasets.py                # 现有（保留，逻辑移植至 task_registry.py）
│   ├── rl_datasets.py                 # 现有（不变，prompt 模板逻辑在 rl.py 中处理）
│   ├── eval_datasets.py               # 现有（不变）
│   └── task_registry.py               # 新增：SFT 任务注册表（4 种内置 + 扩展机制）
├── training/
│   ├── __main__.py                    # 扩展：prepare_sft 命令注册
│   ├── sft.py                         # 重构：JSONL 输入 + completion-only loss + freeze_llm + checkpoint 续训
│   ├── prepare_sft.py                 # 新增：CSV+index.json → JSONL 预构建
│   ├── rl.py                          # 扩展：prompt_template 参数 + reward registry 查找
│   ├── trainer.py                     # 现有（扩展训练期间评估回调）
│   └── rewards.py                     # 重构：装饰器注册机制 + 组合奖励
├── evaluation/
│   ├── evaluate.py                    # 扩展：支持训练期间调用（子集评估）
│   ├── logit_processor.py             # 现有（不变）
│   ├── metrics.py                     # 现有（不变）
│   └── training_evaluator.py          # 新增：训练期间评估回调（HF Trainer callback）
└── models/
    └── sasrec.py                      # 现有（不变）

templates/                              # 新增：prompt 模板目录
├── sid_seq.txt                        # 序列推荐默认模板
├── item_feat.txt                      # 特征描述默认模板
├── fusion.txt                         # 融合推荐默认模板
└── sid_to_title.txt                   # 反向映射默认模板

tests/
├── test_image_downloader.py           # 新增
├── test_visual_embed.py               # 新增
├── test_sid_registry.py               # 新增
├── test_prepare_sft.py                # 新增
├── test_task_registry.py              # 新增
├── test_reward_registry.py            # 新增
├── test_training_evaluator.py         # 新增
├── test_gated_sae.py                  # 现有
└── test_data.py                       # 现有
```

**Structure Decision**: 遵循现有 SAEGenRec 包结构，在各子模块中新增文件。注册表模式（registry.py、task_registry.py）集中管理扩展点。Prompt 模板作为纯文本文件放在项目根目录 `templates/` 下。

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| IV. Behavioral Parity — 新功能无 MiniOneRec 对应 | 多模态、JSONL SFT、sid_to_title 是功能扩展，非既有功能重构 | 现有 RQ-VAE/SFT/RL 流程保持等价行为，新功能不影响既有行为 |

## Constitution Check — Post-Phase 1 Design Re-evaluation

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Modularity-First | ✅ PASS | 设计验证通过。新模块（image_downloader、visual_embed、prepare_sft、task_registry、rewards registry、training_evaluator）均为独立文件，可单独 import 测试。注册表模式（SID_METHODS、SFT_TASKS、_REWARD_REGISTRY）通过字典查找解耦调用方与实现方。contracts/ 中每个 CLI 命令有明确的输入/输出契约。 |
| II. Pipeline Composability | ✅ PASS | 新增阶段边界均为持久化文件：图像→`.jpg`、视觉特征→`.npy`、SFT数据→`.jsonl`+`meta.json`、SID→`.index.json`。data-model.md 定义了每种文件的 schema 和命名约定。quickstart.md 验证了任意阶段可独立替换。 |
| III. Config-Driven | ✅ PASS | 新增 `VisualEmbConfig`、`PrepSFTConfig` dataclass。所有新参数（VISION_MODEL、SID_TYPE、TASK、PROMPT_TEMPLATE、REWARD_TYPE、FREEZE_LLM、EVAL_REC_*）通过 CLI/dataclass 管理。`meta.json` 记录生成参数支持实验复现。 |
| IV. Behavioral Parity | ⚠️ JUSTIFIED | contracts/ 明确标注了向后兼容：`make embed` 保留为 `embed_text` 别名、`make build_sae_sid` 保留为 `build_sid METHOD=gated_sae` 别名（FR-026）。现有 rewards.py 的 5 个函数仅增加 `@register_reward` 装饰器，内部逻辑不变。SFT 重构后的 completion-only loss 与现有 `labels = [-100] * prompt_len + completion_tokens` 模式一致。 |
| V. Incremental Migration | ✅ PASS | quickstart.md 中 6 个场景验证了渐进式流程。US1(数据)→US2(SID)→US3(SFT)→US4(RL)→US5(评估) 依赖链清晰，每步有独立测试标准。多阶段 SFT 场景验证了阶段间 checkpoint 传递的正确性。 |

## Generated Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| research.md | `specs/003-modular-framework-extension/research.md` | 10 项技术决策（图像下载、视觉提取API、JSONL加载、SID注册表、奖励注册、训练评估回调、模板机制、token格式、checkpoint续训、冻结策略） |
| data-model.md | `specs/003-modular-framework-extension/data-model.md` | 8 个实体定义（EmbeddingFile、ImageDirectory、SIDMethod、SFTTask、SFTDataFile、PromptTemplate、RewardFunction、TrainingEvaluator）及关系图 |
| contracts/data_process.md | `specs/003-modular-framework-extension/contracts/data_process.md` | 3 个 CLI 命令（download_images、embed_text、extract_visual） |
| contracts/sid_builder.md | `specs/003-modular-framework-extension/contracts/sid_builder.md` | 3 个 CLI 命令（build_sid、train_sid、generate_sid）+ 向后兼容映射 |
| contracts/sft.md | `specs/003-modular-framework-extension/contracts/sft.md` | 3 个 CLI 命令（prepare_sft、sft、list_sft_tasks） |
| contracts/rl.md | `specs/003-modular-framework-extension/contracts/rl.md` | 2 个 CLI 命令（rl、list_rewards）+ 奖励注册 API |
| contracts/evaluation.md | `specs/003-modular-framework-extension/contracts/evaluation.md` | 1 个 CLI 命令（evaluate）+ TrainingEvaluator callback 规格 |
| quickstart.md | `specs/003-modular-framework-extension/quickstart.md` | 6 个集成场景（端到端、GatedSAE对比、多阶段SFT、自定义奖励、训练评估、自定义任务） |
