# Implementation Plan: GatedSAE SID 生成

**Branch**: `002-gated-sae-sid` | **Date**: 2026-03-08 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-gated-sae-sid/spec.md`

## Summary

在现有 SID 构建流程中新增基于 GatedSAE 的 SID 生成方式，作为 RQ-VAE 的替代选项。使用 SAELens 库的 GatedSAE 实现将 item text embedding 编码为稀疏特征，选择激活值最大的 K 个特征索引作为 SID token。输出格式与 RQ-VAE 完全兼容，可直接接入下游 SFT/RL/评估流程。

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: SAELens (sae-lens), PyTorch, NumPy, loguru, fire
**Storage**: .npy（输入 embedding）、safetensors + JSON（GatedSAE checkpoint）、JSON（.index.json 输出）
**Testing**: pytest
**Target Platform**: Linux (单 GPU, RTX 5080 16GB)
**Project Type**: CLI / library
**Performance Goals**: Beauty 数据集（~12K items）GatedSAE 训练 < 30 分钟，SID 生成 < 1 分钟
**Constraints**: 16GB VRAM，字典大小默认 4x 输入维度
**Scale/Scope**: 10K-100K items 级别的推荐数据集

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Pre-design | Post-design | Notes |
|-----------|-----------|-------------|-------|
| I. Modularity-First | PASS | PASS | 新模块 `gated_sae.py` 和 `generate_sae_indices.py` 独立于其他阶段，可单独 import 和调用 |
| II. Pipeline Composability | PASS | PASS | 输出 .index.json 格式与 RQ-VAE 一致，下游模块无需任何修改 |
| III. Config-Driven | PASS | PASS | 新增 `GatedSAEConfig` dataclass，所有超参数通过配置管理 |
| IV. Behavioral Parity | N/A | N/A | 新功能，非参考实现的模块化重构 |
| V. Incremental Migration | PASS | PASS | 添加新文件，不修改现有 SID 构建模块 |

## Project Structure

### Documentation (this feature)

```text
specs/002-gated-sae-sid/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0: research decisions
├── data-model.md        # Phase 1: entity definitions
├── quickstart.md        # Phase 1: usage guide
├── contracts/
│   └── cli.md           # CLI command contracts
└── checklists/
    └── requirements.md  # Spec quality checklist
```

### Source Code (repository root)

```text
SAEGenRec/
├── config.py                          # [MODIFY] 新增 GatedSAEConfig dataclass
└── sid_builder/
    ├── __main__.py                    # [MODIFY] 注册 gated_sae_train + generate_sae_indices 命令
    ├── gated_sae.py                   # [NEW] GatedSAE 训练：NpyDataProvider + SAELens SAETrainer 包装
    └── generate_sae_indices.py        # [NEW] SAE-based SID 生成：encode → top-K → dedup → .index.json

tests/
└── test_gated_sae.py                  # [NEW] GatedSAE 训练和 SID 生成测试

Makefile                               # [MODIFY] 新增 build_sae_sid target
pyproject.toml                         # [MODIFY] 添加 sae-lens 依赖
```

**Structure Decision**: 遵循现有 `sid_builder/` 模块结构，新增两个文件分别负责训练和 SID 生成，与现有 `rqvae.py` + `generate_indices.py` 对应。

## Key Implementation Details

### 1. NpyDataProvider（数据适配层）

SAELens `SAETrainer` 接受 `DataProvider = Iterator[torch.Tensor]`。创建 `NpyDataProvider` 将 .npy 文件包装为批次迭代器：

- 加载 .npy 到内存（item embedding 通常 < 1GB）
- 每次 `__next__` 返回 `(batch_size, d_in)` 的 tensor
- 支持多 epoch 循环（按 `total_training_samples` 控制）
- 每 epoch 随机打乱顺序

### 2. GatedSAE 训练流程

直接使用 SAELens 组件：
- 创建 `GatedTrainingSAEConfig(d_in=d_in, d_sae=d_in*expansion_factor, l1_coefficient=...)`
- 实例化 `GatedTrainingSAE`
- 创建 `SAETrainer(sae=sae, data_provider=npy_provider, cfg=trainer_cfg)`
- 调用 `trainer.fit()` 训练
- 使用 SAELens 的 `save_inference_model()` 保存推理模型
- 额外保存 `training_config.json`（含 k, expansion_factor 等项目特定参数）

### 3. SAE-based SID 生成

- 加载 GatedSAE checkpoint（`SAE.load_from_disk()`）
- 批量编码所有 item embedding：`feature_acts = sae.encode(batch)`
- 对每个 item 取 top-K：`torch.topk(feature_acts, k=K)`
- 转换为 token 字符串：`[{chr(97+pos)}_{feature_index}]`
- 去碰撞：冲突 item 尝试 top-(K+1), top-(K+2) 特征替换最后位置
- 输出 .index.json
- 默认 K=8，SID 由 8 个 token 组成（a-h 前缀）

### 4. Token 兼容性

SID token 格式 `[a_x]` 中 x 的范围从 `[0, 255]`（RQ-VAE）扩展为 `[0, d_sae-1]`（GatedSAE）。这不影响下游组件：
- `TokenExtender` 动态从 .index.json 提取唯一 token
- `build_prefix_tree` 动态构建前缀树
- `_parse_sid_sequence` 正则 `(\[[a-z]_\d+\])+` 已兼容任意数字

## Complexity Tracking

无 Constitution 违规，无需记录。
