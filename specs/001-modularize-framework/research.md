# Research: Modularize MiniOneRec Framework

**Phase**: 0 | **Date**: 2026-03-07

## R-001: .inter 文件格式设计

**Decision**: 最小化列 `user_id`, `item_id`, `item_asin`, `timestamp`, `rating`，TSV 格式，首行表头。
**Rationale**: `.inter` 是数据预处理（FR-001）与数据转换（FR-006）之间的中间格式。不含 title/SID 信息，由 convert_dataset 阶段通过 `.item.json` 和 `.index.json` 注入。这实现了数据预处理与 SID 构建的完全解耦（Constitution I: Modularity-First）。
**Alternatives considered**:
- 完整 CSV（含 title）: 违反模块间解耦原则，convert_dataset 需做冗余校验
- 与 MiniOneRec CSV 完全相同: MiniOneRec 直接生成最终格式，无中间层

## R-002: Review Embedding 下游消费

**Decision**: 仅在 P1 阶段生成并存储 `.emb-{model}-review.npy`，当前 SFT/RL/评估模块不消费。
**Rationale**: Review embedding 是 MiniOneRec 之外的扩展功能。在初始模块化中保持训练模块与参考实现完全一致（Constitution IV: Behavioral Parity）。预留扩展点供未来添加 review-aware dataset 类或 reward 函数。
**Alternatives considered**:
- 在 SFT dataset 中使用 review embedding 作为额外输入: 破坏行为对等
- 作为新 reward 类型: 增加复杂度，偏离初始模块化范围

## R-003: 多 GPU / DeepSpeed 支持

**Decision**: SFT（FR-008）和 RL（FR-009）支持可选 DeepSpeed 集成，config 作为参数传入，单 GPU 时自动退化。
**Rationale**: MiniOneRec 通过 `deepspeed` 启动训练脚本，使用 ZeRO stage 2。保持一致（Constitution IV）。通过可选参数实现，不影响单 GPU 用户体验。
**Alternatives considered**:
- 仅单 GPU: 限制实际使用场景
- HuggingFace Accelerate 抽象: 增加额外抽象层，偏离参考实现

## R-004: Amazon 数据集版本支持

**Decision**: 初始版本仅支持 Amazon 2015 格式（MiniOneRec `process.py` 使用的格式）。
**Rationale**: 2015 是 MiniOneRec 主脚本使用的格式。2018 格式（`amazon18_data_process.py`）和 2023 格式差异较大，可作为后续迭代。这控制了初始模块化的范围（Constitution V: Incremental Migration）。
**Alternatives considered**:
- 同时支持 2015+2018: 增加 ~30% 工作量，双 parser 维护
- 全版本支持: 范围过大，2023 格式差异根本性

## R-005: CLI 入口设计

**Decision**: `python -m SAEGenRec.{module}` + `fire.Fire()`，同时提供 Makefile `make` targets。
**Rationale**: 与 MiniOneRec 完全一致的 CLI 模式（Constitution IV）。`fire.Fire()` 自动将函数参数映射为 CLI 参数，零额外代码。Makefile targets 提供便捷的高层封装。
**Alternatives considered**:
- click/typer: 更规范但增加依赖和模板代码
- 纯 Python API: 不满足 FR-015 独立执行要求

## R-006: 配置管理架构

**Decision**: 每个 pipeline 阶段定义自己的 `@dataclass` 配置类，集中在 `SAEGenRec/config.py` 中。类别名映射作为 `CATEGORY_MAP` 常量定义一次。
**Rationale**: Constitution III (Config-Driven) 要求。参考实现中类别字典在 sft.py、rl.py、evaluate.py 中重复三次。通过集中配置消除冗余。dataclass 支持序列化、CLI 覆盖、类型检查。
**Alternatives considered**:
- YAML 配置文件: 对于此项目过于重量级，参考实现使用命令行参数
- 分散在各模块: 违反 FR-012

## R-007: RQ-VAE 模型架构迁移

**Decision**: 将 `references/MiniOneRec/rq/` 目录结构完整迁移到 `SAEGenRec/sid_builder/models/`，保持类名和接口一致。训练入口在 `SAEGenRec/sid_builder/rqvae.py`。
**Rationale**: RQ-VAE 架构（RQVAE → ResidualQuantizer → VectorQuantizer → layers）是经过验证的实现。修改架构会破坏行为对等（Constitution IV）。
**Alternatives considered**:
- 重写为更 Pythonic 的实现: 引入行为差异风险
- 直接 import 参考实现: 违反模块化原则

## R-008: 数据目录约定

**Decision**: 遵循 Constitution 数据目录约定：`data/raw/`（原始只读）→ `data/interim/`（.inter, .item.json, .review.json, .npy, .index.json）→ `data/processed/`（CSV + info TXT）→ `models/`（checkpoints）。
**Rationale**: Constitution II (Pipeline Composability) + Cookiecutter Data Science 模板约定。分层目录使每个阶段的输入输出位置明确。
**Alternatives considered**:
- 平铺目录: 文件混杂难以管理
- 每个阶段独立输出目录: 与 Cookiecutter 约定不一致
