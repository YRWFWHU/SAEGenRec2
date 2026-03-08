<!--
SYNC IMPACT REPORT
==================
Version change: N/A (initial fill) → 1.0.0
Modified principles: N/A (first ratification)
Added sections:
  - Core Principles (I–V)
  - Technology Stack Constraints
  - Modularization Workflow
  - Governance
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ no changes required (generic)
  - .specify/templates/spec-template.md ✅ no changes required
  - .specify/templates/tasks-template.md ✅ no changes required
Deferred TODOs: none
-->

# SAEGenRec Constitution

## Core Principles

### I. Modularity-First

每个流水线阶段（数据预处理、SID 构建、数据集、SFT、RL、评估）MUST 作为独立的
Python 子模块实现于 `SAEGenRec/` 包内，可单独 `import` 和调用，无需依赖其他阶段的
运行时状态。

- 模块间通信只能通过明确定义的文件格式（CSV、JSON、NPY）或 Python 函数接口，
  不得通过共享全局变量或隐式副作用传递状态。
- 每个模块 MUST 可在不启动完整训练流水线的前提下独立运行和测试。
- 禁止跨模块循环依赖；依赖方向须与流水线顺序一致。

**Rationale**: MiniOneRec 参考实现的所有逻辑集中在少数脚本中，模块化是本项目的核心
价值，确保各组件可独立替换（如用 RQ-Kmeans 替换 RQ-VAE）。

### II. Pipeline Composability

流水线各阶段的输入输出格式 MUST 显式文档化，且阶段边界 MUST 是持久化文件
（不依赖内存对象在阶段间传递）：

- 数据预处理 → `{dataset}.inter` + `{dataset}.item.json` + `{dataset}.review.json`
- 文本编码 → `{dataset}.emb-{model}-td.npy` + `{dataset}.emb-{model}-review.npy`
- SID 构建 → `{dataset}.index.json`
- 数据转换 → `train/valid/test CSV` + `info/{dataset}.txt`
- SFT/RL 训练 → HuggingFace 兼容 checkpoint 目录

任意阶段 MUST 可被替代实现替换，只要其输出格式符合上述约定。

**Rationale**: 文件边界隔离使每个阶段可独立重跑、调试和替换，符合数据科学实验的
迭代习惯。

### III. Config-Driven

所有超参数、路径和实验设置 MUST 通过 Python `dataclass` 或 YAML 配置文件管理，
不得在业务逻辑代码中硬编码（路径字符串、学习率、数据集类别名等）。

- 类别名映射（如 `"Industrial_and_Scientific"` → `"industrial and scientific items"`）
  MUST 集中在配置层，不得在多个模块中重复定义。
- 命令行参数 MUST 通过配置 dataclass 传入，支持覆盖默认值。
- 配置对象 MUST 可序列化（便于实验复现）。

**Rationale**: MiniOneRec 中类别字典在 `sft.py`、`rl.py`、`evaluate.py` 中重复三次，
配置驱动消除此类冗余并降低维护成本。

### IV. Behavioral Parity with MiniOneRec

模块化实现的每个阶段 MUST 与 `references/MiniOneRec/` 中对应脚本产生等价结果
（在相同输入、相同随机种子下）：

- 新模块与参考实现的行为差异 MUST 在 spec 中显式说明并有充分理由。
- 随机种子 MUST 通过统一的 `set_seed()` 工具函数设置，覆盖 `random`、`numpy`、
  `torch`。
- 数据处理步骤 MUST 是幂等的：对相同输入多次运行产生相同输出。
- `references/MiniOneRec/` 目录为只读参考，MUST NOT 修改。

**Rationale**: 作为参考实现的模块化版本，行为一致性是验证正确性的唯一基准。

### V. Incremental Migration

模块化工作 MUST 按以下顺序渐进推进，每步完成后独立验证再进入下一步：

1. 数据预处理（`SAEGenRec/data_process/`）
2. 文本编码与 SID 构建（`SAEGenRec/sid_builder/`：RQ-VAE / RQ-Kmeans 变体）
3. 数据集类（`SAEGenRec/datasets/`）
4. SFT 训练（`SAEGenRec/training/sft.py`）
5. RL 训练（`SAEGenRec/training/rl.py`）
6. 评估（`SAEGenRec/evaluation/`）

不得跳跃阶段或同时重构多个阶段，以控制回归风险。

**Rationale**: 渐进迁移允许在任意中间状态停止，并确保每个已完成模块的正确性在
后续工作开始前已得到验证。

## Technology Stack Constraints

本项目技术栈 MUST 与 MiniOneRec 参考实现保持兼容：

- **Python**: 3.12（项目环境），参考实现为 3.11，接口设计须兼容两者
- **深度学习框架**: PyTorch + HuggingFace Transformers + TRL（GRPO）
- **数据格式约定**:
  - 交互数据：TSV（`.inter`，首行为表头）
  - 物品元数据：JSON（`.item.json`，key 为字符串 item_id）
  - 物品嵌入：NumPy NPY（`.emb-{model}-td.npy`）
  - SID 索引：JSON（`.index.json`，key 为字符串 item_id，value 为 token 列表）
  - 训练数据：CSV（含 `history_item_sid`、`target_item_sid`、`history_item_title`、
    `target_item_title`、`history_item_id`、`target_item_id`、`user_id` 字段）
  - 物品信息文件：制表符分隔 TXT（`semantic_id\titem_title\titem_id`）
- **包管理**: `flit`，可编辑安装（`pip install -e .`）
- **Lint/Format**: `ruff`（行长 99，isort，`SAEGenRec` 为第一方模块）

引入新的核心依赖前 MUST 在对应 spec 中说明必要性。

## Modularization Workflow

### 模块完成标准（Definition of Done）

一个模块被认为"完成"当且仅当满足全部条件：

1. 代码位于 `SAEGenRec/` 包的对应子模块中，可通过 `from SAEGenRec.xxx import ...` 导入
2. 配置参数通过 dataclass 定义，无硬编码路径或超参数
3. 与参考实现产生等价输出（有可运行的对比验证脚本或测试用例）
4. `tests/` 目录有对应的基础测试

### 数据目录约定

```
data/raw/          # 原始数据（只读）
data/interim/      # 中间结果（.inter, .item.json, .npy, .index.json）
data/processed/    # 最终训练数据（CSV + info TXT）
models/            # SID 量化器 checkpoint 和 LLM checkpoint
```

### SID Token 格式

SID 由多层量化 token 拼接而成，每个 token 带方括号，如 `[A][B][C]`。
token 通过 `TokenExtender` 加入 LLM 词表。此格式为项目核心数据契约，不得随意变更。

## Governance

本 Constitution 是 SAEGenRec 项目所有开发决策的最高准则，优先于任何临时约定。

- 修改本文件 MUST 通过 `speckit.constitution` 命令并按语义版本规则更新版本号。
- MAJOR：原则删除或根本性重定义（需显式迁移说明）。
- MINOR：新增原则或章节（需在 Sync Impact Report 中说明影响范围）。
- PATCH：措辞修正、格式调整（可直接提交）。
- 所有功能 spec 和 plan MUST 在 "Constitution Check" 节中验证符合五项核心原则。
- 运行时开发指引参见 `CLAUDE.md`。

**Version**: 1.0.0 | **Ratified**: 2026-03-07 | **Last Amended**: 2026-03-07
