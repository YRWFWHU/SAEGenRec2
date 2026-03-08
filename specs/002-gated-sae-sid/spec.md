# Feature Specification: GatedSAE SID 生成

**Feature Branch**: `002-gated-sae-sid`
**Created**: 2026-03-08
**Status**: Draft
**Input**: 设计新的 SID 生成方式，使用 GatedSAE 将 text embedding 转换为稀疏特征，选择激活值最大的 K 个特征作为 SID

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 训练 GatedSAE 模型 (Priority: P1)

研究者在已有 item text embedding（.npy 文件）的基础上，训练一个 GatedSAE 模型，将稠密嵌入转换为稀疏特征表示。训练完成后，GatedSAE checkpoint 被保存到指定目录，供后续 SID 生成使用。

**Why this priority**: GatedSAE 模型是整个新 SID 生成方案的核心组件，必须先有可用的模型才能进行后续步骤。

**Independent Test**: 可通过在 Beauty 数据集的 embedding 上运行训练命令来独立测试，验证模型能收敛且 checkpoint 正确保存。

**Acceptance Scenarios**:

1. **Given** 已有 item text embedding 文件（.npy 格式），**When** 研究者运行 GatedSAE 训练命令，**Then** 系统训练 GatedSAE 模型并保存 checkpoint 到指定目录
2. **Given** GatedSAE 训练完成，**When** 研究者查看训练日志，**Then** 可以看到重构损失和稀疏性指标随训练逐步改善
3. **Given** 训练好的 GatedSAE checkpoint，**When** 研究者重新加载模型，**Then** 模型可以正确对输入 embedding 进行编码和解码

---

### User Story 2 - 生成 SAE-based SID (Priority: P1)

研究者使用训练好的 GatedSAE 模型，对所有 item embedding 进行编码，选择每个 item 激活值最大的 K 个特征索引作为 SID token，生成 .index.json 文件。该文件格式与现有 RQ-VAE 生成的 .index.json 兼容，可直接用于下游 SFT/RL 训练。

**Why this priority**: SID 生成是连接 GatedSAE 与下游训练流程的关键步骤，与训练同为 P1。

**Independent Test**: 使用训练好的 GatedSAE checkpoint 对 Beauty 数据集生成 .index.json，验证每个 item 都获得了由 K 个特征索引组成的 SID。

**Acceptance Scenarios**:

1. **Given** 训练好的 GatedSAE 模型和 item embedding 文件，**When** 研究者运行 SID 生成命令，**Then** 系统生成 .index.json 文件，其中每个 item 对应一个 K-token SID
2. **Given** 生成的 .index.json 文件，**When** 查看 SID 格式，**Then** 每个 SID 由 K 个 token 组成（默认 K=8，格式如 `[a_x][b_y]...[h_z]`），其中 token 按激活值从大到小排列
3. **Given** 同一数据集的所有 item，**When** 生成 SID 后检查唯一性，**Then** 不存在两个不同 item 拥有完全相同的 SID

---

### User Story 3 - 端到端流水线集成 (Priority: P2)

研究者可以通过 CLI 命令或 Makefile 一键运行 GatedSAE SID 生成流程（训练 + 生成），作为 RQ-VAE 的替代选项。生成的 SID 可无缝接入后续的 convert_dataset → SFT → RL → 评估流程。

**Why this priority**: 集成到流水线使功能完整可用，但核心算法已在 P1 完成。

**Independent Test**: 从 embedding 文件出发，执行 GatedSAE 训练 + SID 生成 + convert_dataset，验证输出 CSV 和 info 文件格式正确。

**Acceptance Scenarios**:

1. **Given** 已有 embedding 文件，**When** 研究者运行 GatedSAE SID 生成流水线命令，**Then** 系统依次完成 GatedSAE 训练和 SID 生成
2. **Given** GatedSAE 生成的 .index.json，**When** 运行 convert_dataset，**Then** 输出的 CSV 和 info 文件格式与 RQ-VAE 生成的完全一致，可直接用于 SFT 训练

---

### User Story 4 - 对比实验 (Priority: P3)

研究者可以在相同数据集上分别使用 RQ-VAE 和 GatedSAE 生成 SID，然后走完完整的 SFT → RL → 评估流程，对比两种方法在推荐指标（HR@K, NDCG@K）上的表现。

**Why this priority**: 对比实验是验证新方法有效性的必要手段，但依赖前面的功能全部就绪。

**Independent Test**: 在 Beauty 数据集上分别用两种 SID 方法训练 SFT 模型，评估并对比指标。

**Acceptance Scenarios**:

1. **Given** 同一数据集分别用 RQ-VAE 和 GatedSAE 生成的 SID，**When** 在相同 LLM 上完成 SFT 训练和评估，**Then** 可以获得两组可比较的 HR@K 和 NDCG@K 指标

---

### Edge Cases

- 当某些 item embedding 存在 NaN 或 Inf 值时，GatedSAE 应能正确处理（跳过或填零）
- 当多个 item 的 top-K 特征完全相同时（SID 冲突），系统需有去重策略
- 当 K 值大于实际非零激活特征数时，应使用所有非零特征并用零填充剩余位置
- 当字典大小远大于 item 数量时，部分特征可能从未被激活为 top-K，属于正常现象

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统 MUST 基于 SAELens 库的 GatedSAE 实现，在 item text embedding 上训练 GatedSAE 模型，学习将稠密向量映射为稀疏特征表示
- **FR-002**: 系统 MUST 支持配置 GatedSAE 的关键超参数：字典大小（特征数量）、输入维度、训练轮数、学习率、稀疏性权重
- **FR-003**: 系统 MUST 在训练过程中记录并输出关键指标：重构损失、稀疏性（L0 范数）、dead neuron 比例
- **FR-004**: 系统 MUST 能从训练好的 GatedSAE checkpoint 对所有 item embedding 进行编码，提取每个 item 激活值最大的 K 个特征索引
- **FR-005**: 系统 MUST 将 top-K 特征索引转换为与现有 SID 格式兼容的 token 字符串（如 K=8 时 `[a_x][b_y][c_z][d_w][e_v][f_u][g_t][h_s]`），其中每个位置的 token 按激活值降序排列
- **FR-006**: 系统 MUST 输出 .index.json 文件，格式与 RQ-VAE 生成的一致：`{"item_id": ["[a_x]", "[b_y]", ...], ...}`（key 为 item_id 字符串，value 为 K 个 token 字符串的列表）
- **FR-007**: 系统 MUST 在生成 SID 时检测并处理冲突（不同 item 产生相同 SID），提供去重机制
- **FR-008**: 系统 MUST 提供 CLI 入口，通过 `python -m SAEGenRec.sid_builder` 统一调用 GatedSAE 训练和 SID 生成
- **FR-009**: 系统 MUST 保存 GatedSAE 训练 checkpoint（含模型权重和配置），支持后续加载复用
- **FR-010**: 系统 MUST 支持任意 K 值配置，默认 K=8。系统自动生成对应数量的位置前缀 token（K=8 时使用 a/b/c/d/e/f/g/h）。下游组件（TokenExtender、build_prefix_tree、_parse_sid_sequence）预期天然兼容任意 K 值，实施时需验证此兼容性

### Key Entities

- **GatedSAE 模型**: 门控稀疏自编码器，包含编码器（含门控机制）和解码器，将 d_in 维稠密 embedding 映射为 d_sae 维稀疏特征
- **稀疏特征激活**: 每个 item 通过 GatedSAE 编码后的稀疏激活向量，大部分维度为零，非零维度代表该 item 的显著语义特征
- **SID (Semantic ID)**: 由 top-K 激活特征的索引组成的结构化标识符，用于在 LLM 中表示 item
- **GatedSAE 配置**: 训练和推理相关的超参数集合，包括字典大小、输入维度、K 值、稀疏性权重等

## Clarifications

### Session 2026-03-08

- Q: GatedSAE 实现来源（自行实现 vs 复用 SAELens）？ → A: 基于 SAELens 库，引入 SAELens 作为依赖或直接复用其 GatedSAE 模块，仅编写 SID 生成逻辑
- Q: 字典大小（特征数量）应为多少？ → A: 可配置，默认 4x 输入维度（如 384→1536），由研究者根据实验调整
- Q: SAE top-K 特征缺乏 RQ-VAE 的层次语义，是否需要额外处理？ → A: 接受差异，按激活值降序排列即可，不额外引入层次语义，前缀树照常使用

## Assumptions

- 输入 embedding 来自现有 text2emb 步骤，格式为 .npy 文件（与 RQ-VAE 共用相同数据源）
- GatedSAE 字典大小可配置，默认为输入维度的 4 倍（如 384 维输入 → 1536 个特征），研究者可根据实验调整
- K 默认为 8，生成的 SID token 使用位置前缀 `[a_x][b_y]...[h_z]` 以兼容下游前缀树约束解码
- top-K 特征按激活值降序排列：第 1 位（a）为最强特征，依次递减至第 K 位。此排序不具备 RQ-VAE 的粗→细层次语义，属于已接受的设计差异
- GatedSAE 训练在单 GPU 上完成，不需要分布式训练支持
- SID 冲突去重策略与现有 RQ-VAE generate_indices 中的去重逻辑类似

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: GatedSAE 训练在 Beauty 数据集上可在 30 分钟内完成，重构误差（MSE）低于原始 embedding 方差的 10%
- **SC-002**: 生成的 SID 覆盖 100% 的 item（每个 item 都获得唯一 SID），SID 冲突率低于 5%
- **SC-003**: 使用 GatedSAE SID 训练的 SFT 模型在推荐指标（HR@10, NDCG@10）上与 RQ-VAE 基线的差距不超过 20%
- **SC-004**: 端到端流水线（GatedSAE 训练 + SID 生成 + convert_dataset）可通过单条 CLI 命令完成，无需手动干预
- **SC-005**: GatedSAE 编码后的稀疏特征平均 L0 范数不超过 K 的 3 倍（即大部分激活集中在少量特征上）
