# Feature Specification: Strong SAE-GenRec Beauty Baseline

**Feature Branch**: `004-sae-beauty-baseline`
**Created**: 2026-03-11
**Status**: Draft
**Input**: Strong SAE-GenRec baseline experiment on Beauty dataset: LOO split, max_history=20, HR@10 target=0.8, SAE K=6 ef=4x l1=0.2, LLM=Qwen3.5-0.8B, embedding=Qwen3-Embedding-0.6B, beam_size=30, eval every epoch

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 端到端实验流水线执行 (Priority: P1)

研究人员运行完整实验流水线：从 Beauty 原始数据，经过嵌入、SAE 训练、SID 生成、SFT 训练，最终在测试集上取得推荐指标。每个阶段产出可复现的 checkpoint 和日志。

**Why this priority**: 这是实验的核心目标，其他一切都为此服务。

**Independent Test**: 执行完整流水线后，`results/` 目录中存在 `metrics.json`，HR@10 字段有值。

**Acceptance Scenarios**:

1. **Given** Beauty 原始数据已就位，**When** 执行全流水线脚本，**Then** 所有阶段依次完成，无致命错误，最终输出 HR@K / NDCG@K 指标。
2. **Given** 某阶段产出已缓存，**When** 跳过该阶段重新运行，**Then** 后续阶段正常消费缓存结果。
3. **Given** 流水线完成，**When** 检查日志，**Then** 每个阶段的关键参数（K=6、ef=4、l1=0.2、beam_size=30 等）均有记录。

---

### User Story 2 - 每 Epoch 训练与推荐指标双重评估 (Priority: P1)

每完成一个训练 epoch，自动触发两类评估：(a) 训练集上的损失 / 困惑度，(b) 验证集上的 HR@K / NDCG@K，结果写入日志，供实验对比使用。

**Why this priority**: 需要观测训练动态，判断何时停止以及超参数是否合理。

**Independent Test**: 训练日志中每个 epoch 均有两类评估结果。

**Acceptance Scenarios**:

1. **Given** SFT 训练开始，**When** 第 1 个 epoch 结束，**Then** 日志中同时出现该 epoch 的训练损失和 HR@10。
2. **Given** 多 epoch 训练，**When** 训练完成，**Then** 可绘制 HR@10 随 epoch 的变化曲线。

---

### User Story 3 - 实验过程记录与改动追踪 (Priority: P2)

每次实验（超参数变更、模型替换、数据配置调整）均有结构化记录，包含实验编号、改动说明、关键指标，便于后续复盘和论文写作。

**Why this priority**: 研究过程中需要对比多次实验，记录是理解改进方向的依据。

**Independent Test**: `roadmap/experiments/` 目录下存在本次实验记录文件，包含参数和结果。

**Acceptance Scenarios**:

1. **Given** 实验完成，**When** 查看实验记录，**Then** 能找到每次改动的说明和对应指标变化。
2. **Given** 需要复现某次实验，**When** 按记录中的参数重跑，**Then** 结果在允许误差内一致。

---

### Edge Cases

- 当 Qwen3-Embedding-0.6B 的嵌入维度与 SAE 期望输入维度不匹配时，流水线应报错并提示维度。
- 当物品数量超出前缀树容量时，约束 beam search 应正常降级而不崩溃。
- 当某用户历史长度不足 1 时，数据集应跳过该用户，不影响整体评估。
- 当 beam_size=30 导致 OOM 时，应有清晰的错误提示和降低 beam_size 的建议。
- 当 SAE K=6 生成的 SID 碰撞率过高时，流水线应报告碰撞统计并触发去重迭代。

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 数据预处理 MUST 使用 LOO（Leave-One-Out）划分，历史序列截断为最多 20 个物品。
- **FR-002**: 文本嵌入 MUST 使用 Qwen3-Embedding-0.6B 模型生成物品嵌入。
- **FR-003**: GatedSAE 训练 MUST 使用 K=6、expansion_factor=4、l1_coefficient=0.2 配置。
- **FR-004**: SID 生成 MUST 基于训练好的 GatedSAE，为每个物品分配 6 个特征 token（`[f_*]` 格式）。
- **FR-005**: SFT 训练 MUST 使用 Qwen3.5-0.8B 作为生成模型。
- **FR-006**: 评估 MUST 使用宽度为 30 的约束 beam search，计算 HR@1/5/10/20 和 NDCG@1/5/10/20。
- **FR-007**: SFT 训练 MUST 每个 epoch 结束后触发一次推荐指标评估（在验证集上）。
- **FR-008**: 实验全程 MUST 记录关键参数配置和每次改动原因，存入 `roadmap/experiments/` 结构化日志。
- **FR-009**: 流水线 MUST 支持断点续跑（各阶段产出 checkpoint 后可跳过已完成步骤）。

### Key Entities

- **实验记录（ExperimentLog）**: 实验编号、日期、改动说明、参数配置快照、关键指标（HR@10、NDCG@10）
- **SID 索引（SIDIndex）**: 物品 ASIN → 6 个特征 token 的映射，由 GatedSAE K=6 生成
- **训练 Checkpoint**: 每 epoch 保存的模型权重，附带对应的验证集指标

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 在 Beauty 测试集（LOO）上，HR@10 达到目标 0.8（拉伸目标，需迭代改进，当前 baseline ~0.024）。
- **SC-002**: 每个 epoch 的推荐指标评估在 epoch 结束后自动完成，结果写入日志。
- **SC-003**: 实验日志完整覆盖所有超参数变更，每次实验可在 5 分钟内定位对应参数和指标。
- **SC-004**: SAE 训练后物品 SID 碰撞率 < 5%（K=6 配置下）。
- **SC-005**: 端到端流水线任意阶段可单独重跑，结果可复现（误差 < 0.001）。

---

## Assumptions

- Beauty 原始数据已存在于 `data/raw/` 目录。
- Qwen3.5-0.8B 和 Qwen3-Embedding-0.6B 模型已缓存本地（HuggingFace cache）。
- 实验环境为单卡 RTX 5080（16GB VRAM），必要时使用 LoRA + gradient checkpointing。
- HR@10=0.8 是拉伸目标（stretch goal），当前 SOTA 在 Beauty 上约为 0.05-0.15，需通过迭代改进逼近。
- 每个 epoch 的推荐指标评估使用验证集，若验证集过大可抽样（最多 1000 条）以控制评估时间。
- 实验记录以 Markdown 文件形式保存在 `roadmap/experiments/` 目录。
