# Research: GatedSAE SID 生成

**Date**: 2026-03-08
**Feature**: 002-gated-sae-sid

## Decision 1: SAELens 集成方式

**Decision**: 引入 SAELens 作为依赖，使用其 `GatedTrainingSAE` + `SAETrainer` 进行训练，仅编写数据适配层和 SID 生成逻辑。

**Rationale**:
- SAELens 的 GatedSAE 实现成熟（含门控机制、辅助损失、dead neuron 检测）
- `SAETrainer` 支持任意 `DataProvider = Iterator[torch.Tensor]`，可自定义数据源
- 避免重复实现门控编码、L1 稀疏性调度、梯度裁剪等训练基础设施

**Alternatives considered**:
- 自行实现 GatedSAE：代码量大，易出错，维护成本高
- Vendor（复制 SAELens 代码）：失去上游更新，需自行维护

## Decision 2: .npy 数据适配

**Decision**: 创建自定义 `NpyDataProvider`（实现 `Iterator[torch.Tensor]`），直接从 .npy 文件加载 embedding 批次，传入 SAELens 的 `SAETrainer`。

**Rationale**:
- SAELens `ActivationsStore` 设计用于 TransformerLens 模型或 HuggingFace Dataset，不支持 .npy
- `SAETrainer` 的 `data_provider` 参数接受任何 `Iterator[torch.Tensor]`
- 自定义 DataProvider 最轻量，无需转换数据格式

**Alternatives considered**:
- 转换 .npy → HuggingFace Dataset → ActivationsStore：多一层间接，增加磁盘占用
- 修改 ActivationsStore：侵入 SAELens 内部，不利于维护

## Decision 3: SID 冲突去重策略

**Decision**: 对冲突 item，尝试使用 top-(K+1)、top-(K+2)... 特征替换最后一个位置的特征索引。最多尝试 `max_dedup_iters` 次。

**Rationale**:
- GatedSAE 没有 Sinkhorn 量化可用，无法像 RQ-VAE 那样调整量化过程
- SAE 编码是确定性的（无随机噪声），对同一输入总返回相同激活
- 用 top-(K+j) 特征替换是最自然的去碰撞方式：下一个最强特征作为区分信号

**Alternatives considered**:
- 添加微小噪声后重新编码：破坏确定性，不可复现
- 拼接额外特征（K+1 token）：改变 SID 长度，不兼容下游

## Decision 4: 字典大小与 token 词汇量

**Decision**: 字典大小可配置（默认 4x 输入维度）。SID token 格式保持 `[a_x]` 但 x 的范围扩展为 `[0, dict_size-1]`。

**Rationale**:
- 4x 是 SAE 研究中标准的扩展比，平衡稀疏性和表达力
- TokenExtender 从 .index.json 动态提取所有唯一 token，无需预定义范围
- 前缀树也是动态构建，自适应 token 词汇量

**Alternatives considered**:
- 固定 dict_size=256：过小，SAE 特征不够稀疏
- 固定 dict_size=16x：默认过大，多数特征可能为 dead neuron

## Decision 5: 训练跳过 LLM 评估

**Decision**: GatedSAE 训练时不使用 SAELens 的 LLM-based 评估器（`LLMSaeEvaluator`），仅跟踪重构损失、L0 范数、dead neuron 比例。

**Rationale**:
- SAELens 的评估器需要 TransformerLens 模型，本场景无 LLM 可 hook
- 对于 item embedding 的 SAE 训练，重构质量和稀疏性是充分的训练信号
- 下游推荐指标（HR@K, NDCG@K）在完整流水线评估中体现

**Alternatives considered**:
- 自定义评估器计算 SID 唯一性：增加训练复杂度，且训练中 K 值依赖后处理
