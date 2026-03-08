# Data Model: Modularize MiniOneRec Framework

**Phase**: 1 | **Date**: 2026-03-07

## Entities

### Item

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| item_id | int | 内部整数 ID（0-indexed） | 唯一，由预处理阶段分配 |
| item_asin | str | Amazon 产品标识符 | 唯一，原始数据中的 `asin` 字段 |
| title | str | 产品标题 | 非空，≤20 词，无 HTML span 标签 |
| description | str | 产品描述 | 可为空 |

**Lifecycle**: 原始 JSON → k-core 过滤 → item_id 分配 → `.item.json` 持久化
**Identity**: `item_asin` 为自然键，`item_id` 为系统生成的代理键

### Review

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| user_id | str | 用户标识符（reviewerID） | 非空 |
| item_id | int | 关联的 item 内部 ID | 必须存在于 .item.json |
| item_asin | str | 关联的 Amazon 产品标识符 | 非空 |
| timestamp | int | Unix 时间戳 | > 0 |
| rating | float | 评分 | 1.0 - 5.0 |
| review_text | str | 评论正文 | 可为空（缺失时为空字符串） |
| summary | str | 评论摘要 | 可为空 |

**Lifecycle**: 原始 JSON → k-core 过滤 → `.review.json` 持久化 → text2emb 编码为 `.npy`
**Identity**: `(user_id, item_asin, timestamp)` 三元组唯一标识一条交互
**Ordering**: `.review.json` 中的行序与 `.emb-{model}-review.npy` 的行序一一对应

### InteractionSample

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| user_id | str | 用户标识符 | 非空 |
| history_item_ids | list[int] | 历史 item_id 列表 | 长度 ≤ max_history_len，按时间排序 |
| history_item_asins | list[str] | 历史 item_asin 列表 | 与 history_item_ids 等长 |
| target_item_id | int | 目标 item_id | 必须存在于 .item.json |
| target_item_asin | str | 目标 item_asin | 非空 |
| target_rating | float | 目标评分 | 1.0 - 5.0 |
| target_timestamp | int | 目标时间戳 | > 所有 history 时间戳 |

**Lifecycle**: 由 preprocess 阶段从过滤后的交互序列生成，持久化为 `.inter` 文件
**Note**: 这是滑动窗口生成的训练样本，一个用户可产生多个 InteractionSample

### SID (Semantic Item Description)

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| item_id | str (as key) | 物品 ID（字符串形式） | 必须存在于 .item.json |
| tokens | list[str] | SID token 序列 | 长度 = 量化层数（默认 3） |

**Lifecycle**: item embedding → RQ-VAE/RQ-Kmeans 量化 → `.index.json` 持久化 → TokenExtender 注入 LLM 词表
**Format**: 每个 token 形如 `[{level_prefix}_{code_index}]`，例如 `[a_42]`
**Uniqueness**: 理想情况下唯一，但允许碰撞（collision rate 作为质量指标追踪）

### PrefixTree (hash_dict)

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| prefix | tuple[int] | 已生成 token ID 的元组 | 长度 0 到 num_levels-1 |
| valid_next | set[int] | 合法的下一个 token ID 集合 | 非空 |

**Lifecycle**: 运行时从 `.index.json` + tokenizer 构建，不持久化
**Usage**: ConstrainedLogitsProcessor 在 beam search 中查询

## Relationships

```
Item (1) ──── (N) Review          # 一个 item 有多条来自不同用户的 review
Item (1) ──── (1) SID             # 一个 item 映射到恰好一个 SID（可能碰撞）
Item (1) ──── (1) Embedding       # 一个 item 有一个 td embedding 行
Review (1) ── (1) ReviewEmbedding # 一条 review 有一个 review embedding 行
User (1) ──── (N) Review          # 一个用户有多条 review
User (1) ──── (N) InteractionSample # 一个用户生成多个训练样本（滑动窗口）
SID (N) ────── PrefixTree         # 所有 SID 构成前缀树
```

## State Transitions

### Pipeline Data Flow

```
raw JSON ─── [preprocess] ──→ .inter + .item.json + .review.json
                                │
                                ├── [text2emb] ──→ .emb-{model}-td.npy
                                │                  .emb-{model}-review.npy
                                │
                                └── .emb-td.npy ── [rqvae/rqkmeans] ──→ checkpoint
                                                         │
                                                    [generate_indices] ──→ .index.json
                                                         │
                                    .inter + .item.json + .index.json
                                         │
                                    [convert_dataset] ──→ CSV + info TXT
                                         │
                                    CSV + info ── [sft] ──→ SFT checkpoint
                                         │
                                    SFT ckpt ── [rl] ──→ RL checkpoint
                                         │
                                    checkpoint ── [evaluate] ──→ HR@K, NDCG@K
```
