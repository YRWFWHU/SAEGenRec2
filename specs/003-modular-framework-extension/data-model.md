# Data Model: 模块化框架扩展

**Branch**: `003-modular-framework-extension` | **Date**: 2026-03-09

## Entities

### 1. EmbeddingFile

物品特征向量的持久化文件（.npy 格式）。

| Field | Type | Description |
|-------|------|-------------|
| path | str | 文件路径，如 `data/interim/Beauty.emb-all-MiniLM-L6-v2-text.npy` |
| modality | enum(text, visual) | 特征模态 |
| model_name | str | 特征提取器模型名（slug 形式，如 `all-MiniLM-L6-v2`） |
| n_items | int | 行数（物品数量） |
| d_embed | int | 向量维度 |
| item_order | str | 行序定义，与 `{category}.item.json` 中 item_id 顺序一致 |

**Validation**:
- `n_items` 必须等于 `.item.json` 中的物品数
- 行序与文本嵌入一致（visual 与 text 的第 i 行对应同一 item）
- 不含 NaN/Inf（加载时 `np.nan_to_num` 清理）

**Naming Convention**: `{category}.emb-{model_slug}-{modality}.npy`
- 文本：`Beauty.emb-all-MiniLM-L6-v2-text.npy`
- 视觉：`Beauty.emb-clip-vit-base-patch32-visual.npy`

---

### 2. ImageDirectory

物品图像的本地存储目录。

| Field | Type | Description |
|-------|------|-------------|
| base_dir | str | 目录路径，如 `data/interim/Beauty/images/` |
| category | str | 商品类别名 |
| format | str | 图像格式，固定 `JPEG` |
| naming | str | 文件命名，`{item_asin}.jpg` |

**Validation**:
- 每个 `item_asin` 最多一个图像文件
- 缺失图像不阻断流程，下游用零向量替代

---

### 3. SIDMethod (Abstract)

SID 生成方法的统一接口。

| Field | Type | Description |
|-------|------|-------------|
| name | str | 方法名：`rqvae`, `rqkmeans`, `gated_sae` |
| token_format | str | token 前缀格式：`auto`, `a-h`(位置), `f`(统一), 或自定义单字符 |
| default_k | int | 默认 SID token 数（RQ: 3, SAE: 8） |
| config_class | type | 对应的 dataclass 配置类 |

**Interface**:
- `train(embedding_path, output_dir, **config) -> str`: 训练模型，返回 checkpoint 路径
- `generate(checkpoint, embedding_path, output_path, k, token_format) -> str`: 生成 .index.json

**State Transitions**:
```
未训练 → [train()] → 已训练(checkpoint) → [generate()] → 已生成(.index.json)
```

**Registry**: `SID_METHODS: Dict[str, Type[SIDMethod]]`
- `"rqvae"` → `RQVAEMethod`
- `"rqkmeans"` → `RQKMeansMethod`
- `"gated_sae"` → `GatedSAEMethod`

---

### 4. SFTTask (Abstract)

SFT 任务类型定义。

| Field | Type | Description |
|-------|------|-------------|
| name | str | 任务名：`sid_seq`, `item_feat`, `fusion`, `sid_to_title` |
| default_template | str | 默认 prompt 模板文件路径（`templates/{name}.txt`） |
| required_inputs | List[str] | 所需输入文件类型：`csv`, `index_json`, `item_json` |
| required_placeholders | List[str] | 模板中必须出现的占位符 |

**Builtin Tasks**:

| Task | Input | Output(completion) | Required Placeholders |
|------|-------|---------------------|----------------------|
| `sid_seq` | CSV (history_item_sid) | target SID | `{history}` |
| `item_feat` | index.json + item.json | SID for given title | `{title}` |
| `fusion` | CSV + index.json + item.json | target SID | `{history}`, `{titles}` |
| `sid_to_title` | index.json + item.json | title for given SID | `{sid}` |

**Registry**: `SFT_TASKS: Dict[str, Type[SFTTask]]`

---

### 5. SFTDataFile

持久化的 SFT JSONL 训练数据。

| Field | Type | Description |
|-------|------|-------------|
| directory | str | `data/processed/{sid_type}/{dataset}/{category}/{task}/` |
| sid_type | str | SID 方法名（rqvae, rqkmeans, gated_sae） |
| dataset | str | 数据集名（默认 `Amazon`） |
| category | str | 商品类别名 |
| task | str | SFT 任务名 |
| split | enum(train, valid, test) | 数据分割 |
| format | str | 固定 `JSONL` |

**JSONL Line Schema**:
```json
{"prompt": "<instruction + user input>", "completion": "<target output>"}
```

**Meta File** (`meta.json`):
```json
{
  "sid_type": "rqvae",
  "dataset": "Amazon",
  "category": "Beauty",
  "task": "sid_seq",
  "template": "templates/sid_seq.txt",
  "created_at": "2026-03-09T12:00:00",
  "n_train": 12345,
  "n_valid": 1234,
  "n_test": 1234
}
```

**Validation**:
- 每行必须包含 `prompt` 和 `completion` 两个字段
- `completion` 不可为空字符串（空则跳过并记录警告）
- 不同 TASK 的数据目录独立，互不覆盖

---

### 6. PromptTemplate

Prompt 模板文件（纯文本 + 占位符）。

| Field | Type | Description |
|-------|------|-------------|
| path | str | 文件路径，如 `templates/sid_seq.txt` |
| task | str | 关联的 SFT 任务名 |
| placeholders | List[str] | 使用的占位符列表 |
| format | str | Python `str.format_map()` 语法 |

**Builtin Templates**:
- `templates/sid_seq.txt`: `{history}` → target SID
- `templates/item_feat.txt`: `{title}` → SID
- `templates/fusion.txt`: `{history}`, `{titles}` → target SID
- `templates/sid_to_title.txt`: `{sid}` → title

**Validation**:
- 模板文件必须包含任务所需的全部 required_placeholders
- `prepare_sft` 阶段验证，缺失则报错提示

---

### 7. RewardFunction

RL 奖励函数。

| Field | Type | Description |
|-------|------|-------------|
| name | str | 函数名：`rule`, `prefix`, `ranking`, `semantic`, `sasrec` |
| fn | Callable | 签名：`(predictions: List[str], target: str, **kwargs) -> List[float]` |
| requires_extra | List[str] | 需要的额外参数（如 `embeddings`, `sasrec_model`） |

**Registry**: `_REWARD_REGISTRY: Dict[str, RewardFunction]`

**Combination**: `CombinedReward(names: List[str], weights: List[float])`
- 按权重加权求和各奖励函数输出
- `reward_type="rule+semantic"` + `reward_weights="0.6,0.4"` → `CombinedReward(["rule", "semantic"], [0.6, 0.4])`

---

### 8. TrainingEvaluator

训练期间推荐指标评估器。

| Field | Type | Description |
|-------|------|-------------|
| enabled | bool | 是否启用（`EVAL_REC` 参数） |
| eval_steps | float | 评估间隔（`EVAL_REC_STEPS`，如 0.1 = 每 10% 步） |
| num_beams | int | 约束波束搜索 beam 数（`EVAL_REC_BEAMS`，默认 10） |
| n_samples | int | 评估样本数（`EVAL_REC_SAMPLES`，默认 200） |
| k_values | List[int] | HR@K/NDCG@K 的 K 值列表 |
| info_file | str | SID → title 映射文件路径 |
| test_csv | str | 测试数据 CSV 路径 |

**Implementation**: HuggingFace `TrainerCallback`
- `on_evaluate` 或 `on_step_end` 触发
- 复用 `evaluate.py` 的 `constrained_beam_search` + `compute_hr_ndcg`
- 日志输出 HR@K、NDCG@K 指标

## Relationship Diagram

```
EmbeddingFile(.npy) ──────┐
                          ▼
ImageDirectory(images/) → SIDMethod.train() → checkpoint
                                                  │
                                                  ▼
EmbeddingFile(.npy) ────→ SIDMethod.generate() → .index.json
                                                      │
                          CSV + .item.json ──────────┤
                                                      ▼
PromptTemplate(.txt) ──→ prepare_sft ──────────→ SFTDataFile(.jsonl)
                                                      │
                                                      ▼
                          SFT Trainer ◄── TrainingEvaluator (callback)
                               │
                               ▼
                          SFT checkpoint
                               │
                               ▼
RewardFunction ─────────→ RL Trainer ◄── TrainingEvaluator (callback)
                               │
                               ▼
                          RL checkpoint → evaluate → metrics.json
```
