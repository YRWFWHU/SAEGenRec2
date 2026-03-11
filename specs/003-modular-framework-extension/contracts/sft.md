# CLI Contract: SFT (prepare + train)

**Module**: `python -m SAEGenRec.training`

## Commands

### prepare_sft

将 CSV + index.json 预构建为 prompt-completion JSONL。

```bash
python -m SAEGenRec.training prepare_sft \
    --category=Beauty \
    --sid_type=rqvae \
    [--task=sid_seq] \
    [--dataset=Amazon] \
    [--prompt_template=templates/sid_seq.txt] \
    [--data_dir=data/processed] \
    [--interim_dir=data/interim] \
    [--overwrite=False]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| category | str | **required** | 商品类别名 |
| sid_type | str | **required** | SID 方法名（rqvae, rqkmeans, gated_sae） |
| task | str | `sid_seq` | SFT 任务类型 |
| dataset | str | `Amazon` | 数据集名称 |
| prompt_template | str | auto（按 task 查找默认模板） | prompt 模板文件路径 |
| data_dir | str | `data/processed` | 输出基路径 |
| interim_dir | str | `data/interim` | 中间数据目录（含 CSV、index.json、item.json） |
| overwrite | bool | False | 是否覆盖已存在的数据 |
| tasks | str | None | 多任务混合（`sid_seq+item_feat`） |
| task_weights | str | None | 混合权重（`0.7,0.3`） |

**Input**:
- `data/processed/{category}.{train,valid,test}.csv`
- `data/interim/{category}.index.json`
- `data/interim/{category}.item.json`（item_feat/fusion/sid_to_title 任务需要）
- `templates/{task}.txt`（prompt 模板）

**Output**:
```
data/processed/{sid_type}/{dataset}/{category}/{task}/
├── train.jsonl
├── valid.jsonl
├── test.jsonl
└── meta.json
```

**JSONL Line Format**:
```json
{"prompt": "Below is an instruction...\n### User Input:\nThe user has interacted with items [a_1][b_2][c_3]...\n\n### Response:\n", "completion": "[a_7][b_8][c_9]"}
```

**Behavior**:
1. 从任务注册表查找 `task` 对应的 `SFTTask` 实现
2. 加载 CSV 数据和 index.json
3. 使用 prompt 模板构造每条数据的 prompt + completion
4. 按 train/valid/test 分割写入 JSONL
5. 写入 meta.json 记录生成参数
6. completion 为空的样本跳过并记录警告
7. 目标目录已存在时：若 `--overwrite=False` 则报错提示

**Multi-task mixing** (`--tasks=sid_seq+item_feat --task_weights=0.7,0.3`):
- 分别生成各任务数据
- 按权重比例采样合并到 `data/processed/{sid_type}/{dataset}/{category}/mixed/`

---

### sft

从持久化 JSONL 训练 SFT 模型。

```bash
python -m SAEGenRec.training sft \
    --model_path=models/llm \
    --sft_data_dir=data/processed/rqvae/Amazon/Beauty/sid_seq \
    [--category=Beauty] \
    [--output_dir=models/sft/Beauty] \
    [--freeze_llm=False] \
    [--num_epochs=10] \
    [--batch_size=128] \
    [--micro_batch_size=4] \
    [--learning_rate=3e-4] \
    [--sample=-1] \
    [--eval_rec=False] \
    [--eval_rec_steps=0.1] \
    [--eval_rec_beams=10] \
    [--eval_rec_samples=200]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_path | str | **required** | 基座 LLM 或 SFT checkpoint 路径 |
| sft_data_dir | str | **required** | JSONL 数据目录（含 train.jsonl、valid.jsonl） |
| category | str | auto-detect from meta.json | 商品类别名 |
| output_dir | str | `models/sft/{category}` | 模型输出目录 |
| freeze_llm | bool | False | 冻结 LLM 参数，仅训练 SID token embedding |
| num_epochs | int | 10 | 训练轮数 |
| batch_size | int | 128 | 有效批量大小 |
| micro_batch_size | int | 4 | 微批量大小（梯度累积） |
| learning_rate | float | 3e-4 | 学习率 |
| sample | int | -1 | 采样数（-1 = 全部） |
| eval_rec | bool | False | 是否启用训练期间推荐指标评估 |
| eval_rec_steps | float | 0.1 | 评估间隔（占总步数比例） |
| eval_rec_beams | int | 10 | 评估 beam 数 |
| eval_rec_samples | int | 200 | 评估样本数 |

**Input**: `{sft_data_dir}/train.jsonl`, `{sft_data_dir}/valid.jsonl`
**Output**: `{output_dir}/final_checkpoint/`

**Behavior**:
1. 加载 model_path 的 tokenizer，检测是否已含 SID token
2. 若未含 SID token：从 sft_data_dir 推断 SID token 集合，执行 TokenExtender 扩展
3. 若已含 SID token（从 checkpoint 续训）：跳过扩展，保留已有 embedding 权重
4. 加载 JSONL 数据（每行 `{"prompt": "...", "completion": "..."}`），拼接为完整文本
5. 使用 TRL `SFTTrainer` + `DataCollatorForCompletionOnlyLM(response_template, tokenizer)` 自动实现 completion-only loss（prompt 部分 labels 自动设为 -100，无需手动 masking）
6. `response_template` 从 prompt 模板中提取分隔标记（如 `"### Response:\n"`）
7. `freeze_llm=True`：冻结全部参数，仅 embedding 矩阵的 SID token 行可训练
8. `eval_rec=True`：注册 `TrainingEvaluator` callback
9. 训练并保存 checkpoint

**Checkpoint resume** (`model_path` 指向 SFT checkpoint):
- 自动加载 tokenizer（含 SID token）+ 模型权重
- `TokenExtender` 检测到已有 SID token 后跳过扩展
- 可使用不同的 `sft_data_dir`（多阶段 SFT）

---

### list_sft_tasks

列出所有已注册的 SFT 任务类型。

```bash
python -m SAEGenRec.training list_sft_tasks
```

**Output** (stdout):
```
Available SFT Tasks:
  sid_seq      - 历史 SID 序列 → 目标 SID (template: templates/sid_seq.txt)
  item_feat    - item title → SID (template: templates/item_feat.txt)
  fusion       - 历史 title + SID → 目标 SID (template: templates/fusion.txt)
  sid_to_title - SID → item title (template: templates/sid_to_title.txt)
```
