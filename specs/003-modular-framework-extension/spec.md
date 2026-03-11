# Feature Specification: 模块化框架扩展

**Feature Branch**: `003-modular-framework-extension`
**Created**: 2026-03-08
**Status**: Draft
**Input**: 数据处理多模态支持、SID统一接口、SFT/RL任务自定义、训练期间评估

## Clarifications

### Session 2026-03-09

- Q: 不同 TASK 的 SFT 数据应该如何隔离？ → A: 增加 TASK 子目录：`data/processed/{SID_type}/{大数据集}/{分类}/{TASK}/train.jsonl`
- Q: "大数据集"参数的含义？ → A: 默认 "Amazon" 但暴露 `DATASET=Amazon` 可选参数，支持未来扩展其他平台
- Q: 多模态特征融合方式？ → A: 不做框架级融合，提供文本和视觉两种 embedding 接口供用户自行设计融合策略，默认仅使用文本 embedding（视觉 embedding 不默认提供）
- Q: RL 数据是否也采用 JSONL 持久化？ → A: RL 保持 CSV 输入，但复用 SFT 的 prompt 模板机制（`--prompt_template` 参数）
- Q: 框架应内置哪些 SFT 任务类型？ → A: 内置 4 种：`sid_seq`（历史SID → 目标SID）、`item_feat`（title → SID）、`fusion`（历史title+SID → 目标SID）、`sid_to_title`（SID → title，反向映射任务）

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 多模态数据处理 (Priority: P1)

研究者需要为推荐系统物品获取多模态特征。当前系统仅支持文本嵌入（title + description），无法利用物品图像信息。研究者希望：
1. 从 Amazon 原始元数据中下载每个物品的最高分辨率 MAIN 图像
2. 使用可配置的视觉特征提取器（如 CLIP、DINOv2）将图像处理为视觉特征向量
3. 将文本和图像处理命令分离，可独立执行
4. 在 Makefile 中提供命令让用户选择提取哪些特征、使用什么模型、以及数据集划分方式

数据在 interim 目录中按 `DATASET/CATEGORY/data` 格式组织。例如 `data/interim/Amazon/Beauty/images/`。

**Why this priority**: 多模态特征是后续所有模块（SID生成、SFT、RL）的数据基础。没有图像特征，后续的多模态SID生成和融合推荐无法进行。

**Independent Test**: 运行 `make download_images CATEGORY=Beauty` 下载图像，再运行 `make extract_visual CATEGORY=Beauty VISION_MODEL=openai/clip-vit-base-patch32` 提取视觉特征，验证 `data/interim/Beauty.emb-clip-vit-base-patch32-visual.npy` 文件存在且形状为 (n_items, d_visual)。

**Make 命令设计**:

```bash
# 下载物品图像（从 Amazon 元数据中提取最高分辨率 MAIN 图像）
make download_images CATEGORY=Beauty [CONCURRENCY=8] [IMAGE_DIR=data/interim/Beauty/images]

# 提取文本嵌入（从现有 embed 命令重构，支持自定义模型）
make embed_text CATEGORY=Beauty [TEXT_MODEL=sentence-transformers/all-MiniLM-L6-v2]

# 提取视觉特征（从下载的图像中提取视觉嵌入）
make extract_visual CATEGORY=Beauty [VISION_MODEL=openai/clip-vit-base-patch32]

# 一键提取所有特征（文本 + 视觉）
make embed_all CATEGORY=Beauty [TEXT_MODEL=...] [VISION_MODEL=...]

# 保留原有 embed 命令作为 embed_text 的别名（向后兼容）
make embed CATEGORY=Beauty
```

**Acceptance Scenarios**:

1. **Given** Amazon 原始元数据包含 imageURLHighRes 字段, **When** 研究者运行 `make download_images CATEGORY=Beauty`, **Then** 系统下载每个物品的最高分辨率 MAIN 图像到 `data/interim/{category}/images/{item_asin}.jpg`，跳过已下载的文件，记录下载成功/失败统计
2. **Given** 已下载的物品图像目录, **When** 研究者运行 `make extract_visual CATEGORY=Beauty VISION_MODEL=openai/clip-vit-base-patch32`, **Then** 系统输出 `{dataset}.emb-clip-vit-base-patch32-visual.npy` 文件，行序与 item_id 对齐（与文本嵌入一致）
3. **Given** 用户希望更换文本嵌入模型, **When** 运行 `make embed_text CATEGORY=Beauty TEXT_MODEL=sentence-transformers/all-mpnet-base-v2`, **Then** 系统使用指定模型生成文本嵌入，输出文件名包含模型标识
4. **Given** 某些物品没有图像URL或下载失败, **When** 提取视觉特征时, **Then** 缺失图像的物品使用零向量填充，日志记录缺失数量

---

### User Story 2 - SID 生成统一接口 (Priority: P1)

研究者需要一个统一的 SID 生成接口，支持 RQ-VAE、RQ-KMeans 和 GatedSAE 三种方法，可灵活选择输入特征（文本嵌入或视觉嵌入），并自定义 token 格式。框架不做特征融合，仅提供文本和视觉两种 embedding 接口，用户可自行设计融合策略后将处理好的 .npy 传入。默认仅使用文本 embedding。当前系统中三种方法的接口不统一，参数风格和输出格式各异。

**Why this priority**: SID 是连接特征提取与下游训练的核心桥梁。统一接口让研究者可以快速对比不同SID生成方法的效果，无需理解每种方法的内部细节。与US1同为P1因为两者共同构成数据准备层。

**Independent Test**: 运行 `make build_sid CATEGORY=Beauty METHOD=rqvae`，验证输出格式正确；更换 `METHOD=gated_sae` 或 `METHOD=rqkmeans` 后同样能生成格式一致的 .index.json。

**Make 命令设计**:

```bash
# 统一 SID 生成入口（通过 METHOD 参数选择方法）
make build_sid CATEGORY=Beauty METHOD=rqvae [EMB_PATH=data/interim/Beauty.emb-*.npy] [K=3] [TOKEN_FORMAT=auto]
make build_sid CATEGORY=Beauty METHOD=gated_sae [EMB_PATH=...] [K=8]
make build_sid CATEGORY=Beauty METHOD=rqkmeans [EMB_PATH=...] [K=3]

# 仅训练 SID 模型（不生成 .index.json）
make train_sid CATEGORY=Beauty METHOD=rqvae [EMB_PATH=...] [SID_MODEL_DIR=models/sid/Beauty]

# 仅从已训练模型生成 .index.json
make generate_sid CATEGORY=Beauty METHOD=rqvae [SID_MODEL_DIR=...] [EMB_PATH=...]

# 保留原有 build_sae_sid 命令作为 build_sid METHOD=gated_sae 的别名（向后兼容）
make build_sae_sid CATEGORY=Beauty
```

**Acceptance Scenarios**:

1. **Given** 文本嵌入文件已生成, **When** 研究者运行 `make build_sid CATEGORY=Beauty METHOD=rqvae`, **Then** 系统训练 RQ-VAE 并输出 .index.json，token 格式为 `[a_x][b_y][c_z]`（位置感知，每层不同codebook）
2. **Given** 文本嵌入文件已生成, **When** 研究者运行 `make build_sid CATEGORY=Beauty METHOD=gated_sae`, **Then** 系统训练 GatedSAE 并输出 .index.json，token 格式为 `[f_x][f_y]...[f_z]`（位置无关，单一codebook）
3. **Given** 视觉嵌入文件已生成（或用户自行融合的特征文件）, **When** 研究者指定 `EMB_PATH=data/interim/Beauty.emb-clip-visual.npy`, **Then** 系统使用指定的 .npy 文件进行 SID 生成（框架不做融合，仅接受单个 .npy 输入）
4. **Given** 研究者希望自定义 token 前缀格式, **When** 指定 `TOKEN_FORMAT=v`, **Then** 生成的 .index.json 使用自定义格式（如 `[v_x]` 表示视觉特征）

---

### User Story 3 - 可自定义 SFT 任务 (Priority: P2)

研究者需要方便地自定义 SFT 训练任务，包括 prompt 模板、数据集组合方式、以及训练策略。当前系统的 prompt 模板硬编码在 Dataset 类中，添加新任务需修改多个文件，且每次训练时动态拼接 prompt-completion 对，无法复用。

**核心设计变更**：
1. **数据持久化**：自定义 SFT 任务数据预构建后持久化保存在 processed 路径下，目录结构为 `data/processed/{SID_type}/{DATASET}/{CATEGORY}/{TASK}/`，例如 `data/processed/rqvae/Amazon/Beauty/sid_seq/train.jsonl`。SID_type 对应生成 SID 所用的方法（rqvae、rqkmeans、gated_sae），TASK 对应任务类型（sid_seq、item_feat 等），不同任务的数据独立存放，支持多阶段训练。
2. **统一格式**：SFT 数据统一为 prompt-completion 对格式（JSONL），每行包含 `{"prompt": "...", "completion": "..."}`。
3. **Completion-only loss**：训练时仅在 completion 部分计算 loss，使用 TRL `SFTTrainer` + `DataCollatorForCompletionOnlyLM` 自动将 prompt 部分的 token 标签设为 -100，确保模型只学习生成目标而非背诵指令（无需手动 masking）。
4. **冻结 LLM 参数**：支持 `FREEZE_LLM=True` 模式，冻结 LLM 原始参数，仅训练新添加的 SID token embedding。这种模式下训练速度更快、显存占用更低，适合快速验证 SID 编码方案的有效性。冻结时仅 SID token（如 `[a_0]`、`[f_0]` 等通过 TokenExtender 新增的 token）对应的 embedding 参与梯度更新，LLM 原始词表的 embedding 和所有 transformer 层参数保持不变。
5. **多阶段渐进式 SFT**：`MODEL_PATH` 支持指向已有的 SFT checkpoint（不仅限于基座 LLM），从而实现"先简单任务后复杂任务"的课程学习（Curriculum Learning）。例如：阶段1 `item_feat`（title → SID）+ `sid_to_title`（SID → title）学习双向语义映射 → 阶段2 `fusion`（历史title+SID → 目标SID）学习融合推荐 → 阶段3 `sid_seq`（历史SID → 目标SID）精调序列推荐。每个阶段使用不同的 `SFT_DATA_DIR`（指向对应 TASK 子目录），输出到不同的 `OUTPUT_DIR`，前一阶段的输出 checkpoint 作为下一阶段的 `MODEL_PATH`。

**Why this priority**: SFT 是推荐模型的核心训练阶段。预构建 + 持久化的 prompt-completion 数据让研究者可以：(a) 复用已生成的数据避免重复计算；(b) 直观检查训练数据质量；(c) 跨实验对比不同 prompt 设计；(d) 多阶段渐进式训练。依赖 US1/US2 的数据输出。

**Independent Test**: 运行 `make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae TASK=sid_seq`，验证 `data/processed/rqvae/Amazon/Beauty/sid_seq/train.jsonl` 存在且每行包含 prompt + completion 字段；再运行 `make sft SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/sid_seq MODEL_PATH=models/llm`，验证训练正常收敛且 loss 仅在 completion 上计算。

**Make 命令设计**:

```bash
# 预构建 SFT 数据（从 CSV + index.json → prompt-completion JSONL）
make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae \
    [TASK=sid_seq] \
    [PROMPT_TEMPLATE=templates/my_prompt.txt] \
    [DATASET=Amazon]

# 预构建多任务混合 SFT 数据
make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae \
    TASKS=sid_seq+item_feat \
    [TASK_WEIGHTS=0.7,0.3]

# SFT 训练（从持久化的 prompt-completion 数据训练，全参数微调）
make sft CATEGORY=Beauty MODEL_PATH=models/llm \
    SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/sid_seq \
    [FREEZE_LLM=False] \
    [SAMPLE=-1]

# SFT 训练（冻结 LLM，仅训练 SID token embedding）
make sft CATEGORY=Beauty MODEL_PATH=models/llm \
    SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/sid_seq \
    FREEZE_LLM=True

# 多阶段渐进式 SFT（Curriculum Learning）
# 阶段1：在简单任务（item 特征描述）上 SFT，学习 SID token 语义
make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae TASK=item_feat
make sft MODEL_PATH=models/llm \
    SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/item_feat \
    OUTPUT_DIR=models/sft/stage1

# 阶段2：从阶段1的 checkpoint 继续，在复杂任务（序列推荐）上 SFT
make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae TASK=sid_seq
make sft MODEL_PATH=models/sft/stage1/final_checkpoint \
    SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/sid_seq \
    OUTPUT_DIR=models/sft/stage2

# 快速验证 SFT（小样本 + 少 epoch）
make sft_quick CATEGORY=Beauty MODEL_PATH=models/llm \
    SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/sid_seq \
    [SAMPLE=500] [NUM_EPOCHS=1]

# 列出可用的 SFT 任务类型和 prompt 模板
make list_sft_tasks
```

**数据目录结构**:

```
data/processed/
├── rqvae/
│   └── Amazon/
│       └── Beauty/
│           ├── sid_seq/                 # 序列推荐任务
│           │   ├── train.jsonl          # {"prompt": "...", "completion": "..."}
│           │   ├── valid.jsonl
│           │   ├── test.jsonl
│           │   └── meta.json            # 生成参数记录（SID_TYPE, TASK, TEMPLATE, 时间戳）
│           ├── item_feat/               # Item 特征描述任务
│           │   ├── train.jsonl
│           │   ├── valid.jsonl
│           │   ├── test.jsonl
│           │   └── meta.json
│           └── info.txt                 # semantic_id → title → item_id 映射（共享）
├── gated_sae/
│   └── Amazon/
│       └── Beauty/
│           ├── sid_seq/
│           │   └── ...
│           └── info.txt
└── rqkmeans/
    └── ...
```

**JSONL 行格式示例**:

```json
{"prompt": "The user has interacted with items [a_1][b_2][c_3], [a_4][b_5][c_6] in chronological order. Can you predict the next possible item?", "completion": "[a_7][b_8][c_9]"}
```

**Acceptance Scenarios**:

1. **Given** 研究者运行 `make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae TASK=sid_seq`, **When** CSV 和 index.json 已存在, **Then** 系统在 `data/processed/rqvae/Amazon/Beauty/sid_seq/` 生成 train/valid/test.jsonl，每行为 prompt-completion 对
2. **Given** 研究者指定自定义 prompt 模板, **When** 运行 `make prepare_sft PROMPT_TEMPLATE=templates/my_prompt.txt`, **Then** 生成的 JSONL 使用自定义模板构造 prompt 字段
3. **Given** 持久化的 SFT JSONL 数据已存在, **When** 运行 `make sft SFT_DATA_DIR=data/processed/rqvae/Amazon/Beauty/sid_seq`, **Then** 训练直接读取 JSONL 数据，仅在 completion 部分计算 loss（prompt 部分标签为 -100）
4. **Given** 研究者希望混合多种任务, **When** 运行 `make prepare_sft TASKS=sid_seq+item_feat TASK_WEIGHTS=0.7,0.3`, **Then** 系统按比例采样并合并到同一 JSONL 文件
5. **Given** 研究者希望快速验证 SID 编码方案, **When** 运行 `make sft FREEZE_LLM=True`, **Then** LLM 原始参数全部冻结，仅 SID token embedding 参与梯度更新，训练速度显著加快且显存占用降低
6. **Given** 研究者希望渐进式训练, **When** 先运行 `make sft MODEL_PATH=models/llm SFT_DATA_DIR=.../Beauty/item_feat OUTPUT_DIR=models/sft/stage1`，再运行 `make sft MODEL_PATH=models/sft/stage1/final_checkpoint SFT_DATA_DIR=.../Beauty/sid_seq OUTPUT_DIR=models/sft/stage2`, **Then** 阶段2从阶段1的 checkpoint 继续训练，模型保留已学到的 SID token 语义知识，在序列推荐任务上进一步优化
7. **Given** 研究者加载已有 SFT checkpoint 继续训练, **When** checkpoint 中的 tokenizer 已包含 SID token, **Then** 系统自动检测并跳过重复的 token 扩展，直接加载已有 embedding 权重
8. **Given** 研究者创建了新的 SFT 任务类型, **When** 通过注册机制添加, **Then** `make prepare_sft TASK=新任务名` 可用，无需修改核心代码

---

### User Story 4 - 可自定义 RL 奖励函数 (Priority: P2)

研究者需要方便地定义和注册自己的 RL 奖励函数。当前系统支持 rule、prefix、ranking、semantic、sasrec 五种奖励函数，但添加新函数需要修改 rewards.py 和 rl.py 两处代码。RL 训练保持 CSV 输入（GRPO 算法需要对每个 prompt 生成多个 completion 并计算奖励，数据流与 SFT 不同），但复用 SFT 的 prompt 模板机制，通过 `--prompt_template` 参数定制 prompt 构造方式。

**Why this priority**: 奖励函数是 RL 训练效果的核心决定因素。可定制的奖励函数让研究者可以快速实验不同的奖励信号设计。与 US3 同为 P2，因为两者都是训练定制化。

**Independent Test**: 运行 `make rl CATEGORY=Beauty REWARD_TYPE=rule+prefix REWARD_WEIGHTS=0.6,0.4`，验证训练正常运行且日志显示使用了组合奖励。

**Make 命令设计**:

```bash
# RL 训练（扩展现有命令，增加奖励函数配置）
make rl CATEGORY=Beauty MODEL_PATH=models/sft/Beauty/final_checkpoint \
    [REWARD_TYPE=rule] \
    [REWARD_WEIGHTS=1.0] \
    [NUM_GENERATIONS=16] \
    [BETA=0.04] \
    [PROMPT_TEMPLATE=templates/my_prompt.txt] \
    [SAMPLE=-1]

# 组合奖励函数
make rl CATEGORY=Beauty MODEL_PATH=... REWARD_TYPE=rule+semantic REWARD_WEIGHTS=0.6,0.4

# 快速验证 RL
make rl_quick CATEGORY=Beauty MODEL_PATH=... [SAMPLE=500] [NUM_GENERATIONS=4]

# 列出所有已注册的奖励函数
make list_rewards
```

**Acceptance Scenarios**:

1. **Given** 研究者编写了符合统一签名的奖励函数, **When** 通过注册装饰器注册, **Then** 该函数可通过 `make rl REWARD_TYPE=函数名` 使用
2. **Given** 研究者定义了需要额外参数的奖励函数, **When** 通过 CLI 传入额外参数, **Then** 系统将参数传递给奖励函数
3. **Given** 研究者希望组合多个奖励函数, **When** 运行 `make rl REWARD_TYPE=rule+semantic REWARD_WEIGHTS=0.6,0.4`, **Then** 系统按权重加权组合多个奖励信号

---

### User Story 5 - 训练期间推荐指标评估 (Priority: P3)

研究者需要在 SFT 和 RL 训练过程中选择性地对推荐指标（HR@K、NDCG@K）进行验证，以便及时发现模型退化或选择最佳checkpoint。当前系统仅在训练结束后进行独立评估，训练过程中无法观察推荐指标变化。

**Why this priority**: 训练期间评估是实验效率的改善，而非核心功能阻塞。RL 训练已有 `test_during_training` 参数的初步支持，但 SFT 训练完全没有。

**Independent Test**: 运行 `make sft CATEGORY=Beauty EVAL_REC=True EVAL_REC_STEPS=0.1`，验证训练日志中每 10% 步骤出现 HR@10、NDCG@10 指标；运行 `make rl` 同理。

**Make 命令设计**:

```bash
# SFT 训练 + 训练期间推荐指标评估
make sft CATEGORY=Beauty MODEL_PATH=... \
    EVAL_REC=True \
    [EVAL_REC_STEPS=0.1] \
    [EVAL_REC_BEAMS=10] \
    [EVAL_REC_SAMPLES=200]

# RL 训练 + 训练期间推荐指标评估
make rl CATEGORY=Beauty MODEL_PATH=... \
    EVAL_REC=True \
    [EVAL_REC_STEPS=0.1] \
    [EVAL_REC_BEAMS=10] \
    [EVAL_REC_SAMPLES=200]

# 独立评估（保留现有命令，增加参数灵活性）
make evaluate CATEGORY=Beauty MODEL_PATH=models/rl/Beauty \
    [NUM_BEAMS=50] \
    [K_VALUES=1,3,5,10,20] \
    [BATCH_SIZE=4]

# 全流程 pipeline（支持所有新参数）
make pipeline CATEGORY=Beauty MODEL_PATH=models/llm \
    [METHOD=rqvae] [TEXT_MODEL=...] [VISION_MODEL=...] \
    [EVAL_REC=True]
```

**Acceptance Scenarios**:

1. **Given** 研究者运行 `make sft EVAL_REC=True EVAL_REC_STEPS=0.1`, **When** SFT 训练到达 10% 步骤, **Then** 系统在验证集上执行约束波束搜索并报告 HR@K、NDCG@K
2. **Given** 研究者运行 `make rl EVAL_REC=True`, **When** RL 训练到达指定步骤, **Then** 系统同样报告推荐指标（复用现有 evaluate 逻辑）
3. **Given** 研究者未指定 `EVAL_REC`（默认关闭）, **When** 运行 `make sft` 或 `make rl`, **Then** 行为与当前系统完全一致，无性能影响

---

### Edge Cases

- 物品图像URL列表为空或全部下载失败时，视觉特征文件应全为零向量，不阻断流程
- 用户传入的 .npy 嵌入文件维度与 SID 模型期望维度不匹配时，系统应在训练前报错并提示实际维度
- 自定义 prompt 模板中缺少必要占位符（如 `{history}` 或 `{target}`）时，系统应在 `prepare_sft` 阶段报错并提示
- `prepare_sft` 生成的 JSONL 中 completion 为空字符串时（如 target SID 缺失），应跳过该样本并记录警告
- 重复运行 `prepare_sft` 时，若目标目录已存在数据，应提示用户确认覆盖或使用 `--overwrite` 参数
- 自定义奖励函数抛出异常时，系统应捕获并记录，使用默认奖励值（0.0）继续训练
- 训练期间评估的约束波束搜索可能很慢（num_beams=50），应支持用户指定较小的 beam 数（如 num_beams=10）和较少的评估样本数
- 并发下载图像时需处理网络超时、429限流等情况，支持断点续传（跳过已下载文件）
- SID统一接口在切换方法时，若前一方法的checkpoint已存在，应提示用户是否复用

## Requirements *(mandatory)*

### Functional Requirements

**数据处理模块（US1）**：
- **FR-001**: 系统 MUST 支持从 Amazon 元数据 JSON 中提取物品图像 URL 并下载最高分辨率的 MAIN 图像
- **FR-002**: 系统 MUST 支持通过 CLI 参数指定视觉特征提取器模型名称（如 CLIP、DINOv2 等 HuggingFace 模型）
- **FR-003**: 系统 MUST 将文本嵌入和视觉嵌入命令分离为独立的 Makefile 目标（`embed_text`、`extract_visual`）
- **FR-004**: 系统 MUST 保证视觉嵌入 .npy 文件的行序与 item_id 对齐（与文本嵌入一致）
- **FR-005**: 系统 MUST 支持跳过已下载的图像文件（断点续传）
- **FR-006**: 系统 MUST 在图像缺失时使用零向量填充对应行，并记录缺失数量

**SID 生成模块（US2）**：
- **FR-007**: 系统 MUST 提供统一的 `build_sid` CLI 入口，通过 `--method` 参数选择 SID 生成方法（rqvae、rqkmeans、gated_sae）
- **FR-008**: 系统 MUST 为每种 SID 方法保持各自的 token 格式（RQ-VAE/RQ-KMeans 使用位置前缀 `[a_x][b_y][c_z]`，GatedSAE 使用统一前缀 `[f_x]`）
- **FR-009**: 系统 MUST 支持用户通过 `--token_format` 自定义 token 前缀（如 `--token_format=v` 生成 `[v_x]`）
- **FR-010**: 系统 MUST 支持用户指定任意 .npy 嵌入文件作为 SID 生成的输入（文本、视觉、或用户自行处理的特征），框架不提供内置融合功能，默认使用文本 embedding

**SFT 任务模块（US3）**：
- **FR-011**: 系统 MUST 提供 `prepare_sft` 命令，将 CSV + index.json 预构建为 prompt-completion 格式的 JSONL 文件
- **FR-012**: SFT 数据 MUST 持久化保存在 `data/processed/{SID_type}/{DATASET}/{CATEGORY}/{TASK}/` 目录下，按 train/valid/test 分割，不同 TASK 的数据独立存放互不覆盖
- **FR-013**: JSONL 每行 MUST 包含 `prompt` 和 `completion` 两个字段，训练时仅在 completion 部分计算 loss（prompt 标签设为 -100）
- **FR-014**: 系统 MUST 支持用户通过 CLI 参数或模板文件指定 prompt 模板，用于 `prepare_sft` 阶段的 prompt 构造
- **FR-015**: 系统 MUST 内置 4 种 SFT 任务类型并提供注册机制允许用户扩展：
  - `sid_seq`：历史 SID 序列 → 目标 SID（序列推荐，对应现有 SidSFTDataset）
  - `item_feat`：item title → SID（特征描述，对应现有 SidItemFeatDataset）
  - `fusion`：历史 title + SID → 目标 SID（融合推荐，对应现有 FusionSeqRecDataset）
  - `sid_to_title`：SID → item title（反向映射，新增任务，帮助模型学习 SID 与语义的双向关联）
- **FR-016**: 系统 MUST 支持通过配置指定多个任务类型的混合比例（如 `TASKS=sid_seq+item_feat TASK_WEIGHTS=0.7,0.3`）
- **FR-017**: `make sft` 命令 MUST 直接从持久化的 JSONL 数据训练，无需再传入 CSV/index.json 路径
- **FR-017a**: 系统 MUST 支持 `FREEZE_LLM=True` 模式，冻结 LLM 全部原始参数（包括原始词表 embedding 和 transformer 层），仅训练通过 TokenExtender 新增的 SID token embedding
- **FR-017b**: `MODEL_PATH` MUST 同时支持基座 LLM 路径和已有 SFT checkpoint 路径，从 checkpoint 继续训练时自动加载已有的 tokenizer（含 SID token）和模型权重，实现多阶段渐进式 SFT
- **FR-017c**: 从 SFT checkpoint 继续训练时，系统 MUST 自动检测 tokenizer 中已存在的 SID token，跳过重复扩展，保留已学习的 embedding 权重

**RL 任务模块（US4）**：
- **FR-018**: 系统 MUST 提供奖励函数注册机制（装饰器模式），允许用户定义符合统一签名的奖励函数
- **FR-019**: 系统 MUST 支持通过 `--reward_type` 参数引用已注册的奖励函数名称
- **FR-020**: 系统 MUST 支持通过 `+` 语法组合多个奖励函数（如 `rule+semantic`）并指定权重
- **FR-020a**: RL 训练 MUST 复用 SFT 的 prompt 模板机制，支持通过 `--prompt_template` 参数定制 prompt 构造方式，保持 CSV 作为数据输入格式

**评估模块（US5）**：
- **FR-021**: 系统 MUST 支持在 SFT 训练过程中按指定步骤间隔执行推荐指标评估（通过 `EVAL_REC=True` 启用）
- **FR-022**: 系统 MUST 支持在 RL 训练过程中按指定步骤间隔执行推荐指标评估
- **FR-023**: 训练期间评估 MUST 默认关闭，仅在用户显式启用时执行
- **FR-024**: 训练期间评估 MUST 支持配置评估样本数和 beam 数以控制评估速度

**Make 命令体系（跨模块）**：
- **FR-025**: 所有新增 make 命令 MUST 使用 `[PARAM=default]` 风格的可选参数，与现有命令保持一致
- **FR-026**: 现有的 `make embed`、`make build_sid`、`make build_sae_sid`、`make pipeline` 命令 MUST 保持向后兼容
- **FR-027**: 系统 MUST 提供 `make pipeline` 的扩展版本，支持通过参数选择完整流程中的各环节配置
- **FR-028**: 系统 MUST 提供 `make list_sft_tasks`、`make list_rewards` 等发现命令，帮助用户了解可用选项

### Make 命令完整汇总

| 模块 | 命令 | 说明 | 关键参数 |
|------|------|------|---------|
| **数据处理** | `make preprocess` | k-core 过滤 + 分割（现有） | `CATEGORY`, `SPLIT_METHOD` |
| | `make download_images` | 下载物品图像 | `CATEGORY`, `CONCURRENCY` |
| | `make embed_text` | 提取文本嵌入 | `CATEGORY`, `TEXT_MODEL` |
| | `make extract_visual` | 提取视觉特征 | `CATEGORY`, `VISION_MODEL` |
| | `make embed_all` | 提取全部特征（文本+视觉） | `CATEGORY`, `TEXT_MODEL`, `VISION_MODEL` |
| | `make embed` | embed_text 的别名（向后兼容） | `CATEGORY` |
| **SID 生成** | `make build_sid` | 统一 SID 生成（训练+生成） | `CATEGORY`, `METHOD`, `EMB_PATH`, `K`, `TOKEN_FORMAT` |
| | `make train_sid` | 仅训练 SID 模型 | `CATEGORY`, `METHOD`, `EMB_PATH` |
| | `make generate_sid` | 仅从模型生成 .index.json | `CATEGORY`, `METHOD`, `SID_MODEL_DIR` |
| | `make build_sae_sid` | build_sid METHOD=gated_sae 别名（向后兼容） | `CATEGORY` |
| **数据转换** | `make convert` | .inter → CSV + info TXT（现有） | `CATEGORY` |
| **SFT 数据** | `make prepare_sft` | CSV + index.json → prompt-completion JSONL | `CATEGORY`, `SID_TYPE`, `TASK`, `PROMPT_TEMPLATE`, `DATASET`(默认Amazon) |
| **SFT 训练** | `make sft` | 从 JSONL 训练（completion-only loss） | `CATEGORY`, `MODEL_PATH`, `SFT_DATA_DIR`, `EVAL_REC` |
| | `make sft_quick` | 快速验证 SFT | `CATEGORY`, `MODEL_PATH`, `SFT_DATA_DIR`, `SAMPLE` |
| | `make list_sft_tasks` | 列出可用任务类型和模板 | — |
| **RL 训练** | `make rl` | GRPO RL 训练（扩展） | `CATEGORY`, `MODEL_PATH`, `REWARD_TYPE`, `REWARD_WEIGHTS`, `PROMPT_TEMPLATE`, `EVAL_REC` |
| | `make rl_quick` | 快速验证 RL | `CATEGORY`, `MODEL_PATH`, `SAMPLE` |
| | `make list_rewards` | 列出已注册奖励函数 | — |
| **评估** | `make evaluate` | 约束波束搜索评估（扩展） | `CATEGORY`, `MODEL_PATH`, `NUM_BEAMS`, `K_VALUES` |
| **CF 模型** | `make sasrec` | SASRec 训练（现有） | `CATEGORY` |
| **全流程** | `make pipeline` | 端到端（扩展） | 所有模块参数 |

### Key Entities

- **EmbeddingFile**: 物品特征向量文件（.npy），关键属性：模态（text/visual）、模型名称、维度、物品数量。行序与 item_id 对齐。
- **SIDMethod**: SID 生成方法的抽象，关键属性：方法名称（rqvae/rqkmeans/gated_sae）、token 格式、模型参数。实现统一的 train→generate 接口。
- **SFTDataFile**: 持久化的 SFT 训练数据（JSONL），关键属性：所在目录（`data/processed/{SID_type}/{DATASET}/{CATEGORY}/{TASK}/`）、split（train/valid/test）、生成参数（meta.json）。每行包含 `prompt` 和 `completion` 字段。不同 TASK 独立存放，支持多阶段训练。
- **PromptTemplate**: SFT prompt 模板，关键属性：instruction 文本、input 格式（含占位符 `{history}`、`{target}`、`{category}`）。用于 `prepare_sft` 阶段构造 prompt 字段。
- **RewardFunction**: RL 奖励函数，关键属性：函数名称、签名 `(predictions, target, **kwargs) -> List[float]`、是否需要额外参数。
- **TrainingEvaluator**: 训练期间评估器，关键属性：评估间隔、评估样本数、beam 数、k_values。

## Assumptions

- Amazon 元数据 JSON 中的 `imageURLHighRes` 字段包含物品图像 URL 列表，MAIN 图像通常是列表中的第一张或最大分辨率的图像
- 视觉特征提取器使用 HuggingFace transformers 库的预训练模型，通过 `AutoModel.from_pretrained()` 加载
- 图像下载使用并发请求（线程池），默认并发数为 8，支持用户配置
- 自定义 prompt 模板支持 Python format string 语法，占位符包括 `{history}`、`{target}`、`{category}`
- SFT 数据持久化为 JSONL 格式（每行一条 JSON），便于流式读取和调试检查
- `prepare_sft` 阶段将数据构造与训练解耦：研究者可先生成数据、检查质量、再启动训练
- Completion-only loss 通过 TRL `SFTTrainer` + `DataCollatorForCompletionOnlyLM(response_template="### Response:\n")` 自动实现，无需手动设置 label 为 -100
- `data/processed/{SID_type}/{DATASET}/{CATEGORY}/{TASK}/meta.json` 记录生成参数（SID_type、TASK、TEMPLATE 路径、时间戳），支持实验复现
- 奖励函数注册机制使用 Python 装饰器模式（如 `@register_reward("my_reward")`），无需修改框架核心代码
- 训练期间评估复用现有的 `evaluate.py` 逻辑，但使用较小的 beam 数和样本数以控制耗时
- 所有新增 CLI 命令遵循现有的 `python-fire` 风格
- 现有的 `text2emb` 命令重命名为 `embed_text`（保留向后兼容的别名）

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 研究者可以在 5 分钟内完成从原始数据到多模态特征（文本+视觉）的提取流程（不含下载时间），无需修改框架代码
- **SC-002**: 研究者可以通过单个 CLI 命令切换 SID 生成方法（RQ-VAE/RQ-KMeans/GatedSAE），生成格式统一的 .index.json
- **SC-003**: 研究者可以在不修改框架源代码的情况下，定义并使用自定义 SFT prompt 模板
- **SC-004**: 研究者可以在不修改框架源代码的情况下，注册并使用自定义 RL 奖励函数
- **SC-005**: 训练期间推荐指标（HR@10、NDCG@10）与训练后独立评估的结果差异不超过 5%（在相同评估参数下）
- **SC-006**: 所有新增功能覆盖单元测试，核心流程有集成测试
- **SC-007**: 现有的 RQ-VAE 和 GatedSAE 流程在重构后保持完全兼容，已有的 Makefile 命令不受影响
