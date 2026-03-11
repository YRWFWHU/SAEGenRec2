# Phase 0 Research: 模块化框架扩展

**Branch**: `003-modular-framework-extension` | **Date**: 2026-03-09

## 1. 图像下载并发策略

**Decision**: 使用 `concurrent.futures.ThreadPoolExecutor`，默认 `max_workers=8`

**Rationale**:
- 图像下载为 I/O 密集型任务，线程池比进程池更轻量
- ThreadPoolExecutor 是标准库，无需额外依赖
- 默认 8 并发兼顾速度与 Amazon 限流（避免 429）
- 已有 `requests` 库可直接使用

**Alternatives considered**:
- `asyncio + aiohttp`：性能更优但增加异步复杂度，不值得
- `multiprocessing.Pool`：下载任务非 CPU 密集，进程开销不必要

## 2. 视觉特征提取 API

**Decision**: 使用 HuggingFace `transformers.AutoModel` + `AutoProcessor` 统一接口

**Rationale**:
- CLIP (`openai/clip-vit-base-patch32`) 和 DINOv2 (`facebook/dinov2-base`) 都支持 `AutoModel.from_pretrained()`
- `AutoProcessor` 自动处理图像预处理（resize、normalize）
- 与项目已有的 HuggingFace 生态一致
- CLIP 使用 `model.get_image_features()`，DINOv2 使用 `model(pixel_values).last_hidden_state[:, 0]`
- 需要封装统一的 `extract_features(model, processor, images) -> np.ndarray` 接口

**Alternatives considered**:
- `torchvision.models`：不支持 CLIP，接口不统一
- `timm`：良好但不含 CLIP，且与现有 HF 生态不一致
- `sentence-transformers`（CLIP 模式）：仅限 CLIP 系列，不够通用

## 3. JSONL 数据集加载与 Completion-Only Loss

**Decision**: 使用 TRL `SFTTrainer` + `DataCollatorForCompletionOnlyLM` 实现 completion-only masking

**Rationale**:
- 项目已依赖 TRL（GRPO RL 训练），无需新增依赖
- `SFTTrainer` 原生支持 JSONL/HF Dataset 输入，配合 `DataCollatorForCompletionOnlyLM` 自动将 prompt 部分的 label 设为 -100
- 消除手动在 Dataset 中计算 `labels = [-100] * prompt_len + completion_tokens` 的样板代码
- `DataCollatorForCompletionOnlyLM(response_template, tokenizer)` 通过 response_template（如 `"### Response:\n"`）自动识别 completion 起始位置
- JSONL 格式：每行 `{"prompt": "...", "completion": "..."}`，加载后拼接为完整文本，由 collator 处理 masking
- `SFTTrainer` 兼容 HuggingFace `TrainerCallback`，训练期间评估回调可无缝集成

**Alternatives considered**:
- 手动 `labels = [-100] * prompt_len + completion_tokens`：可行但样板代码多，每个 Dataset 类都需重复实现 masking 逻辑
- `datasets.load_dataset("json")`+ 自定义 Trainer：不如 SFTTrainer 开箱即用

## 4. SID 方法注册表模式

**Decision**: 基类 `SIDMethod` + 子类注册字典，在 `registry.py` 中维护 `{name: class}` 映射

**Rationale**:
- 与 spec 中设计的 `@register_reward` 装饰器模式一致
- 三种方法（RQ-VAE、RQ-KMeans、GatedSAE）有共同的 `train()` + `generate()` 接口
- 注册表 + 工厂模式让 `build_sid --method=xxx` 只需查表调用
- 新方法只需继承 `SIDMethod` 并注册，无需修改 `__main__.py`

**Alternatives considered**:
- `importlib` 动态导入：过于魔法，调试困难
- `if/elif` 分支：不可扩展，违反 OCP

## 5. 奖励函数注册机制

**Decision**: `@register_reward("name")` 装饰器 + 全局注册字典 `_REWARD_REGISTRY`

**Rationale**:
- 现有 rewards.py 已定义 5 个函数（rule、prefix、ranking、semantic、sasrec），签名统一
- 装饰器模式零侵入：现有函数只需加一行 `@register_reward("rule")`
- 组合奖励通过 `CombinedReward(names=["rule", "semantic"], weights=[0.6, 0.4])` 实现
- `list_rewards()` 直接遍历注册字典

**Alternatives considered**:
- Python `entry_points`：用于跨包插件，单包内过度设计
- `ABC` 基类强制接口：现有函数已是简单函数签名，类化不必要

## 6. 训练期间评估回调

**Decision**: HuggingFace `TrainerCallback`，在 `on_evaluate` 或 `on_step_end` 中触发子集评估

**Rationale**:
- HuggingFace Trainer 原生支持 Callback 注入
- 可复用现有 `evaluate.py` 的 `constrained_beam_search` + `compute_hr_ndcg` 逻辑
- 通过 `EVAL_REC_SAMPLES` 和 `EVAL_REC_BEAMS` 控制评估开销
- 默认关闭（`EVAL_REC=False`），零性能影响

**Alternatives considered**:
- 自定义 training loop：破坏现有 Trainer 集成
- `evaluate` 命令每 N 步外部调用：需额外进程管理，复杂度高

## 7. Prompt 模板机制

**Decision**: 纯文本文件 + Python `str.format_map()` 占位符替换

**Rationale**:
- 模板文件放在 `templates/` 目录，易于查看和修改
- `{history}`, `{target}`, `{category}`, `{title}`, `{sid}` 等占位符直观
- `str.format_map()` 对缺失键友好（可用 `defaultdict` 处理可选占位符）
- 与 `prepare_sft --prompt_template=templates/my_prompt.txt` CLI 接口匹配

**Alternatives considered**:
- Jinja2：对简单替换过重，增加依赖
- Python f-string / eval：安全风险

## 8. Token 格式自定义

**Decision**: `--token_format` 参数控制 token 前缀，默认 `auto`（根据方法自动选择）

**Rationale**:
- RQ-VAE/RQ-KMeans 默认位置前缀 `a/b/c/...`（每层不同 codebook）
- GatedSAE 默认统一前缀 `f`（单一 codebook）
- `--token_format=v` 可覆盖为 `[v_0][v_1]...`（视觉特征场景）
- `auto` 模式下 RQ 系列用位置前缀，SAE 系列用 `f`

**Alternatives considered**:
- JSON 配置文件指定格式：对单参数过重
- 固定格式不可改：限制研究灵活性

## 9. SFT Checkpoint 续训与 Token 检测

**Decision**: 加载 tokenizer 时检查 vocab 中是否已含 SID token pattern（如 `[a_0]`、`[f_0]`），若存在则跳过 `TokenExtender` 扩展

**Rationale**:
- 多阶段 SFT 需要从 stage1 checkpoint 继续训练
- stage1 的 tokenizer 已包含 SID token，重复添加会导致 embedding 矩阵维度错误
- 检测逻辑简单：`if "[a_0]" in tokenizer.get_vocab() or "[f_0]" in tokenizer.get_vocab(): skip_extend()`
- 保留已学习的 embedding 权重是课程学习的核心价值

**Alternatives considered**:
- 总是重新扩展并随机初始化：丢失已学习的 SID 语义
- 配置文件标记是否已扩展：增加状态管理复杂度

## 10. 冻结 LLM 参数策略

**Decision**: `FREEZE_LLM=True` 时，遍历 `model.named_parameters()` 对非 SID token embedding 的参数设 `requires_grad=False`

**Rationale**:
- TokenExtender 新增的 SID token 在 embedding 矩阵最后几行
- 冻结策略：所有参数 `requires_grad=False`，然后仅对 embedding 权重的 SID token 行恢复 `requires_grad=True`
- 实现：`model.get_input_embeddings().weight` 的 `[original_vocab_size:]` 部分可训练
- 训练速度提升 ~3-5x，显存降低 ~60%

**Alternatives considered**:
- LoRA：增加 peft 依赖，与现有 TokenExtender 机制不兼容
- 仅冻结 transformer 层（保留全部 embedding 可训练）：不够精确
