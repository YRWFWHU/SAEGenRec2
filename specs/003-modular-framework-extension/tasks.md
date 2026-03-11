# Tasks: 模块化框架扩展

**Input**: Design documents from `/specs/003-modular-framework-extension/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Included — plan.md specifies 7 test files.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Package**: `SAEGenRec/` at repository root
- **Tests**: `tests/` at repository root
- **Templates**: `templates/` at repository root
- **Config**: `SAEGenRec/config.py`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add dependencies, config dataclasses, and prompt templates shared across user stories

- [X] T001 Add `Pillow` and `requests` to dependencies in `pyproject.toml`
- [X] T002 [P] Add `VisualEmbConfig` dataclass (vision_model, batch_size, device, data_dir) and `PrepSFTConfig` dataclass (category, sid_type, task, dataset, prompt_template, data_dir, interim_dir, overwrite) to `SAEGenRec/config.py`
- [X] T003 [P] Create 4 prompt template files: `templates/sid_seq.txt` (placeholder: `{history}`), `templates/item_feat.txt` (placeholder: `{title}`), `templates/fusion.txt` (placeholders: `{history}`, `{titles}`), `templates/sid_to_title.txt` (placeholder: `{sid}`). Each uses the existing instruction format from `SidSFTDataset.pre()` with `### Response:\n` as the response separator. Include the instruction header and `### User Input:` / `### Response:` markers.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Prompt template loading utility used by US3, US4, US5

**⚠️ CRITICAL**: US3、US4、US5 的实现需要此阶段完成。US1 和 US2 不依赖此阶段，可在 Phase 1 完成后立即开始。

- [X] T004 Create prompt template loader utility function `load_template(path: str) -> str` and `render_template(template: str, **kwargs) -> str` (using `str.format_map` with `defaultdict(str)` for missing keys) in `SAEGenRec/datasets/template_utils.py`. Include validation that required placeholders are present.

**Checkpoint**: Foundation ready — user story implementation can now begin

---

## Phase 3: User Story 1 — 多模态数据处理 (Priority: P1) 🎯 MVP

**Goal**: 从 Amazon 元数据下载物品图像并提取视觉特征，重构 embed_text 命令

**Independent Test**: `make download_images CATEGORY=Beauty` 下载图像，`make extract_visual CATEGORY=Beauty` 生成 `.emb-*-visual.npy`，`make embed_text CATEGORY=Beauty` 生成 `.emb-*-text.npy`

### Tests for US1

- [X] T005 [P] [US1] Write unit tests for image downloader in `tests/test_image_downloader.py`: test URL extraction from meta JSON, test skip existing files, test zero-vector fallback for missing images, test concurrency parameter, test retry on failure (mock requests)
- [X] T006 [P] [US1] Write unit tests for visual embed in `tests/test_visual_embed.py`: test output shape (n_items, d_visual), test zero-vector for missing images, test row order matches item.json, test NaN/Inf cleaning

### Implementation for US1

- [X] T007 [P] [US1] Implement `download_images()` function in `SAEGenRec/data_process/image_downloader.py`: parse `imageURLHighRes` from `data/raw/meta_{category}.json`, use `concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)` to download MAIN image (first URL), save as `data/interim/{category}/images/{item_asin}.jpg`, skip existing files, retry 3 times on network errors, log success/failure/skip statistics via loguru. Parameters per contracts/data_process.md.
- [X] T008 [P] [US1] Implement `extract_visual()` function in `SAEGenRec/data_process/visual_embed.py`: load HuggingFace vision model via `AutoModel.from_pretrained(vision_model)` + `AutoProcessor.from_pretrained(vision_model)`, iterate items in `{category}.item.json` order, load each `{item_asin}.jpg` from image dir, batch process on GPU, output `{category}.emb-{model_slug}-visual.npy`. Use zero vector for missing/corrupt images. Log missing count. Parameters per contracts/data_process.md.
- [X] T009 [US1] Refactor `SAEGenRec/sid_builder/text2emb.py`: extract core embedding logic into new `SAEGenRec/data_process/embed_text.py` with `embed_text()` function, output naming `{category}.emb-{model_slug}-text.npy` (model_slug strips `sentence-transformers/` prefix). Keep `text2emb` in `sid_builder/text2emb.py` as thin wrapper calling `embed_text()` for backward compatibility.
- [X] T010 [US1] Update `SAEGenRec/data_process/__main__.py`: register `download_images` (from image_downloader), `embed_text` (from embed_text), `extract_visual` (from visual_embed) in fire.Fire() dispatch dict. Keep existing `preprocess` and `convert_dataset`.

**Checkpoint**: US1 complete — image download + visual/text embedding extraction functional

---

## Phase 4: User Story 2 — SID 生成统一接口 (Priority: P1)

**Goal**: 通过 `--method` 参数在 RQ-VAE/RQ-KMeans/GatedSAE 之间切换，统一 train+generate 接口

**Independent Test**: `make build_sid CATEGORY=Beauty METHOD=rqvae` 和 `make build_sid CATEGORY=Beauty METHOD=gated_sae` 均能生成 `.index.json`

### Tests for US2

- [X] T011 [P] [US2] Write unit tests for SID registry in `tests/test_sid_registry.py`: test all 3 methods registered, test registry lookup by name, test unknown method raises error, test SIDMethod base class interface (train/generate signatures), test `token_format=auto` resolution (rqvae→position prefix, gated_sae→`f` prefix), test custom token_format override

### Implementation for US2

- [X] T012 [P] [US2] Create `SIDMethod` abstract base class in `SAEGenRec/sid_builder/base.py`: define `name`, `default_k`, `token_format` class attributes and abstract methods `train(embedding_path, output_dir, **config) -> str` and `generate(checkpoint, embedding_path, output_path, k, token_format) -> str`. See data-model.md Entity 3.
- [X] T013 [P] [US2] Create SID method registry in `SAEGenRec/sid_builder/registry.py`: `SID_METHODS: Dict[str, Type[SIDMethod]]` dict, `register_sid_method(name)` decorator, `get_sid_method(name) -> SIDMethod` lookup function, `list_sid_methods() -> List[str]` helper.
- [X] T014 [P] [US2] Adapt `SAEGenRec/sid_builder/rqvae.py`: create `RQVAEMethod(SIDMethod)` class wrapping existing `rqvae_train` as `train()` and existing `generate_indices` as `generate()`. Set `name="rqvae"`, `default_k=3`. Register via `@register_sid_method("rqvae")`. Keep `rqvae_train()` function for backward compat.
- [X] T015 [P] [US2] Adapt `SAEGenRec/sid_builder/rqkmeans.py`: create `RQKMeansMethod(SIDMethod)` class wrapping existing `rqkmeans_faiss`/`rqkmeans_constrained`/`rqkmeans_plus` as `train()` and `generate_indices` as `generate()`. Set `name="rqkmeans"`, `default_k=3`. Register via `@register_sid_method("rqkmeans")`. Keep existing functions for backward compat.
- [X] T016 [P] [US2] Adapt `SAEGenRec/sid_builder/gated_sae.py`: create `GatedSAEMethod(SIDMethod)` class wrapping existing `gated_sae_train` as `train()` and `generate_sae_indices` as `generate()`. Set `name="gated_sae"`, `default_k=8`. Register via `@register_sid_method("gated_sae")`. Keep existing functions for backward compat.
- [X] T017 [P] [US2] Add `token_format` parameter to `generate_indices()` in `SAEGenRec/sid_builder/generate_indices.py`: when `token_format="auto"` use existing `a/b/c/...` positional prefixes, when single char (e.g. `"v"`) use `[v_0][v_1]...` uniform prefix. Default `"auto"`.
- [X] T018 [P] [US2] Add `token_format` parameter to `generate_sae_indices()` in `SAEGenRec/sid_builder/generate_sae_indices.py`: when `token_format="auto"` use existing `f` prefix, when custom char use that char. Default `"auto"`.
- [X] T019 [US2] Implement `build_sid()`, `train_sid()`, `generate_sid()` CLI dispatch functions in `SAEGenRec/sid_builder/__main__.py`: `build_sid` looks up method from registry → calls `method.train()` then `method.generate()`. Add `build_sid`, `train_sid`, `generate_sid` to fire.Fire() dict. Keep all existing command names for backward compat. Validate input .npy embedding dimensions match method expectations before training (raise early error with actual vs expected shape). When switching methods, check if previous method's checkpoint exists and log info message.

**Checkpoint**: US2 complete — unified SID interface with 3 methods and token format customization

---

## Phase 5: User Story 3 — 可自定义 SFT 任务 (Priority: P2)

**Goal**: JSONL 持久化 SFT 数据 + SFTTrainer completion-only loss + freeze LLM + 多阶段训练

**Independent Test**: `make prepare_sft CATEGORY=Beauty SID_TYPE=rqvae TASK=sid_seq` 生成 JSONL，`make sft SFT_DATA_DIR=... MODEL_PATH=models/llm` 训练正常收敛

### Tests for US3

- [X] T020 [P] [US3] Write unit tests for task registry in `tests/test_task_registry.py`: test 4 builtin tasks registered (sid_seq, item_feat, fusion, sid_to_title), test custom task registration via decorator, test required_placeholders validation, test unknown task raises error
- [X] T021 [P] [US3] Write unit tests for prepare_sft in `tests/test_prepare_sft.py`: test JSONL output format (each line has `prompt` + `completion` keys), test train/valid/test split, test meta.json written with correct parameters, test empty completion lines skipped, test overwrite protection, test template placeholder rendering

### Implementation for US3

- [X] T022 [US3] Implement SFT task registry with 4 builtin tasks in `SAEGenRec/datasets/task_registry.py`: define `SFTTask` base class with `name`, `default_template`, `required_inputs`, `required_placeholders` attributes and `build_examples(csv_data, index_json, item_json, template, **kwargs) -> List[dict]` method. Implement `SidSeqTask`, `ItemFeatTask`, `FusionTask`, `SidToTitleTask`. Use `@register_sft_task(name)` decorator + `_SFT_TASK_REGISTRY` dict + `get_sft_task(name)` + `list_sft_tasks()`. Port prompt construction logic from existing `SidSFTDataset`, `SidItemFeatDataset`, `FusionSeqRecDataset` in `SAEGenRec/datasets/sft_datasets.py`, using templates from `templates/` via `template_utils.py`.
- [X] T023 [US3] Implement `prepare_sft()` function in `SAEGenRec/training/prepare_sft.py`: load CSV splits + index.json + item.json, look up task from registry, render each example using prompt template, write train/valid/test.jsonl to `data/processed/{sid_type}/{dataset}/{category}/{task}/`, write meta.json, skip empty completions with warning, check overwrite flag. Parameters per contracts/sft.md. Support multi-task mixing (`tasks` + `task_weights` params).
- [X] T024 [US3] Refactor `sft()` in `SAEGenRec/training/sft.py` to accept `sft_data_dir` (JSONL) instead of CSV paths: load train.jsonl + valid.jsonl, concatenate prompt+completion into full text, use TRL `SFTTrainer` with `DataCollatorForCompletionOnlyLM(response_template="### Response:\n", tokenizer=tokenizer)` for automatic completion-only loss. Remove old CSV-based dataset construction (SidSFTDataset/SidItemFeatDataset/FusionSeqRecDataset direct usage in sft.py). Auto-detect `category` from meta.json if not provided.
- [X] T025 [US3] Implement SID token auto-detection in `SAEGenRec/training/sft.py`: after loading tokenizer from `model_path`, check if SID tokens already exist by scanning vocab for any token matching pattern `\[.+_\d+\]` (e.g. `[a_0]`, `[f_0]`, `[v_0]` etc.). If yes: skip `TokenExtender`, log "SID tokens already present, skipping extension". If no: scan JSONL data to extract SID token set (all `[x_N]` patterns from completions), add via `tokenizer.add_tokens()` + `model.resize_token_embeddings()`.
- [X] T026 [US3] Ensure `freeze_llm` logic in `SAEGenRec/training/sft.py` works with SFTTrainer: freeze all params, then unfreeze embedding rows `[original_vocab_size:]` for SID tokens, register gradient mask hook. Verify existing implementation from `sft()` function still works under SFTTrainer.
- [X] T027 [US3] Implement `list_sft_tasks()` function in `SAEGenRec/training/prepare_sft.py`: print all registered tasks with name, description, and default template path. Register `prepare_sft` and `list_sft_tasks` in `SAEGenRec/training/__main__.py` fire.Fire() dict.

**Checkpoint**: US3 complete — JSONL SFT data pipeline + SFTTrainer + freeze_llm + multi-stage training

---

## Phase 6: User Story 4 — 可自定义 RL 奖励函数 (Priority: P2)

**Goal**: 装饰器注册奖励函数 + `+` 语法组合奖励 + prompt 模板支持

**Independent Test**: `make rl REWARD_TYPE=rule+prefix REWARD_WEIGHTS=0.6,0.4` 训练正常运行且日志显示组合奖励

### Tests for US4

- [X] T028 [P] [US4] Write unit tests for reward registry in `tests/test_reward_registry.py`: test 5 builtin rewards registered (rule, prefix, ranking, semantic, sasrec), test custom reward registration via `@register_reward`, test `get_reward_fn("rule")` returns correct function, test `CombinedReward` weighted aggregation, test `reward_type="rule+semantic"` parsing, test unknown reward raises error with available list, test exception handling returns 0.0

### Implementation for US4

- [X] T029 [US4] Add reward registry infrastructure to `SAEGenRec/training/rewards.py`: add `_REWARD_REGISTRY: Dict[str, Callable]` dict, `register_reward(name)` decorator function, `get_reward_fn(name) -> Callable` lookup, `list_rewards() -> Dict[str, str]` helper. Add `@register_reward("rule")` / `@register_reward("prefix")` / `@register_reward("ranking")` / `@register_reward("semantic")` / `@register_reward("sasrec")` decorators to existing 5 functions (no logic changes).
- [X] T030 [US4] Implement `CombinedReward` class in `SAEGenRec/training/rewards.py`: `__init__(names: List[str], weights: List[float])` resolves each name from registry, `__call__(predictions, target, **kwargs) -> List[float]` calls each reward and returns weighted sum. Add `parse_reward_type(reward_type: str, reward_weights: str) -> Callable` that splits on `+`, parses weights, returns single fn or CombinedReward.
- [X] T031 [US4] Extend `rl()` in `SAEGenRec/training/rl.py`: replace hardcoded reward dispatch with `parse_reward_type(reward_type, reward_weights)` call. Add `prompt_template` parameter: if provided, load template file and use `render_template()` from template_utils to construct prompts (replacing `{history}`, `{category}` etc.). Add `reward_weights` parameter. Keep all existing parameters unchanged.
- [X] T032 [US4] Register `list_rewards` command in `SAEGenRec/training/__main__.py`: add `list_rewards` function that calls `rewards.list_rewards()` and prints formatted output. Add to fire.Fire() dict.

**Checkpoint**: US4 complete — reward registry + combined rewards + prompt template in RL

---

## Phase 7: User Story 5 — 训练期间推荐指标评估 (Priority: P3)

**Goal**: SFT/RL 训练中可选 HR@K/NDCG@K 评估回调

**Independent Test**: `make sft EVAL_REC=True EVAL_REC_STEPS=0.1` 日志中每 10% 步出现 HR@10/NDCG@10

### Tests for US5

- [X] T033 [P] [US5] Write unit tests for training evaluator in `tests/test_training_evaluator.py`: test callback creation with parameters, test disabled by default (eval_rec=False), test step interval calculation, test metrics logging format, test integration with Trainer mock (callback triggers at correct steps)

### Implementation for US5

- [X] T034 [US5] Refactor `SAEGenRec/evaluation/evaluate.py`: extract core evaluation logic into `evaluate_subset(model, tokenizer, test_data, info_file, num_beams, n_samples, k_values, device) -> Dict[str, float]` function that can be called programmatically (not just as CLI). Keep existing `evaluate()` CLI function unchanged, have it call `evaluate_subset` internally.
- [X] T035 [US5] Implement `TrainingEvaluator` callback in `SAEGenRec/evaluation/training_evaluator.py`: subclass `transformers.TrainerCallback`, accept `eval_rec_steps`, `eval_rec_beams`, `eval_rec_samples`, `info_file`, `test_csv`, `k_values` in `__init__`. In `on_step_end`: check if current step matches interval, if so call `evaluate_subset()` with trainer's model and tokenizer, log metrics via loguru and `trainer.log()`. Format: `[TrainingEvaluator] Step {step}/{total}: HR@1=..., HR@5=..., HR@10=..., NDCG@5=..., NDCG@10=...`.
- [X] T036 [US5] Integrate `TrainingEvaluator` in `SAEGenRec/training/sft.py`: add `eval_rec`, `eval_rec_steps`, `eval_rec_beams`, `eval_rec_samples` parameters. When `eval_rec=True`: create `TrainingEvaluator` callback, derive `test_csv` and `info_file` from `sft_data_dir/meta.json`, add callback to `SFTTrainer(callbacks=[...])`.
- [X] T037 [US5] Integrate `TrainingEvaluator` in `SAEGenRec/training/rl.py`: add same `eval_rec*` parameters. When `eval_rec=True`: create `TrainingEvaluator` callback, add to `ReReTrainer(callbacks=[...])`.

**Checkpoint**: US5 complete — optional training-time evaluation for both SFT and RL

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Makefile targets, backward compatibility, and end-to-end validation

- [X] T038 Update `Makefile` with all new targets per spec.md Make 命令完整汇总 table: `download_images`, `embed_text`, `extract_visual`, `embed_all`, `build_sid` (unified with METHOD param), `train_sid`, `generate_sid`, `prepare_sft`, `sft` (updated for SFT_DATA_DIR), `sft_quick`, `list_sft_tasks`, `rl` (updated for REWARD_TYPE/REWARD_WEIGHTS/PROMPT_TEMPLATE), `rl_quick`, `list_rewards`, `evaluate` (updated params), `pipeline` (extended). Keep `embed` as alias for `embed_text`, `build_sae_sid` as alias for `build_sid METHOD=gated_sae`.
- [X] T039 [P] Add backward-compat alias for `text2emb` in `SAEGenRec/sid_builder/__main__.py`: keep `"text2emb": text2emb` in fire.Fire() dict pointing to original function. Verify all existing CLI commands still work: `rqvae_train`, `generate_indices`, `gated_sae_train`, `generate_sae_indices`, `rqkmeans_*`.
- [X] T040 [P] Run `python -m pytest tests/` to verify no regressions on existing tests (`test_gated_sae.py`, `test_data.py`)
- [X] T041 Run `make lint` and `make format` to ensure code passes ruff checks
- [X] T042 Validate quickstart.md scenario 1 (end-to-end) by dry-running the command sequence with `--help` flags to verify all new CLI commands are registered and accept expected parameters

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — template_utils is used by US3, US4
- **US1 (Phase 3)**: Depends on Phase 1 (config dataclasses) — no dependency on Phase 2
- **US2 (Phase 4)**: Depends on Phase 1 — no dependency on other stories
- **US3 (Phase 5)**: Depends on Phase 2 (template_utils) and Phase 4 (SID registry for SID token extraction)
- **US4 (Phase 6)**: Depends on Phase 2 (template_utils)
- **US5 (Phase 7)**: Depends on Phase 5 (sft.py refactored) and Phase 6 (rl.py extended)
- **Polish (Phase 8)**: Depends on all user stories complete

### User Story Dependencies

- **US1 (P1)**: Can start after Phase 1 — fully independent
- **US2 (P1)**: Can start after Phase 1 — fully independent, parallelizable with US1
- **US3 (P2)**: Requires Phase 2 (template_utils). Soft dependency on US2 (needs SID token patterns for auto-detection, but can use hardcoded patterns for initial implementation)
- **US4 (P2)**: Requires Phase 2 (template_utils). Independent of US1-US3
- **US5 (P3)**: Requires US3 (sft.py refactored with SFTTrainer) and US4 (rl.py extended). Must be last user story.

### Within Each User Story

- Tests first (marked [P] where possible)
- Base classes / registries before implementations
- Implementations before CLI command registration
- CLI registration last within each story

### Parallel Opportunities

- **Phase 1**: T002 [P] and T003 [P] can run in parallel
- **Phase 3 (US1)**: T005 [P] + T006 [P] (tests), T007 [P] + T008 [P] (implementations on different files)
- **Phase 4 (US2)**: T014 [P] + T015 [P] + T016 [P] (adapting 3 SID methods in parallel), T017 [P] + T018 [P] (token_format on different files)
- **Phase 5 (US3)**: T020 [P] + T021 [P] (tests)
- **US1 and US2 can run entirely in parallel** (different modules, no shared files)

---

## Parallel Example: US1 + US2 (Phase 3 + Phase 4)

```
# These two user stories can be worked on simultaneously:

# US1 stream (data_process module):
T005, T006 (tests in parallel)
T007, T008 (image_downloader + visual_embed in parallel)
T009 (embed_text refactor)
T010 (CLI registration)

# US2 stream (sid_builder module):
T011 (tests)
T012, T013 (base + registry in parallel)
T014, T015, T016 (3 method adapters in parallel)
T017, T018 (token_format on 2 files in parallel)
T019 (CLI registration)
```

---

## Implementation Strategy

### MVP First (US1 + US2 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004)
3. Complete Phase 3: US1 — 多模态数据处理 (T005-T010)
4. Complete Phase 4: US2 — SID 统一接口 (T011-T019)
5. **STOP and VALIDATE**: `make embed_text`, `make extract_visual`, `make build_sid METHOD=rqvae`, `make build_sid METHOD=gated_sae` all work

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. US1 + US2 → Data pipeline complete (MVP!)
3. US3 → SFT customization with JSONL + multi-stage
4. US4 → RL reward customization
5. US5 → Training-time evaluation monitoring
6. Polish → Makefile, backward compat, validation

### Single Developer Strategy

Phase 1 → Phase 2 → Phase 3 & 4 (US1 + US2 in parallel or sequential) → Phase 5 (US3) → Phase 6 (US4) → Phase 7 (US5) → Phase 8 (Polish)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- SFT refactoring (T024) switches from `transformers.Trainer` to TRL `SFTTrainer` — this is the biggest single change
- Backward compatibility is critical: all existing `make` commands and CLI functions must continue working
