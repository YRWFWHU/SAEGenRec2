# Tasks: GatedSAE SID 生成

**Input**: Design documents from `/specs/002-gated-sae-sid/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 添加 SAELens 依赖和配置数据类

- [X] T001 Add sae-lens dependency to pyproject.toml in `[project.optional-dependencies]` or `[project.dependencies]`
- [X] T002 Add GatedSAEConfig dataclass to SAEGenRec/config.py with fields: embedding_path, d_in, expansion_factor, k, l1_coefficient, lr, total_training_samples, train_batch_size, output_dir, device, seed, max_dedup_iters (see data-model.md for defaults)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 数据适配层，US1 和 US2 共同依赖

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 Implement NpyDataProvider class in SAEGenRec/sid_builder/gated_sae.py — load .npy file into memory, implement `__iter__` and `__next__` returning `(batch_size, d_in)` tensors, support multi-epoch cycling with shuffle, handle NaN/Inf values (replace with 0)

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - 训练 GatedSAE 模型 (Priority: P1) 🎯 MVP

**Goal**: 研究者可以在 item text embedding 上训练 GatedSAE 模型，保存 checkpoint 供后续使用

**Independent Test**: 在 Beauty 数据集 embedding 上运行 `python -m SAEGenRec.sid_builder gated_sae_train --embedding_path=data/interim/Beauty.emb-all-MiniLM-L6-v2-td.npy --output_dir=models/gated_sae/Beauty`，验证模型收敛且 checkpoint 目录包含 sae_weights.safetensors、cfg.json、training_config.json

### Implementation for User Story 1

- [X] T004 [US1] Implement gated_sae_train() function in SAEGenRec/sid_builder/gated_sae.py — call set_seed() first for reproducibility, auto-detect d_in from .npy shape, create GatedTrainingSAEConfig(d_in, d_sae=d_in*expansion_factor, l1_coefficient), instantiate GatedTrainingSAE, create SAETrainer(sae, data_provider=NpyDataProvider, cfg=SAETrainerConfig), call trainer.fit(), log MSE loss/L0/dead neuron ratio via loguru
- [X] T005 [US1] Implement checkpoint saving in gated_sae_train() in SAEGenRec/sid_builder/gated_sae.py — call sae.save_inference_model() for SAELens standard checkpoint, additionally save training_config.json with project-specific params (k, expansion_factor, embedding_path, seed)
- [X] T006 [US1] Register gated_sae_train command in SAEGenRec/sid_builder/__main__.py — add to fire.Fire() dispatch dict, accepting all GatedSAEConfig params as CLI args

**Checkpoint**: GatedSAE 训练功能完整可用，可通过 CLI 训练并保存模型

---

## Phase 4: User Story 2 - 生成 SAE-based SID (Priority: P1)

**Goal**: 从训练好的 GatedSAE checkpoint 生成 .index.json，每个 item 获得 K-token SID

**Independent Test**: 运行 `python -m SAEGenRec.sid_builder generate_sae_indices --checkpoint=models/gated_sae/Beauty --embedding_path=data/interim/Beauty.emb-all-MiniLM-L6-v2-td.npy --output_path=data/interim/Beauty.index.json`，验证输出 JSON 中每个 item 有 K=8 个 token，格式为 `[a_x][b_y]...[h_z]`

**Dependencies**: US1 must be complete (need trained checkpoint)

### Implementation for User Story 2

- [X] T007 [US2] Implement core encoding logic in SAEGenRec/sid_builder/generate_sae_indices.py — load GatedSAE via SAE.load_from_disk(), batch encode all embeddings with sae.encode(), extract top-K indices via torch.topk(), convert to token strings `[{chr(97+pos)}_{feature_index}]`
- [X] T008 [US2] Implement SID dedup in SAEGenRec/sid_builder/generate_sae_indices.py — detect collisions (same SID for different items), resolve by replacing last position with top-(K+j) feature, iterate up to max_dedup_iters times, log collision rate and remaining conflicts
- [X] T009 [US2] Implement generate_sae_indices() entry function in SAEGenRec/sid_builder/generate_sae_indices.py — orchestrate load→encode→topk→dedup→save pipeline, write .index.json in format `{"item_id": ["[a_x]", "[b_y]", ...], ...}` (default K=8), handle edge case where K > number of non-zero activations (zero-fill)
- [X] T010 [US2] Register generate_sae_indices command in SAEGenRec/sid_builder/__main__.py — add to fire.Fire() dispatch dict with params: checkpoint, embedding_path, k(default=8), output_path, max_dedup_iters, batch_size, device
- [X] T011 [US2] Verify K=8 downstream compatibility — generate .index.json with K=8, run convert_dataset, verify: (1) _parse_sid_sequence regex matches 8-token SID strings, (2) TokenExtender extracts all unique tokens including a-h prefixes, (3) build_prefix_tree constructs 8-level deep tree correctly. Fix any hardcoded K=3 assumptions found

**Checkpoint**: 完整的 GatedSAE → SID 流程可用，K=8 兼容性已验证

---

## Phase 5: User Story 3 - 端到端流水线集成 (Priority: P2)

**Goal**: 通过 Makefile 一键完成 GatedSAE 训练 + SID 生成，无缝接入下游流程

**Independent Test**: 运行 `make build_sae_sid CATEGORY=Beauty` 后，验证 data/interim/Beauty.index.json 已生成，再运行 `make convert CATEGORY=Beauty` 验证输出 CSV 和 info 文件格式正确

**Dependencies**: US1 and US2 must be complete

### Implementation for User Story 3

- [X] T012 [US3] Add build_sae_sid target to Makefile — chain gated_sae_train and generate_sae_indices commands with $(CATEGORY) variable, following existing build_sid target pattern
- [X] T013 [US3] Verify convert_dataset compatibility by running convert_dataset with GatedSAE-generated .index.json (K=8) — confirm output CSV history_item_sid and target_item_sid fields parse correctly, info TXT format matches expectations (no code changes expected, validation only)

**Checkpoint**: 研究者可通过 `make build_sae_sid` + `make convert` 完成数据准备

---

## Phase 6: User Story 4 - 对比实验 (Priority: P3)

**Goal**: 在相同数据集上对比 RQ-VAE 和 GatedSAE SID 的推荐效果

**Independent Test**: 在 Beauty 数据集上分别用两种方法完成 SID→SFT→评估，比较 HR@10 和 NDCG@10

**Dependencies**: US3 must be complete, existing RQ-VAE pipeline must work

### Implementation for User Story 4

- [ ] T014 [US4] Run RQ-VAE baseline experiment on Beauty dataset — execute full pipeline (build_sid → convert → sft → evaluate), record metrics to results/Beauty_rqvae/metrics.json
- [ ] T015 [US4] Run GatedSAE experiment on Beauty dataset — execute full pipeline (build_sae_sid → convert → sft → evaluate), record metrics to results/Beauty_gated_sae/metrics.json
- [ ] T016 [US4] Compare and document results — tabulate HR@K and NDCG@K for both methods, calculate relative difference, verify SC-003 (gap ≤ 20%)

**Checkpoint**: 两种 SID 方法的对比数据完整，可用于论文或后续研究

---

## Phase 7: Tests

**Purpose**: 基础测试（Constitution 要求 tests/ 目录有对应测试）

- [X] T017 [P] Write test for NpyDataProvider in tests/test_gated_sae.py — verify: batch shape correctness, NaN/Inf handling, multi-epoch cycling with shuffle, total sample count control
- [X] T018 [P] Write test for SID generation format in tests/test_gated_sae.py — verify: output .index.json structure (K=8 tokens per item, correct prefix letters a-h, valid feature indices), dedup produces unique SIDs
- [X] T019 Write test for set_seed() determinism in tests/test_gated_sae.py — verify: same seed + same input → identical GatedSAE encode output and identical SID assignment

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: 代码质量和文档完善

- [X] T020 Run `make lint` and `make format` to ensure code style compliance across all new files
- [X] T021 Run quickstart.md validation — execute all commands in specs/002-gated-sae-sid/quickstart.md end-to-end

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational (Phase 2)
- **US2 (Phase 4)**: Depends on US1 completion (need trained checkpoint)
- **US3 (Phase 5)**: Depends on US1 + US2 completion
- **US4 (Phase 6)**: Depends on US3 completion + working RQ-VAE baseline
- **Tests (Phase 7)**: Can start after US2 completion (needs code to test)
- **Polish (Phase 8)**: Depends on all desired user stories and tests being complete

### User Story Dependencies

- **US1 (P1)**: Foundational → US1（独立可测试）
- **US2 (P1)**: Foundational → US1 → US2（依赖 US1 的 checkpoint）
- **US3 (P2)**: US1 + US2 → US3（集成层，依赖核心功能）
- **US4 (P3)**: US3 → US4（需完整流水线）

### Within Each User Story

- Config/data layer before training logic
- Training before inference
- Core logic before CLI registration
- Story complete before moving to next priority

### Parallel Opportunities

- T001 and T002 can run in parallel (different files)
- T007 and T008 can be implemented in parallel within generate_sae_indices.py (different functions, but same file — sequentially recommended)
- T014 and T015 can run in parallel (independent experiment runs)
- T017, T018, T019 can run in parallel (different test functions in same file)

---

## Parallel Example: User Story 1

```bash
# Phase 1 setup tasks can run in parallel:
Task T001: "Add sae-lens dependency to pyproject.toml"
Task T002: "Add GatedSAEConfig to SAEGenRec/config.py"

# US1 tasks are sequential (same file gated_sae.py):
Task T004: "Implement gated_sae_train() in gated_sae.py"
Task T005: "Implement checkpoint saving in gated_sae.py"
Task T006: "Register CLI command in __main__.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T002)
2. Complete Phase 2: Foundational (T003)
3. Complete Phase 3: User Story 1 (T004-T006)
4. **STOP and VALIDATE**: 运行 `python -m SAEGenRec.sid_builder gated_sae_train` 验证训练和保存
5. Continue to US2 for SID 生成

### Incremental Delivery

1. Setup + Foundational → 基础设施就绪
2. US1 → 训练可用 → 验证模型收敛
3. US2 → SID 生成可用 → 验证 .index.json 格式
4. US3 → 流水线集成 → 验证 Makefile 一键运行
5. US4 → 对比实验 → 验证推荐效果

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- US1 和 US2 都是 P1 优先级，但有顺序依赖（训练→生成）
- US4 是实验任务而非纯代码任务，主要是运行流水线并记录结果
- 所有新代码文件遵循 ruff 格式化规则（行长 99，isort）
- SAELens 集成不修改 SAELens 源码，仅使用其公开 API
