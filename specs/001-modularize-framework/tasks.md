# Tasks: Modularize MiniOneRec Framework

**Input**: Design documents from `/specs/001-modularize-framework/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Included per SC-005 requirement. Core data transformation logic must have test coverage.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, package structure, and dependency configuration

- [X] T001 Create package directory structure per plan.md: `SAEGenRec/data_process/`, `SAEGenRec/sid_builder/`, `SAEGenRec/sid_builder/models/`, `SAEGenRec/datasets/`, `SAEGenRec/training/`, `SAEGenRec/evaluation/`, `SAEGenRec/models/`
- [X] T002 Create all `__init__.py` files for package structure: `SAEGenRec/__init__.py`, `SAEGenRec/data_process/__init__.py`, `SAEGenRec/sid_builder/__init__.py`, `SAEGenRec/sid_builder/models/__init__.py`, `SAEGenRec/datasets/__init__.py`, `SAEGenRec/training/__init__.py`, `SAEGenRec/evaluation/__init__.py`, `SAEGenRec/models/__init__.py`
- [X] T003 Update `pyproject.toml` to add runtime dependencies: torch, transformers, trl, deepspeed, faiss-cpu, fire, loguru, sentence-transformers, numpy, tqdm, scikit-learn

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Implement centralized configuration module in `SAEGenRec/config.py`: define `CATEGORY_MAP` dict (FR-012), `set_seed()` function covering random/numpy/torch/cudnn (FR-013), and base dataclass `BaseConfig` with common fields (seed, device, output_dir)
- [X] T005 [P] Implement `DataProcessConfig` dataclass in `SAEGenRec/config.py`: fields for category, k_core, st_year, st_month, ed_year, ed_month, split_method (TO/LOO), max_history_len (default=50), raw_data_dir, output_dir
- [X] T006 [P] Implement `TextEmbConfig` dataclass in `SAEGenRec/config.py`: fields for model_name, item_json_path, review_json_path, output_dir, batch_size, device
- [X] T007 [P] Implement `RQVAEConfig` dataclass in `SAEGenRec/config.py`: fields for embedding_path, num_levels (default=3), codebook_size (default=256), layers (default=[2048,1024,512,256,128,64]), lr, epochs, batch_size, beta, sk_epsilon, sk_iters, kmeans_init, output_dir
- [X] T008 [P] Implement `SFTConfig`, `RLConfig`, `EvalConfig`, `SASRecConfig` dataclasses in `SAEGenRec/config.py`: fields matching reference implementation hyperparameters for each training/eval stage
- [X] T009 [P] Create test file `tests/test_config.py`: test set_seed() determinism, CATEGORY_MAP completeness, dataclass serialization, config override from dict

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Preprocess Raw Amazon Data (Priority: P1) MVP

**Goal**: Filter raw Amazon 2015 review data by k-core/date range, produce .inter + .item.json + .review.json files, generate text embeddings as .npy files

**Independent Test**: Run preprocessing on Amazon Beauty dataset, verify output file formats, column schemas, and row counts. Run text embedding, verify .npy shapes.

### Implementation for User Story 1

- [X] T010 [US1] Implement data preprocessing core logic in `SAEGenRec/data_process/preprocess.py`: port `gao()` function from `references/MiniOneRec/data/process.py` — k-core filtering loop, title cleaning, item2id assignment, interaction sequence generation with `max_history_len` sliding window. Output `.inter` files (TSV: user_id, item_id, item_asin, timestamp, rating) per contracts/inter-format.md
- [X] T011 [US1] Implement TO (Temporal Order) split strategy in `SAEGenRec/data_process/preprocess.py`: global timestamp sort of all interaction samples, then 8:1:1 positional split into train/valid/test .inter files
- [X] T012 [US1] Implement LOO (Leave-One-Out) split strategy in `SAEGenRec/data_process/preprocess.py`: per-user split — last interaction → test, second-to-last → valid, rest → train
- [X] T013 [US1] Implement .item.json generation in `SAEGenRec/data_process/preprocess.py`: extract item metadata (item_asin, title, description) for all surviving items, keyed by string item_id, per contracts/item-json.md
- [X] T014 [US1] Implement .review.json generation in `SAEGenRec/data_process/preprocess.py`: extract per-interaction review metadata (user_id, item_id, item_asin, timestamp, rating, review_text, summary) for all surviving interactions, per contracts/review-json.md
- [X] T015 [US1] Implement CLI entry point in `SAEGenRec/data_process/__main__.py`: wire `fire.Fire()` to preprocess function, accepting DataProcessConfig parameters
- [X] T016 [US1] Implement text embedding module in `SAEGenRec/sid_builder/text2emb.py`: port text encoding logic from reference — load sentence-transformer model, encode item titles+descriptions → `{dataset}.emb-{model}-td.npy`, encode review texts → `{dataset}.emb-{model}-review.npy` with row ordering matching .review.json
- [X] T017 [P] [US1] Create test file `tests/test_data_process.py`: test k-core filtering correctness, TO split 8:1:1 ratio, LOO split per-user correctness, .inter column schema validation, .item.json structure, .review.json row count matches interactions, max_history_len truncation, idempotency (same input → same output)

**Checkpoint**: Data preprocessing pipeline fully functional. Can produce .inter, .item.json, .review.json, and .npy embedding files from raw Amazon 2015 data.

---

## Phase 4: User Story 2 - Construct Semantic Item Descriptions (SIDs) (Priority: P2)

**Goal**: Train RQ-VAE/RQ-Kmeans on item embeddings, generate .index.json mapping items to SID tokens, convert dataset to final CSV training format

**Independent Test**: Train RQ-VAE on Beauty embeddings, verify collision rate and .index.json structure. Run convert_dataset, verify CSV columns and info TXT format.

### Implementation for User Story 2

- [X] T018 [P] [US2] Port VectorQuantizer to `SAEGenRec/sid_builder/models/vq.py`: copy from `references/MiniOneRec/rq/models/vq.py` — VectorQuantizer class with kmeans init, Sinkhorn balanced assignment, commitment/codebook loss
- [X] T019 [P] [US2] Port helper layers to `SAEGenRec/sid_builder/models/layers.py`: copy from `references/MiniOneRec/rq/models/layers.py` — kmeans(), sinkhorn_algorithm() functions
- [X] T020 [US2] Port ResidualQuantizer to `SAEGenRec/sid_builder/models/rq.py`: copy from `references/MiniOneRec/rq/models/rq.py` — ResidualQuantizer using VectorQuantizer layers
- [X] T021 [US2] Port RQVAE model to `SAEGenRec/sid_builder/models/rqvae.py`: copy from `references/MiniOneRec/rq/models/rqvae.py` — RQVAE class with encoder/decoder MLP and ResidualQuantizer
- [X] T022 [US2] Implement RQ-VAE training entry point in `SAEGenRec/sid_builder/rqvae.py`: port training loop from `references/MiniOneRec/rq/trainer.py` — EmbDataset loading, train/eval loops, collision rate tracking, checkpoint saving. Use RQVAEConfig dataclass
- [X] T023 [US2] Implement RQ-Kmeans variants in `SAEGenRec/sid_builder/rqkmeans.py`: port from `references/MiniOneRec/rq/rqvae.py` — RQ-Kmeans (FAISS), Constrained RQ-Kmeans (balanced), RQ-Kmeans+ (dedup layer) functions
- [X] T024 [US2] Implement index generation in `SAEGenRec/sid_builder/generate_indices.py`: load trained quantizer checkpoint, encode all item embeddings, produce `{dataset}.index.json` mapping string item_id → SID token list, per contracts/index-json.md
- [X] T025 [US2] Implement dataset conversion in `SAEGenRec/data_process/convert_dataset.py`: port from `references/MiniOneRec/convert_dataset.py` — read .inter + .item.json + .index.json, produce train/valid/test CSV files (columns per contracts/training-csv.md) and info TXT file (per contracts/info-txt.md)
- [X] T026 [US2] Implement CLI entry point in `SAEGenRec/sid_builder/__main__.py`: wire `fire.Fire()` to rqvae_train, rqkmeans_train, generate_indices functions
- [X] T027 [P] [US2] Create test file `tests/test_sid_builder.py`: test VectorQuantizer forward pass shape, RQ-VAE training 1 epoch loss decrease, collision rate computation, .index.json structure validation, convert_dataset CSV column schema, info TXT format

**Checkpoint**: SID construction pipeline fully functional. Can train quantizers, generate .index.json, and convert to final CSV training format.

---

## Phase 5: User Story 3 - Fine-tune LLM with SFT (Priority: P3)

**Goal**: Extend LLM tokenizer with SID tokens, train with multi-dataset SFT (3 dataset types), support freeze_LLM and DeepSpeed

**Independent Test**: Run SFT on Beauty preprocessed data for 1 epoch with small model, verify tokenizer extension, embedding resize, loss decrease, checkpoint save/load.

### Implementation for User Story 3

- [X] T028 [P] [US3] Implement base dataset classes in `SAEGenRec/datasets/base.py`: port BaseDataset and CSVBaseDataset from `references/MiniOneRec/data.py` — CSV loading, prompt template formatting, tokenization
- [X] T029 [P] [US3] Implement SFT dataset classes in `SAEGenRec/datasets/sft_datasets.py`: port SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset from `references/MiniOneRec/data.py` — each with their specific prompt templates and instruction formats
- [X] T030 [US3] Implement TokenExtender in `SAEGenRec/training/sft.py`: port from `references/MiniOneRec/sft.py` — extract unique SID tokens from info file, add to tokenizer vocabulary, resize model embedding matrix
- [X] T031 [US3] Implement SFT training logic in `SAEGenRec/training/sft.py`: port from `references/MiniOneRec/sft.py` — multi-dataset concatenation (SidSFTDataset + SidItemFeatDataset + FusionSeqRecDataset), HuggingFace Trainer with cosine LR schedule, early stopping (patience=3), optional freeze_LLM with gradient mask, optional DeepSpeed config
- [X] T032 [US3] Implement CLI entry point in `SAEGenRec/training/__main__.py`: wire `fire.Fire()` to sft and rl training functions
- [X] T033 [P] [US3] Create test file `tests/test_datasets.py`: test prompt template formatting matches reference, tokenization output shapes, SidSFTDataset/SidItemFeatDataset/FusionSeqRecDataset __getitem__ output structure
- [X] T034 [P] [US3] Create test file `tests/test_training.py`: test TokenExtender adds correct number of tokens, embedding matrix resize, freeze_LLM gradient mask correctness

**Checkpoint**: SFT training pipeline fully functional. Can fine-tune any HuggingFace causal LLM on SID recommendation task.

---

## Phase 5.5: Shared Infrastructure (Post-SFT)

**Purpose**: Components shared by US4 (RL) and US5 (Evaluation), MUST complete before either can proceed

- [X] T035 Implement ConstrainedLogitsProcessor in `SAEGenRec/evaluation/logit_processor.py`: port from `references/MiniOneRec/LogitProcessor.py` — prefix tree (hash_dict) construction from info file, logit masking for valid next tokens, EOS forcing when no valid tokens

**Checkpoint**: ConstrainedLogitsProcessor ready — US4 and US5 can now proceed in parallel

---

## Phase 6: User Story 4 - RL Training with GRPO (Priority: P4)

**Goal**: Implement GRPO-based RL with ReReTrainer, 4 reward types, constrained beam search generation

**Independent Test**: Run 1 epoch RL with rule reward on Beauty data, verify constrained generation produces valid SIDs, rewards computed correctly.

### Implementation for User Story 4

- [X] T036 [P] [US4] Implement RL dataset classes in `SAEGenRec/datasets/rl_datasets.py`: port SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset from `references/MiniOneRec/data.py` — prompt→history and history→target mappings
- [X] T037 [P] [US4] Implement reward functions in `SAEGenRec/training/rewards.py`: port from `references/MiniOneRec/rl.py` — rule_reward (binary), ranking_reward (NDCG-aware), semantic_reward (cosine similarity), sasrec_reward (CF scores)
- [X] T038 [US4] Implement ReReTrainer in `SAEGenRec/training/trainer.py`: port from `references/MiniOneRec/minionerec_trainer.py` — extends trl.GRPOTrainer with constrained generation, online evaluation, dynamic sampling, reward computation integration
- [X] T039 [US4] Implement RL training entry point in `SAEGenRec/training/rl.py`: port from `references/MiniOneRec/rl.py` — 3 combined RL datasets, reward type selection, ReReTrainer initialization, DeepSpeed support, checkpoint saving
- [X] T040 [P] [US4] Create test file `tests/test_rl.py`: test 4 reward functions (rule binary correctness, ranking NDCG penalty, semantic cosine, sasrec score shape), ReReTrainer constrained generation produces valid SIDs only

**Checkpoint**: RL training pipeline fully functional. Can optimize SFT checkpoint with GRPO and 4 reward types.

---

## Phase 7: User Story 5 - Evaluation with Constrained Beam Search (Priority: P5)

**Goal**: Offline evaluation with constrained beam search, compute HR@K and NDCG@K metrics

**Independent Test**: Run evaluation on Beauty test set with trained checkpoint, verify all generated sequences are valid SIDs, metrics computed correctly.

### Implementation for User Story 5

- [X] T041 [P] [US5] Implement EvalSidDataset in `SAEGenRec/datasets/eval_datasets.py`: port from `references/MiniOneRec/data.py` — test data loading with prompt formatting for evaluation
- [X] T042 [P] [US5] Implement metrics computation in `SAEGenRec/evaluation/metrics.py`: port from `references/MiniOneRec/calc.py` — HR@K and NDCG@K for K={1,3,5,10,20}
- [X] T043 [US5] Implement evaluation entry point in `SAEGenRec/evaluation/evaluate.py`: port from `references/MiniOneRec/evaluate.py` — load model + info file, build prefix tree, constrained beam search with ConstrainedLogitsProcessor (num_beams configurable), compute metrics per test sample, aggregate and report
- [X] T044 [US5] Implement CLI entry point in `SAEGenRec/evaluation/__main__.py`: wire `fire.Fire()` to evaluate function
- [X] T045 [P] [US5] Create test file `tests/test_evaluation.py`: test prefix tree completeness (all SIDs reachable), ConstrainedLogitsProcessor masking correctness, HR@K/NDCG@K computation against known values, deterministic evaluation (same input → same output)

**Checkpoint**: Evaluation pipeline fully functional. Can evaluate any trained model with constrained beam search and report HR@K/NDCG@K metrics.

---

## Phase 8: User Story 6 - SASRec CF Model (Priority: P6)

**Goal**: Train SASRec collaborative filtering model for use as CF reward in RL

**Independent Test**: Train SASRec on Beauty dataset, verify NDCG@20 on validation set, checkpoint save/load.

### Implementation for User Story 6

- [X] T046 [US6] Implement SASRec model (with GRU and Caser variants) in `SAEGenRec/models/sasrec.py`: port from `references/MiniOneRec/sasrec.py` — SASRec (self-attention), GRU, Caser architectures with BCE loss, evaluation loop, early stopping on NDCG@20
- [X] T047 [US6] Add SASRec training CLI to `SAEGenRec/models/__init__.py` or extend `SAEGenRec/training/__main__.py`: wire `fire.Fire()` to sasrec_train function with SASRecConfig
- [X] T048 [P] [US6] Create test file `tests/test_sasrec.py`: test SASRec forward pass output shape, BCE loss computation, GRU/Caser variant initialization, early stopping trigger on NDCG@20

**Checkpoint**: SASRec training functional. Can train CF model and use as sasrec reward in RL.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Makefile targets, documentation, integration validation

- [X] T049 Add Makefile targets for all pipeline stages in `Makefile`: make preprocess, make embed, make build_sid, make convert, make sft, make rl, make evaluate, make pipeline (full end-to-end)
- [X] T050 [P] Update `CLAUDE.md` with complete module documentation: package structure, CLI commands, data flow, configuration system
- [X] T051 [P] Add `fire` dependency guard in each `__main__.py`: graceful error if fire not installed
- [X] T052 Run full pipeline end-to-end on Beauty dataset and validate SC-001 (behavioral parity) and SC-002 (metrics within 1% of reference)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 completion - BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Phase 2 — no dependencies on other stories
- **US2 (Phase 4)**: Depends on Phase 2 + US1 (needs .inter, .item.json, .npy files as input)
- **US3 (Phase 5)**: Depends on Phase 2 + US2 (needs CSV + info files as input)
- **Shared (Phase 5.5)**: Depends on US3 — ConstrainedLogitsProcessor (T035), shared by US4 and US5
- **US4 (Phase 6)**: Depends on Phase 2 + US3 + Phase 5.5 (needs SFT checkpoint + ConstrainedLogitsProcessor)
- **US5 (Phase 7)**: Depends on Phase 2 + US3 + Phase 5.5 (needs trained model checkpoint + ConstrainedLogitsProcessor)
- **US6 (Phase 8)**: Depends on Phase 2 + US2 (needs CSV files with item_id columns)
- **Polish (Phase 9)**: Depends on all user stories being complete

### User Story Dependencies

```
Phase 1 (Setup) → Phase 2 (Foundational)
                      │
                      ▼
                   US1 (P1: Data Preprocessing)
                      │
                      ▼
                   US2 (P2: SID Construction)
                      │
                      ├──────────────┐
                      ▼              ▼
                   US3 (P3: SFT)   US6 (P6: SASRec) [parallel]
                      │
                      ▼
                   Phase 5.5 (ConstrainedLogitsProcessor)
                      │
                      ├──────────────┐
                      ▼              ▼
                   US4 (P4: RL)    US5 (P5: Evaluation) [parallel]
                      │              │
                      ▼              ▼
                   Phase 9 (Polish)
```

### Within Each User Story

- Models/layers before training logic
- Core implementation before CLI entry point
- Tests can run in parallel with implementation (marked [P])
- Story complete before downstream stories begin

### Parallel Opportunities

- T005, T006, T007, T008: All config dataclasses (different sections of same file, but independent)
- T018, T019: VQ and layers (independent model files)
- T028, T029: Base datasets and SFT datasets (different files)
- T035, T036: RL datasets and reward functions (different files)
- T040, T041: Eval dataset and metrics (different files)
- US3 and US6 can run in parallel after US2 completes
- US4 and US5 can run in parallel after US3 completes
- All test files can be written in parallel with their implementations

---

## Parallel Example: User Story 2

```bash
# Launch model ports in parallel (T018, T019):
Task: "Port VectorQuantizer to SAEGenRec/sid_builder/models/vq.py"
Task: "Port helper layers to SAEGenRec/sid_builder/models/layers.py"

# After models complete, sequential (T020 → T021 → T022):
Task: "Port ResidualQuantizer to SAEGenRec/sid_builder/models/rq.py"
Task: "Port RQVAE model to SAEGenRec/sid_builder/models/rqvae.py"
Task: "Implement RQ-VAE training in SAEGenRec/sid_builder/rqvae.py"

# Independent of RQ-VAE (T023):
Task: "Implement RQ-Kmeans in SAEGenRec/sid_builder/rqkmeans.py"

# Tests in parallel with implementation:
Task: "Create tests/test_sid_builder.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (config.py with set_seed, CATEGORY_MAP, dataclasses)
3. Complete Phase 3: User Story 1 (data preprocessing + text embedding)
4. **STOP and VALIDATE**: Run preprocessing on Beauty dataset, verify .inter/.item.json/.review.json/.npy outputs
5. Verify idempotency and TO/LOO split correctness

### Incremental Delivery

1. Setup + Foundational → config infrastructure ready
2. US1 (Data Preprocessing) → can produce training data from raw Amazon data
3. US2 (SID Construction) → can build SIDs and convert to CSV format
4. US3 (SFT) → can fine-tune LLM for recommendation
5. US5 (Evaluation) → can evaluate model quality (can run in parallel with US4/US6)
6. US4 (RL) + US6 (SASRec) → full pipeline with RL optimization
7. Polish → Makefile, docs, end-to-end validation

### Key Milestone: SC-002 Validation

After US5 completes, run end-to-end pipeline on Industrial_and_Scientific dataset and compare HR@K/NDCG@K metrics with reference implementation. This validates behavioral parity (SC-001) and metric equivalence (SC-002).

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Port logic from `references/MiniOneRec/` preserving exact behavior — do NOT refactor algorithms
- Use `SAEGenRec/config.py` dataclasses for all parameters — no hardcoded values in business logic
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
