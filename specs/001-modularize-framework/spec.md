# Feature Specification: Modularize MiniOneRec Framework

**Feature Branch**: `001-modularize-framework`
**Created**: 2026-03-07
**Status**: Draft
**Input**: User description: "帮我实现整个框架，保持逻辑与Minionerec相同"

## Clarifications

### Session 2026-03-07

- Q: `.inter` 文件的列模式是什么？ → A: 最小化列 `user_id`, `item_id`, `item_asin`, `timestamp`, `rating`，不含 title/SID（由 convert_dataset 阶段通过 `.item.json` 和 `.index.json` 注入）。
- Q: Review embedding 在下游如何消费？ → A: 仅在 P1 阶段生成并存储，当前训练/推理模块不消费。作为扩展预留点，训练模块保持与 MiniOneRec 完全一致。
- Q: 是否支持多 GPU / DeepSpeed 分布式训练？ → A: 支持 DeepSpeed（与 MiniOneRec 一致），DeepSpeed config 作为可选参数传入，单 GPU 时自动退化。
- Q: 初始版本需要支持哪些 Amazon 数据集版本？ → A: 仅支持 2015 版本（MiniOneRec 主脚本 `process.py` 使用的格式），2018/2023 作为后续迭代。
- Q: 各 pipeline 阶段的 CLI 入口方式？ → A: `python -m SAEGenRec.{module}` + `fire.Fire()`（与 MiniOneRec 一致），同时在 Makefile 中提供 `make` 命令封装各阶段调用。

## Assumptions

- The reference implementation at `references/MiniOneRec/` is the single source of truth for all logic and behavior.
- Modularization preserves exact behavioral parity: given the same inputs and random seeds, the modularized code MUST produce identical outputs to the reference scripts.
- The initial modularization targets Amazon 2015 review dataset format only. Support for 2018 and 2023 formats is deferred to future iterations.
- The LLM base model is assumed to be a HuggingFace-compatible causal language model (e.g., Qwen2.5, LLaMA).
- SID construction defaults to 3-level quantization with 256 codes per level, matching the reference implementation's default configuration.
- GPR variants (`sft_gpr.py`, `rl_gpr.py`, `convert_dataset_gpr.py`) are out of scope for the initial modularization — only the core pipeline is included.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Preprocess Raw Amazon Data into Training Format (Priority: P1)

A researcher downloads raw Amazon review data and wants to preprocess it into the standardized format required for SID construction and model training. They run a single command specifying the dataset name, filtering parameters, split strategy, maximum history length, and output path. The system filters interactions by k-core, date range, splits data into train/valid/test using the chosen strategy, and produces `.inter` files, `.item.json` (item metadata), `.review.json` (user review metadata including review text, rating, timestamp per interaction), and generates text embeddings as `.npy` files for both item descriptions and review text.

The system supports two dataset split strategies:

- **Temporal Order (TO)**: All interaction sequences across all users are sorted globally by timestamp, then split 8:1:1 by position. This is MiniOneRec's default strategy — the same user may appear in train/valid/test, but earlier interactions are always in earlier splits.
- **Leave-One-Out (LOO)**: For each user, the last interaction is placed in the test set, the second-to-last in the validation set, and all remaining in the training set. This ensures every user appears in all three splits.

The maximum history sequence length is configurable (default: 50, matching MiniOneRec). This controls the sliding window size when generating interaction samples — each sample's history contains at most `max_history_len` most recent items.

**Why this priority**: Data preprocessing is the foundation of the entire pipeline. No downstream task can proceed without properly formatted data. This is the entry point for any new dataset.

**Independent Test**: Can be fully tested by running the preprocessing module on a small subset of Amazon data (e.g., All items from Beauty) and verifying the output file formats, column structures, and row counts match the reference implementation's output.

**Acceptance Scenarios**:

1. **Given** raw Amazon 2015 review JSON files, **When** the user runs the data preprocessing module with k-core=5, a date range, and `split_method="TO"`, **Then** the system produces `{dataset}.train.inter`, `{dataset}.valid.inter`, `{dataset}.test.inter`, `{dataset}.item.json`, and `{dataset}.review.json` files with the correct column schemas. The TO split sorts all interaction samples globally by timestamp and splits 8:1:1 by position.
2. **Given** the same raw data, **When** the user runs preprocessing with `split_method="LOO"`, **Then** for each user the last interaction goes to test, the second-to-last to valid, and the rest to train. Each user appears in all three splits.
3. **Given** `max_history_len=20`, **When** generating interaction samples, **Then** each sample's history sequence contains at most 20 items (the most recent ones). The default value is 50, matching MiniOneRec.
4. **Given** the `.item.json` and a text encoder model path, **When** the user runs the text embedding module, **Then** the system produces a `{dataset}.emb-{model}-td.npy` file with shape `(num_items, embedding_dim)` for item text, and a `{dataset}.emb-{model}-review.npy` file with shape `(num_reviews, embedding_dim)` for review text representations.
5. **Given** the same raw data and same parameters, **When** the preprocessing is run twice, **Then** the outputs are byte-identical (idempotency).

---

### User Story 2 - Construct Semantic Item Descriptions (SIDs) (Priority: P2)

A researcher has item embeddings (`.npy`) and wants to construct SIDs — compact discrete token sequences that uniquely identify each item. They choose a quantization method (RQ-VAE, RQ-Kmeans, Constrained RQ-Kmeans, or RQ-Kmeans+), train the quantizer, and generate an index mapping from item IDs to SID token sequences. They then convert the dataset into the final CSV training format with SID-based history and targets.

**Why this priority**: SID construction is the core innovation that bridges item representations with LLM vocabulary. It is the second pipeline stage and a prerequisite for all training.

**Independent Test**: Can be tested by training an RQ-VAE on the reference Industrial dataset embeddings and verifying that: (a) the collision rate matches the reference implementation, (b) the generated `.index.json` has the same structure (item_id → list of SID tokens), and (c) the converted CSV files contain the correct fields.

**Acceptance Scenarios**:

1. **Given** item embeddings as a `.npy` file, **When** the user trains an RQ-VAE with default hyperparameters (3 levels, 256 codes each, layers=[2048,1024,512,256,128,64]), **Then** the system produces a trained model checkpoint and reports loss and collision rate metrics matching the reference implementation.
2. **Given** a trained RQ-VAE checkpoint, **When** the user runs index generation, **Then** the system produces a `{dataset}.index.json` mapping each item ID (as string) to a list of SID tokens (e.g., `["[a_42]", "[b_128]", "[c_7]"]`).
3. **Given** `.inter` files, `.item.json`, and `.index.json`, **When** the user runs dataset conversion, **Then** the system produces train/valid/test CSV files with columns `user_id`, `history_item_sid`, `target_item_sid`, `history_item_title`, `target_item_title`, `history_item_id`, `target_item_id`, and an info TXT file with tab-separated `semantic_id\titem_title\titem_id`.

---

### User Story 3 - Fine-tune LLM with Supervised Learning (SFT) (Priority: P3)

A researcher has the converted training data (CSV + info files) and a base LLM. They want to fine-tune the LLM to learn sequential recommendation via next-SID prediction. The system extends the tokenizer with SID tokens, loads multiple training tasks (SID sequence prediction, SID-item feature alignment, fusion sequence-to-SID), and trains with HuggingFace Trainer including early stopping.

**Why this priority**: SFT is the first training stage that produces a recommendation-capable model. It depends on Stories 1 and 2 being complete.

**Independent Test**: Can be tested by running SFT on the P1 preprocessed data (e.g., Beauty dataset from User Story 1) for 1 epoch with a small base model and verifying that: (a) the tokenizer correctly adds new SID tokens, (b) the model's embedding matrix is correctly resized, (c) training loss decreases, and (d) the saved checkpoint can be loaded for inference.

**Acceptance Scenarios**:

1. **Given** a base LLM and SID index file, **When** the SFT module initializes, **Then** the tokenizer vocabulary is extended with all unique SID tokens from the index, and the model's embedding matrix is resized accordingly.
2. **Given** train and validation CSV files, **When** SFT training runs with 3 combined datasets (SidSFTDataset + SidItemFeatDataset + FusionSeqRecDataset), **Then** the training loop produces decreasing loss and saves the best checkpoint based on validation loss with early stopping (patience=3).
3. **Given** `freeze_LLM=True`, **When** SFT training runs, **Then** only the newly added SID token embeddings receive gradient updates, while all original LLM parameters remain frozen (verified by gradient mask).

---

### User Story 4 - Train with Recommendation-Oriented Reinforcement Learning (Priority: P4)

A researcher has an SFT checkpoint and wants to further optimize it using GRPO-based reinforcement learning with recommendation-specific reward functions. They choose a reward type (rule-based, ranking-aware, semantic similarity, or collaborative filtering via SASRec), and the system trains with constrained beam search to ensure all generated candidates are valid SIDs.

**Why this priority**: RL training is an optional but important optimization stage. It builds on the SFT checkpoint and provides the final performance boost.

**Independent Test**: Can be tested by running 1 epoch of RL training on the reference dataset with rule-based reward and verifying that: (a) the ReReTrainer correctly applies constrained generation, (b) rewards are computed correctly for matching/non-matching predictions, and (c) the model checkpoint is saved.

**Acceptance Scenarios**:

1. **Given** an SFT checkpoint and info file listing all valid SIDs, **When** RL training initializes, **Then** the system loads 3 combined RL datasets (SidDataset + RLTitle2SidDataset + RLSeqTitle2SidDataset) and constructs prompt→history and history→target mappings for reward computation.
2. **Given** `reward_type="ranking"`, **When** completions are generated for a batch, **Then** rewards combine binary accuracy (1.0 for correct, 0.0 for incorrect) with NDCG-aware ranking penalties (higher-ranked incorrect items receive larger negative rewards).
3. **Given** `beam_search=True`, **When** the trainer generates candidate completions during RL, **Then** constrained beam search ensures all generated sequences are valid SID sequences from the item vocabulary (via prefix tree / hash_dict constraints).

---

### User Story 5 - Evaluate Model with Constrained Beam Search (Priority: P5)

A researcher has a trained model checkpoint and wants to evaluate its recommendation quality. They run offline evaluation with constrained beam search that guarantees all generated recommendations are valid items. The system computes HR@K and NDCG@K metrics.

**Why this priority**: Evaluation is the final pipeline stage that validates model quality. It depends on having a trained model.

**Independent Test**: Can be tested by running evaluation on the reference Industrial test set with a known checkpoint and comparing HR@K and NDCG@K metrics against published reference results.

**Acceptance Scenarios**:

1. **Given** a trained model and an info file, **When** the evaluation module builds a prefix tree (hash_dict), **Then** every valid SID sequence is reachable through the tree, and no invalid sequences can be generated during constrained beam search.
2. **Given** test data and a model, **When** evaluation runs with `num_beams=50`, **Then** the system generates top-50 recommendations per test sample using `ConstrainedLogitsProcessor` and computes HR@{1,3,5,10,20} and NDCG@{1,3,5,10,20}.
3. **Given** the same model, data, and seed, **When** evaluation is run twice, **Then** the metrics are identical (deterministic evaluation).

---

### User Story 6 - Train SASRec Collaborative Filtering Model (Priority: P6)

A researcher wants to train a SASRec collaborative filtering model on the converted dataset to be used as a CF reward signal during RL training. They specify the dataset and hyperparameters, and the system trains SASRec with early stopping, saving the best model checkpoint.

**Why this priority**: SASRec is an auxiliary model used only for the `sasrec` reward type in RL. It is optional and only needed when using CF-based rewards.

**Independent Test**: Can be tested by training SASRec on the reference Industrial dataset and verifying HR@K/NDCG@K metrics on the test set.

**Acceptance Scenarios**:

1. **Given** train/valid/test CSV files with `history_item_id` and `item_id` columns, **When** SASRec training runs, **Then** the model trains with BCE loss, evaluates on validation data each epoch, and applies early stopping when NDCG@20 stops improving.
2. **Given** a trained SASRec checkpoint, **When** used as CF reward in RL, **Then** the model correctly scores candidate items given user history sequences.

---

### Edge Cases

- What happens when an item has no title or description in the metadata? The system MUST fall back to a placeholder string (e.g., `"Item_{item_id}"`), matching the reference implementation's behavior.
- What happens when an interaction has no review text (empty or missing)? The system MUST store an empty string in `.review.json` and produce a zero vector in the review embedding `.npy` file for that entry.
- What happens when RQ-VAE quantization produces collisions (multiple items mapping to the same SID)? The system MUST report the collision rate and continue training — collisions are expected and tracked as a quality metric.
- What happens when constrained beam search produces no valid next tokens for a beam? The system MUST force the EOS token to terminate that beam, matching `ConstrainedLogitsProcessor` behavior.
- What happens when the user provides a dataset category not in the predefined mapping? The system MUST raise a clear error listing valid categories.
- What happens when using LOO split and a user has fewer than 3 interactions? The system MUST skip users with fewer than 3 interactions (since k-core filtering with K≥5 already guarantees at least 5 interactions per user, this should not occur under normal usage but MUST be handled gracefully).
- What happens when `max_history_len` exceeds a user's actual history length? The system MUST use the full available history without padding — shorter sequences are valid inputs.
- What happens when the info file contains items whose SID tokens are not in the tokenizer vocabulary? The `TokenExtender` MUST add all SID tokens before model initialization.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a `SAEGenRec.data_process` module that filters raw Amazon review data by k-core and date range, producing `.inter` (columns: `user_id`, `item_id`, `item_asin`, `timestamp`, `rating`), `.item.json`, and `.review.json` files. The module MUST support two split strategies via `split_method` parameter: (a) **TO (Temporal Order)** — global timestamp sort + 8:1:1 positional split (MiniOneRec default); (b) **LOO (Leave-One-Out)** — per-user last-item for test, second-to-last for valid, rest for train. The module MUST accept a `max_history_len` parameter (default: 50) controlling the sliding window size for interaction sample generation. The `.review.json` file MUST preserve per-interaction review metadata including review text, rating, timestamp, and summary for each user-item pair that survives filtering.
- **FR-002**: System MUST provide a `SAEGenRec.sid_builder.text2emb` module that encodes text into dense vectors using a specified text encoder model, outputting `.npy` files. It MUST support two modes: (a) item text embeddings from title + description (`{dataset}.emb-{model}-td.npy` with shape `(num_items, embedding_dim)`), and (b) per-interaction review text embeddings (`{dataset}.emb-{model}-review.npy` with shape `(num_reviews, embedding_dim)`). In mode (b), each row corresponds to one `(user_id, item_id, timestamp)` interaction record from `.review.json`, in the same order. Review embeddings are generated and stored as an extension point for future use (e.g., review-aware datasets, review-semantic rewards); current SFT/RL/evaluation modules do NOT consume them and maintain full behavioral parity with MiniOneRec.
- **FR-003**: System MUST provide a `SAEGenRec.sid_builder.rqvae` module implementing RQ-VAE training with configurable number of quantization levels, codebook sizes, encoder/decoder layers, and Sinkhorn-based balanced assignment.
- **FR-004**: System MUST provide a `SAEGenRec.sid_builder.rqkmeans` module implementing RQ-Kmeans (via FAISS), Constrained RQ-Kmeans (balanced clusters), and RQ-Kmeans+ (extra deduplication layer) variants.
- **FR-005**: System MUST provide a `SAEGenRec.sid_builder.generate_indices` module that applies a trained quantizer to all items and produces a `{dataset}.index.json` file mapping item IDs to SID token lists.
- **FR-006**: System MUST provide a `SAEGenRec.data_process.convert_dataset` module that combines `.inter`, `.item.json`, and `.index.json` to produce CSV training files and info TXT files.
- **FR-007**: System MUST provide a `SAEGenRec.datasets` module containing all dataset classes (`SidSFTDataset`, `SidItemFeatDataset`, `FusionSeqRecDataset`, `SidDataset`, `RLTitle2SidDataset`, `RLSeqTitle2SidDataset`, `EvalSidDataset`) with identical prompt templates and tokenization logic as the reference.
- **FR-008**: System MUST provide a `SAEGenRec.training.sft` module implementing SFT training with tokenizer extension (`TokenExtender`), multi-dataset concatenation, cosine learning rate schedule, optional LLM freezing with gradient masks, and optional DeepSpeed integration (config passed as parameter, single-GPU fallback when not provided).
- **FR-009**: System MUST provide a `SAEGenRec.training.rl` module implementing GRPO-based RL training with the `ReReTrainer`, supporting all four reward types (rule, ranking, semantic, sasrec) and optional DeepSpeed integration.
- **FR-010**: System MUST provide a `SAEGenRec.evaluation` module implementing constrained beam search evaluation via `ConstrainedLogitsProcessor` and metric computation (HR@K, NDCG@K) via `calc.py` logic.
- **FR-011**: System MUST provide a `SAEGenRec.models.sasrec` module implementing the SASRec collaborative filtering model (with GRU and Caser variants) for CF reward computation.
- **FR-012**: System MUST centralize all category name mappings (e.g., `"Industrial_and_Scientific"` → `"industrial and scientific items"`) in a single configuration location, eliminating duplication across modules.
- **FR-013**: System MUST provide a unified `set_seed()` utility that sets seeds for `random`, `numpy`, `torch` (CPU + CUDA), and `cudnn` determinism, used consistently across all modules.
- **FR-014**: All configurable parameters (hyperparameters, paths, model names) MUST be exposed through Python dataclasses, allowing override from code or command line.
- **FR-015**: Each pipeline stage MUST be independently executable via `python -m SAEGenRec.{module}` with `fire.Fire()` CLI, and also via corresponding `make` targets in the Makefile. A user can run any single stage given the correct input files, without importing or running any other stage.

### Key Entities

- **Item**: A product in the dataset, identified by `item_id` (integer), with attributes `title` and `description`. Each item maps to exactly one SID after quantization.
- **Review**: A user's review of an item, keyed by the `(user_id, item_id, timestamp)` triple (per-interaction granularity). Each review record contains `review_text`, `rating`, `timestamp`, and `summary`. Reviews are stored in `.review.json` as a list where each entry corresponds to one interaction that survived k-core filtering. The ordering matches the row indices of `{dataset}.emb-{model}-review.npy`, enabling direct positional lookup: the review embedding at row `i` corresponds to the review record at index `i` in `.review.json`. This per-interaction design allows each position in a user's interaction sequence to carry its own review embedding.
- **SID (Semantic Item Description)**: A sequence of discrete tokens (e.g., `[a_42][b_128][c_7]`) that uniquely identifies an item. Produced by quantizing the item's text embedding through RQ-VAE or RQ-Kmeans. SID tokens are added to the LLM vocabulary.
- **Interaction Sequence**: A chronologically ordered list of item SIDs representing a user's purchase history. Used as input to the recommendation model.
- **Prompt**: A formatted text string containing the user's interaction history and an instruction to predict the next item. Follows the template: `### User Input:\n{instruction_with_history}\n\n### Response:\n{target_sid}`.
- **Index Mapping**: A JSON dictionary mapping item IDs (strings) to lists of SID tokens. Core data contract between SID construction and all downstream stages.
- **Info File**: A tab-separated text file listing all valid items as `semantic_id\titem_title\titem_id`. Used to build the prefix tree for constrained decoding.
- **Prefix Tree (hash_dict)**: An in-memory dictionary mapping SID token prefixes to valid next tokens. Enables constrained beam search during evaluation and RL training.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Given the same input data and random seed, every modularized pipeline stage produces byte-identical or numerically equivalent output (within floating-point tolerance of 1e-6) compared to the corresponding reference script.
- **SC-002**: A new user can install the package (`pip install -e .`), run the full pipeline end-to-end on the included reference dataset (Industrial_and_Scientific), and obtain HR@K/NDCG@K metrics within 1% of the reference results.
- **SC-003**: Each of the 6 pipeline stages (data preprocessing, text embedding, SID construction, dataset conversion, training, evaluation) can be imported and executed independently without errors when provided correct input files.
- **SC-004**: Zero hardcoded file paths or hyperparameter values exist in the business logic modules — all are sourced from configuration dataclasses.
- **SC-005**: All existing tests pass (`make test`) and new module-level tests achieve coverage of core data transformation logic (file format parsing, prompt generation, SID tokenization, metric calculation).
- **SC-006**: The `references/MiniOneRec/` directory remains completely unmodified throughout the modularization process.
