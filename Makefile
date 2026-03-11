#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = SAEGenRec
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	pip install -e .
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# Pipeline configuration — override via: make preprocess CATEGORY=Beauty DATA_DIR=data/raw
CATEGORY ?= Beauty
DATASET ?= Amazon
DATA_DIR ?= data
INTERIM_DIR ?= data/interim
OUTPUT_DIR ?= data/processed
MODEL_PATH ?= models/sft
RL_MODEL_PATH ?= models/rl
SASREC_MODEL_PATH ?= models/sasrec
TRAIN_CSV ?= $(OUTPUT_DIR)/$(CATEGORY).train.csv
VALID_CSV ?= $(OUTPUT_DIR)/$(CATEGORY).valid.csv
TEST_CSV ?= $(OUTPUT_DIR)/$(CATEGORY).test.csv
INFO_FILE ?= $(OUTPUT_DIR)/info/$(CATEGORY).txt
EMB_DIR ?= $(OUTPUT_DIR)/emb
INDEX_FILE ?= $(INTERIM_DIR)/$(CATEGORY).index.json
RESULTS_DIR ?= results

# SID options
METHOD ?= rqvae
TOKEN_FORMAT ?= auto
SID_TYPE ?= $(METHOD)

# Embedding options
TEXT_MODEL ?= sentence-transformers/all-MiniLM-L6-v2
VISION_MODEL ?= openai/clip-vit-base-patch32
CONCURRENCY ?= 8

# SFT options
SFT_DATA_DIR ?= $(OUTPUT_DIR)/$(SID_TYPE)/$(DATASET)/$(CATEGORY)/sid_seq
TASK ?= sid_seq
SFT_MODEL_PATH ?= $(MODEL_PATH)

# RL options
REWARD_TYPE ?= rule
REWARD_WEIGHTS ?=
PROMPT_TEMPLATE ?=

## Preprocess raw Amazon review data (k-core filter + TO/LOO split)
.PHONY: preprocess
preprocess:
	$(PYTHON_INTERPRETER) -m SAEGenRec.data_process preprocess \
		--data_dir=$(DATA_DIR)/raw \
		--output_dir=$(OUTPUT_DIR) \
		--category=$(CATEGORY)

## Download item images from Amazon metadata (MAIN image, highest resolution)
.PHONY: download_images
download_images:
	$(PYTHON_INTERPRETER) -m SAEGenRec.data_process download_images \
		--category=$(CATEGORY) \
		--data_dir=$(DATA_DIR) \
		--concurrency=$(CONCURRENCY)

## Generate text embeddings with sentence-transformers
.PHONY: embed_text
embed_text:
	$(PYTHON_INTERPRETER) -m SAEGenRec.data_process embed_text \
		--category=$(CATEGORY) \
		--model=$(TEXT_MODEL) \
		--data_dir=$(INTERIM_DIR)

## Extract visual features from item images
.PHONY: extract_visual
extract_visual:
	$(PYTHON_INTERPRETER) -m SAEGenRec.data_process extract_visual \
		--category=$(CATEGORY) \
		--vision_model=$(VISION_MODEL) \
		--data_dir=$(INTERIM_DIR)

## Generate both text and visual embeddings
.PHONY: embed_all
embed_all: embed_text extract_visual

## Backward-compat alias for embed_text
.PHONY: embed
embed: embed_text

## Build semantic IDs via unified SID interface (METHOD=rqvae|rqkmeans|gated_sae)
.PHONY: build_sid
build_sid:
	$(PYTHON_INTERPRETER) -m SAEGenRec.sid_builder build_sid \
		--method=$(METHOD) \
		--category=$(CATEGORY) \
		--data_dir=$(INTERIM_DIR) \
		--output_dir=$(INTERIM_DIR) \
		--token_format=$(TOKEN_FORMAT)

## Train SID model only (no index generation)
.PHONY: train_sid
train_sid:
	$(PYTHON_INTERPRETER) -m SAEGenRec.sid_builder train_sid \
		--method=$(METHOD) \
		--category=$(CATEGORY) \
		--data_dir=$(INTERIM_DIR) \
		--output_dir=$(INTERIM_DIR)

## Generate SID index from trained checkpoint
.PHONY: generate_sid
generate_sid:
	$(PYTHON_INTERPRETER) -m SAEGenRec.sid_builder generate_sid \
		--method=$(METHOD) \
		--category=$(CATEGORY) \
		--data_dir=$(INTERIM_DIR) \
		--output_dir=$(INTERIM_DIR) \
		--token_format=$(TOKEN_FORMAT)

## Build semantic IDs via GatedSAE (backward-compat alias for build_sid METHOD=gated_sae)
SAE_MODEL_DIR ?= models/gated_sae/$(CATEGORY)
.PHONY: build_sae_sid
build_sae_sid:
	$(MAKE) build_sid METHOD=gated_sae

## List all available SID generation methods
.PHONY: list_sid_methods
list_sid_methods:
	$(PYTHON_INTERPRETER) -m SAEGenRec.sid_builder list_sid_methods

## Convert inter files to CSV + info TXT
.PHONY: convert
convert:
	$(PYTHON_INTERPRETER) -m SAEGenRec.data_process convert_dataset \
		--dataset=$(CATEGORY) \
		--data_dir=$(OUTPUT_DIR) \
		--index_file=$(INDEX_FILE) \
		--output_dir=$(OUTPUT_DIR)

## Prepare SFT data (CSV + index → JSONL)
.PHONY: prepare_sft
prepare_sft:
	$(PYTHON_INTERPRETER) -m SAEGenRec.training prepare_sft \
		--category=$(CATEGORY) \
		--sid_type=$(SID_TYPE) \
		--task=$(TASK) \
		--dataset=$(DATASET) \
		--interim_dir=$(INTERIM_DIR) \
		--data_dir=$(OUTPUT_DIR)

## List all available SFT task types
.PHONY: list_sft_tasks
list_sft_tasks:
	$(PYTHON_INTERPRETER) -m SAEGenRec.training list_sft_tasks

## Run SFT training (JSONL mode via SFT_DATA_DIR, or CSV mode for backward compat)
.PHONY: sft
sft:
	$(PYTHON_INTERPRETER) -m SAEGenRec.training sft \
		--model_path=$(MODEL_PATH) \
		--sft_data_dir=$(SFT_DATA_DIR) \
		--category=$(CATEGORY)

## Run SFT quick validation (100 samples)
.PHONY: sft_quick
sft_quick:
	$(PYTHON_INTERPRETER) -m SAEGenRec.training sft \
		--model_path=$(MODEL_PATH) \
		--sft_data_dir=$(SFT_DATA_DIR) \
		--category=$(CATEGORY) \
		--sample=100 \
		--num_epochs=1

## Run RL (GRPO) training
.PHONY: rl
rl:
	$(PYTHON_INTERPRETER) -m SAEGenRec.training rl \
		--model_path=$(MODEL_PATH) \
		--train_csv=$(TRAIN_CSV) \
		--eval_csv=$(VALID_CSV) \
		--info_file=$(INFO_FILE) \
		--category=$(CATEGORY) \
		--output_dir=$(RL_MODEL_PATH) \
		--reward_type=$(REWARD_TYPE) \
		$(if $(REWARD_WEIGHTS),--reward_weights=$(REWARD_WEIGHTS),) \
		$(if $(PROMPT_TEMPLATE),--prompt_template=$(PROMPT_TEMPLATE),)

## Run RL quick validation (500 samples, 4 generations)
.PHONY: rl_quick
rl_quick:
	$(PYTHON_INTERPRETER) -m SAEGenRec.training rl \
		--model_path=$(MODEL_PATH) \
		--train_csv=$(TRAIN_CSV) \
		--eval_csv=$(VALID_CSV) \
		--info_file=$(INFO_FILE) \
		--category=$(CATEGORY) \
		--output_dir=$(RL_MODEL_PATH) \
		--reward_type=$(REWARD_TYPE) \
		--sample=500 \
		--num_generations=4

## List all available reward functions
.PHONY: list_rewards
list_rewards:
	$(PYTHON_INTERPRETER) -m SAEGenRec.training list_rewards

## Evaluate model with constrained beam search
.PHONY: evaluate
evaluate:
	$(PYTHON_INTERPRETER) -m SAEGenRec.evaluation evaluate \
		--model_path=$(RL_MODEL_PATH) \
		--test_csv=$(TEST_CSV) \
		--info_file=$(INFO_FILE) \
		--category=$(CATEGORY) \
		--output_dir=$(RESULTS_DIR)

## Train SASRec CF model
.PHONY: sasrec
sasrec:
	$(PYTHON_INTERPRETER) -m SAEGenRec.models sasrec_train \
		--train_csv=$(TRAIN_CSV) \
		--valid_csv=$(VALID_CSV) \
		--test_csv=$(TEST_CSV) \
		--output_dir=$(SASREC_MODEL_PATH)

## Run full pipeline end-to-end (preprocess → embed → build_sid → convert → prepare_sft → sft → rl → evaluate)
.PHONY: pipeline
pipeline: preprocess embed_text build_sid convert prepare_sft sft rl evaluate


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
