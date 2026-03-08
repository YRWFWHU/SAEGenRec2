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
DATA_DIR ?= data
OUTPUT_DIR ?= data/processed
MODEL_PATH ?= models/sft
RL_MODEL_PATH ?= models/rl
SASREC_MODEL_PATH ?= models/sasrec
TRAIN_CSV ?= $(OUTPUT_DIR)/$(CATEGORY).train.csv
VALID_CSV ?= $(OUTPUT_DIR)/$(CATEGORY).valid.csv
TEST_CSV ?= $(OUTPUT_DIR)/$(CATEGORY).test.csv
INFO_FILE ?= $(OUTPUT_DIR)/info/$(CATEGORY).txt
EMB_DIR ?= $(OUTPUT_DIR)/emb
INDEX_FILE ?= $(OUTPUT_DIR)/$(CATEGORY).index.json
RESULTS_DIR ?= results

## Preprocess raw Amazon review data (k-core filter + TO/LOO split)
.PHONY: preprocess
preprocess:
	$(PYTHON_INTERPRETER) -m SAEGenRec.data_process preprocess \
		--data_dir=$(DATA_DIR)/raw \
		--output_dir=$(OUTPUT_DIR) \
		--category=$(CATEGORY)

## Generate text embeddings with sentence-transformers
.PHONY: embed
embed:
	$(PYTHON_INTERPRETER) -m SAEGenRec.sid_builder text2emb \
		--dataset=$(CATEGORY) \
		--data_dir=$(OUTPUT_DIR) \
		--output_dir=$(EMB_DIR)

## Build semantic IDs via RQ-VAE training
.PHONY: build_sid
build_sid:
	$(PYTHON_INTERPRETER) -m SAEGenRec.sid_builder rqvae_train \
		--dataset=$(CATEGORY) \
		--emb_dir=$(EMB_DIR) \
		--output_dir=$(OUTPUT_DIR)
	$(PYTHON_INTERPRETER) -m SAEGenRec.sid_builder generate_indices \
		--dataset=$(CATEGORY) \
		--emb_dir=$(EMB_DIR) \
		--checkpoint_dir=$(OUTPUT_DIR) \
		--output_dir=$(OUTPUT_DIR)

## Convert inter files to CSV + info TXT
.PHONY: convert
convert:
	$(PYTHON_INTERPRETER) -m SAEGenRec.data_process convert_dataset \
		--dataset=$(CATEGORY) \
		--data_dir=$(OUTPUT_DIR) \
		--index_file=$(INDEX_FILE) \
		--output_dir=$(OUTPUT_DIR)

## Run SFT training
.PHONY: sft
sft:
	$(PYTHON_INTERPRETER) -m SAEGenRec.training sft \
		--model_path=$(MODEL_PATH) \
		--train_csv=$(TRAIN_CSV) \
		--eval_csv=$(VALID_CSV) \
		--info_file=$(INFO_FILE) \
		--category=$(CATEGORY)

## Run RL (GRPO) training
.PHONY: rl
rl:
	$(PYTHON_INTERPRETER) -m SAEGenRec.training rl \
		--model_path=$(MODEL_PATH) \
		--train_csv=$(TRAIN_CSV) \
		--eval_csv=$(VALID_CSV) \
		--info_file=$(INFO_FILE) \
		--category=$(CATEGORY) \
		--output_dir=$(RL_MODEL_PATH)

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

## Run full pipeline end-to-end (preprocess → embed → build_sid → convert → sft → rl → evaluate)
.PHONY: pipeline
pipeline: preprocess embed build_sid convert sft rl evaluate


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
