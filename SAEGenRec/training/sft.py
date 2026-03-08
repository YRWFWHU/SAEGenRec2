"""SFT 训练模块：TokenExtender + SFT 训练逻辑。

Ported from references/MiniOneRec/sft.py
"""

import json
import math
import os
import random
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import transformers
from datasets import Dataset as HFDataset
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)

from SAEGenRec.config import CATEGORY_MAP
from SAEGenRec.datasets.sft_datasets import FusionSeqRecDataset, SidItemFeatDataset, SidSFTDataset


class TokenExtender:
    """从 .index.json 提取唯一 SID tokens，扩展分词器词表。

    Ported from references/MiniOneRec/sft.py TokenExtender
    """

    def __init__(self, index_file: str):
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None

    def _load_data(self):
        with open(self.index_file) as f:
            self.indices = json.load(f)

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
        if self.indices is None:
            self._load_data()
        token_set = set()
        for token_list in self.indices.values():
            for token in token_list:
                token_set.add(token)
        self.new_tokens = sorted(list(token_set))
        return self.new_tokens


def _get_cosine_schedule_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def sft(
    model_name: str = "",
    train_csv: str = "",
    valid_csv: str = "",
    info_file: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    output_dir: str = "models/sft",
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    group_by_length: bool = False,
    freeze_llm: bool = False,
    train_from_scratch: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: Optional[str] = None,
    deepspeed: Optional[str] = None,
    sample: int = -1,
    seed: int = 42,
    category: str = "",
) -> None:
    """SFT 训练入口。

    Args:
        model_name: HuggingFace 模型名称或路径
        train_csv: 训练 CSV 路径
        valid_csv: 验证 CSV 路径
        sid_index_path: .index.json 路径（用于 TokenExtender）
        item_meta_path: .item.json 路径（用于 SidItemFeatDataset 和 FusionSeqRecDataset）
        output_dir: 输出目录
        freeze_llm: 是否冻结 LLM 参数，只训练新增 token 嵌入
        deepspeed: DeepSpeed 配置文件路径（可选）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["WANDB_PROJECT"] = wandb_project

    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    else:
        device_map = "auto"

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device_map
        )
    else:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)
        logger.info("Training from scratch!")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    original_vocab_size = len(tokenizer)
    new_tokens = []

    if sid_index_path and os.path.exists(sid_index_path):
        logger.info(f"Loading index: {sid_index_path}")
        extender = TokenExtender(index_file=sid_index_path)
        new_tokens = extender.get_new_tokens()
        if new_tokens:
            logger.info(f"Adding {len(new_tokens)} new tokens")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

    if freeze_llm:
        logger.info("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad):
                    grad[:original_vocab_size].zero_()
                    return grad

                embedding_layer.weight.register_hook(mask_grad)
                logger.info(
                    f"Unfrozen {len(new_tokens)} new token embeddings "
                    f"(indices {original_vocab_size} to {len(tokenizer)-1})"
                )
        else:
            logger.warning("freeze_llm=True but no new tokens added. All parameters are frozen!")

    # Build datasets
    train_datasets = []

    sft_ds = SidSFTDataset(
        train_file=train_csv,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=CATEGORY_MAP.get(category, category),
    )
    train_datasets.append(sft_ds)

    if item_meta_path and sid_index_path and os.path.exists(item_meta_path) and os.path.exists(sid_index_path):
        feat_ds = SidItemFeatDataset(
            item_file=item_meta_path,
            index_file=sid_index_path,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=sample,
            seed=seed,
            category=CATEGORY_MAP.get(category, category),
        )
        train_datasets.append(feat_ds)

        fusion_ds = FusionSeqRecDataset(
            train_file=train_csv,
            item_file=item_meta_path,
            index_file=sid_index_path,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=sample,
            seed=seed,
            category=CATEGORY_MAP.get(category, category),
        )
        train_datasets.append(fusion_ds)

    train_data = ConcatDataset(train_datasets)
    val_data = SidSFTDataset(
        train_file=valid_csv,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=CATEGORY_MAP.get(category, category),
    )

    logger.info("Data loaded")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model.gradient_checkpointing_enable()

    hf_train = HFDataset.from_dict(
        {k: [v[k] for v in train_data] for k in train_data[0].keys()}
    ).shuffle(seed=42)
    hf_val = HFDataset.from_dict(
        {k: [v[k] for v in val_data] for k in val_data[0].keys()}
    ).shuffle(seed=42)

    eval_step = 0.05
    training_args = transformers.TrainingArguments(
        run_name=wandb_run_name,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=20,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=eval_step,
        save_strategy="steps",
        save_steps=eval_step,
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        # group_by_length removed in newer transformers
        report_to="none",
        deepspeed=deepspeed,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)

    final_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved final model to: {final_dir}")
