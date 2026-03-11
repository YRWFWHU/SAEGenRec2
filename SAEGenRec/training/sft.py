"""SFT 训练模块：TokenExtender + SFT 训练逻辑。

支持两种模式：
1. JSONL 模式（新）：从 sft_data_dir 读取 train.jsonl/valid.jsonl，使用 TRL SFTTrainer
2. CSV 模式（旧，向后兼容）：从 train_csv/valid_csv 读取，使用 transformers.Trainer
"""

from functools import partial
import json
import math
import os
import random
import re
from typing import Optional

from loguru import logger
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)

from SAEGenRec.config import CATEGORY_MAP


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


def _detect_sid_tokens_in_vocab(tokenizer) -> bool:
    """检测 tokenizer 词表中是否已包含 SID token（如 [a_0], [f_0], [v_0]）。"""
    vocab = tokenizer.get_vocab()
    sid_pattern = re.compile(r"^\[.+_\d+\]$")
    for token in vocab:
        if sid_pattern.match(token):
            return True
    return False


def _extract_sid_tokens_from_jsonl(jsonl_path: str) -> list:
    """从 JSONL completion 中提取所有 SID token。"""
    sid_pattern = re.compile(r"\[[^\]]+_\d+\]")
    token_set = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                completion = obj.get("completion", "")
                tokens = sid_pattern.findall(completion)
                token_set.update(tokens)
            except json.JSONDecodeError:
                pass
    return sorted(token_set)


def _apply_freeze_llm(model, tokenizer, original_vocab_size: int, new_tokens: list):
    """冻结 LLM 参数，仅训练 SID token embedding。"""
    if not new_tokens:
        logger.warning("freeze_llm=True but no new tokens added. All parameters are frozen!")
        for param in model.parameters():
            param.requires_grad = False
        return

    for param in model.parameters():
        param.requires_grad = False

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
    model_path: str = "",
    sft_data_dir: str = "",
    category: str = "",
    output_dir: str = "",
    freeze_llm: bool = False,
    num_epochs: int = 10,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    sample: int = -1,
    seed: int = 42,
    deepspeed: Optional[str] = None,
    wandb_project: str = "",
    wandb_run_name: str = "",
    eval_rec: bool = False,
    eval_rec_steps: float = 0.1,
    eval_rec_beams: int = 10,
    eval_rec_samples: int = 200,
    # 向后兼容参数（CSV 模式）
    model_name: str = "",
    train_csv: str = "",
    valid_csv: str = "",
    info_file: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    group_by_length: bool = False,
    train_from_scratch: bool = False,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    """SFT 训练入口（JSONL 模式）。

    Args:
        model_path: 基座 LLM 或 SFT checkpoint 路径（优先）
        sft_data_dir: JSONL 数据目录（含 train.jsonl、valid.jsonl、meta.json）
        category: 商品类别名（为空时从 meta.json 自动推断）
        output_dir: 模型输出目录
        freeze_llm: 是否冻结 LLM，只训练 SID token embedding
        eval_rec: 是否启用训练期间推荐指标评估
    """
    # 向后兼容：若未提供 model_path 则使用 model_name
    if not model_path and model_name:
        model_path = model_name

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

    # ---- JSONL 模式 ----
    if sft_data_dir and os.path.exists(sft_data_dir):
        _sft_jsonl(
            model_path=model_path,
            sft_data_dir=sft_data_dir,
            category=category,
            output_dir=output_dir,
            freeze_llm=freeze_llm,
            num_epochs=num_epochs,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            learning_rate=learning_rate,
            cutoff_len=cutoff_len,
            sample=sample,
            seed=seed,
            deepspeed=deepspeed,
            wandb_run_name=wandb_run_name,
            eval_rec=eval_rec,
            eval_rec_steps=eval_rec_steps,
            eval_rec_beams=eval_rec_beams,
            eval_rec_samples=eval_rec_samples,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device_map=device_map,
            ddp=ddp,
        )
    else:
        # ---- CSV 兼容模式 ----
        _sft_csv_compat(
            model_name=model_path,
            train_csv=train_csv,
            valid_csv=valid_csv,
            sid_index_path=sid_index_path,
            item_meta_path=item_meta_path,
            output_dir=output_dir or "models/sft",
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            cutoff_len=cutoff_len,
            freeze_llm=freeze_llm,
            train_from_scratch=train_from_scratch,
            wandb_run_name=wandb_run_name,
            resume_from_checkpoint=resume_from_checkpoint,
            deepspeed=deepspeed,
            sample=sample,
            seed=seed,
            category=category,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device_map=device_map,
            ddp=ddp,
        )


def _sft_jsonl(
    model_path: str,
    sft_data_dir: str,
    category: str,
    output_dir: str,
    freeze_llm: bool,
    num_epochs: int,
    batch_size: int,
    micro_batch_size: int,
    learning_rate: float,
    cutoff_len: int,
    sample: int,
    seed: int,
    deepspeed: Optional[str],
    wandb_run_name: str,
    eval_rec: bool,
    eval_rec_steps: float,
    eval_rec_beams: int,
    eval_rec_samples: int,
    gradient_accumulation_steps: int,
    device_map,
    ddp: bool,
) -> None:
    """JSONL 模式 SFT 训练：使用 TRL SFTTrainer（prompt/completion dict 格式，自动 completion-only loss）。"""
    from trl import SFTConfig, SFTTrainer

    from datasets import Dataset as HFDataset

    # 读取 meta.json 推断 category
    meta_path = os.path.join(sft_data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if not category:
            category = meta.get("category", "")

    if not output_dir:
        output_dir = os.path.join("models", "sft", category)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    original_vocab_size = len(tokenizer)

    # 注册 SID 边界特殊 token
    from SAEGenRec.datasets.task_registry import SID_SPECIAL_TOKENS
    tokenizer.add_special_tokens({"additional_special_tokens": SID_SPECIAL_TOKENS})
    logger.info(f"Added SID boundary special tokens: {SID_SPECIAL_TOKENS}")

    # SID token 检测与扩展
    if _detect_sid_tokens_in_vocab(tokenizer):
        logger.info("SID tokens already present in tokenizer vocab, skipping extension.")
        new_tokens = []
    else:
        # 从 JSONL 中提取 SID token
        train_jsonl = os.path.join(sft_data_dir, "train.jsonl")
        new_tokens = _extract_sid_tokens_from_jsonl(train_jsonl)
        if new_tokens:
            logger.info(f"Adding {len(new_tokens)} SID tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device_map
    )
    # resize for all newly added tokens (SID boundary + SID content tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 冻结 LLM（可选）
    if freeze_llm:
        logger.info("freeze_llm=True: freezing LLM, only training SID token embeddings")
        _apply_freeze_llm(model, tokenizer, original_vocab_size, new_tokens)

    # 加载 JSONL 数据（保留 prompt/completion 分离，TRL 0.29.0 自动处理 completion-only loss）
    def _load_jsonl(path: str, sample_n: int = -1) -> HFDataset:
        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                examples.append({"prompt": obj["prompt"], "completion": obj["completion"]})
        if sample_n > 0 and sample_n < len(examples):
            import random as _random
            examples = _random.sample(examples, sample_n)
        return HFDataset.from_list(examples)

    train_path = os.path.join(sft_data_dir, "train.jsonl")
    valid_path = os.path.join(sft_data_dir, "valid.jsonl")
    train_dataset = _load_jsonl(train_path, sample)
    eval_dataset = _load_jsonl(valid_path)

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # 训练 callbacks
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    if eval_rec:
        try:
            from SAEGenRec.evaluation.training_evaluator import TrainingEvaluator
            # 从 meta.json 推断 test_csv 和 info_file
            eval_test_csv = ""
            eval_info_file = ""
            if os.path.exists(meta_path):
                processed_base = os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(sft_data_dir)
                )))
                eval_test_csv = os.path.join(processed_base, f"{category}.test.csv")
                eval_info_file = os.path.join(processed_base, "info", f"{category}.txt")

            # 将 float 比例转换为整数步数（基于 dataset 大小估算）
            n_train = len(train_dataset)
            steps_per_epoch = max(1, n_train // (micro_batch_size * gradient_accumulation_steps))
            total_steps = steps_per_epoch * num_epochs
            if isinstance(eval_rec_steps, float) and eval_rec_steps < 1.0:
                eval_rec_steps_int = max(1, int(total_steps * eval_rec_steps))
            else:
                eval_rec_steps_int = int(eval_rec_steps)

            evaluator = TrainingEvaluator(
                eval_rec_steps=eval_rec_steps_int,
                eval_rec_beams=eval_rec_beams,
                eval_rec_samples=eval_rec_samples,
                test_csv=eval_test_csv,
                info_file=eval_info_file,
                category=category,
            )
            callbacks.append(evaluator)
            logger.info(f"TrainingEvaluator callback registered (every {eval_rec_steps_int} steps).")
        except Exception as e:
            logger.warning(f"Could not create TrainingEvaluator: {e}")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model.gradient_checkpointing_enable()

    eval_step = 0.05
    training_args = SFTConfig(
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
        report_to="none",
        deepspeed=deepspeed,
        max_length=cutoff_len,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        processing_class=tokenizer,
    )
    model.config.use_cache = False

    trainer.train()
    trainer.save_model(output_dir)

    final_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved final model to: {final_dir}")


def _sft_csv_compat(
    model_name: str,
    train_csv: str,
    valid_csv: str,
    sid_index_path: str,
    item_meta_path: str,
    output_dir: str,
    batch_size: int,
    micro_batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    freeze_llm: bool,
    train_from_scratch: bool,
    wandb_run_name: str,
    resume_from_checkpoint: Optional[str],
    deepspeed: Optional[str],
    sample: int,
    seed: int,
    category: str,
    gradient_accumulation_steps: int,
    device_map,
    ddp: bool,
) -> None:
    """CSV 兼容模式：保留旧 SFT 训练逻辑。"""
    from torch.utils.data import ConcatDataset

    from datasets import Dataset as HFDataset
    from SAEGenRec.datasets.sft_datasets import (
        FusionSeqRecDataset,
        SidItemFeatDataset,
        SidSFTDataset,
    )

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
        _apply_freeze_llm(model, tokenizer, original_vocab_size, new_tokens)

    cat_name = CATEGORY_MAP.get(category, category)
    train_datasets = []

    sft_ds = SidSFTDataset(
        train_file=train_csv,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=cat_name,
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
            category=cat_name,
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
            category=cat_name,
        )
        train_datasets.append(fusion_ds)

    train_data = ConcatDataset(train_datasets)
    val_data = SidSFTDataset(
        train_file=valid_csv,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=cat_name,
    )

    logger.info("Data loaded (CSV compat mode)")

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
