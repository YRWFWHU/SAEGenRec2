"""RL (GRPO) 训练入口。

Ported from references/MiniOneRec/rl.py
"""

import os
import random
from typing import Optional

from loguru import logger
import numpy as np
import torch
from trl import GRPOConfig

from SAEGenRec.config import CATEGORY_MAP
from SAEGenRec.datasets.rl_datasets import RLSeqTitle2SidDataset, RLTitle2SidDataset, SidDataset
from SAEGenRec.evaluation.logit_processor import build_prefix_tree
from SAEGenRec.training.rewards import parse_reward_type
from SAEGenRec.training.trainer import ReReTrainer


def rl(
    model_path: str = "",
    seed: int = 42,
    train_csv: str = "",
    eval_csv: str = "",
    info_file: str = "",
    category: str = "",
    output_dir: str = "models/rl",
    sample: int = -1,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,
    temperature: float = 1.0,
    num_generations: int = 16,
    eval_step: float = 0.199,
    reward_type: str = "rule",
    reward_weights: str = "",
    prompt_template: str = "",
    beam_search: bool = False,
    test_beam: int = 20,
    dynamic_sampling: bool = False,
    add_gt: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_during_training: bool = True,
    sample_train: bool = False,
    dapo: bool = False,
    gspo: bool = False,
    deepspeed: Optional[str] = None,
    wandb_project: str = "",
    wandb_run_name: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    cf_path: str = "",
    ada_path: str = "",
    eval_rec: bool = False,
    eval_rec_steps: int = 100,
    eval_rec_beams: int = 10,
    eval_rec_samples: int = 200,
    eval_test_csv: str = "",
    eval_info_file: str = "",
) -> None:
    """RL 训练入口。

    Args:
        model_path: SFT checkpoint 路径
        train_csv: 训练 CSV 路径
        info_file: info TXT 路径（用于构建前缀树）
        reward_type: 奖励函数名称，支持 '+' 组合，如 "rule+prefix"
        reward_weights: 组合奖励权重，如 "0.7,0.3"
        prompt_template: 自定义 prompt 模板路径（空字符串时使用数据集默认）
    """
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ["WANDB_PROJECT"] = wandb_project

    cat_name = CATEGORY_MAP.get(category, category)

    # Build datasets
    train_datasets = []

    if train_csv and os.path.exists(train_csv):
        sid_ds = SidDataset(train_file=train_csv, seed=seed, category=cat_name, sample=sample)
        train_datasets.append(sid_ds)

    if item_meta_path and sid_index_path and os.path.exists(item_meta_path) and os.path.exists(sid_index_path):
        title_ds = RLTitle2SidDataset(
            item_file=item_meta_path,
            index_file=sid_index_path,
            seed=seed,
            category=cat_name,
            sample=sample,
        )
        train_datasets.append(title_ds)

        seq_ds = RLSeqTitle2SidDataset(
            train_file=train_csv,
            seed=seed,
            category=cat_name,
            sample=sample,
        )
        train_datasets.append(seq_ds)

    if not train_datasets:
        raise ValueError("No training data found. Please provide train_csv and/or item_meta_path.")

    from datasets import Dataset as HFDataset

    all_samples = []
    for ds in train_datasets:
        for item in ds:
            if item is not None:
                all_samples.append(item)

    hf_train = HFDataset.from_list(all_samples)
    logger.info(f"RL training samples: {len(hf_train)}")

    # Load model and tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build prefix tree for constrained generation
    prefix_allowed_tokens_fn = None
    if info_file and os.path.exists(info_file) and beam_search:
        hash_dict = build_prefix_tree(info_file, tokenizer)

        def prefix_fn(batch_id: int, hash_key: list) -> list:
            return hash_dict.get(tuple(hash_key), [])

        prefix_allowed_tokens_fn = prefix_fn

    # Load custom prompt template if provided
    if prompt_template:
        from SAEGenRec.datasets.template_utils import load_template
        _prompt_tmpl = load_template(prompt_template)
        logger.info(f"Loaded prompt template from: {prompt_template}")
    else:
        _prompt_tmpl = None

    # Select reward function (supports '+' combined syntax)
    reward_fn = parse_reward_type(reward_type, reward_weights)
    logger.info(f"Reward function: {reward_type}" + (f" (weights: {reward_weights})" if reward_weights else ""))

    def compute_rewards(prompts, completions, **kwargs):
        rewards = []
        targets = kwargs.get("target", [""] * len(completions))
        for i, completion in enumerate(completions):
            target = targets[i] if isinstance(targets, (list, tuple)) else targets
            r = reward_fn([completion], target)[0]
            rewards.append(r)
        return rewards

    # TrainingEvaluator callback (optional)
    rl_callbacks = []
    if eval_rec:
        try:
            from SAEGenRec.evaluation.training_evaluator import TrainingEvaluator
            evaluator = TrainingEvaluator(
                eval_rec_steps=eval_rec_steps,
                eval_rec_beams=eval_rec_beams,
                eval_rec_samples=eval_rec_samples,
                test_csv=eval_test_csv or "",
                info_file=eval_info_file or info_file,
                category=category,
            )
            rl_callbacks.append(evaluator)
            logger.info(f"TrainingEvaluator callback registered (every {eval_rec_steps} steps).")
        except Exception as e:
            logger.warning(f"Could not create TrainingEvaluator: {e}")

    # GRPOConfig
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
        temperature=temperature,
        num_generations=num_generations,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=eval_step,
        bf16=True,
        report_to="none",
        run_name=wandb_run_name,
        deepspeed=deepspeed,
    )

    trainer = ReReTrainer(
        model=model_path,
        args=grpo_config,
        train_dataset=hf_train,
        processing_class=tokenizer,
        reward_funcs=[compute_rewards],
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=test_beam,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        dynamic_sampling=dynamic_sampling,
        reward_type=reward_type,
        info_file=info_file,
        base_model_name=model_path,
        callbacks=rl_callbacks if rl_callbacks else None,
    )

    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"RL training done. Saved to: {output_dir}")
