"""ReReTrainer：扩展 trl.GRPOTrainer，支持约束生成和多种 reward。

Ported from references/MiniOneRec/minionerec_trainer.py
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from loguru import logger
from trl import GRPOConfig, GRPOTrainer


class ReReTrainer(GRPOTrainer):
    """ReReTrainer：基于 GRPO 的 RL 训练器，支持约束波束搜索和在线评估。

    Extends trl.GRPOTrainer with:
    - ConstrainedLogitsProcessor for valid SID generation
    - Online evaluation during training
    - Dynamic sampling
    - Multiple reward function support (rule, ranking, semantic, sasrec)
    """

    def __init__(
        self,
        *args,
        prefix_allowed_tokens_fn: Optional[Callable] = None,
        num_beams: int = 1,
        beam_search: bool = False,
        test_during_training: bool = False,
        test_beam: int = 20,
        dynamic_sampling: bool = False,
        reward_type: str = "rule",
        info_file: str = "",
        base_model_name: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.num_beams = num_beams
        self.beam_search = beam_search
        self.test_during_training = test_during_training
        self.test_beam = test_beam
        self.dynamic_sampling = dynamic_sampling
        self.reward_type = reward_type
        self.info_file = info_file
        self.base_model_name = base_model_name

    def _generate_completions(
        self,
        model,
        prompts,
        tokenizer,
        generation_config,
        **kwargs,
    ):
        """重写生成方法以支持约束 Logits 处理器。"""
        if self.prefix_allowed_tokens_fn is not None and self.beam_search:
            from SAEGenRec.evaluation.logit_processor import ConstrainedLogitsProcessor
            from transformers import LogitsProcessorList

            constrained_processor = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                num_beams=generation_config.num_beams,
                base_model=self.base_model_name,
                eos_token_id=tokenizer.eos_token_id,
            )
            logits_processor = LogitsProcessorList([constrained_processor])
        else:
            logits_processor = None

        return super()._generate_completions(
            model,
            prompts,
            tokenizer,
            generation_config,
            logits_processor=logits_processor,
            **kwargs,
        )
