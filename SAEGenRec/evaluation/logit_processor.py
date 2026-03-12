"""约束 Logits 处理器：基于前缀树限制波束搜索输出为合法 SID。

Ported from references/MiniOneRec/LogitProcessor.py
"""

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import torch
from transformers.generation import LogitsProcessor


def build_prefix_tree(info_file: str, tokenizer) -> Dict[Tuple, List[int]]:
    """从 info TXT 文件构建前缀树（hash_dict）。

    info TXT 格式：semantic_id \\t item_title \\t item_id（无表头）

    Returns:
        hash_dict: {prefix_token_ids_tuple → valid_next_token_ids_list}
    """
    hash_dict: Dict[Tuple, List[int]] = {}

    with open(info_file) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split("\t")
        if not parts:
            continue
        semantic_id = parts[0].strip()
        if not semantic_id:
            continue

        # 将 semantic_id 字符串 tokenize
        token_ids = tokenizer.encode(semantic_id, add_special_tokens=False)

        # 构建前缀树：对长度为 0..n-1 的前缀，记录合法的下一个 token
        for prefix_len in range(len(token_ids)):
            prefix = tuple(token_ids[:prefix_len])
            next_token = token_ids[prefix_len]
            if prefix not in hash_dict:
                hash_dict[prefix] = []
            if next_token not in hash_dict[prefix]:
                hash_dict[prefix].append(next_token)

    return hash_dict


class ConstrainedLogitsProcessor(LogitsProcessor):
    """约束 Logits 处理器：使用前缀树限制波束搜索只生成合法 SID token 序列。

    Ported from references/MiniOneRec/LogitProcessor.py
    """

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, List[int]], List[int]],
        num_beams: int,
        base_model: str = "",
        eos_token_id: Optional[int] = None,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count = 0
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        if base_model and base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))

        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    # Step 0: use empty prefix to allow any valid first SID token
                    hash_key = []
                else:
                    hash_key = sent[-self.count :].tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)

                if len(prefix_allowed_tokens) == 0:
                    warnings.warn(
                        f"No valid tokens found for hash_key {hash_key} at step {self.count}. "
                        f"This indicates the model generated an unexpected token."
                    )
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue

                vocab_size = scores.shape[-1]
                valid_tokens = [t for t in prefix_allowed_tokens if t < vocab_size]
                if valid_tokens:
                    mask[batch_id * self._num_beams + beam_id, valid_tokens] = 0

        self.count += 1
        scores = scores + mask
        return scores
