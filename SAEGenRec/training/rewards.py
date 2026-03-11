"""RL reward 函数：rule、prefix、ranking、semantic、sasrec。

Ported from references/MiniOneRec/rl.py reward functions.
"""

import math
import re
from typing import Callable, Dict, List, Optional

from loguru import logger
import numpy as np
import torch

# ---- Reward Registry ----

_REWARD_REGISTRY: Dict[str, Callable] = {}


def register_reward(name: str):
    """注册 reward 函数的装饰器。"""
    def decorator(fn: Callable) -> Callable:
        _REWARD_REGISTRY[name] = fn
        return fn
    return decorator


def get_reward_fn(name: str) -> Callable:
    """通过名称获取 reward 函数，包装异常处理。"""
    if name not in _REWARD_REGISTRY:
        available = list(_REWARD_REGISTRY.keys())
        raise ValueError(f"Unknown reward: '{name}'. Available: {available}")
    raw_fn = _REWARD_REGISTRY[name]

    def safe_fn(predictions, target, **kwargs):
        try:
            return raw_fn(predictions, target, **kwargs)
        except Exception as e:
            logger.warning(f"Reward '{name}' raised exception: {e}. Returning zeros.")
            return [0.0] * len(predictions)

    return safe_fn


def list_rewards() -> Dict[str, str]:
    """列出所有已注册的 reward 函数及其文档字符串。"""
    return {name: (fn.__doc__ or "").split("\n")[0].strip() for name, fn in _REWARD_REGISTRY.items()}


class CombinedReward:
    """组合多个 reward 函数，按权重加权求和。"""

    def __init__(self, names: List[str], weights: List[float]):
        assert len(names) == len(weights), "names 和 weights 数量不匹配"
        self._fns = [get_reward_fn(n) for n in names]
        total = sum(weights)
        self._weights = [w / total for w in weights]  # normalize

    def __call__(self, predictions: List[str], target: str, **kwargs) -> List[float]:
        n = len(predictions)
        result = [0.0] * n
        for fn, w in zip(self._fns, self._weights):
            rewards = fn(predictions, target, **kwargs)
            for i, r in enumerate(rewards):
                result[i] += w * r
        return result


def parse_reward_type(reward_type: str, reward_weights: str) -> Callable:
    """解析 reward_type 字符串（支持 '+' 组合语法）。

    Args:
        reward_type: 如 'rule' 或 'rule+prefix'
        reward_weights: 如 '' 或 '0.7,0.3'

    Returns:
        单个 reward 函数或 CombinedReward 实例
    """
    names = [n.strip() for n in reward_type.split("+") if n.strip()]
    if len(names) == 1:
        return get_reward_fn(names[0])
    if reward_weights:
        weights = [float(w.strip()) for w in reward_weights.split(",")]
    else:
        weights = [1.0 / len(names)] * len(names)
    return CombinedReward(names=names, weights=weights)


# ---- Builtin reward functions ----

@register_reward("rule")
def rule_reward(
    predictions: List[str],
    target: str,
    item_dict: Optional[Dict[str, List[int]]] = None,
    **kwargs,
) -> List[float]:
    """Rule reward：预测与目标完全匹配则奖励 1，否则 0。

    Args:
        predictions: 模型生成的多个预测（对应 num_generations 个）
        target: 目标 SID 字符串
        item_dict: SID → item_id 列表映射（用于验证是否是合法 SID，可选）

    Returns:
        rewards: 每个预测对应的奖励（0 或 1）
    """
    target = target.strip().strip('"').strip("\n")
    rewards = []
    for pred in predictions:
        pred = pred.strip().strip('"').strip("\n")
        rewards.append(1.0 if pred == target else 0.0)
    return rewards


@register_reward("prefix")
def prefix_reward(
    predictions: List[str],
    target: str,
    item_dict: Optional[Dict[str, List[int]]] = None,
    **kwargs,
) -> List[float]:
    """Prefix reward：对 SID 各分量前缀匹配给予部分奖励，引入 reward 方差。

    SID 格式 [a_x][b_y][c_z]，按分量匹配：
    - 完全匹配：1.0
    - 前两分量匹配：0.5
    - 仅第一分量匹配：0.2
    - 不匹配：0.0

    与 rule_reward 相比，partial match 可产生组内 reward 方差，
    使 GRPO 在精确匹配难以实现时也能获得学习信号。
    """
    _BRACKET = re.compile(r'\[[^\]]+\]')

    target_clean = target.strip().strip('"').strip("\n")
    target_parts = _BRACKET.findall(target_clean)
    n = len(target_parts)

    rewards = []
    for pred in predictions:
        pred_clean = pred.strip().strip('"').strip("\n")
        if pred_clean == target_clean:
            rewards.append(1.0)
            continue
        pred_parts = _BRACKET.findall(pred_clean)
        matches = sum(1 for a, b in zip(target_parts, pred_parts) if a == b)
        if n == 0 or matches == 0:
            rewards.append(0.0)
        elif matches == n - 1:
            rewards.append(0.5)
        elif matches >= 1:
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


@register_reward("ranking")
def ranking_reward(
    predictions: List[str],
    target: str,
    item_dict: Optional[Dict[str, List[int]]] = None,
    k: int = 20,
    **kwargs,
) -> List[float]:
    """Ranking reward：目标在预测列表中的 NDCG-aware 排名奖励。

    与 rule_reward 不同：对所有预测一起评分（基于在列表中的排名）。
    Predictions 被视为有序列表（beam search 排名），返回每个预测的奖励。

    References:
        SC-001 行为对等：参考 references/MiniOneRec/rl.py ranking_reward
    """
    target = target.strip().strip('"').strip("\n")
    preds_clean = [p.strip().strip('"').strip("\n") for p in predictions]

    # 找到目标的最高排名
    rank = None
    for i, pred in enumerate(preds_clean):
        if pred == target:
            rank = i
            break

    rewards = []
    for i, pred in enumerate(preds_clean):
        if pred == target:
            if rank is not None and rank < k:
                # NDCG-style reward
                reward = 1.0 / math.log2(rank + 2)
            else:
                reward = 0.0
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards


@register_reward("semantic")
def semantic_reward(
    predictions: List[str],
    target: str,
    embeddings: Optional[np.ndarray] = None,
    item2idx: Optional[Dict[str, int]] = None,
    **kwargs,
) -> List[float]:
    """Semantic reward：预测与目标嵌入的余弦相似度。

    Args:
        predictions: 模型生成的预测 SID 字符串
        target: 目标 SID 字符串
        embeddings: item 嵌入矩阵 (N_items × dim)
        item2idx: SID → embedding 行索引映射

    Returns:
        rewards: 余弦相似度 [0, 1]
    """
    target = target.strip().strip('"').strip("\n")

    if embeddings is None or item2idx is None:
        # Fallback to rule reward if no embeddings
        return rule_reward(predictions, target)

    target_idx = item2idx.get(target)
    if target_idx is None:
        return [0.0] * len(predictions)

    target_emb = embeddings[target_idx]
    target_norm = np.linalg.norm(target_emb)

    rewards = []
    for pred in predictions:
        pred = pred.strip().strip('"').strip("\n")
        pred_idx = item2idx.get(pred)
        if pred_idx is None:
            rewards.append(0.0)
            continue
        pred_emb = embeddings[pred_idx]
        pred_norm = np.linalg.norm(pred_emb)
        if target_norm == 0 or pred_norm == 0:
            rewards.append(0.0)
        else:
            cos_sim = np.dot(target_emb, pred_emb) / (target_norm * pred_norm)
            rewards.append(float(max(0.0, cos_sim)))

    return rewards


@register_reward("sasrec")
def sasrec_reward(
    predictions: List[str],
    target: str,
    sasrec_model=None,
    history_item_ids: Optional[List[int]] = None,
    item2id: Optional[Dict[str, int]] = None,
    device: str = "cpu",
    **kwargs,
) -> List[float]:
    """SASRec reward：使用训练好的 SASRec 模型对预测 SID 评分。

    Args:
        predictions: 模型生成的预测 SID 字符串
        target: 目标 SID 字符串（未使用，保持接口一致）
        sasrec_model: 训练好的 SASRec 模型（torch.nn.Module）
        history_item_ids: 用户历史 item_id 列表
        item2id: SID → item_id 映射
        device: 计算设备

    Returns:
        rewards: SASRec 预测分数
    """
    if sasrec_model is None or item2id is None or history_item_ids is None:
        return [0.0] * len(predictions)

    dev = torch.device(device)
    sasrec_model.eval()

    rewards = []
    with torch.no_grad():
        for pred in predictions:
            pred = pred.strip().strip('"').strip("\n")
            pred_item_id = item2id.get(pred)
            if pred_item_id is None:
                rewards.append(0.0)
                continue

            try:
                history_tensor = torch.tensor([history_item_ids], dtype=torch.long).to(dev)
                pred_tensor = torch.tensor([[pred_item_id]], dtype=torch.long).to(dev)
                score = sasrec_model.predict(history_tensor, pred_tensor)
                rewards.append(float(score.cpu().item()))
            except Exception:
                rewards.append(0.0)

    return rewards
