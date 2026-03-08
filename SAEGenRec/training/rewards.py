"""RL reward 函数：rule、ranking、semantic、sasrec。

Ported from references/MiniOneRec/rl.py reward functions.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import torch


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
