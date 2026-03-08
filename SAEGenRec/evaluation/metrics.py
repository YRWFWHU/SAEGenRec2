"""评估指标：HR@K 和 NDCG@K。

Ported from references/MiniOneRec/calc.py
"""

import math
from typing import Dict, List, Optional

import numpy as np


def compute_hr_ndcg(
    predictions: List[List[str]],
    targets: List[str],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """计算 HR@K 和 NDCG@K。

    Args:
        predictions: 每个样本的预测列表（beam search 输出，按排名排序）
        targets: 每个样本的真实目标（字符串）
        k_values: K 值列表，默认 [1, 3, 5, 10, 20]

    Returns:
        dict: {"HR@1": ..., "NDCG@1": ..., ...}
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    n = len(predictions)
    if n == 0:
        return {f"HR@{k}": 0.0 for k in k_values} | {f"NDCG@{k}": 0.0 for k in k_values}

    max_beams = len(predictions[0]) if predictions else 0
    valid_ks = [k for k in k_values if k <= max_beams]

    all_hr = {k: 0.0 for k in valid_ks}
    all_ndcg = {k: 0.0 for k in valid_ks}

    for preds, target in zip(predictions, targets):
        target = target.strip().strip('"').strip()
        # 找到目标在预测列表中的位置
        rank = None
        for i, pred in enumerate(preds):
            pred = pred.strip().strip('"').strip()
            if pred == target:
                rank = i
                break

        for k in valid_ks:
            if rank is not None and rank < k:
                all_hr[k] += 1.0
                all_ndcg[k] += 1.0 / math.log2(rank + 2)

    results = {}
    log2_2 = math.log2(2)  # normalizer: ideal rank=0 → log2(2)=1
    for k in valid_ks:
        results[f"HR@{k}"] = all_hr[k] / n
        results[f"NDCG@{k}"] = all_ndcg[k] / n / log2_2

    # Fill missing ks with 0
    for k in k_values:
        if f"HR@{k}" not in results:
            results[f"HR@{k}"] = 0.0
        if f"NDCG@{k}" not in results:
            results[f"NDCG@{k}"] = 0.0

    return results


def hr_at_k(predictions: List[List[str]], targets: List[str], k: int) -> float:
    """计算 HR@K。"""
    results = compute_hr_ndcg(predictions, targets, k_values=[k])
    return results[f"HR@{k}"]


def ndcg_at_k(predictions: List[List[str]], targets: List[str], k: int) -> float:
    """计算 NDCG@K。"""
    results = compute_hr_ndcg(predictions, targets, k_values=[k])
    return results[f"NDCG@{k}"]
