"""从训练好的 GatedSAE checkpoint 生成 .index.json 文件。"""

import json
import os

from loguru import logger
import numpy as np
from sae_lens import SAE
import torch


def _encode_all(
    sae: SAE,
    embeddings: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """批量编码所有 item embedding，返回 (n_items, d_sae) feature activations。"""
    n = embeddings.shape[0]
    all_acts = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = torch.from_numpy(embeddings[start : start + batch_size]).float().to(device)
            acts = sae.encode(batch)
            all_acts.append(acts.cpu())
    return torch.cat(all_acts, dim=0)


def _build_sids(topk_indices: torch.Tensor, k: int) -> list[list[int]]:
    """从 top-k 索引构建 SID 列表（每个 item 一个 k 元素列表）。"""
    return topk_indices[:, :k].tolist()


def _sids_to_str(sids: list[list[int]]) -> list[str]:
    return [str(sid) for sid in sids]


def generate_sae_indices(
    checkpoint: str = "",
    embedding_path: str = "",
    k: int = 8,
    output_path: str = "",
    max_dedup_iters: int = 20,
    batch_size: int = 256,
    device: str = "cuda:0",
) -> str:
    """从 GatedSAE checkpoint 生成所有 item 的 SID token 索引。

    Args:
        checkpoint: GatedSAE checkpoint 目录路径（含 sae_weights.safetensors + cfg.json）
        embedding_path: item embedding .npy 路径
        k: top-K 特征数（SID token 数），默认 8
        output_path: 输出 .index.json 路径
        max_dedup_iters: 最大去碰撞迭代次数
        batch_size: 编码 batch 大小
        device: 计算设备

    Returns:
        output_path
    """
    dev_str = device if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_str)

    # 加载 GatedSAE（推理模型）
    sae = SAE.load_from_disk(checkpoint, device=dev_str)
    sae.eval()
    logger.info(f"Loaded GatedSAE from {checkpoint}: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # 加载 embedding
    raw = np.load(embedding_path).astype(np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    n_items = raw.shape[0]
    logger.info(f"Loaded embeddings: {raw.shape}")

    # 编码所有 item
    feature_acts = _encode_all(sae, raw, batch_size=batch_size, device=dev)
    # shape: (n_items, d_sae)

    # 预计算 top-(K + max_dedup_iters) 以便去重时有备用特征
    extra = max_dedup_iters
    fetch_k = min(k + extra, sae.cfg.d_sae)
    topk_values, topk_indices = torch.topk(feature_acts, k=fetch_k, dim=1)
    # topk_indices: (n_items, fetch_k)

    # 处理 K > 非零激活数的情况：用 0 填充（实际上 topk 会返回 0 值对应的索引，已足够）
    # 初始 SID = top-K 索引
    item_sids: list[list[int]] = topk_indices[:, :k].tolist()

    # ——— SID 去重 ———
    for iteration in range(max_dedup_iters):
        # 检测碰撞
        sid_to_items: dict[str, list[int]] = {}
        for item_id, sid in enumerate(item_sids):
            key = str(sid)
            if key not in sid_to_items:
                sid_to_items[key] = []
            sid_to_items[key].append(item_id)

        collision_groups = [g for g in sid_to_items.values() if len(g) > 1]
        if not collision_groups:
            logger.info(f"Dedup complete after {iteration} iteration(s). No collisions remaining.")
            break

        n_conflicts = sum(len(g) - 1 for g in collision_groups)
        logger.info(
            f"Dedup iteration {iteration}: {len(collision_groups)} collision groups, "
            f"{n_conflicts} conflicting items"
        )

        # 对每个碰撞组，让非首位 item 尝试使用备用特征替换最后一个位置
        j = iteration + 1  # 使用第 K+j 个特征（0-indexed: K+j-1）
        candidate_col = k + j - 1  # topk_indices 中的列索引
        if candidate_col >= fetch_k:
            logger.warning(f"Exhausted backup features at iteration {iteration}, stopping dedup.")
            break

        for group in collision_groups:
            # 保留第一个 item，对其余 item 替换最后一个位置
            for item_id in group[1:]:
                new_last = int(topk_indices[item_id, candidate_col].item())
                new_sid = item_sids[item_id].copy()
                new_sid[-1] = new_last
                item_sids[item_id] = new_sid

    # 最终碰撞统计
    final_strs = _sids_to_str(item_sids)
    n_unique = len(set(final_strs))
    collision_rate = (n_items - n_unique) / n_items
    logger.info(
        f"Final: {n_items} items, {n_unique} unique SIDs, collision_rate={collision_rate:.4f}"
    )
    if collision_rate > 0.05:
        logger.warning(
            f"Collision rate {collision_rate:.4f} exceeds 5% threshold. "
            "Consider increasing max_dedup_iters or expansion_factor."
        )

    # 转换为 token 字符串格式：[f_{feature_index}]
    # GatedSAE 所有特征来自同一 codebook，使用统一前缀 "f"（feature），
    # 与 RQ-VAE 的位置感知前缀（a/b/c 对应不同 codebook）区别开来。
    index_dict: dict[str, list[str]] = {}
    for item_id, sid in enumerate(item_sids):
        tokens = [f"[f_{feat_idx}]" for feat_idx in sid]
        index_dict[str(item_id)] = tokens

    # 保存 .index.json
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
    )
    with open(output_path, "w") as f:
        json.dump(index_dict, f)
    logger.info(f"Saved: {output_path}")

    return output_path
