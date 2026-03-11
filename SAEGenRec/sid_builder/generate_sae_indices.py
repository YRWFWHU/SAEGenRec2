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



def generate_sae_indices(
    checkpoint: str = "",
    embedding_path: str = "",
    k: int = 8,
    output_path: str = "",
    max_dedup_iters: int = 20,
    batch_size: int = 256,
    device: str = "cuda:0",
    token_format: str = "auto",
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
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # 加载 GatedSAE（推理模型）
    sae = SAE.load_from_disk(checkpoint, device=str(dev))
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

    # 预计算所有特征的排序索引（按激活值降序），供去重使用
    # 取 min(d_sae, k + 512) 以兼顾效率与去重能力
    fetch_k = min(k + 512, sae.cfg.d_sae)
    _, topk_indices = torch.topk(feature_acts, k=fetch_k, dim=1)
    # topk_indices: (n_items, fetch_k)，按激活值降序排列

    # 初始 SID = top-K 索引（有序列表，顺序即激活强度排名）
    item_sids: list[list[int]] = topk_indices[:, :k].tolist()

    # 去重前碰撞统计
    from collections import Counter as _Counter

    _pre_counts = _Counter(str(sid) for sid in item_sids)
    pre_n_unique = len(_pre_counts)
    pre_colliding = sum(cnt for cnt in _pre_counts.values() if cnt > 1)
    pre_collision_rate = pre_colliding / n_items
    logger.info(
        f"Pre-dedup: {n_items} items, {pre_n_unique} unique SIDs, "
        f"colliding_items={pre_colliding}, collision_rate={pre_collision_rate:.4f}"
    )

    # ——— SID 去重 ———
    # 使用全局已占用 SID 集合，避免跨组分配冲突导致的振荡。
    # 每次为碰撞 item 分配新 SID 时，从全局集合检查并更新，
    # 确保单次迭代不会制造新碰撞。
    for iteration in range(max_dedup_iters):
        sid_to_items: dict[str, list[int]] = {}
        for item_id, sid in enumerate(item_sids):
            sid_to_items.setdefault(str(sid), []).append(item_id)

        collision_groups = [g for g in sid_to_items.values() if len(g) > 1]
        if not collision_groups:
            logger.info(f"Dedup complete after {iteration} iteration(s). No collisions remaining.")
            break

        n_conflicts = sum(len(g) for g in collision_groups)
        logger.info(
            f"Dedup iteration {iteration}: {len(collision_groups)} collision groups, "
            f"{n_conflicts} items involved"
        )

        # 全局已占用 SID 集合（所有 item 的当前 SID）
        global_used: set[tuple] = {tuple(s) for s in item_sids}

        for group in collision_groups:
            # 首位 item 不动，其余 item 逐一在全局集合中找空位
            for item_id in group[1:]:
                old_sid = tuple(item_sids[item_id])
                prefix = item_sids[item_id][:-1]  # 前 k-1 个 token 不变

                for col in range(k, fetch_k):
                    candidate = int(topk_indices[item_id, col].item())
                    new_sid = tuple(prefix + [candidate])
                    if new_sid not in global_used:
                        global_used.discard(old_sid)
                        item_sids[item_id][-1] = candidate
                        global_used.add(new_sid)
                        break
                else:
                    logger.warning(
                        f"Item {item_id}: exhausted {fetch_k - k} backup features, "
                        "collision unresolved."
                    )

    # 最终碰撞统计（collision_rate = 参与碰撞的物品比例）
    from collections import Counter
    sid_counts = Counter(str(sid) for sid in item_sids)
    n_unique = len(sid_counts)
    colliding_items = sum(cnt for cnt in sid_counts.values() if cnt > 1)
    collision_rate = colliding_items / n_items
    logger.info(
        f"Final: {n_items} items, {n_unique} unique SIDs, "
        f"colliding_items={colliding_items}, collision_rate={collision_rate:.4f}"
    )
    if collision_rate > 0.05:
        logger.warning(
            f"Collision rate {collision_rate:.4f} exceeds 5% threshold. "
            "Consider increasing expansion_factor or max_dedup_iters."
        )

    # 转换为 token 字符串格式
    # token_format: 'auto' → 使用 'f' 前缀（GatedSAE 默认）；单字符 → 使用该字符
    prefix = "f" if token_format == "auto" else token_format
    index_dict: dict[str, list[str]] = {}
    for item_id, sid in enumerate(item_sids):
        tokens = [f"[{prefix}_{feat_idx}]" for feat_idx in sid]
        index_dict[str(item_id)] = tokens

    # 保存 .index.json
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
    )
    with open(output_path, "w") as f:
        json.dump(index_dict, f)
    logger.info(f"Saved: {output_path}")

    # 保存伴随 meta 文件（含去重前统计）
    meta = {
        "pre_dedup_n_unique": pre_n_unique,
        "pre_dedup_colliding_items": pre_colliding,
        "pre_dedup_collision_rate": pre_collision_rate,
        "post_dedup_n_unique": n_unique,
        "post_dedup_colliding_items": colliding_items,
        "post_dedup_collision_rate": collision_rate,
    }
    meta_path = output_path + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return output_path
