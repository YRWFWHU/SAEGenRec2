"""从训练好的 RQ-VAE checkpoint 生成 .index.json 文件。

Ported from references/MiniOneRec/rq/models/generate_indices.py
"""

import json
import os

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from SAEGenRec.sid_builder.models.rqvae import RQVAE
from SAEGenRec.sid_builder.rqvae import EmbDataset


def _load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """从 checkpoint 加载 RQVAE 模型，返回 (model, config)。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]

    model = RQVAE(
        in_dim=config["in_dim"],
        num_emb_list=config["num_emb_list"],
        e_dim=config["e_dim"],
        layers=config["layers"],
        dropout_prob=config.get("dropout_prob", 0.0),
        beta=config.get("beta", 0.25),
        kmeans_init=config.get("kmeans_init", False),
        kmeans_iters=config.get("kmeans_iters", 100),
        sk_epsilons=config.get("sk_epsilons", [0.0] * len(config["num_emb_list"])),
        sk_iters=config.get("sk_iters", 100),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def generate_indices(
    checkpoint: str = "",
    embedding_path: str = "",
    output_path: str = "",
    batch_size: int = 64,
    device: str = "cuda:0",
    max_dedup_iters: int = 20,
    num_workers: int = 4,
    token_format: str = "auto",
) -> str:
    """从 RQ-VAE checkpoint 生成所有 item 的 SID token 索引。

    SID 碰撞处理：对碰撞的 items 使用 Sinkhorn（use_sk=True）重新量化，
    最多迭代 max_dedup_iters 次（与参考实现一致）。

    Args:
        checkpoint: RQ-VAE checkpoint .pth 路径
        embedding_path: item embedding .npy 路径（行 i = item_id i）
        output_path: 输出 .index.json 路径
        batch_size: 编码 batch 大小
        device: 计算设备
        max_dedup_iters: 最大碰撞去重迭代次数
        num_workers: DataLoader 工作线程数

    Returns:
        output_path
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, config = _load_model_from_checkpoint(checkpoint, dev)

    data = EmbDataset(embedding_path)
    data_loader = DataLoader(
        data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    prefix_list = list("abcdefghijklmnopqrstuvwxyz")
    num_levels = len(config["num_emb_list"])

    # Initial encoding without Sinkhorn
    all_indices = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(dev)
            indices = model.get_indices(batch, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            all_indices.extend(indices.tolist())

    all_indices = np.array(all_indices, dtype=np.int32)
    all_strs = np.array([str(list(idx)) for idx in all_indices])

    # Enable last-level Sinkhorn for dedup (参考实现逻辑)
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    # Iterative collision resolution
    for tt in range(max_dedup_iters):
        # Check for collisions
        collision_map: dict = {}
        for i, s in enumerate(all_strs):
            if s not in collision_map:
                collision_map[s] = []
            collision_map[s].append(i)

        collision_groups = [g for g in collision_map.values() if len(g) > 1]
        if not collision_groups:
            logger.info("No collisions remaining.")
            break

        logger.info(f"Iteration {tt}: {len(collision_groups)} collision groups")

        with torch.no_grad():
            for group in collision_groups:
                group_data = data[group].to(dev)
                indices = model.get_indices(group_data, use_sk=True)
                indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                for item_i, idx in zip(group, indices):
                    all_indices[item_i] = idx
                    all_strs[item_i] = str(list(idx))

    logger.info(
        f"Total items: {len(all_indices)}, "
        f"unique: {len(set(all_strs.tolist()))}, "
        f"collision_rate: {(len(all_strs) - len(set(all_strs.tolist()))) / len(all_strs):.4f}"
    )

    # Convert to contract format: {str(item_id): [token, ...]}
    # token_format: 'auto' → 位置前缀 a/b/c/...；单字符 → [char_N] 统一前缀
    index_dict = {}
    for item_id, idx in enumerate(all_indices):
        if token_format == "auto":
            tokens = [f"[{prefix_list[i]}_{int(c)}]" for i, c in enumerate(idx)]
        else:
            tokens = [f"[{token_format}_{int(c)}]" for c in idx]
        index_dict[str(item_id)] = tokens

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(index_dict, f)
    logger.info(f"Saved: {output_path}")

    return output_path
