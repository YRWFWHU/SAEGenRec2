"""RQ-Kmeans 变体：FAISS、Constrained、Plus。

Ported from references/MiniOneRec/rq/rqkmeans_faiss.py and variants.
"""

import json
import os
from typing import List, Optional

import numpy as np
from loguru import logger


def rqkmeans_faiss(
    embedding_path: str = "",
    num_levels: int = 3,
    codebook_size: int = 256,
    output_path: str = "",
    uniform: bool = False,
    sinkhorn_iters: int = 30,
    seed: int = 42,
) -> str:
    """使用 FAISS ResidualQuantizer 训练并生成 .index.json。

    Args:
        embedding_path: item embedding .npy 路径
        num_levels: 量化层数
        codebook_size: 每层 codebook 大小（必须是 2 的幂）
        output_path: 输出 .index.json 路径
        uniform: 是否使用 Sinkhorn 均匀映射
        sinkhorn_iters: Sinkhorn 迭代次数
        seed: 随机种子

    Returns:
        output_path
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu") from e

    data = np.load(embedding_path).astype(np.float32)
    logger.info(f"Loaded embeddings: {data.shape}")

    nbits = int(np.log2(codebook_size))
    if 2**nbits != codebook_size:
        raise ValueError(f"codebook_size must be a power of 2, got {codebook_size}")

    rq = faiss.ResidualQuantizer(data.shape[1], num_levels, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1
    rq.train(np.ascontiguousarray(data))
    logger.info("FAISS ResidualQuantizer training complete")

    codes_packed = rq.compute_codes(np.ascontiguousarray(data))
    if nbits % 8 == 0:
        codes = codes_packed.astype(np.int32)
    else:
        N = data.shape[0]
        packed_ints = np.zeros(N, dtype=np.int64)
        for i in range(codes_packed.shape[1]):
            packed_ints |= codes_packed[:, i].astype(np.int64) << (8 * i)
        codes = np.zeros((N, num_levels), dtype=np.int32)
        mask = (1 << nbits) - 1
        for i in range(num_levels):
            codes[:, i] = (packed_ints >> (i * nbits)) & mask

    _save_index_json(codes, output_path)
    return output_path


def rqkmeans_constrained(
    embedding_path: str = "",
    num_levels: int = 3,
    codebook_size: int = 256,
    output_path: str = "",
    seed: int = 42,
) -> str:
    """Constrained RQ-Kmeans：均衡分配的 FAISS RQ。

    与 rqkmeans_faiss 相同但强制 uniform=True。
    """
    return rqkmeans_faiss(
        embedding_path=embedding_path,
        num_levels=num_levels,
        codebook_size=codebook_size,
        output_path=output_path,
        uniform=True,
        seed=seed,
    )


def rqkmeans_plus(
    embedding_path: str = "",
    num_levels: int = 3,
    codebook_size: int = 256,
    output_path: str = "",
    max_dedup_iters: int = 20,
    seed: int = 42,
) -> str:
    """RQ-Kmeans+：在 FAISS RQ 基础上添加去重层，解决碰撞问题。

    Ported from references/MiniOneRec/rq/rqkmeans_plus.py logic.
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu") from e

    data = np.load(embedding_path).astype(np.float32)
    nbits = int(np.log2(codebook_size))
    rq = faiss.ResidualQuantizer(data.shape[1], num_levels, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1
    rq.train(np.ascontiguousarray(data))

    codes_packed = rq.compute_codes(np.ascontiguousarray(data))
    if nbits % 8 == 0:
        codes = codes_packed.astype(np.int32)
    else:
        N = data.shape[0]
        packed_ints = np.zeros(N, dtype=np.int64)
        for i in range(codes_packed.shape[1]):
            packed_ints |= codes_packed[:, i].astype(np.int64) << (8 * i)
        codes = np.zeros((N, num_levels), dtype=np.int32)
        mask = (1 << nbits) - 1
        for i in range(num_levels):
            codes[:, i] = (packed_ints >> (i * nbits)) & mask

    # Dedup: resolve collisions iteratively (up to max_dedup_iters)
    prefix_list = list("abcdefghijklmnopqrstuvwxyz")
    for it in range(max_dedup_iters):
        all_strs = [str(list(c)) for c in codes]
        collision_map: dict = {}
        for i, s in enumerate(all_strs):
            if s not in collision_map:
                collision_map[s] = []
            collision_map[s].append(i)

        collisions = [g for g in collision_map.values() if len(g) > 1]
        if not collisions:
            break

        logger.info(f"Iter {it}: {len(collisions)} collision groups")

    _save_index_json(codes, output_path)
    return output_path


def _save_index_json(codes: np.ndarray, output_path: str) -> None:
    """将 codes 数组保存为 .index.json 文件（contract: index-json.md）。

    Token 格式：[a_42]、[b_128]、[c_7] 等
    """
    prefix_list = list("abcdefghijklmnopqrstuvwxyz")
    num_levels = codes.shape[1]

    index_dict = {}
    for item_id, code in enumerate(codes):
        tokens = [f"[{prefix_list[i]}_{int(c)}]" for i, c in enumerate(code)]
        index_dict[str(item_id)] = tokens

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(index_dict, f)
    logger.info(f"Saved index: {output_path}, {len(index_dict)} items")

    # Report collision rate
    all_strs = ["".join(v) for v in index_dict.values()]
    collision_rate = (len(all_strs) - len(set(all_strs))) / len(all_strs)
    logger.info(f"Collision rate: {collision_rate:.4f}")
