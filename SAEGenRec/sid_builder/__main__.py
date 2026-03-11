"""CLI 入口：python -m SAEGenRec.sid_builder <command> [args]"""

import os

try:
    import fire
except ImportError as e:
    raise ImportError("fire is required for CLI usage. Install with: pip install fire") from e

from SAEGenRec.sid_builder.gated_sae import gated_sae_train
from SAEGenRec.sid_builder.generate_indices import generate_indices
from SAEGenRec.sid_builder.generate_sae_indices import generate_sae_indices
from SAEGenRec.sid_builder.rqkmeans import rqkmeans_constrained, rqkmeans_faiss, rqkmeans_plus
from SAEGenRec.sid_builder.rqvae import rqvae_train
from SAEGenRec.sid_builder.text2emb import text2emb

# 触发注册表加载（导入含 @register_sid_method 的模块）
import SAEGenRec.sid_builder.rqvae  # noqa: F401
import SAEGenRec.sid_builder.rqkmeans  # noqa: F401
import SAEGenRec.sid_builder.gated_sae  # noqa: F401


def build_sid(
    category: str = "",
    method: str = "rqvae",
    emb_path: str = "",
    output_dir: str = "",
    k: int = None,
    token_format: str = "auto",
    sid_model_dir: str = "",
    **kwargs,
) -> None:
    """统一 SID 生成入口：训练 + 生成 .index.json。

    Args:
        category: 商品类别名（用于推断路径）
        method: SID 生成方法（rqvae / rqkmeans / gated_sae）
        emb_path: 输入嵌入文件路径（默认推断）
        output_dir: SID 模型输出目录
        k: SID token 数（None 时使用方法默认值）
        token_format: token 前缀格式（'auto' / 单字符如 'v'）
        sid_model_dir: 指定已有模型目录（跳过训练，直接生成）
    """
    from loguru import logger
    from SAEGenRec.sid_builder.registry import get_sid_method

    sid_method = get_sid_method(method)
    if k is None:
        k = sid_method.default_k

    # 推断嵌入路径
    if not emb_path:
        interim_dir = os.path.join("data", "interim")
        # 按优先级查找 text 嵌入
        candidates = [
            os.path.join(interim_dir, f"{category}.emb-all-MiniLM-L6-v2-text.npy"),
            os.path.join(interim_dir, f"{category}.emb-all-MiniLM-L6-v2-td.npy"),
        ]
        for c in candidates:
            if os.path.exists(c):
                emb_path = c
                break
        if not emb_path:
            raise FileNotFoundError(
                f"Could not find embedding file for category '{category}'. "
                "Please specify --emb_path explicitly."
            )

    # 验证嵌入维度
    import numpy as np
    emb = np.load(emb_path, mmap_mode="r")
    logger.info(f"Embedding shape: {emb.shape}")

    # 推断输出路径
    if not output_dir:
        output_dir = os.path.join("models", f"sid_{method}", category)
    index_path = os.path.join("data", "interim", f"{category}.{method}.index.json")

    # 检查已有 checkpoint
    if sid_model_dir and os.path.exists(sid_model_dir):
        logger.info(f"Using existing checkpoint: {sid_model_dir}")
        checkpoint = sid_model_dir
    else:
        if sid_model_dir:
            logger.info(f"Checkpoint dir '{sid_model_dir}' not found. Will train from scratch.")
        checkpoint = sid_method.train(
            embedding_path=emb_path, output_dir=output_dir, **kwargs
        )

    sid_method.generate(
        checkpoint=checkpoint,
        embedding_path=emb_path,
        output_path=index_path,
        k=k,
        token_format=token_format,
    )
    logger.info(f"SID generation complete: {index_path}")


def train_sid(
    category: str = "",
    method: str = "rqvae",
    emb_path: str = "",
    sid_model_dir: str = "",
    **kwargs,
) -> None:
    """仅训练 SID 模型，不生成 .index.json。"""
    from loguru import logger
    from SAEGenRec.sid_builder.registry import get_sid_method

    sid_method = get_sid_method(method)
    if not emb_path:
        emb_path = os.path.join("data", "interim", f"{category}.emb-all-MiniLM-L6-v2-text.npy")
    if not sid_model_dir:
        sid_model_dir = os.path.join("models", f"sid_{method}", category)

    checkpoint = sid_method.train(
        embedding_path=emb_path, output_dir=sid_model_dir, **kwargs
    )
    logger.info(f"Training complete: {checkpoint}")


def generate_sid(
    category: str = "",
    method: str = "rqvae",
    emb_path: str = "",
    sid_model_dir: str = "",
    k: int = None,
    token_format: str = "auto",
    **kwargs,
) -> None:
    """仅从已训练模型生成 .index.json。"""
    from loguru import logger
    from SAEGenRec.sid_builder.registry import get_sid_method

    sid_method = get_sid_method(method)
    if k is None:
        k = sid_method.default_k
    if not emb_path:
        emb_path = os.path.join("data", "interim", f"{category}.emb-all-MiniLM-L6-v2-text.npy")
    if not sid_model_dir:
        sid_model_dir = os.path.join("models", f"sid_{method}", category)

    index_path = os.path.join("data", "interim", f"{category}.{method}.index.json")
    sid_method.generate(
        checkpoint=sid_model_dir,
        embedding_path=emb_path,
        output_path=index_path,
        k=k,
        token_format=token_format,
    )
    logger.info(f"Generation complete: {index_path}")


def list_sid_methods_cmd() -> None:
    """列出所有已注册的 SID 方法。"""
    from SAEGenRec.sid_builder.registry import SID_METHODS

    print("Available SID Methods:")
    for name, cls in SID_METHODS.items():
        instance = cls()
        print(f"  {name:<12} k={instance.default_k}, token_format={instance.token_format}")


if __name__ == "__main__":
    fire.Fire(
        {
            # 新增统一命令
            "build_sid": build_sid,
            "train_sid": train_sid,
            "generate_sid": generate_sid,
            "list_sid_methods": list_sid_methods_cmd,
            # 向后兼容：保留所有旧命令
            "text2emb": text2emb,
            "rqvae_train": rqvae_train,
            "rqkmeans_faiss": rqkmeans_faiss,
            "rqkmeans_constrained": rqkmeans_constrained,
            "rqkmeans_plus": rqkmeans_plus,
            "generate_indices": generate_indices,
            "gated_sae_train": gated_sae_train,
            "generate_sae_indices": generate_sae_indices,
        }
    )
