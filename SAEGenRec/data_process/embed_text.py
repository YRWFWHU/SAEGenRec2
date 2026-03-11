"""文本嵌入模块：将 item title+description 编码为 .npy 文件。

命名约定: {category}.emb-{model_slug}-text.npy
"""

import json
import os
import re
from typing import Optional

import numpy as np
import torch
from loguru import logger


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    return text.strip()


def embed_text(
    category: str = "",
    item_json_path: str = "",
    review_json_path: str = "",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "data/interim",
    batch_size: int = 256,
    device: str = "cuda",
) -> str:
    """将 item title+description 编码为 .npy 嵌入文件。

    Args:
        category: 商品类别名（用于推断文件路径和输出文件名）
        item_json_path: {category}.item.json 路径（为空时自动推断）
        review_json_path: {category}.review.json 路径（可选）
        model_name: sentence-transformers 模型名
        output_dir: 输出目录
        batch_size: 编码批量大小
        device: 计算设备

    Returns:
        output_path: 输出 .npy 文件路径
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from e

    if not item_json_path:
        item_json_path = os.path.join("data", "interim", f"{category}.item.json")

    model_slug = model_name.split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)
    text_path = os.path.join(output_dir, f"{category}.emb-{model_slug}-text.npy")

    logger.info(f"Loading model: {model_name}")
    dev = device if torch.cuda.is_available() and "cuda" in device else "cpu"
    model = SentenceTransformer(model_name, device=dev)

    with open(item_json_path) as f:
        item_json = json.load(f)

    max_item_id = max(int(k) for k in item_json.keys())
    item_texts = []
    for i in range(max_item_id + 1):
        meta = item_json.get(str(i), {})
        title = _clean_text(meta.get("title", ""))
        desc = _clean_text(meta.get("description", ""))
        text = f"{title} {desc}".strip() if desc else title
        if not text:
            text = "unknown item"
        item_texts.append(text)

    logger.info(f"Encoding {len(item_texts)} item texts ...")
    embs = model.encode(
        item_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    np.save(text_path, embs)
    logger.info(f"Saved: {text_path}, shape={embs.shape}")

    # 可选：review 嵌入
    if review_json_path and os.path.exists(review_json_path):
        review_path = os.path.join(output_dir, f"{category}.emb-{model_slug}-review.npy")
        with open(review_json_path) as f:
            review_json = json.load(f)

        review_texts = []
        for record in review_json:
            text = _clean_text(record.get("review_text", ""))
            if not text:
                text = _clean_text(record.get("summary", "")) or "no review"
            review_texts.append(text)

        logger.info(f"Encoding {len(review_texts)} review texts ...")
        review_embs = model.encode(
            review_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        np.save(review_path, review_embs)
        logger.info(f"Saved: {review_path}, shape={review_embs.shape}")

    return text_path
