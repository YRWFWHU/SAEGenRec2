"""文本嵌入模块：将 item title+description 和 review text 编码为 .npy 文件。

Ported from references/MiniOneRec/rq/text2emb/amazon_text2emb.py
使用 sentence-transformers 编码，输出按 item_id 排序。
"""

import json
import os
from typing import Optional

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


def _clean_text(text: str) -> str:
    if not text:
        return ""
    import re

    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    return text.strip()


def text2emb(
    item_json_path: str = "",
    review_json_path: str = "",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "data/interim",
    batch_size: int = 256,
    device: str = "cuda",
    dataset: Optional[str] = None,
) -> None:
    """将 item titles+descriptions 和 review texts 编码为 .npy 文件。

    输出:
        {dataset}.emb-{model_short}-td.npy   — item title+description embeddings (N_items × dim)
        {dataset}.emb-{model_short}-review.npy — review text embeddings (N_reviews × dim)

    行序：
        td.npy 的行 i 对应 item_id=i 的 item
        review.npy 的行 i 对应 .review.json 的第 i 条记录
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from e

    # 推断 dataset 名称（用于输出文件名）
    if dataset is None:
        basename = os.path.basename(item_json_path)
        dataset = basename.replace(".item.json", "")

    model_short = model_name.split("/")[-1]

    os.makedirs(output_dir, exist_ok=True)
    td_path = os.path.join(output_dir, f"{dataset}.emb-{model_short}-td.npy")
    review_path = os.path.join(output_dir, f"{dataset}.emb-{model_short}-review.npy")

    logger.info(f"Loading model: {model_name}")
    dev = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    model = SentenceTransformer(model_name, device=dev)

    # ---- item title+description embeddings ----
    with open(item_json_path) as f:
        item_json = json.load(f)

    # 按 item_id 升序排列
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

    logger.info(f"Encoding {len(item_texts)} item title+description texts ...")
    td_embs = model.encode(
        item_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    np.save(td_path, td_embs)
    logger.info(f"Saved: {td_path}, shape={td_embs.shape}")

    # ---- review text embeddings ----
    if review_json_path and os.path.exists(review_json_path):
        with open(review_json_path) as f:
            review_json = json.load(f)

        review_texts = []
        for record in review_json:
            text = _clean_text(record.get("review_text", ""))
            if not text:
                summary = _clean_text(record.get("summary", ""))
                text = summary if summary else "no review"
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
    else:
        logger.warning("review_json_path not provided or not found, skipping review embeddings.")
