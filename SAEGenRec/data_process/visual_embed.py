"""视觉特征提取模块：将物品图像转为视觉嵌入向量。"""

import json
import os
from typing import List, Optional

import numpy as np
import torch
from loguru import logger


def extract_visual(
    category: str = "",
    vision_model: str = "openai/clip-vit-base-patch32",
    item_json_path: str = "",
    image_dir: str = "",
    output_path: str = "",
    output_dir: str = "data/interim",
    batch_size: int = 64,
    device: str = "cuda",
) -> int:
    """从物品图像中提取视觉特征向量，行序与 item.json 的 item_id 对齐。

    Args:
        category: 商品类别名
        vision_model: HuggingFace 视觉模型名（如 openai/clip-vit-base-patch32）
        item_json_path: {category}.item.json 路径
        image_dir: 图像目录（含 {asin}.jpg 文件）
        output_path: 输出 .npy 路径（优先）
        output_dir: 输出目录（output_path 为空时使用）
        batch_size: GPU 批量大小
        device: 计算设备

    Returns:
        missing_count: 缺失图像的物品数量
    """
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as e:
        raise ImportError("transformers is required for visual embedding.") from e

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow is required for image loading. Install with: pip install Pillow") from e

    # 推断路径
    if not item_json_path:
        item_json_path = os.path.join("data", "interim", f"{category}.item.json")
    if not image_dir:
        image_dir = os.path.join("data", "interim", category, "images")
    if not output_path:
        model_slug = vision_model.replace("/", "-").replace("openai-", "").replace("facebook-", "")
        output_path = os.path.join(output_dir, f"{category}.emb-{model_slug}-visual.npy")

    # 加载 item.json（按 item_id 升序排列）
    with open(item_json_path) as f:
        item_json = json.load(f)

    max_item_id = max(int(k) for k in item_json.keys())
    n_items = max_item_id + 1

    # 加载模型
    dev = device if torch.cuda.is_available() and "cuda" in device else "cpu"
    logger.info(f"Loading vision model: {vision_model} on {dev}")

    processor = AutoProcessor.from_pretrained(vision_model)
    model = AutoModel.from_pretrained(vision_model).to(dev)
    model.eval()

    # 推断特征维度（试跑一个黑色图像）
    try:
        _dummy = Image.new("RGB", (224, 224))
        _inputs = processor(images=_dummy, return_tensors="pt").to(dev)
        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                _out = model.get_image_features(**_inputs)
            else:
                _out = model(**_inputs).last_hidden_state[:, 0]
        d_visual = _out.shape[-1]
    except Exception:
        d_visual = 512  # fallback

    embeddings = np.zeros((n_items, d_visual), dtype=np.float32)
    missing_count = 0

    # 按批处理
    item_ids = list(range(n_items))
    for batch_start in range(0, n_items, batch_size):
        batch_ids = item_ids[batch_start : batch_start + batch_size]
        images = []
        valid_ids = []

        for item_id in batch_ids:
            meta = item_json.get(str(item_id), {})
            asin = meta.get("asin", str(item_id))
            img_path = os.path.join(image_dir, f"{asin}.jpg")

            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    valid_ids.append(item_id)
                except Exception:
                    missing_count += 1
            else:
                missing_count += 1

        if images:
            try:
                inputs = processor(images=images, return_tensors="pt", padding=True).to(dev)
                with torch.no_grad():
                    if hasattr(model, "get_image_features"):
                        feats = model.get_image_features(**inputs)
                    else:
                        feats = model(**inputs).last_hidden_state[:, 0]
                feats = feats.cpu().float().numpy()
                for i, item_id in enumerate(valid_ids):
                    embeddings[item_id] = feats[i]
            except Exception as e:
                logger.warning(f"Batch failed at start={batch_start}: {e}")
                for item_id in valid_ids:
                    missing_count += 1

        if (batch_start // batch_size + 1) % 10 == 0:
            logger.info(f"Progress: {batch_start + len(batch_ids)}/{n_items}")

    # 清理 NaN/Inf
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    np.save(output_path, embeddings)
    logger.info(f"Saved: {output_path}, shape={embeddings.shape}, missing={missing_count}")

    return missing_count
