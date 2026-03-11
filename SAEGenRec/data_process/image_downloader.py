"""并发图像下载模块：从 Amazon 元数据中下载最高分辨率 MAIN 图像。"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from loguru import logger


def _extract_main_image_url(meta: dict) -> Optional[str]:
    """从物品元数据中提取最高分辨率 MAIN 图像 URL。"""
    urls = meta.get("imageURLHighRes") or meta.get("imURLl") or []
    if isinstance(urls, str):
        urls = [urls]
    for url in urls:
        if url and isinstance(url, str) and url.startswith("http"):
            return url
    return None


def _should_skip(output_path: str) -> bool:
    """判断文件是否已存在（断点续传）。"""
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def _download_with_retry(url: str, output_path: str, retries: int = 3) -> bool:
    """下载单个 URL，失败时重试。返回是否成功。"""
    import requests

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        except Exception as e:
            if attempt == retries - 1:
                logger.warning(f"Failed to download {url} after {retries} attempts: {e}")
    return False


def _download_one(asin: str, url: Optional[str], image_dir: str) -> Dict:
    """下载单个物品图像，返回状态字典。"""
    if not url:
        return {"asin": asin, "status": "skipped_no_url"}

    output_path = os.path.join(image_dir, f"{asin}.jpg")
    if _should_skip(output_path):
        return {"asin": asin, "status": "skipped_exists"}

    success = _download_with_retry(url, output_path)
    return {"asin": asin, "status": "success" if success else "failed"}


def download_images(
    category: str = "",
    data_dir: str = "data/raw",
    image_dir: str = "",
    concurrency: int = 8,
) -> Dict:
    """从 Amazon 元数据 JSON 中并发下载物品图像。

    Args:
        category: 商品类别名（用于查找 meta_{category}.json）
        data_dir: 原始数据目录（包含 meta_{category}.json）
        image_dir: 图像输出目录（默认为 data/interim/{category}/images）
        concurrency: 最大并发下载数

    Returns:
        stats: {'success': N, 'failed': N, 'skipped_exists': N, 'skipped_no_url': N}
    """
    if not image_dir:
        image_dir = os.path.join("data", "interim", category, "images")

    os.makedirs(image_dir, exist_ok=True)

    # 查找 meta 文件（支持多种命名）
    meta_path = None
    for name in [f"meta_{category}.json", f"meta_{category}.jsonl"]:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            meta_path = candidate
            break

    if meta_path is None:
        raise FileNotFoundError(
            f"Meta file not found in {data_dir}. Expected: meta_{category}.json"
        )

    logger.info(f"Loading meta from: {meta_path}")

    # 加载元数据（支持 JSON list 和 JSONL 两种格式）
    items = []
    with open(meta_path) as f:
        try:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    logger.info(f"Found {len(items)} items in meta file")

    # 构建 (asin, url) 对列表
    tasks = []
    for meta in items:
        asin = meta.get("asin", "")
        if not asin:
            continue
        url = _extract_main_image_url(meta)
        tasks.append((asin, url))

    # 并发下载
    stats = {"success": 0, "failed": 0, "skipped_exists": 0, "skipped_no_url": 0}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_download_one, asin, url, image_dir): asin
            for asin, url in tasks
        }
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            status = result["status"]
            stats[status] = stats.get(status, 0) + 1
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(tasks)} — {stats}")

    logger.info(
        f"Download complete: success={stats['success']}, "
        f"failed={stats['failed']}, "
        f"skipped_exists={stats['skipped_exists']}, "
        f"skipped_no_url={stats['skipped_no_url']}"
    )
    return stats
