"""数据集转换：将 .inter + .item.json + .index.json 转换为 CSV 训练格式。

Ported from references/MiniOneRec/convert_dataset.py
输出格式遵循 contracts/training-csv.md 和 contracts/info-txt.md。
"""

import csv
import json
import os
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


def _load_inter_file(path: str) -> List[Dict]:
    """读取 .inter TSV 文件，返回字典列表。"""
    records = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append(row)
    return records


def _tokens_to_sid(tokens: List[str]) -> str:
    """将 SID token 列表拼接为字符串，如 ['[a_1]', '[b_2]', '[c_3]'] → '[a_1][b_2][c_3]'。"""
    return "".join(tokens)


def _convert_split(
    inter_records: List[Dict],
    item_json: Dict[str, Dict],
    index_json: Dict[str, List[str]],
) -> List[Dict]:
    """将 .inter 记录转换为 CSV 行格式。

    每行 .inter 代表一个交互样本（target item）。
    由于 .inter 只包含 target 信息，history 需要从同用户的多行记录重建。
    """
    # 按用户分组，并按 timestamp 排序，重建历史序列
    user_records: Dict[str, List[Dict]] = {}
    for rec in inter_records:
        uid = rec["user_id"]
        if uid not in user_records:
            user_records[uid] = []
        user_records[uid].append(rec)

    # 对每个用户按时间排序
    for uid in user_records:
        user_records[uid].sort(key=lambda x: int(x["timestamp"]))

    rows = []
    for uid, recs in user_records.items():
        # 对于每个 target（除了第一个，没有 history），生成一行
        for i in range(1, len(recs)):
            history = recs[:i]
            target = recs[i]

            history_ids = [int(r["item_id"]) for r in history]
            history_asins = [r["item_asin"] for r in history]
            target_id = int(target["item_id"])
            target_asin = target["item_asin"]

            # SID tokens
            history_sids = []
            for hid in history_ids:
                if str(hid) in index_json:
                    history_sids.append(_tokens_to_sid(index_json[str(hid)]))

            if str(target_id) not in index_json:
                continue  # skip if no SID for target
            target_sid = _tokens_to_sid(index_json[str(target_id)])

            # Titles
            history_titles = [
                item_json.get(str(hid), {}).get("title", f"Item_{hid}")
                for hid in history_ids
            ]
            target_title = item_json.get(str(target_id), {}).get("title", f"Item_{target_id}")

            rows.append(
                {
                    "user_id": uid,
                    "history_item_sid": " ".join(history_sids),
                    "target_item_sid": target_sid,
                    "history_item_title": ",".join(history_titles),
                    "target_item_title": target_title,
                    "history_item_id": ",".join(str(x) for x in history_ids),
                    "target_item_id": target_id,
                }
            )

    return rows


def _write_info_txt(
    item_json: Dict[str, Dict],
    index_json: Dict[str, List[str]],
    output_path: str,
) -> None:
    """写入 info TXT 文件（contract: info-txt.md）。

    格式：semantic_id \\t item_title \\t item_id（无表头）
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item_id, meta in item_json.items():
            if item_id in index_json:
                sid = _tokens_to_sid(index_json[item_id])
                title = meta.get("title", f"Item_{item_id}")
                f.write(f"{sid}\t{title}\t{item_id}\n")
    logger.info(f"Saved info TXT: {output_path}")


def convert_dataset(
    inter_dir: str = "",
    item_json: str = "",
    index_json: str = "",
    dataset: str = "",
    output_dir: str = "data/processed",
    splits: str = "train,valid,test",
) -> None:
    """将 .inter + .item.json + .index.json 转换为 CSV 训练格式。

    Args:
        inter_dir: 包含 {dataset}.{split}.inter 文件的目录
        item_json: {dataset}.item.json 文件路径
        index_json: {dataset}.index.json 文件路径
        dataset: 数据集名称（用于输出文件命名）
        output_dir: 输出目录
        splits: 逗号分隔的划分名称（默认 "train,valid,test"）
    """
    logger.info(f"Loading item metadata: {item_json}")
    with open(item_json) as f:
        items = json.load(f)

    logger.info(f"Loading index: {index_json}")
    with open(index_json) as f:
        index = json.load(f)

    split_list = [s.strip() for s in splits.split(",")]
    os.makedirs(output_dir, exist_ok=True)
    info_dir = os.path.join(output_dir, "info")
    os.makedirs(info_dir, exist_ok=True)

    # Write info TXT
    info_path = os.path.join(info_dir, f"{dataset}.txt")
    _write_info_txt(items, index, info_path)

    for split in split_list:
        inter_path = os.path.join(inter_dir, f"{dataset}.{split}.inter")
        if not os.path.exists(inter_path):
            logger.warning(f"Not found: {inter_path}, skipping")
            continue

        records = _load_inter_file(inter_path)
        logger.info(f"Loaded {len(records)} records from {inter_path}")

        rows = _convert_split(records, items, index)
        logger.info(f"Converted to {len(rows)} CSV rows for {split}")

        csv_path = os.path.join(output_dir, f"{dataset}.{split}.csv")
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved: {csv_path}")
        else:
            logger.warning(f"No rows for {split}, skipping CSV creation")
