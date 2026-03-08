"""数据预处理模块：从 Amazon 2015 原始 JSON 生成 .inter、.item.json、.review.json 文件。

Ported from references/MiniOneRec/data/process.py with modular refactoring:
- 输出格式改为 .inter (TSV) + .item.json + .review.json 三个中间文件
- 支持 TO (Temporal Order) 和 LOO (Leave-One-Out) 两种划分方式
- 历史窗口长度由 max_history_len 控制（参考实现固定为 10）
"""

import ast
import datetime
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from SAEGenRec.config import DataProcessConfig


def get_timestamp_start(year: int, month: int) -> int:
    return int(
        datetime.datetime(
            year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
    )


def _clean_title(title: str) -> str:
    return title.replace("&quot;", '"').replace("&amp;", "&").strip(" ").strip('"')


def _load_raw_data(
    raw_data_dir: str, category: str
) -> Tuple[List[Dict], List[Dict]]:
    """加载原始 Amazon 2015 元数据和评论数据。"""
    meta_path = os.path.join(raw_data_dir, f"meta_{category}.json")
    with open(meta_path) as f:
        metadata = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                metadata.append(json.loads(line))
            except json.JSONDecodeError:
                # Amazon 2015 meta files use Python literal syntax
                metadata.append(ast.literal_eval(line))

    review_path = os.path.join(raw_data_dir, f"{category}_5.json")
    if not os.path.exists(review_path):
        review_path = os.path.join(raw_data_dir, f"{category}.json")
    with open(review_path) as f:
        reviews = [json.loads(line) for line in f]

    return metadata, reviews


def _kcore_filter(
    metadata: List[Dict],
    reviews: List[Dict],
    k: int,
    start_timestamp: int,
    end_timestamp: int,
) -> Tuple[List[Dict], Dict[str, str], Dict[str, int]]:
    """执行 k-core 过滤，返回过滤后的评论、id_title 映射和 item2id 映射。

    移植自 references/MiniOneRec/data/process.py gao()，支持递归扩大时间窗口。
    """
    # 构建 id_title 映射（同时过滤无效标题）
    remove_items: set = set()
    id_title: Dict[str, str] = {}
    for meta in metadata:
        if ("title" not in meta) or (meta["title"].find("<span id") > -1):
            remove_items.add(meta["asin"])
            continue
        title = _clean_title(meta["title"])
        if len(title) > 1 and len(title.split(" ")) <= 20:
            id_title[meta["asin"]] = title
        else:
            remove_items.add(meta["asin"])

    for review in reviews:
        if review["asin"] not in id_title:
            remove_items.add(review["asin"])

    remove_users: set = set()

    while True:
        users: Dict[str, int] = {}
        items: Dict[str, int] = {}
        new_reviews: List[Dict] = []
        flag = False

        for review in tqdm(reviews, desc="k-core filtering"):
            ts = int(review["unixReviewTime"])
            if ts < start_timestamp or ts > end_timestamp:
                continue
            if review["reviewerID"] in remove_users or review["asin"] in remove_items:
                continue
            users[review["reviewerID"]] = users.get(review["reviewerID"], 0) + 1
            items[review["asin"]] = items.get(review["asin"], 0) + 1
            new_reviews.append(review)

        for user, cnt in users.items():
            if cnt < k:
                remove_users.add(user)
                flag = True
        for item, cnt in items.items():
            if cnt < k:
                remove_items.add(item)
                flag = True

        total = sum(users.values())
        density = total / (len(users) * len(items)) if users and items else 0
        logger.info(
            f"users: {len(users)}, items: {len(items)}, reviews: {total}, "
            f"density: {density:.6f}"
        )

        if not flag:
            break
        reviews = new_reviews

    # 过滤后的 items 列表
    surviving_items = list(items.keys())
    return new_reviews, id_title, surviving_items, remove_users, remove_items


def _assign_item_ids(items: List[str], seed: int = 42) -> Dict[str, int]:
    """随机打乱后分配 item_id（与参考实现保持一致）。"""
    items = list(items)
    random.seed(seed)
    random.shuffle(items)
    return {asin: idx for idx, asin in enumerate(items)}


def _build_interactions(
    reviews: List[Dict],
    item2id: Dict[str, int],
    id_title: Dict[str, str],
    max_history_len: int,
) -> List[List]:
    """构建交互样本列表（滑动窗口）。

    每个样本: [user_id, history_asins, target_asin, history_ids, target_id,
               history_ratings, target_rating, history_timestamps, target_timestamp]
    """
    # 按用户聚合，并按时间排序
    interact: Dict[str, Dict[str, List]] = {}
    for review in tqdm(reviews, desc="building interactions"):
        user = review["reviewerID"]
        if user not in interact:
            interact[user] = {
                "items": [],
                "ratings": [],
                "timestamps": [],
            }
        interact[user]["items"].append(review["asin"])
        interact[user]["ratings"].append(review["overall"])
        interact[user]["timestamps"].append(int(review["unixReviewTime"]))

    interaction_list = []
    for user, data in tqdm(interact.items(), desc="sliding window"):
        combined = sorted(
            zip(data["items"], data["ratings"], data["timestamps"]),
            key=lambda x: x[2],
        )
        items, ratings, timestamps = zip(*combined)
        items, ratings, timestamps = list(items), list(ratings), list(timestamps)
        item_ids = [item2id[asin] for asin in items]

        for i in range(1, len(items)):
            st = max(i - max_history_len, 0)
            interaction_list.append(
                [
                    user,
                    items[st:i],       # history_asins
                    items[i],          # target_asin
                    item_ids[st:i],    # history_item_ids
                    item_ids[i],       # target_item_id
                    ratings[st:i],     # history_ratings
                    ratings[i],        # target_rating
                    timestamps[st:i],  # history_timestamps
                    timestamps[i],     # target_timestamp
                ]
            )

    return interaction_list


def _write_inter_file(
    path: str,
    samples: List[List],
) -> None:
    """写入 .inter TSV 文件（contract: inter-format.md）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("user_id\titem_id\titem_asin\ttimestamp\trating\n")
        for s in samples:
            user_id = s[0]
            target_asin = s[2]
            target_item_id = s[4]
            target_rating = s[6]
            target_timestamp = s[8]
            f.write(
                f"{user_id}\t{target_item_id}\t{target_asin}\t"
                f"{target_timestamp}\t{target_rating}\n"
            )


def _split_to(
    interaction_list: List[List],
) -> Tuple[List[List], List[List], List[List]]:
    """TO (Temporal Order) 划分：按 target_timestamp 全局排序，8:1:1 分割。"""
    sorted_list = sorted(interaction_list, key=lambda x: int(x[8]))
    n = len(sorted_list)
    train = sorted_list[: int(n * 0.8)]
    valid = sorted_list[int(n * 0.8) : int(n * 0.9)]
    test = sorted_list[int(n * 0.9) :]
    return train, valid, test


def _split_loo(
    interaction_list: List[List],
) -> Tuple[List[List], List[List], List[List]]:
    """LOO (Leave-One-Out) 划分：每个用户最后一条→test，倒数第二条→valid，其余→train。"""
    # 按用户分组
    user_samples: Dict[str, List] = {}
    for s in interaction_list:
        user = s[0]
        if user not in user_samples:
            user_samples[user] = []
        user_samples[user].append(s)

    train, valid, test = [], [], []
    for user, samples in user_samples.items():
        # 按 target_timestamp 排序
        samples = sorted(samples, key=lambda x: int(x[8]))
        if len(samples) >= 2:
            test.append(samples[-1])
            valid.append(samples[-2])
            train.extend(samples[:-2])
        elif len(samples) == 1:
            train.extend(samples)

    return train, valid, test


def _write_item_json(
    path: str,
    item2id: Dict[str, int],
    id_title: Dict[str, str],
    metadata: List[Dict],
) -> None:
    """写入 .item.json 文件（contract: item-json.md）。"""
    # 构建 asin → description 映射
    asin_to_desc: Dict[str, str] = {}
    for meta in metadata:
        desc = ""
        if "description" in meta:
            d = meta["description"]
            if isinstance(d, list):
                desc = " ".join(str(x) for x in d if x)
            elif isinstance(d, str):
                desc = d
        asin_to_desc[meta["asin"]] = desc

    item_json: Dict[str, Dict] = {}
    for asin, item_id in item2id.items():
        item_json[str(item_id)] = {
            "item_asin": asin,
            "title": id_title.get(asin, ""),
            "description": asin_to_desc.get(asin, ""),
        }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(item_json, f, ensure_ascii=False)


def _write_review_json(
    path: str,
    reviews: List[Dict],
    item2id: Dict[str, int],
) -> None:
    """写入 .review.json 文件（contract: review-json.md）。

    行序与 .emb-{model}-review.npy 的行序一一对应。
    """
    records = []
    for review in reviews:
        asin = review["asin"]
        if asin not in item2id:
            continue
        records.append(
            {
                "user_id": review["reviewerID"],
                "item_id": item2id[asin],
                "item_asin": asin,
                "timestamp": int(review["unixReviewTime"]),
                "rating": float(review["overall"]),
                "review_text": review.get("reviewText", "") or "",
                "summary": review.get("summary", "") or "",
            }
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, ensure_ascii=False)


def preprocess(
    category: str = "",
    k_core: int = 5,
    st_year: int = 2017,
    st_month: int = 10,
    ed_year: int = 2018,
    ed_month: int = 11,
    split_method: str = "TO",
    max_history_len: int = 50,
    raw_data_dir: str = "data/raw",
    output_dir: str = "data/interim",
    seed: int = 42,
    metadata: Optional[List[Dict]] = None,
    reviews: Optional[List[Dict]] = None,
) -> None:
    """从原始 Amazon 2015 数据生成 .inter、.item.json、.review.json 文件。

    Args:
        category: Amazon 数据集类别名（如 All_Beauty）
        k_core: k-core 过滤阈值
        st_year/st_month: 时间窗口起始年月
        ed_year/ed_month: 时间窗口结束年月
        split_method: "TO" (Temporal Order) 或 "LOO" (Leave-One-Out)
        max_history_len: 历史序列最大长度
        raw_data_dir: 原始数据目录（含 meta_{category}.json 和 {category}_5.json）
        output_dir: 输出目录
        seed: 随机种子（用于 item_id 分配）
        metadata/reviews: 可选，直接传入数据（用于测试）
    """
    if st_year < 1996:
        return

    start_timestamp = get_timestamp_start(st_year, st_month)
    end_timestamp = get_timestamp_start(ed_year, ed_month)
    logger.info(f"Time range: {start_timestamp} to {end_timestamp}")

    if metadata is None or reviews is None:
        metadata, reviews = _load_raw_data(raw_data_dir, category)

    logger.info(f"Loaded: {len(metadata)} items, {len(reviews)} reviews")

    new_reviews, id_title, surviving_items, remove_users, remove_items = _kcore_filter(
        metadata, reviews, k_core, start_timestamp, end_timestamp
    )

    # 若过滤后 items 不足 3000，尝试向前扩展一年（与参考实现一致）
    if st_year > 1996 and len(surviving_items) < 3000:
        logger.info(
            f"Only {len(surviving_items)} items after filtering, "
            f"retrying with st_year={st_year - 1}"
        )
        preprocess(
            category=category,
            k_core=k_core,
            st_year=st_year - 1,
            st_month=st_month,
            ed_year=ed_year,
            ed_month=ed_month,
            split_method=split_method,
            max_history_len=max_history_len,
            raw_data_dir=raw_data_dir,
            output_dir=output_dir,
            seed=seed,
            metadata=metadata,
            reviews=reviews,
        )
        return

    logger.info(
        f"After k-core: {len(surviving_items)} items, "
        f"{len(new_reviews)} reviews"
    )

    item2id = _assign_item_ids(surviving_items, seed=seed)

    # 构建交互样本
    interaction_list = _build_interactions(new_reviews, item2id, id_title, max_history_len)
    logger.info(f"Total interaction samples: {len(interaction_list)}")

    # 划分
    if split_method.upper() == "TO":
        train, valid, test = _split_to(interaction_list)
    elif split_method.upper() == "LOO":
        train, valid, test = _split_loo(interaction_list)
    else:
        raise ValueError(f"Unknown split_method: {split_method}. Use 'TO' or 'LOO'.")

    logger.info(
        f"Split: train={len(train)}, valid={len(valid)}, test={len(test)}"
    )

    # 写入 .inter 文件
    prefix = os.path.join(output_dir, category)
    _write_inter_file(f"{prefix}.train.inter", train)
    _write_inter_file(f"{prefix}.valid.inter", valid)
    _write_inter_file(f"{prefix}.test.inter", test)

    # 写入 .item.json
    _write_item_json(f"{prefix}.item.json", item2id, id_title, metadata)

    # 写入 .review.json
    _write_review_json(f"{prefix}.review.json", new_reviews, item2id)

    logger.info(
        f"Done! Output: {prefix}.{{train,valid,test}}.inter, "
        f".item.json, .review.json"
    )
