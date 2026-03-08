"""测试 data_process 预处理逻辑。"""

import json
import os
import tempfile

import pytest

from SAEGenRec.data_process.preprocess import (
    _assign_item_ids,
    _build_interactions,
    _kcore_filter,
    _split_loo,
    _split_to,
    _write_inter_file,
    _write_item_json,
    _write_review_json,
    preprocess,
)

# ---- 测试数据 ----

METADATA = [
    {"asin": "B001", "title": "Widget Alpha", "description": "A great widget"},
    {"asin": "B002", "title": "Widget Beta", "description": ""},
    {"asin": "B003", "title": "Widget Gamma", "description": "Another widget"},
    {"asin": "B004", "title": "Widget Delta", "description": ""},
    {"asin": "B005", "title": "Widget Epsilon", "description": ""},
    {"asin": "B006", "title": "<span id='bad'>bad</span>", "description": ""},  # invalid title
]

# 时间戳都在 2018 年范围内
REVIEWS = [
    # user U1 interacts with B001, B002, B003, B004, B005 (k=2, each appears 2+ times)
    {"reviewerID": "U1", "asin": "B001", "unixReviewTime": "1514764800", "overall": 5.0, "reviewText": "Great", "summary": "Excellent"},
    {"reviewerID": "U1", "asin": "B002", "unixReviewTime": "1514851200", "overall": 4.0, "reviewText": "Good", "summary": "Good"},
    {"reviewerID": "U1", "asin": "B003", "unixReviewTime": "1514937600", "overall": 3.0, "reviewText": "Ok", "summary": "Ok"},
    {"reviewerID": "U1", "asin": "B004", "unixReviewTime": "1515024000", "overall": 2.0, "reviewText": "", "summary": ""},
    {"reviewerID": "U1", "asin": "B005", "unixReviewTime": "1515110400", "overall": 1.0, "reviewText": "", "summary": ""},
    {"reviewerID": "U2", "asin": "B001", "unixReviewTime": "1514764800", "overall": 4.0, "reviewText": "Nice", "summary": "Nice"},
    {"reviewerID": "U2", "asin": "B002", "unixReviewTime": "1514851200", "overall": 3.0, "reviewText": "", "summary": ""},
    {"reviewerID": "U2", "asin": "B003", "unixReviewTime": "1514937600", "overall": 5.0, "reviewText": "Love it", "summary": "Love"},
    {"reviewerID": "U2", "asin": "B004", "unixReviewTime": "1515024000", "overall": 4.0, "reviewText": "", "summary": ""},
    {"reviewerID": "U2", "asin": "B005", "unixReviewTime": "1515110400", "overall": 3.0, "reviewText": "", "summary": ""},
    {"reviewerID": "U3", "asin": "B001", "unixReviewTime": "1514764800", "overall": 3.0, "reviewText": "", "summary": ""},
    {"reviewerID": "U3", "asin": "B002", "unixReviewTime": "1514851200", "overall": 4.0, "reviewText": "", "summary": ""},
    {"reviewerID": "U3", "asin": "B003", "unixReviewTime": "1514937600", "overall": 2.0, "reviewText": "", "summary": ""},
    {"reviewerID": "U3", "asin": "B004", "unixReviewTime": "1515024000", "overall": 5.0, "reviewText": "", "summary": ""},
    {"reviewerID": "U3", "asin": "B005", "unixReviewTime": "1515110400", "overall": 1.0, "reviewText": "", "summary": ""},
]

# 时间范围：2018-01-01 ~ 2019-01-01
START_TS = 1514764800  # 2018-01-01
END_TS = 1546300800   # 2019-01-01


def test_kcore_filter_removes_invalid_titles():
    """无效标题的 item 应被过滤（B006 含 <span id>）。"""
    new_reviews, id_title, surviving, remove_users, remove_items = _kcore_filter(
        METADATA, REVIEWS, k=2, start_timestamp=START_TS, end_timestamp=END_TS
    )
    assert "B006" in remove_items
    assert "B006" not in id_title


def test_kcore_filter_respects_k():
    """k-core 过滤：每个用户/item 必须至少有 k 次交互。"""
    # 使用 k=4，只有 3 个用户，每个 item 有 3 次，所以应该全部被过滤
    reviews_low = [
        {"reviewerID": "U1", "asin": "B001", "unixReviewTime": "1514764800", "overall": 5.0},
        {"reviewerID": "U2", "asin": "B001", "unixReviewTime": "1514764800", "overall": 4.0},
        {"reviewerID": "U3", "asin": "B001", "unixReviewTime": "1514764800", "overall": 3.0},
        {"reviewerID": "U1", "asin": "B002", "unixReviewTime": "1514851200", "overall": 5.0},
        {"reviewerID": "U2", "asin": "B002", "unixReviewTime": "1514851200", "overall": 4.0},
        {"reviewerID": "U3", "asin": "B002", "unixReviewTime": "1514851200", "overall": 3.0},
    ]
    new_reviews, id_title, surviving, remove_users, remove_items = _kcore_filter(
        METADATA, reviews_low, k=4, start_timestamp=START_TS, end_timestamp=END_TS
    )
    # k=4, each user only has 2 interactions => all removed
    assert len(surviving) == 0 or len(new_reviews) == 0


def test_assign_item_ids_unique():
    """item_id 分配应保证唯一性。"""
    items = ["B001", "B002", "B003", "B004", "B005"]
    item2id = _assign_item_ids(items, seed=42)
    assert len(item2id) == len(items)
    assert len(set(item2id.values())) == len(items)
    assert all(0 <= v < len(items) for v in item2id.values())


def test_assign_item_ids_deterministic():
    """相同种子应产生相同的 item_id 分配。"""
    items = ["B001", "B002", "B003", "B004", "B005"]
    id1 = _assign_item_ids(items, seed=42)
    id2 = _assign_item_ids(items, seed=42)
    assert id1 == id2


def test_build_interactions_max_history():
    """滑动窗口应遵守 max_history_len。"""
    # 一个用户有 10 次交互
    reviews = [
        {
            "reviewerID": "U1",
            "asin": f"B00{i:02d}",
            "unixReviewTime": str(1514764800 + i * 86400),
            "overall": 5.0,
        }
        for i in range(10)
    ]
    metadata_local = [
        {"asin": f"B00{i:02d}", "title": f"Item {i}", "description": ""}
        for i in range(10)
    ]
    item2id = {f"B00{i:02d}": i for i in range(10)}
    id_title = {f"B00{i:02d}": f"Item {i}" for i in range(10)}

    samples = _build_interactions(reviews, item2id, id_title, max_history_len=3)
    # 每个样本的历史长度不超过 3
    for s in samples:
        assert len(s[1]) <= 3  # history_asins


def test_split_to_ratio():
    """TO 划分应接近 8:1:1。"""
    # 构造 100 个样本
    samples = [
        ["U1", [], f"B{i:03d}", [], i, [], 5.0, [], 1514764800 + i * 3600]
        for i in range(100)
    ]
    train, valid, test = _split_to(samples)
    assert len(train) == 80
    assert len(valid) == 10
    assert len(test) == 10
    assert len(train) + len(valid) + len(test) == 100


def test_split_to_sorted_by_timestamp():
    """TO 划分后 train/valid/test 应按时间有序。"""
    import random as rng

    samples = [
        ["U1", [], f"B{i:03d}", [], i, [], 5.0, [], 1514764800 + rng.randint(0, 1000000)]
        for i in range(100)
    ]
    train, valid, test = _split_to(samples)
    timestamps = [s[8] for s in train + valid + test]
    assert timestamps == sorted(timestamps)


def test_split_loo_per_user():
    """LOO 划分：每个用户最后→test，倒二→valid，其余→train。"""
    # U1 有 5 次交互，U2 有 3 次
    samples = []
    for i in range(5):
        samples.append(["U1", [], f"B{i:03d}", [], i, [], 5.0, [], 1514764800 + i * 3600])
    for i in range(3):
        samples.append(["U2", [], f"C{i:03d}", [], i, [], 4.0, [], 1514764800 + i * 3600])

    train, valid, test = _split_loo(samples)

    # U1: 3 train, 1 valid, 1 test; U2: 1 train, 1 valid, 1 test
    test_users = [s[0] for s in test]
    valid_users = [s[0] for s in valid]
    assert "U1" in test_users
    assert "U2" in test_users
    assert "U1" in valid_users
    assert "U2" in valid_users
    assert len(test) == 2  # 一个用户一条
    assert len(valid) == 2
    assert len(train) == 4  # U1: 3, U2: 1


def test_write_inter_file_schema():
    """写入的 .inter 文件应有正确的 TSV 格式和列名。"""
    samples = [
        ["U1", ["B001"], "B002", [0], 1, [5.0], 4.0, [1514764800], 1514851200],
    ]
    with tempfile.NamedTemporaryFile(suffix=".inter", mode="r", delete=False) as f:
        path = f.name

    try:
        _write_inter_file(path, samples)
        with open(path) as f:
            lines = f.readlines()
        assert lines[0].strip() == "user_id\titem_id\titem_asin\ttimestamp\trating"
        cols = lines[1].strip().split("\t")
        assert len(cols) == 5
        assert cols[0] == "U1"
        assert cols[1] == "1"   # target_item_id
        assert cols[2] == "B002"  # target_asin
    finally:
        os.unlink(path)


def test_write_item_json_structure():
    """写入的 .item.json 应有正确的键值结构。"""
    item2id = {"B001": 0, "B002": 1}
    id_title = {"B001": "Widget Alpha", "B002": "Widget Beta"}
    meta = [
        {"asin": "B001", "title": "Widget Alpha", "description": "Desc A"},
        {"asin": "B002", "title": "Widget Beta"},
    ]
    with tempfile.NamedTemporaryFile(suffix=".json", mode="r", delete=False) as f:
        path = f.name

    try:
        _write_item_json(path, item2id, id_title, meta)
        with open(path) as f:
            data = json.load(f)
        assert "0" in data
        assert "1" in data
        assert data["0"]["item_asin"] == "B001"
        assert data["0"]["title"] == "Widget Alpha"
        assert "description" in data["0"]
    finally:
        os.unlink(path)


def test_review_json_row_count_matches_interactions():
    """review.json 的行数应等于通过 k-core 的交互数。"""
    new_reviews, id_title, surviving, _, _ = _kcore_filter(
        METADATA, REVIEWS, k=2, start_timestamp=START_TS, end_timestamp=END_TS
    )
    item2id = _assign_item_ids(surviving, seed=42)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="r", delete=False) as f:
        path = f.name

    try:
        _write_review_json(path, new_reviews, item2id)
        with open(path) as f:
            records = json.load(f)
        # 所有 new_reviews 中 asin 在 item2id 中的都应被写入
        valid_count = sum(1 for r in new_reviews if r["asin"] in item2id)
        assert len(records) == valid_count
    finally:
        os.unlink(path)


def test_preprocess_idempotency():
    """相同输入和种子应产生相同输出。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        preprocess(
            category="TestCat",
            k_core=2,
            st_year=2018,
            st_month=1,
            ed_year=2019,
            ed_month=1,
            split_method="TO",
            max_history_len=50,
            output_dir=tmpdir,
            seed=42,
            metadata=METADATA,
            reviews=REVIEWS,
        )

        # 读取第一次结果
        train1 = open(os.path.join(tmpdir, "TestCat.train.inter")).read()
        item1 = open(os.path.join(tmpdir, "TestCat.item.json")).read()

        preprocess(
            category="TestCat",
            k_core=2,
            st_year=2018,
            st_month=1,
            ed_year=2019,
            ed_month=1,
            split_method="TO",
            max_history_len=50,
            output_dir=tmpdir,
            seed=42,
            metadata=METADATA,
            reviews=REVIEWS,
        )

        train2 = open(os.path.join(tmpdir, "TestCat.train.inter")).read()
        item2 = open(os.path.join(tmpdir, "TestCat.item.json")).read()

        assert train1 == train2
        assert item1 == item2


def test_preprocess_output_files_exist():
    """preprocess 应生成所有预期的输出文件。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        preprocess(
            category="TestCat",
            k_core=2,
            st_year=2018,
            st_month=1,
            ed_year=2019,
            ed_month=1,
            split_method="TO",
            max_history_len=50,
            output_dir=tmpdir,
            seed=42,
            metadata=METADATA,
            reviews=REVIEWS,
        )
        assert os.path.exists(os.path.join(tmpdir, "TestCat.train.inter"))
        assert os.path.exists(os.path.join(tmpdir, "TestCat.valid.inter"))
        assert os.path.exists(os.path.join(tmpdir, "TestCat.test.inter"))
        assert os.path.exists(os.path.join(tmpdir, "TestCat.item.json"))
        assert os.path.exists(os.path.join(tmpdir, "TestCat.review.json"))


def test_preprocess_loo_split():
    """LOO 划分应产生正确的每用户划分。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        preprocess(
            category="TestCat",
            k_core=2,
            st_year=2018,
            st_month=1,
            ed_year=2019,
            ed_month=1,
            split_method="LOO",
            max_history_len=50,
            output_dir=tmpdir,
            seed=42,
            metadata=METADATA,
            reviews=REVIEWS,
        )
        test_file = os.path.join(tmpdir, "TestCat.test.inter")
        assert os.path.exists(test_file)
        with open(test_file) as f:
            lines = f.readlines()
        # 应有表头 + 若干数据行
        assert lines[0].strip().startswith("user_id")
        assert len(lines) > 1


def test_inter_column_schema():
    """train.inter 所有行应有 5 列。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        preprocess(
            category="TestCat",
            k_core=2,
            st_year=2018,
            st_month=1,
            ed_year=2019,
            ed_month=1,
            split_method="TO",
            max_history_len=50,
            output_dir=tmpdir,
            seed=42,
            metadata=METADATA,
            reviews=REVIEWS,
        )
        with open(os.path.join(tmpdir, "TestCat.train.inter")) as f:
            lines = f.readlines()
        for line in lines[1:]:  # skip header
            cols = line.strip().split("\t")
            assert len(cols) == 5, f"Expected 5 columns, got {len(cols)}: {line}"
