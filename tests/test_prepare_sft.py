"""Unit tests for prepare_sft module."""

import json
import os
import tempfile

import pandas as pd
import pytest


def _make_csv(tmpdir, split="train", n=5, category="Beauty"):
    """Create minimal CSV for testing (filename: {category}.{split}.csv)."""
    rows = []
    for i in range(n):
        rows.append({
            "user_id": i,
            "history_item_sid": "[a_1][b_2][c_3] [a_4][b_5][c_6]",
            "target_item_sid": "[a_7][b_8][c_9]",
            "history_item_id": "0 1",
            "item_id": "2",
            "item_asin": f"B{i:03d}",
            "history_item_title": "Item A, Item B",
            "target_item_title": "Item C",
        })
    df = pd.DataFrame(rows)
    path = os.path.join(tmpdir, f"{category}.{split}.csv")
    df.to_csv(path, index=False)
    return path


def _make_index_json(tmpdir):
    """Create minimal index.json."""
    data = {"0": ["[a_1]", "[b_2]", "[c_3]"], "1": ["[a_4]", "[b_5]", "[c_6]"], "2": ["[a_7]", "[b_8]", "[c_9]"]}
    path = os.path.join(tmpdir, "Beauty.index.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _make_item_json(tmpdir):
    """Create minimal item.json."""
    data = {
        "0": {"title": "Item A", "asin": "B000"},
        "1": {"title": "Item B", "asin": "B001"},
        "2": {"title": "Item C", "asin": "B002"},
    }
    path = os.path.join(tmpdir, "Beauty.item.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def test_jsonl_output_format(tmp_path):
    """Test that JSONL output has prompt + completion keys."""
    train_csv = _make_csv(str(tmp_path), "train")
    _make_csv(str(tmp_path), "valid", n=2)
    _make_csv(str(tmp_path), "test", n=2)
    index_path = _make_index_json(str(tmp_path))
    _make_item_json(str(tmp_path))

    out_dir = str(tmp_path / "out")
    from SAEGenRec.training.prepare_sft import prepare_sft

    prepare_sft(
        category="Beauty",
        sid_type="rqvae",
        task="sid_seq",
        interim_dir=str(tmp_path),
        data_dir=out_dir,
        overwrite=True,
    )

    jsonl_path = os.path.join(out_dir, "rqvae", "Amazon", "Beauty", "sid_seq", "train.jsonl")
    assert os.path.exists(jsonl_path), f"JSONL not found at {jsonl_path}"
    with open(jsonl_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) > 0
    for line in lines:
        assert "prompt" in line
        assert "completion" in line


def test_train_valid_test_split(tmp_path):
    """Test that all three splits are generated."""
    _make_csv(str(tmp_path), "train", n=5)
    _make_csv(str(tmp_path), "valid", n=2)
    _make_csv(str(tmp_path), "test", n=2)
    _make_index_json(str(tmp_path))
    _make_item_json(str(tmp_path))

    out_dir = str(tmp_path / "out")
    from SAEGenRec.training.prepare_sft import prepare_sft

    prepare_sft(
        category="Beauty",
        sid_type="rqvae",
        task="sid_seq",
        interim_dir=str(tmp_path),
        data_dir=out_dir,
        overwrite=True,
    )

    task_dir = os.path.join(out_dir, "rqvae", "Amazon", "Beauty", "sid_seq")
    for split in ["train", "valid", "test"]:
        assert os.path.exists(os.path.join(task_dir, f"{split}.jsonl"))


def test_meta_json_written(tmp_path):
    """Test that meta.json is written with correct parameters."""
    _make_csv(str(tmp_path), "train", n=3)
    _make_csv(str(tmp_path), "valid", n=2)
    _make_csv(str(tmp_path), "test", n=2)
    _make_index_json(str(tmp_path))
    _make_item_json(str(tmp_path))

    out_dir = str(tmp_path / "out")
    from SAEGenRec.training.prepare_sft import prepare_sft

    prepare_sft(
        category="Beauty",
        sid_type="rqvae",
        task="sid_seq",
        interim_dir=str(tmp_path),
        data_dir=out_dir,
        overwrite=True,
    )

    meta_path = os.path.join(out_dir, "rqvae", "Amazon", "Beauty", "sid_seq", "meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["category"] == "Beauty"
    assert meta["sid_type"] == "rqvae"
    assert meta["task"] == "sid_seq"


def test_overwrite_protection(tmp_path):
    """Test that existing data is protected when overwrite=False."""
    _make_csv(str(tmp_path), "train", n=3)
    _make_csv(str(tmp_path), "valid", n=2)
    _make_csv(str(tmp_path), "test", n=2)
    _make_index_json(str(tmp_path))
    _make_item_json(str(tmp_path))

    out_dir = str(tmp_path / "out")
    from SAEGenRec.training.prepare_sft import prepare_sft

    # First run
    prepare_sft(
        category="Beauty",
        sid_type="rqvae",
        task="sid_seq",
        interim_dir=str(tmp_path),
        data_dir=out_dir,
        overwrite=True,
    )

    # Second run without overwrite should raise
    with pytest.raises(FileExistsError):
        prepare_sft(
            category="Beauty",
            sid_type="rqvae",
            task="sid_seq",
            interim_dir=str(tmp_path),
            data_dir=out_dir,
            overwrite=False,
        )


def test_template_placeholder_rendering(tmp_path):
    """Test that prompt templates are rendered with correct placeholders."""
    _make_csv(str(tmp_path), "train", n=3)
    _make_csv(str(tmp_path), "valid", n=2)
    _make_csv(str(tmp_path), "test", n=2)
    _make_index_json(str(tmp_path))
    _make_item_json(str(tmp_path))

    out_dir = str(tmp_path / "out")
    from SAEGenRec.training.prepare_sft import prepare_sft

    prepare_sft(
        category="Beauty",
        sid_type="rqvae",
        task="sid_seq",
        interim_dir=str(tmp_path),
        data_dir=out_dir,
        overwrite=True,
    )

    jsonl_path = os.path.join(out_dir, "rqvae", "Amazon", "Beauty", "sid_seq", "train.jsonl")
    with open(jsonl_path) as f:
        first_line = json.loads(f.readline())
    # Should not contain raw placeholder
    assert "{history}" not in first_line["prompt"]
    assert "### Response:" in first_line["prompt"]
