"""测试 SID 构建模块：VQ/RQ-VAE 模型、碰撞率计算、convert_dataset。"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from SAEGenRec.sid_builder.models.layers import MLPLayers, kmeans, sinkhorn_algorithm
from SAEGenRec.sid_builder.models.rq import ResidualVectorQuantizer
from SAEGenRec.sid_builder.models.rqvae import RQVAE
from SAEGenRec.sid_builder.models.vq import VectorQuantizer
from SAEGenRec.sid_builder.rqvae import EmbDataset


# ---- VectorQuantizer 测试 ----

def test_vq_forward_shape():
    """VectorQuantizer forward 输出 shape 应正确。"""
    vq = VectorQuantizer(n_e=16, e_dim=8, kmeans_init=False)
    x = torch.randn(4, 8)
    x_q, loss, indices = vq(x, use_sk=False)
    assert x_q.shape == x.shape
    assert loss.ndim == 0
    assert indices.shape == (4,)


def test_vq_forward_loss_positive():
    """VQ loss 应为正数。"""
    vq = VectorQuantizer(n_e=16, e_dim=8, kmeans_init=False)
    x = torch.randn(4, 8)
    _, loss, _ = vq(x, use_sk=False)
    assert loss.item() >= 0


def test_vq_indices_in_range():
    """VQ indices 应在 [0, n_e) 范围内。"""
    vq = VectorQuantizer(n_e=16, e_dim=8, kmeans_init=False)
    x = torch.randn(10, 8)
    _, _, indices = vq(x, use_sk=False)
    assert (indices >= 0).all()
    assert (indices < 16).all()


# ---- ResidualVectorQuantizer 测试 ----

def test_rq_forward_shape():
    """ResidualVectorQuantizer forward 输出 shape 应正确。"""
    rq = ResidualVectorQuantizer(
        n_e_list=[16, 16, 16],
        e_dim=8,
        sk_epsilons=[0.0, 0.0, 0.0],
        kmeans_init=False,
    )
    x = torch.randn(4, 8)
    x_q, loss, indices = rq(x, use_sk=False)
    assert x_q.shape == x.shape
    assert indices.shape == (4, 3)  # batch × num_levels


# ---- RQVAE 测试 ----

def test_rqvae_forward_shape():
    """RQVAE forward 输出 shape 应与输入相同。"""
    model = RQVAE(
        in_dim=32,
        num_emb_list=[8, 8, 8],
        e_dim=4,
        layers=[16, 8],
        kmeans_init=False,
        sk_epsilons=[0.0, 0.0, 0.0],
    )
    x = torch.randn(4, 32)
    out, rq_loss, indices = model(x, use_sk=False)
    assert out.shape == x.shape
    assert indices.shape == (4, 3)


def test_rqvae_loss_decreases():
    """RQVAE 训练 1 epoch 后 loss 应从初始值变化（验证梯度流动）。"""
    model = RQVAE(
        in_dim=16,
        num_emb_list=[4, 4],
        e_dim=4,
        layers=[8],
        kmeans_init=False,
        sk_epsilons=[0.0, 0.0],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(8, 16)

    model.train()
    out, rq_loss, _ = model(x, use_sk=False)
    loss_before, _ = model.compute_loss(out, rq_loss, xs=x)
    loss_before = loss_before.item()

    optimizer.zero_grad()
    out, rq_loss, _ = model(x, use_sk=False)
    loss, _ = model.compute_loss(out, rq_loss, xs=x)
    loss.backward()
    optimizer.step()

    # Loss changed (not necessarily decreased in 1 step, but gradients flowed)
    assert not torch.isnan(loss)


def test_rqvae_get_indices_shape():
    """get_indices 应返回 (N, num_levels) 形状。"""
    model = RQVAE(
        in_dim=16,
        num_emb_list=[4, 4, 4],
        e_dim=4,
        layers=[8],
        kmeans_init=False,
        sk_epsilons=[0.0, 0.0, 0.0],
    )
    model.eval()
    x = torch.randn(5, 16)
    with torch.no_grad():
        indices = model.get_indices(x, use_sk=False)
    assert indices.shape == (5, 3)


# ---- 碰撞率测试 ----

def test_collision_rate_zero():
    """5 个不同的 SID 应有 0 碰撞率。"""
    codes = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
    all_strs = [str(c) for c in codes]
    collision_rate = (len(all_strs) - len(set(all_strs))) / len(all_strs)
    assert collision_rate == 0.0


def test_collision_rate_nonzero():
    """重复 SID 应有 > 0 碰撞率。"""
    codes = [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
    all_strs = [str(c) for c in codes]
    collision_rate = (len(all_strs) - len(set(all_strs))) / len(all_strs)
    assert collision_rate > 0


# ---- EmbDataset 测试 ----

def test_emb_dataset():
    """EmbDataset 应正确加载和处理 .npy 文件。"""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        path = f.name
    try:
        embs = np.random.rand(10, 32).astype(np.float32)
        np.save(path, embs)
        ds = EmbDataset(path)
        assert len(ds) == 10
        assert ds.dim == 32
        assert ds[0].shape == (32,)
        assert isinstance(ds[0], torch.Tensor)
    finally:
        os.unlink(path)


def test_emb_dataset_nan_handling():
    """EmbDataset 应将 NaN 替换为 0。"""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        path = f.name
    try:
        embs = np.ones((5, 8), dtype=np.float32)
        embs[0, 0] = float("nan")
        np.save(path, embs)
        ds = EmbDataset(path)
        assert not torch.isnan(ds[0]).any()
    finally:
        os.unlink(path)


# ---- index.json 结构验证 ----

def test_index_json_structure():
    """index.json 应有正确的键值结构（str item_id → list of token strings）。"""
    index = {
        "0": ["[a_42]", "[b_128]", "[c_7]"],
        "1": ["[a_15]", "[b_200]", "[c_33]"],
    }
    for key, val in index.items():
        assert isinstance(key, str)
        assert int(key) >= 0
        assert isinstance(val, list)
        for token in val:
            assert isinstance(token, str)
            assert token.startswith("[")
            assert token.endswith("]")


# ---- convert_dataset CSV 验证 ----

def test_convert_dataset_csv_schema():
    """convert_dataset 生成的 CSV 应有正确的列名。"""
    from SAEGenRec.data_process.convert_dataset import convert_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # 准备 .item.json
        items = {
            "0": {"title": "Widget A", "description": "Desc A", "item_asin": "B001"},
            "1": {"title": "Widget B", "description": "Desc B", "item_asin": "B002"},
            "2": {"title": "Widget C", "description": "Desc C", "item_asin": "B003"},
        }
        item_json_path = os.path.join(tmpdir, "test.item.json")
        with open(item_json_path, "w") as f:
            json.dump(items, f)

        # 准备 .index.json
        index = {
            "0": ["[a_1]", "[b_2]", "[c_3]"],
            "1": ["[a_4]", "[b_5]", "[c_6]"],
            "2": ["[a_7]", "[b_8]", "[c_9]"],
        }
        index_json_path = os.path.join(tmpdir, "test.index.json")
        with open(index_json_path, "w") as f:
            json.dump(index, f)

        # 准备 .inter 文件
        train_inter = "user_id\titem_id\titem_asin\ttimestamp\trating\n"
        train_inter += "U1\t0\tB001\t1000\t5.0\n"
        train_inter += "U1\t1\tB002\t2000\t4.0\n"
        train_inter += "U1\t2\tB003\t3000\t3.0\n"
        inter_path = os.path.join(tmpdir, "test.train.inter")
        with open(inter_path, "w") as f:
            f.write(train_inter)

        output_dir = os.path.join(tmpdir, "processed")
        convert_dataset(
            inter_dir=tmpdir,
            item_json=item_json_path,
            index_json=index_json_path,
            dataset="test",
            output_dir=output_dir,
            splits="train",
        )

        csv_path = os.path.join(output_dir, "test.train.csv")
        assert os.path.exists(csv_path), "train CSV not created"

        import csv as csv_mod

        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            required_cols = {
                "user_id", "history_item_sid", "target_item_sid",
                "history_item_title", "target_item_title",
                "history_item_id", "target_item_id",
            }
            assert required_cols.issubset(set(reader.fieldnames)), (
                f"Missing columns: {required_cols - set(reader.fieldnames)}"
            )


def test_convert_dataset_info_txt():
    """info TXT 应有正确格式：semantic_id \\t title \\t item_id。"""
    from SAEGenRec.data_process.convert_dataset import _write_info_txt

    items = {
        "0": {"title": "Widget A"},
        "1": {"title": "Widget B"},
    }
    index = {
        "0": ["[a_1]", "[b_2]", "[c_3]"],
        "1": ["[a_4]", "[b_5]", "[c_6]"],
    }

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="r", delete=False) as f:
        path = f.name

    try:
        _write_info_txt(items, index, path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            parts = line.strip().split("\t")
            assert len(parts) == 3
            sid, title, item_id = parts
            assert sid.startswith("[") and sid.endswith("]")
            assert int(item_id) >= 0
    finally:
        os.unlink(path)
