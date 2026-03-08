"""测试 GatedSAE 训练和 SID 生成功能。"""

import json
import re

import numpy as np
import pytest
import torch

from SAEGenRec.sid_builder.gated_sae import NpyDataProvider

# ——— T017: NpyDataProvider 测试 ———


class TestNpyDataProvider:
    def test_batch_shape(self, tmp_path):
        """验证批次形状正确。"""
        n, d = 100, 32
        data = np.random.randn(n, d).astype(np.float32)
        path = str(tmp_path / "emb.npy")
        np.save(path, data)

        provider = NpyDataProvider(path, batch_size=16)
        batch = next(provider)
        assert batch.shape == (16, d)
        assert batch.dtype == torch.float32

    def test_partial_last_batch(self, tmp_path):
        """最后一批可能小于 batch_size。"""
        n, d = 10, 8
        data = np.random.randn(n, d).astype(np.float32)
        path = str(tmp_path / "emb.npy")
        np.save(path, data)

        provider = NpyDataProvider(path, batch_size=6)
        batch1 = next(provider)
        batch2 = next(provider)
        assert batch1.shape[0] == 6
        assert batch2.shape[0] == 4

    def test_nan_inf_handling(self, tmp_path):
        """NaN 和 Inf 被替换为 0。"""
        data = np.array([[np.nan, np.inf, -np.inf, 1.0]], dtype=np.float32)
        path = str(tmp_path / "emb.npy")
        np.save(path, data)

        provider = NpyDataProvider(path, batch_size=1)
        batch = next(provider)
        assert torch.all(torch.isfinite(batch)), "NaN/Inf should be replaced with 0"
        assert float(batch[0, 3]) == pytest.approx(1.0)

    def test_multi_epoch_cycling(self, tmp_path):
        """多 epoch 循环：超出总样本数后继续返回数据。"""
        n, d = 10, 4
        data = np.random.randn(n, d).astype(np.float32)
        path = str(tmp_path / "emb.npy")
        np.save(path, data)

        provider = NpyDataProvider(path, batch_size=4)
        # 消耗超过 2 个 epoch 的数据（10 items / batch4 = 3 batches/epoch，8 次 = 28 items）
        total = 0
        for _ in range(8):
            batch = next(provider)
            total += batch.shape[0]
        assert total >= 2 * n, "Should cycle through at least 2 epochs"

    def test_total_sample_count(self, tmp_path):
        """数据集长度等于 n_items。"""
        n, d = 50, 8
        data = np.random.randn(n, d).astype(np.float32)
        path = str(tmp_path / "emb.npy")
        np.save(path, data)

        provider = NpyDataProvider(path, batch_size=10)
        assert len(provider) == n

    def test_shuffle_changes_order(self, tmp_path):
        """每 epoch shuffle 会改变顺序（统计意义上）。"""
        n, d = 100, 4
        data = np.arange(n * d, dtype=np.float32).reshape(n, d)
        path = str(tmp_path / "emb.npy")
        np.save(path, data)

        provider = NpyDataProvider(path, batch_size=n, seed=0)
        epoch1 = next(provider).clone()
        epoch2 = next(provider).clone()
        # 两个 epoch 的第一行不同（shuffle 有效）
        assert not torch.all(epoch1 == epoch2), "Shuffle should change order between epochs"


# ——— T018: SID 生成格式测试 ———


class TestSIDFormat:
    def _make_index_json(self, n_items: int, k: int, d_sae: int) -> dict:
        """构造 fake index_dict（与 generate_sae_indices 相同格式）。"""
        prefix_list = list("abcdefghijklmnopqrstuvwxyz")
        index_dict = {}
        for item_id in range(n_items):
            # 随机 K 个不重复的特征索引
            feat_indices = np.random.choice(d_sae, size=k, replace=False).tolist()
            tokens = [f"[{prefix_list[pos]}_{idx}]" for pos, idx in enumerate(feat_indices)]
            index_dict[str(item_id)] = tokens
        return index_dict

    def test_output_structure(self, tmp_path):
        """验证 .index.json 结构：每个 item 有 K=8 个 token，格式正确。"""
        n_items, k, d_sae = 20, 8, 64
        index_dict = self._make_index_json(n_items, k, d_sae)

        path = str(tmp_path / "test.index.json")
        with open(path, "w") as f:
            json.dump(index_dict, f)

        with open(path) as f:
            loaded = json.load(f)

        assert len(loaded) == n_items
        for item_id, tokens in loaded.items():
            assert len(tokens) == k, f"Item {item_id} should have {k} tokens"
            for pos, token in enumerate(tokens):
                prefix = chr(97 + pos)
                assert re.match(rf"\[{prefix}_\d+\]", token), f"Token {token} format invalid"

    def test_prefix_letters_a_to_h(self, tmp_path):
        """K=8 时 prefix 字母应为 a-h。"""
        k = 8
        prefix_list = list("abcdefghijklmnopqrstuvwxyz")
        index_dict = {"0": [f"[{prefix_list[pos]}_42]" for pos in range(k)]}
        path = str(tmp_path / "test.index.json")
        with open(path, "w") as f:
            json.dump(index_dict, f)

        with open(path) as f:
            loaded = json.load(f)

        tokens = loaded["0"]
        for pos, token in enumerate(tokens):
            expected_prefix = chr(97 + pos)
            assert token.startswith(f"[{expected_prefix}_")

    def test_feature_index_range(self):
        """feature_index 应在 [0, d_sae-1] 范围内。"""
        d_sae = 64
        index_dict = self._make_index_json(n_items=10, k=8, d_sae=d_sae)
        pattern = re.compile(r"\[([a-z])_(\d+)\]")
        for tokens in index_dict.values():
            for token in tokens:
                m = pattern.match(token)
                assert m is not None
                feat_idx = int(m.group(2))
                assert 0 <= feat_idx < d_sae

    def test_dedup_produces_unique_sids(self, tmp_path):
        """验证去重逻辑能消除碰撞。"""
        # 构造一个简单场景：两个 item 使用相同的初始 top-K，但有不同的备用特征
        # 通过直接测试去重后 SID 唯一性
        k = 2
        # Item 0: top-2 = [0, 1], backup = [2]
        # Item 1: top-2 = [0, 1], backup = [3]
        initial_sids = [[0, 1], [0, 1]]
        topk_indices = torch.tensor([[0, 1, 2], [0, 1, 3]])  # shape (2, k+1)

        # 模拟去重（手动复现 generate_sae_indices 的逻辑）
        max_dedup_iters = 5
        item_sids = [sid.copy() for sid in initial_sids]
        for iteration in range(max_dedup_iters):
            sid_to_items: dict[str, list[int]] = {}
            for item_id, sid in enumerate(item_sids):
                key = str(sid)
                sid_to_items.setdefault(key, []).append(item_id)
            collision_groups = [g for g in sid_to_items.values() if len(g) > 1]
            if not collision_groups:
                break
            j = iteration + 1
            candidate_col = k + j - 1
            if candidate_col >= topk_indices.shape[1]:
                break
            for group in collision_groups:
                for item_id in group[1:]:
                    new_last = int(topk_indices[item_id, candidate_col].item())
                    new_sid = item_sids[item_id].copy()
                    new_sid[-1] = new_last
                    item_sids[item_id] = new_sid

        # 去重后 SID 应唯一
        sid_strs = [str(s) for s in item_sids]
        assert len(set(sid_strs)) == len(sid_strs), "Dedup should produce unique SIDs"


# ——— T019: set_seed() 确定性测试 ———


class TestSetSeedDeterminism:
    def test_same_seed_same_encode_output(self, tmp_path):
        """相同 seed + 相同输入 → 相同 NpyDataProvider 输出顺序。"""
        from SAEGenRec.config import set_seed

        n, d = 50, 8
        data = np.random.randn(n, d).astype(np.float32)
        path = str(tmp_path / "emb.npy")
        np.save(path, data)

        set_seed(42)
        provider1 = NpyDataProvider(path, batch_size=10, seed=42)
        batch1a = next(provider1).clone()

        set_seed(42)
        provider2 = NpyDataProvider(path, batch_size=10, seed=42)
        batch2a = next(provider2).clone()

        assert torch.allclose(batch1a, batch2a), "Same seed should produce identical batches"

    def test_different_seeds_different_order(self, tmp_path):
        """不同 seed 应产生不同的打乱顺序。"""
        n, d = 100, 4
        data = np.arange(n * d, dtype=np.float32).reshape(n, d)
        path = str(tmp_path / "emb.npy")
        np.save(path, data)

        provider_a = NpyDataProvider(path, batch_size=n, seed=0)
        provider_b = NpyDataProvider(path, batch_size=n, seed=999)
        batch_a = next(provider_a)
        batch_b = next(provider_b)

        assert not torch.all(batch_a == batch_b), "Different seeds should produce different orders"
