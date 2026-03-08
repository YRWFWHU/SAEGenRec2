"""测试 config.py 中的 set_seed()、CATEGORY_MAP 和 dataclass 配置。"""

import random

import numpy as np
import pytest
import torch

from SAEGenRec.config import (
    CATEGORY_MAP,
    BaseConfig,
    DataProcessConfig,
    EvalConfig,
    RLConfig,
    RQVAEConfig,
    SASRecConfig,
    SFTConfig,
    TextEmbConfig,
    set_seed,
)


def test_set_seed_determinism():
    """set_seed 应确保 random/numpy/torch 输出一致。"""
    set_seed(42)
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()

    set_seed(42)
    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.rand(1).item()

    assert r1 == r2
    assert n1 == n2
    assert t1 == t2


def test_set_seed_different_seeds():
    """不同种子产生不同结果。"""
    set_seed(42)
    t1 = torch.rand(1).item()

    set_seed(123)
    t2 = torch.rand(1).item()

    assert t1 != t2


def test_category_map_completeness():
    """CATEGORY_MAP 应包含参考实现使用的核心类别。"""
    required = [
        "Industrial_and_Scientific",
        "Office_Products",
        "Toys_and_Games",
        "Sports",
        "Books",
    ]
    for cat in required:
        assert cat in CATEGORY_MAP, f"{cat} missing from CATEGORY_MAP"
        assert isinstance(CATEGORY_MAP[cat], str)
        assert len(CATEGORY_MAP[cat]) > 0


def test_category_map_values_are_strings():
    for key, val in CATEGORY_MAP.items():
        assert isinstance(key, str)
        assert isinstance(val, str)


def test_base_config_defaults():
    cfg = BaseConfig()
    assert cfg.seed == 42
    assert cfg.device == "cuda"
    assert cfg.output_dir == ""


def test_data_process_config_defaults():
    cfg = DataProcessConfig()
    assert cfg.split_method == "TO"
    assert cfg.max_history_len == 50
    assert cfg.k_core == 5


def test_data_process_config_override():
    cfg = DataProcessConfig(category="All_Beauty", k_core=10, split_method="LOO")
    assert cfg.category == "All_Beauty"
    assert cfg.k_core == 10
    assert cfg.split_method == "LOO"


def test_text_emb_config_defaults():
    cfg = TextEmbConfig()
    assert cfg.batch_size == 256
    assert "MiniLM" in cfg.model_name or cfg.model_name != ""


def test_rqvae_config_defaults():
    cfg = RQVAEConfig()
    assert cfg.num_levels == 3
    assert cfg.codebook_size == 256
    assert cfg.layers == [2048, 1024, 512, 256, 128, 64]
    assert cfg.beta == 0.25
    assert cfg.kmeans_init is True


def test_sft_config_defaults():
    cfg = SFTConfig()
    assert cfg.batch_size == 128
    assert cfg.micro_batch_size == 4
    assert cfg.num_epochs == 10
    assert cfg.freeze_llm is False


def test_rl_config_defaults():
    cfg = RLConfig()
    assert cfg.reward_type == "rule"
    assert cfg.num_generations == 16
    assert cfg.beta == 0.04


def test_eval_config_defaults():
    cfg = EvalConfig()
    assert cfg.num_beams == 50
    assert 20 in cfg.k_values
    assert 10 in cfg.k_values


def test_sasrec_config_defaults():
    cfg = SASRecConfig()
    assert cfg.model == "SASRec"
    assert cfg.epoch == 500
    assert cfg.batch_size == 1024


def test_dataclass_serialization():
    """dataclass 应支持 __dict__ 序列化。"""
    cfg = DataProcessConfig(category="All_Beauty", k_core=5)
    d = cfg.__dict__
    assert d["category"] == "All_Beauty"
    assert d["k_core"] == 5
    assert "split_method" in d


def test_config_override_from_dict():
    """支持通过解包字典覆盖配置。"""
    overrides = {"seed": 99, "output_dir": "/tmp/test"}
    cfg = DataProcessConfig(**overrides)
    assert cfg.seed == 99
    assert cfg.output_dir == "/tmp/test"
