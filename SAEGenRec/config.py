"""集中配置模块：CATEGORY_MAP、set_seed()、各阶段 dataclass 配置。"""

from dataclasses import dataclass, field
import random
from typing import List, Optional

import numpy as np
import torch

# FR-012: 集中类别名映射，消除参考实现中的重复定义
CATEGORY_MAP = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
    "All_Beauty": "beauty products",
    "Beauty": "beauty products",
    "Clothing_Shoes_and_Jewelry": "clothing shoes and jewelry",
    "Electronics": "electronics",
    "Home_and_Kitchen": "home and kitchen",
    "Movies_and_TV": "movies and tv",
    "CDs_and_Vinyl": "cds and vinyl",
    "Video_Games": "video games",
    "Kindle_Store": "kindle store",
    "Apps_for_Android": "apps for android",
    "Musical_Instruments": "musical instruments",
    "Automotive": "automotive",
    "Pet_Supplies": "pet supplies",
    "Tools_and_Home_Improvement": "tools and home improvement",
    "Patio_Lawn_and_Garden": "patio lawn and garden",
    "Baby": "baby products",
    "Grocery_and_Gourmet_Food": "grocery and gourmet food",
    "Cell_Phones_and_Accessories": "cell phones and accessories",
    "Health_and_Personal_Care": "health and personal care",
}


def set_seed(seed: int) -> None:
    """FR-013: 设置所有随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class BaseConfig:
    """FR-014: 所有配置类的公共基类。"""

    seed: int = 42
    device: str = "cuda"
    output_dir: str = ""


@dataclass
class DataProcessConfig(BaseConfig):
    """数据预处理阶段配置。"""

    category: str = ""
    k_core: int = 5
    st_year: int = 2017
    st_month: int = 10
    ed_year: int = 2018
    ed_month: int = 11
    split_method: str = "TO"  # "TO" (Temporal Order) or "LOO" (Leave-One-Out)
    max_history_len: int = 50
    raw_data_dir: str = "data/raw"
    output_dir: str = "data/interim"


@dataclass
class TextEmbConfig(BaseConfig):
    """文本嵌入阶段配置。"""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    item_json_path: str = ""
    review_json_path: str = ""
    output_dir: str = "data/interim"
    batch_size: int = 256
    device: str = "cuda"


@dataclass
class RQVAEConfig(BaseConfig):
    """RQ-VAE 训练配置。"""

    embedding_path: str = ""
    num_levels: int = 3
    codebook_size: int = 256
    layers: List[int] = field(default_factory=lambda: [2048, 1024, 512, 256, 128, 64])
    e_dim: int = 32
    lr: float = 1e-3
    epochs: int = 5000
    batch_size: int = 2048
    beta: float = 0.25
    sk_epsilon: float = 0.0
    sk_iters: int = 50
    kmeans_init: bool = True
    kmeans_iters: int = 100
    learner: str = "AdamW"
    lr_scheduler_type: str = "constant"
    warmup_epochs: int = 50
    eval_step: int = 50
    weight_decay: float = 0.0
    dropout_prob: float = 0.0
    output_dir: str = "models/rqvae"
    device: str = "cuda:0"
    num_workers: int = 4
    save_limit: int = 5


@dataclass
class SFTConfig(BaseConfig):
    """SFT 训练配置。"""

    model_name: str = ""
    train_csv: str = ""
    valid_csv: str = ""
    info_file: str = ""
    sid_index_path: str = ""
    item_meta_path: str = ""
    output_dir: str = "models/sft"
    batch_size: int = 128
    micro_batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 3e-4
    cutoff_len: int = 512
    group_by_length: bool = False
    freeze_llm: bool = False
    train_from_scratch: bool = False
    wandb_project: str = ""
    wandb_run_name: str = ""
    resume_from_checkpoint: Optional[str] = None
    deepspeed: Optional[str] = None
    sample: int = -1


@dataclass
class RLConfig(BaseConfig):
    """RL (GRPO) 训练配置。"""

    model_path: str = ""
    train_csv: str = ""
    eval_csv: str = ""
    info_file: str = ""
    sid_index_path: str = ""
    item_meta_path: str = ""
    cf_path: str = ""
    ada_path: str = ""
    output_dir: str = "models/rl"
    train_batch_size: int = 32
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 1e-6
    beta: float = 0.04
    temperature: float = 1.0
    num_generations: int = 16
    eval_step: float = 0.199
    reward_type: str = "rule"  # rule, ranking, semantic, sasrec
    beam_search: bool = False
    test_beam: int = 20
    dynamic_sampling: bool = False
    add_gt: bool = False
    mask_all_zero: bool = False
    sync_ref_model: bool = False
    test_during_training: bool = True
    sample_train: bool = False
    dapo: bool = False
    gspo: bool = False
    deepspeed: Optional[str] = None
    wandb_project: str = ""
    wandb_run_name: str = ""


@dataclass
class EvalConfig(BaseConfig):
    """评估配置。"""

    model_path: str = ""
    test_csv: str = ""
    info_file: str = ""
    output_dir: str = "results"
    batch_size: int = 4
    num_beams: int = 50
    max_new_tokens: int = 256
    length_penalty: float = 0.0
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])


@dataclass
class GatedSAEConfig(BaseConfig):
    """GatedSAE 训练和 SID 生成配置。"""

    embedding_path: str = ""
    d_in: int = 384
    expansion_factor: int = 4
    k: int = 8
    l1_coefficient: float = 1.0
    lr: float = 3e-4
    total_training_samples: int = 1_000_000
    train_batch_size: int = 4096
    output_dir: str = "models/gated_sae"
    device: str = "cuda:0"
    seed: int = 42
    max_dedup_iters: int = 20


@dataclass
class VisualEmbConfig(BaseConfig):
    """视觉特征提取配置。"""

    vision_model: str = "openai/clip-vit-base-patch32"
    batch_size: int = 64
    device: str = "cuda"
    data_dir: str = "data/interim"
    image_dir: str = ""  # 默认为 data/interim/{category}/images


@dataclass
class PrepSFTConfig(BaseConfig):
    """prepare_sft 数据预构建配置。"""

    category: str = ""
    sid_type: str = "rqvae"
    task: str = "sid_seq"
    dataset: str = "Amazon"
    prompt_template: str = ""  # 空字符串时按 task 自动查找默认模板
    data_dir: str = "data/processed"
    interim_dir: str = "data/interim"
    overwrite: bool = False
    tasks: str = ""  # 多任务混合：'sid_seq+item_feat'
    task_weights: str = ""  # 混合权重：'0.7,0.3'


@dataclass
class SASRecConfig(BaseConfig):
    """SASRec / GRU / Caser 协同过滤模型训练配置。"""

    data: str = ""
    train_csv: str = ""
    valid_csv: str = ""
    test_csv: str = ""
    output_dir: str = "models/sasrec"
    model: str = "SASRec"  # SASRec, GRU, Caser
    epoch: int = 500
    batch_size: int = 1024
    hidden_factor: int = 32
    num_filters: int = 16
    filter_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])
    lr: float = 0.001
    l2_decay: float = 1e-5
    dropout_rate: float = 0.3
    early_stop: int = 20
    eval_num: int = 1
    loss_type: str = "bce"
    sample_num: int = 65536
    save_flag: int = 1
    cuda: int = 0
    r_click: float = 0.2
    r_buy: float = 1.0
    alpha: float = 0.0
    beta: float = 1.0
    debug: bool = False
