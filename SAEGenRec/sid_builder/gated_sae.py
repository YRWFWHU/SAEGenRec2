"""GatedSAE 训练：NpyDataProvider + SAELens SAETrainer 包装。"""

from collections.abc import Iterator
import json
import os

from loguru import logger
import numpy as np
from sae_lens import GatedTrainingSAE, GatedTrainingSAEConfig, LoggingConfig, SAETrainer
from sae_lens.config import SAETrainerConfig
import torch

from SAEGenRec.config import set_seed


class NpyDataProvider(Iterator[torch.Tensor]):
    """将 .npy embedding 文件包装为批次迭代器，供 SAELens SAETrainer 使用。

    每次 __next__ 返回 (batch_size, d_in) 的 CPU tensor。
    支持多 epoch 循环，每 epoch 随机打乱顺序。
    NaN/Inf 值被替换为 0。
    """

    def __init__(self, embedding_path: str, batch_size: int, seed: int = 42):
        raw = np.load(embedding_path).astype(np.float32)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        self._data = torch.from_numpy(raw)
        self._n = self._data.shape[0]
        self._batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._indices: np.ndarray = np.arange(self._n)
        self._pos: int = 0
        self._shuffle_epoch()

    def _shuffle_epoch(self) -> None:
        self._rng.shuffle(self._indices)
        self._pos = 0

    def __iter__(self) -> "NpyDataProvider":
        return self

    def __next__(self) -> torch.Tensor:
        if self._pos >= self._n:
            self._shuffle_epoch()
        end = min(self._pos + self._batch_size, self._n)
        batch_idx = self._indices[self._pos : end]
        self._pos = end
        return self._data[batch_idx]

    @property
    def d_in(self) -> int:
        return self._data.shape[1]

    def __len__(self) -> int:
        return self._n


def gated_sae_train(
    embedding_path: str = "",
    expansion_factor: int = 4,
    k: int = 8,
    l1_coefficient: float = 1.0,
    lr: float = 3e-4,
    total_training_samples: int = 1_000_000,
    train_batch_size: int = 4096,
    output_dir: str = "models/gated_sae",
    device: str = "cuda:0",
    seed: int = 42,
    max_dedup_iters: int = 20,
) -> str:
    """训练 GatedSAE 模型并保存 checkpoint。

    Args:
        embedding_path: 输入 .npy embedding 文件路径
        expansion_factor: 字典大小 = d_in × expansion_factor
        k: top-K 特征数（SID token 数，仅保存到 training_config.json）
        l1_coefficient: L1 稀疏性惩罚权重
        lr: 学习率
        total_training_samples: 总训练样本数（含多 epoch 重复）
        train_batch_size: 训练批次大小
        output_dir: 输出目录（保存 checkpoint）
        device: 训练设备
        seed: 随机种子
        max_dedup_iters: SID 去重最大迭代次数（仅保存到 training_config.json）

    Returns:
        output_dir 路径
    """
    set_seed(seed)

    # 创建 DataProvider（自动检测 d_in，避免二次读文件）
    data_provider = NpyDataProvider(embedding_path, batch_size=train_batch_size, seed=seed)
    d_in = data_provider.d_in
    d_sae = d_in * expansion_factor
    logger.info(f"Embeddings: n={len(data_provider)}, d_in={d_in}, d_sae={d_sae}")

    # 创建 GatedTrainingSAE
    dev = device if torch.cuda.is_available() else "cpu"
    sae_cfg = GatedTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        l1_coefficient=l1_coefficient,
        device=dev,
        dtype="float32",
    )
    sae = GatedTrainingSAE(sae_cfg)

    # 创建 SAETrainer（禁用 wandb）
    trainer_cfg = SAETrainerConfig(
        n_checkpoints=0,
        checkpoint_path=None,
        save_final_checkpoint=False,
        total_training_samples=total_training_samples,
        device=dev,
        autocast=False,
        lr=lr,
        lr_end=lr / 10,
        lr_scheduler_name="constant",
        lr_warm_up_steps=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_decay_steps=0,
        n_restart_cycles=1,
        train_batch_size_samples=train_batch_size,
        dead_feature_window=1000,
        feature_sampling_window=2000,
        logger=LoggingConfig(log_to_wandb=False),
    )
    trainer = SAETrainer(cfg=trainer_cfg, sae=sae, data_provider=data_provider)

    logger.info(
        f"Training GatedSAE: d_in={d_in}, d_sae={d_sae}, "
        f"l1={l1_coefficient}, lr={lr}, samples={total_training_samples:,}"
    )

    trained_sae = trainer.fit()

    # 训练后统计
    dead_count = trainer.dead_neurons.sum().item()
    if trainer.n_frac_active_samples > 0:
        l0_mean = (trainer.act_freq_scores / trainer.n_frac_active_samples).sum().item()
    else:
        l0_mean = 0.0
    logger.info(
        f"Training complete. Dead neurons: {dead_count}/{d_sae} "
        f"({100 * dead_count / d_sae:.1f}%), mean L0: {l0_mean:.2f}"
    )

    # 保存 checkpoint
    os.makedirs(output_dir, exist_ok=True)
    trained_sae.save_inference_model(output_dir)
    logger.info(f"Saved SAELens checkpoint: {output_dir}/sae_weights.safetensors + cfg.json")

    # 保存项目特定配置
    training_config = {
        "embedding_path": embedding_path,
        "d_in": d_in,
        "d_sae": d_sae,
        "expansion_factor": expansion_factor,
        "k": k,
        "l1_coefficient": l1_coefficient,
        "lr": lr,
        "total_training_samples": total_training_samples,
        "train_batch_size": train_batch_size,
        "device": device,
        "seed": seed,
        "max_dedup_iters": max_dedup_iters,
    }
    training_config_path = os.path.join(output_dir, "training_config.json")
    with open(training_config_path, "w") as f:
        json.dump(training_config, f, indent=2)
    logger.info(f"Saved training config: {training_config_path}")

    return output_dir
