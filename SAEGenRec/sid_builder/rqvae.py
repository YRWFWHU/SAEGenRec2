"""RQ-VAE 训练入口。

Ported from references/MiniOneRec/rq/rqvae.py + trainer.py
"""

import heapq
import logging
import os
from datetime import datetime
from time import time

import numpy as np
import torch
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from SAEGenRec.config import RQVAEConfig
from SAEGenRec.sid_builder.models.rqvae import RQVAE


class EmbDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.embeddings = np.load(data_path).astype(np.float32)

        nan_mask = np.isnan(self.embeddings)
        if nan_mask.any():
            logger.warning(f"Found {nan_mask.sum()} NaN values in embeddings, replacing with 0")
            self.embeddings[nan_mask] = 0.0

        inf_mask = np.isinf(self.embeddings)
        if inf_mask.any():
            logger.warning(f"Found {inf_mask.sum()} Inf values in embeddings, replacing with 0")
            self.embeddings[inf_mask] = 0.0

        self.dim = self.embeddings.shape[-1]
        logger.info(f"Loaded embeddings: {self.embeddings.shape}")

    def __getitem__(self, index):
        return torch.FloatTensor(self.embeddings[index])

    def __len__(self):
        return len(self.embeddings)


def _build_optimizer(model, config: RQVAEConfig):
    learner = config.learner.lower()
    lr = config.lr
    wd = config.weight_decay
    params = model.parameters()
    if learner == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    elif learner == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=wd)
    elif learner == "adagrad":
        opt = optim.Adagrad(params, lr=lr, weight_decay=wd)
        dev = torch.device(config.device)
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(dev)
        return opt
    elif learner == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=wd)
    elif learner == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    else:
        logger.warning(f"Unknown optimizer {learner}, using Adam")
        return optim.Adam(params, lr=lr)


def _compute_collision_rate(model, data_loader, device):
    model.eval()
    indices_set = set()
    num_sample = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            num_sample += len(data)
            indices = model.get_indices(data)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for idx in indices:
                code = "-".join(str(int(x)) for x in idx)
                indices_set.add(code)
    collision_rate = (num_sample - len(indices_set)) / num_sample
    return collision_rate


def rqvae_train(
    embedding_path: str = "",
    num_levels: int = 3,
    codebook_size: int = 256,
    layers: str = "2048,1024,512,256,128,64",
    e_dim: int = 32,
    lr: float = 1e-3,
    epochs: int = 5000,
    batch_size: int = 2048,
    beta: float = 0.25,
    sk_epsilon: float = 0.0,
    sk_iters: int = 50,
    kmeans_init: bool = True,
    kmeans_iters: int = 100,
    learner: str = "AdamW",
    lr_scheduler_type: str = "constant",
    warmup_epochs: int = 50,
    eval_step: int = 50,
    weight_decay: float = 0.0,
    dropout_prob: float = 0.0,
    output_dir: str = "models/rqvae",
    device: str = "cuda:0",
    num_workers: int = 4,
    save_limit: int = 5,
    seed: int = 42,
) -> str:
    """训练 RQ-VAE 量化器，返回最优 checkpoint 路径。

    Args:
        embedding_path: item embedding .npy 文件路径
        layers: 逗号分隔的隐藏层大小（字符串形式，fire 兼容）
        返回: 最优 checkpoint 路径（best_collision_model.pth）
    """
    if isinstance(layers, str):
        layer_list = [int(x) for x in layers.split(",")]
    else:
        layer_list = list(layers)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {dev}")

    data = EmbDataset(embedding_path)
    data_loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    num_emb_list = [codebook_size] * num_levels
    sk_epsilons = [sk_epsilon] * num_levels

    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=num_emb_list,
        e_dim=e_dim,
        layers=layer_list,
        dropout_prob=dropout_prob,
        beta=beta,
        kmeans_init=kmeans_init,
        kmeans_iters=kmeans_iters,
        sk_epsilons=sk_epsilons,
        sk_iters=sk_iters,
    ).to(dev)

    optimizer = _build_optimizer(model, RQVAEConfig(
        lr=lr, learner=learner, weight_decay=weight_decay,
        device=device
    ))

    data_num = len(data_loader)
    warmup_steps = warmup_epochs * data_num
    max_steps = epochs * data_num

    if lr_scheduler_type.lower() == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)

    # Checkpoint directory
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = os.path.join(output_dir, run_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = np.inf
    best_collision_rate = np.inf
    best_loss_ckpt = os.path.join(ckpt_dir, "best_loss_model.pth")
    best_collision_ckpt = os.path.join(ckpt_dir, "best_collision_model.pth")

    best_save_heap = []
    newest_save_queue = []

    for epoch_idx in range(epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        t0 = time()

        for data_batch in tqdm(data_loader, desc=f"Train {epoch_idx}", ncols=100):
            data_batch = data_batch.to(dev)
            optimizer.zero_grad()
            out, rq_loss, _ = model(data_batch)
            loss, loss_recon = model.compute_loss(out, rq_loss, xs=data_batch)
            if torch.isnan(loss):
                raise ValueError("Training loss is nan")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_recon += loss_recon.item()

        t1 = time()
        logger.info(
            f"Epoch {epoch_idx} [{t1-t0:.2f}s] loss={total_loss:.4f} recon={total_recon:.4f}"
        )

        if (epoch_idx + 1) % eval_step == 0:
            collision_rate = _compute_collision_rate(model, data_loader, dev)
            logger.info(f"Epoch {epoch_idx} collision_rate={collision_rate:.6f}")

            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(
                    {"epoch": epoch_idx, "state_dict": model.state_dict()},
                    best_loss_ckpt,
                )

            ckpt_path = os.path.join(
                ckpt_dir, f"epoch_{epoch_idx}_collision_{collision_rate:.4f}.pth"
            )
            torch.save(
                {
                    "epoch": epoch_idx,
                    "collision_rate": collision_rate,
                    "state_dict": model.state_dict(),
                    "config": {
                        "in_dim": data.dim,
                        "num_emb_list": num_emb_list,
                        "e_dim": e_dim,
                        "layers": layer_list,
                        "dropout_prob": dropout_prob,
                        "beta": beta,
                        "kmeans_init": kmeans_init,
                        "kmeans_iters": kmeans_iters,
                        "sk_epsilons": sk_epsilons,
                        "sk_iters": sk_iters,
                    },
                },
                ckpt_path,
            )

            if collision_rate < best_collision_rate:
                best_collision_rate = collision_rate
                import shutil

                shutil.copy2(ckpt_path, best_collision_ckpt)

            # Manage checkpoint pool
            now_save = (-collision_rate, ckpt_path)
            if len(newest_save_queue) < save_limit:
                newest_save_queue.append(now_save)
                heapq.heappush(best_save_heap, now_save)
            else:
                old_save = newest_save_queue.pop(0)
                newest_save_queue.append(now_save)
                if collision_rate < -best_save_heap[0][0]:
                    bad_save = heapq.heappop(best_save_heap)
                    heapq.heappush(best_save_heap, now_save)
                    if bad_save not in newest_save_queue and os.path.exists(bad_save[1]):
                        os.remove(bad_save[1])
                if old_save not in best_save_heap and os.path.exists(old_save[1]):
                    os.remove(old_save[1])

    logger.info(
        f"Training done. best_loss={best_loss:.4f}, best_collision_rate={best_collision_rate:.6f}"
    )
    return best_collision_ckpt


# ---- SIDMethod adapter ----

from SAEGenRec.sid_builder.base import SIDMethod
from SAEGenRec.sid_builder.registry import register_sid_method


@register_sid_method("rqvae")
class RQVAEMethod(SIDMethod):
    """RQ-VAE SID 生成方法适配器。"""

    name = "rqvae"
    default_k = 3
    token_format = "auto"  # 使用位置前缀 a/b/c/...

    def train(self, embedding_path: str, output_dir: str = "models/rqvae", **config) -> str:
        return rqvae_train(embedding_path=embedding_path, output_dir=output_dir, **config)

    def generate(
        self,
        checkpoint: str,
        embedding_path: str,
        output_path: str,
        k: int = None,
        token_format: str = "auto",
    ) -> str:
        from SAEGenRec.sid_builder.generate_indices import generate_indices

        return generate_indices(
            checkpoint=checkpoint,
            embedding_path=embedding_path,
            output_path=output_path,
            token_format=token_format,
        )
