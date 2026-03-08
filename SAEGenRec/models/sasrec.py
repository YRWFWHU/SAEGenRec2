"""SASRec, GRU, Caser CF 模型。

Ported from references/MiniOneRec/sasrec.py and SASRecModules_ori.py
"""

import ast
import copy
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Self-attention modules (from SASRecModules_ori.py)
# ---------------------------------------------------------------------------


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0

        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        Q = self.linear_q(queries)
        K = self.linear_k(keys)
        V = self.linear_v(keys)

        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)

        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size**0.5

        # Key masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)
        key_paddings = torch.ones_like(matmul_output) * (-(2**32) + 1)
        matmul_output_m1 = torch.where(
            torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output
        )

        # Causality masking
        diag_vals = torch.ones_like(matmul_output[0, :, :])
        tril = torch.tril(diag_vals)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)
        causality_paddings = torch.ones_like(causality_mask) * (-(2**32) + 1)
        matmul_output_m2 = torch.where(
            torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1
        )

        matmul_output_sm = self.softmax(matmul_output_m2)

        # Query masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])
        matmul_output_qm = matmul_output_sm * query_mask

        matmul_output_dropout = self.dropout(matmul_output_qm)
        output_ws = torch.bmm(matmul_output_dropout, V_)

        output = torch.cat(
            torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2
        )
        return output + queries


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------


class GRU(nn.Module):
    def __init__(self, hidden_size: int, item_num: int, state_size: int, gru_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.item_embeddings = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.s_fc = nn.Linear(hidden_size, item_num)

    def forward(self, states: torch.Tensor, len_states) -> torch.Tensor:
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, len_states, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        return self.s_fc(hidden)

    def forward_eval(self, states: torch.Tensor, len_states) -> torch.Tensor:
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, len_states.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        return self.s_fc(hidden)


class Caser(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        item_num: int,
        state_size: int,
        num_filters: int,
        filter_sizes,
        dropout_rate: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = filter_sizes if isinstance(filter_sizes, list) else eval(filter_sizes)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        self.item_embeddings = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (i, hidden_size)) for i in self.filter_sizes]
        )
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        self.vertical_cnn = nn.Conv2d(1, 1, (state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        self.num_filters_total = num_filters * len(self.filter_sizes)
        self.s_fc = nn.Linear(hidden_size + self.num_filters_total, item_num)
        self.dropout = nn.Dropout(dropout_rate)

    def _encode(self, states: torch.Tensor) -> torch.Tensor:
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)

        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = F.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = F.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)
        h_pool_flat = torch.cat(pooled_outputs, 1).view(-1, self.num_filters_total)

        v_out = F.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        return self.s_fc(self.dropout(out))

    def forward(self, states: torch.Tensor, len_states) -> torch.Tensor:
        return self._encode(states)

    def forward_eval(self, states: torch.Tensor, len_states) -> torch.Tensor:
        return self._encode(states)


class SASRec(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        item_num: int,
        state_size: int,
        dropout: float,
        device,
        num_heads: int = 1,
    ):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.item_embeddings = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(num_embeddings=state_size, embedding_dim=hidden_size)

        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

    def _encode(self, states: torch.Tensor, len_states: torch.Tensor) -> torch.Tensor:
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(
            torch.arange(self.state_size).to(self.device)
        )
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        indices = (len_states - 1).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        state_hidden = torch.gather(ff_out, 1, indices)
        return self.s_fc(state_hidden).squeeze()

    def forward(self, states: torch.Tensor, len_states: torch.Tensor) -> torch.Tensor:
        return self._encode(states, len_states)

    def forward_eval(self, states: torch.Tensor, len_states: torch.Tensor) -> torch.Tensor:
        return self._encode(states, len_states)


# ---------------------------------------------------------------------------
# Dataset and training utilities
# ---------------------------------------------------------------------------


class RecDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame):
        self.data = data_df

    def __getitem__(self, i):
        row = self.data.iloc[i]
        return (
            torch.tensor(row["seq"], dtype=torch.long),
            torch.tensor(row["len_seq"], dtype=torch.long),
            torch.tensor(row["next"], dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)


def _load_csv(csv_path: str, item_num: int, seq_size: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[["history_item_id", "item_id"]].rename(
        columns={"history_item_id": "seq", "item_id": "next"}
    )
    df["seq"] = df["seq"].apply(ast.literal_eval)
    df["len_seq"] = df["seq"].apply(len)
    df["seq"] = df["seq"].apply(lambda x: x + [item_num] * (seq_size - len(x)))
    return df


def _evaluate(
    model: nn.Module,
    csv_path: str,
    item_num: int,
    seq_size: int,
    device,
    topk: List[int],
    batch_size: int = 1024,
) -> tuple:
    df = _load_csv(csv_path, item_num, seq_size)
    hit_all = [0] * len(topk)
    ndcg_all = [0] * len(topk)
    total = len(df)

    model.eval()
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = df.iloc[i : i + batch_size]
            seq = torch.LongTensor(list(batch["seq"])).to(device)
            len_seq = torch.tensor(list(batch["len_seq"])).to(device)
            target = torch.LongTensor(list(batch["next"])).to(device)

            pred = model.forward_eval(seq, len_seq)
            rank_list = pred.shape[1] - 1 - torch.argsort(torch.argsort(pred))
            target_rank = torch.gather(rank_list, 1, target.view(-1, 1)).view(-1)
            ndcg_temp_full = 1.0 / torch.log2(target_rank.float() + 2)
            for j, k in enumerate(topk):
                mask = (target_rank < k).float()
                hit_all[j] += mask.sum().item()
                ndcg_all[j] += (ndcg_temp_full * mask).sum().item()

    hr_list = [hit_all[j] / total for j in range(len(topk))]
    ndcg_list = [ndcg_all[j] / total for j in range(len(topk))]
    return hr_list, ndcg_list


def sasrec_train(
    model_type: str = "SASRec",
    train_csv: str = "",
    valid_csv: str = "",
    test_csv: str = "",
    output_dir: str = "models/sasrec",
    hidden_factor: int = 32,
    num_filters: int = 16,
    filter_sizes: str = "[2,3,4]",
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    l2_decay: float = 1e-5,
    batch_size: int = 1024,
    num_epochs: int = 500,
    eval_num: int = 1,
    early_stop: int = 20,
    topk: str = "1,5,10,20",
    loss_type: str = "bce",
    seq_size: int = 10,
    seed: int = 1,
    save_logits: bool = False,
) -> dict:
    """Train SASRec/GRU/Caser collaborative filtering model with early stopping on NDCG@20.

    Args:
        model_type: "SASRec", "GRU", or "Caser"
        train_csv: training CSV with columns history_item_id, item_id
        valid_csv: validation CSV
        test_csv: test CSV
        output_dir: directory to save best model checkpoint and metrics
        hidden_factor: embedding dimension
        early_stop: epochs without improvement before stopping

    Returns:
        metrics dict with HR@K and NDCG@K for all K in topk
    """
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    topk_list = [int(k.strip()) for k in topk.split(",")]

    # Load training data to determine item_num
    train_df_raw = pd.read_csv(train_csv)
    # item_num = max item_id (0-indexed so item_num is num items, padding idx = item_num)
    item_num = int(train_df_raw["item_id"].max()) + 1

    logger.info(f"item_num={item_num}, seq_size={seq_size}, model={model_type}")

    if model_type == "SASRec":
        model = SASRec(hidden_factor, item_num, seq_size, dropout_rate, device)
    elif model_type == "GRU":
        model = GRU(hidden_factor, item_num, seq_size)
    elif model_type == "Caser":
        model = Caser(hidden_factor, item_num, seq_size, num_filters, filter_sizes, dropout_rate)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=l2_decay)

    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    train_df = _load_csv(train_csv, item_num, seq_size)
    train_dataset = RecDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    ndcg_max = 0.0
    best_epoch = 0
    early_stop_count = 0
    best_model = None
    best_hr = []
    best_ndcg = []

    num_batches = len(train_loader)
    total_step = 0

    for epoch in range(num_epochs):
        model.train()
        for seq, len_seq, target in train_loader:
            seq = seq.to(device)
            target = target.to(device)
            if model_type == "GRU":
                len_seq = len_seq.cpu()
            else:
                len_seq = len_seq.to(device)

            target_neg = torch.randint(0, item_num, target.shape, device=device)

            optimizer.zero_grad()
            model_output = model.forward(seq, len_seq)

            if loss_type == "bce":
                pos_scores = torch.gather(model_output, 1, target.view(-1, 1))
                neg_scores = torch.gather(model_output, 1, target_neg.view(-1, 1))
                scores = torch.cat([pos_scores, neg_scores], 0)
                labels = torch.cat(
                    [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], 0
                )
                loss = criterion(scores, labels)
            else:
                loss = criterion(model_output, target.long())

            loss.backward()
            optimizer.step()
            total_step += 1

            if total_step % (num_batches * eval_num) == 0:
                val_hr, val_ndcg = _evaluate(
                    model, valid_csv, item_num, seq_size, device, topk_list
                )
                # Use NDCG@20 for early stopping (last entry or max k)
                ndcg_last = val_ndcg[-1]
                logger.info(f"Epoch {epoch} | NDCG@{topk_list[-1]}={ndcg_last:.4f}")

                if ndcg_last > ndcg_max:
                    ndcg_max = ndcg_last
                    best_epoch = epoch
                    early_stop_count = 0
                    best_hr = val_hr
                    best_ndcg = val_ndcg
                    best_model = copy.deepcopy(model)
                else:
                    early_stop_count += 1
                    if early_stop_count > early_stop:
                        logger.info(f"Early stop at epoch {epoch}")
                        break
        else:
            continue
        break

    logger.info(f"Best epoch: {best_epoch}, NDCG@{topk_list[-1]}={ndcg_max:.4f}")

    # Final test evaluation
    test_hr, test_ndcg = _evaluate(best_model, test_csv, item_num, seq_size, device, topk_list)

    metrics = {}
    for i, k in enumerate(topk_list):
        metrics[f"HR@{k}"] = test_hr[i]
        metrics[f"NDCG@{k}"] = test_ndcg[i]
    logger.info(f"Test metrics: {metrics}")

    # Save checkpoint and metrics
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"best_{model_type}_state.pth")
    torch.save(best_model.state_dict(), ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    if save_logits:
        logits_path = os.path.join(output_dir, f"{model_type}_logits.npy")
        test_df = _load_csv(test_csv, item_num, seq_size)
        all_logits = []
        best_model.eval()
        with torch.no_grad():
            for i in range(0, len(test_df), 1024):
                batch = test_df.iloc[i : i + 1024]
                seq = torch.LongTensor(list(batch["seq"])).to(device)
                len_seq = torch.tensor(list(batch["len_seq"])).to(device)
                pred = best_model.forward_eval(seq, len_seq)
                all_logits.append(pred.cpu().numpy())
        np.save(logits_path, np.concatenate(all_logits, axis=0))
        logger.info(f"Saved logits: {logits_path}")

    return metrics
