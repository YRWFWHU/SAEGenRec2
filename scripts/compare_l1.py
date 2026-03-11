"""L1 系数对碰撞率影响实验（K=6, ef=8 固定）。

扫描不同 l1_coefficient，记录每个设置下的：
  - 有序碰撞率（去重前/后）
  - 无序集合碰撞率（去重前/后）
  - 死亡神经元比例 & Mean L0（SAE 质量指标）

输出：results/l1_compare/report.md
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ────────────────────────── 配置 ──────────────────────────
EMBEDDING_PATH = "data/interim/Beauty.emb-Qwen3-Embedding-0.6B-text.npy"
CHECKPOINT_BASE = "models/l1_compare"
OUTPUT_DIR = "results/l1_compare"
REPORT_PATH = "results/l1_compare/report.md"

K = 6
EXPANSION_FACTOR = 8
MAX_DEDUP_ITERS = 30

L1_VALUES = [0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.2, 0.5, 1.0, 2.0]

BASE_CONFIG = dict(
    lr=3e-4,
    total_training_samples=2_000_000,
    train_batch_size=4096,
    device="cuda:0",
    seed=42,
)
# ──────────────────────────────────────────────────────────


def ordered_collision_rate(sids):
    counts = Counter(tuple(s) for s in sids)
    n = len(sids)
    return sum(c for c in counts.values() if c > 1) / n


def unordered_collision_rate(sids):
    counts = Counter(frozenset(s) for s in sids)
    n = len(sids)
    return sum(c for c in counts.values() if c > 1) / n


def dedup_ordered(item_sids, topk_indices, k, max_iters=30):
    sids = [list(s) for s in item_sids]
    fetch_k = topk_indices.shape[1]
    for _ in range(max_iters):
        sid_to_items = {}
        for i, s in enumerate(sids):
            sid_to_items.setdefault(str(s), []).append(i)
        groups = [g for g in sid_to_items.values() if len(g) > 1]
        if not groups:
            break
        global_used = {tuple(s) for s in sids}
        for group in groups:
            for item_id in group[1:]:
                old = tuple(sids[item_id])
                prefix = sids[item_id][:-1]
                for col in range(k, fetch_k):
                    candidate = int(topk_indices[item_id, col].item())
                    new = tuple(prefix + [candidate])
                    if new not in global_used:
                        global_used.discard(old)
                        sids[item_id][-1] = candidate
                        global_used.add(new)
                        break
    return sids


def dedup_unordered(item_sids, topk_indices, k, max_iters=30):
    sids = [list(s) for s in item_sids]
    fetch_k = topk_indices.shape[1]
    for _ in range(max_iters):
        set_to_items = {}
        for i, s in enumerate(sids):
            key = frozenset(s)
            set_to_items.setdefault(key, []).append(i)
        groups = [g for g in set_to_items.values() if len(g) > 1]
        if not groups:
            break
        global_used = {frozenset(s) for s in sids}
        for group in groups:
            for item_id in group[1:]:
                old_set = frozenset(sids[item_id])
                sid_set = set(sids[item_id])
                weakest_pos = max(
                    range(k),
                    key=lambda pos: next(
                        col for col in range(fetch_k)
                        if int(topk_indices[item_id, col].item()) == sids[item_id][pos]
                    ),
                )
                remaining = [sids[item_id][p] for p in range(k) if p != weakest_pos]
                for col in range(k, fetch_k):
                    candidate = int(topk_indices[item_id, col].item())
                    if candidate in sid_set:
                        continue
                    new_set = frozenset(remaining + [candidate])
                    if new_set not in global_used:
                        global_used.discard(old_set)
                        sids[item_id][weakest_pos] = candidate
                        global_used.add(new_set)
                        break
    return sids


def encode_all(sae, embeddings, batch_size=512):
    device = next(sae.parameters()).device
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.from_numpy(embeddings[i: i + batch_size]).float().to(device)
            all_acts.append(sae.encode(batch).cpu())
    return torch.cat(all_acts, dim=0)


def run_one(l1: float, raw: np.ndarray) -> dict:
    from SAEGenRec.sid_builder.gated_sae import gated_sae_train
    from sae_lens import SAE

    checkpoint = os.path.join(CHECKPOINT_BASE, f"l1_{l1:.4f}".replace(".", "_"))

    # 训练（如未存在）
    if not os.path.exists(os.path.join(checkpoint, "sae_weights.safetensors")):
        print(f"\n{'='*55}")
        print(f"  Training  l1_coefficient={l1}")
        print(f"{'='*55}")
        gated_sae_train(
            embedding_path=EMBEDDING_PATH,
            output_dir=checkpoint,
            expansion_factor=EXPANSION_FACTOR,
            k=K,
            l1_coefficient=l1,
            max_dedup_iters=MAX_DEDUP_ITERS,
            **BASE_CONFIG,
        )
    else:
        print(f"  [Skip] checkpoint exists: l1={l1}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sae = SAE.load_from_disk(checkpoint, device=device)
    sae.eval()

    feature_acts = encode_all(sae, raw)
    n_items = raw.shape[0]

    # 死亡神经元 & L0（用训练后统计读 training_config，实际推算）
    n_active_features = (feature_acts > 0).any(dim=0).sum().item()
    dead_count = sae.cfg.d_sae - n_active_features
    dead_pct = 100 * dead_count / sae.cfg.d_sae
    mean_l0 = (feature_acts > 0).float().sum(dim=1).mean().item()

    fetch_k = min(K + 512, sae.cfg.d_sae)
    _, topk_indices = torch.topk(feature_acts, k=fetch_k, dim=1)
    raw_sids = topk_indices[:, :K].tolist()

    ord_pre = ordered_collision_rate(raw_sids)
    unord_pre = unordered_collision_rate(raw_sids)

    dedup_ord = dedup_ordered(raw_sids, topk_indices, K, MAX_DEDUP_ITERS)
    ord_post = ordered_collision_rate(dedup_ord)

    dedup_unord = dedup_unordered(raw_sids, topk_indices, K, MAX_DEDUP_ITERS)
    unord_post = unordered_collision_rate(dedup_unord)

    result = {
        "l1": l1,
        "dead_pct": dead_pct,
        "mean_l0": mean_l0,
        "ord_pre_pct": ord_pre * 100,
        "ord_post_pct": ord_post * 100,
        "unord_pre_pct": unord_pre * 100,
        "unord_post_pct": unord_post * 100,
    }
    print(
        f"  l1={l1:<6}  dead={dead_pct:5.1f}%  L0={mean_l0:7.1f}  "
        f"ord_pre={ord_pre*100:6.2f}%  ord_post={ord_post*100:6.2f}%  "
        f"unord_pre={unord_pre*100:6.2f}%  unord_post={unord_post*100:6.2f}%"
    )

    del sae, feature_acts
    torch.cuda.empty_cache()
    return result


def generate_report(rows: list[dict]) -> str:
    lines = ["# L1 系数对 SID 碰撞率影响实验报告\n"]
    lines.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 实验配置\n")
    lines.append("| 参数 | 值 |")
    lines.append("|------|----|")
    lines.append(f"| K | {K} |")
    lines.append(f"| expansion_factor | {EXPANSION_FACTOR}x |")
    lines.append(f"| 数据集 | Beauty (Amazon 2015) |")
    lines.append(f"| 训练样本数 | {BASE_CONFIG['total_training_samples']:,} |")
    lines.append("")

    lines.append("## 结果\n")
    lines.append(
        "| l1_coefficient | 死亡神经元% | Mean L0 "
        "| 有序碰撞(前) | 有序碰撞(后) "
        "| 无序碰撞(前) | 无序碰撞(后) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['l1']} "
            f"| {r['dead_pct']:.1f}% "
            f"| {r['mean_l0']:.1f} "
            f"| {r['ord_pre_pct']:.2f}% "
            f"| {r['ord_post_pct']:.2f}% "
            f"| {r['unord_pre_pct']:.2f}% "
            f"| {r['unord_post_pct']:.2f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_BASE, exist_ok=True)

    raw = np.load(EMBEDDING_PATH).astype(np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Loaded embeddings: {raw.shape}")

    rows = []
    for l1 in L1_VALUES:
        rows.append(run_one(l1, raw))

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    report = generate_report(rows)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved: {REPORT_PATH}")
    print("\n" + report)


if __name__ == "__main__":
    main()
