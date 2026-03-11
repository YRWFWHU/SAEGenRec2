"""SAE SID 扩展系数 × K 值对比实验。

计算不同扩展系数（3x/4x/6x/8x/16x）和不同 K（3~10）下的：
  - 有序碰撞率（去重前）：top-K 特征有序列表完全相同视为碰撞
  - 有序碰撞率（去重后）：运行 last-token 替换去重算法后
  - 无序集合碰撞率：top-K 特征排序后作为集合，集合相同视为碰撞（无需去重）

输出：results/expansion_k_compare/report.md
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
CHECKPOINT_BASE = "models/gated_sae_compare"
OUTPUT_DIR = "results/expansion_k_compare"
REPORT_PATH = "results/expansion_k_compare/report.md"

EXPANSION_FACTORS = [3, 4, 6, 8, 16]
K_VALUES = list(range(3, 11))  # 3 到 10

SAE_BASE_CONFIG = dict(
    l1_coefficient=0.04,
    lr=3e-4,
    total_training_samples=2_000_000,
    train_batch_size=4096,
    device="cuda:0",
    seed=42,
    max_dedup_iters=30,
)
# ──────────────────────────────────────────────────────────


def ordered_collision_rate(item_sids: list[list[int]]) -> float:
    """有序碰撞率：top-K 特征列表（有序）完全相同的物品比例。"""
    counts = Counter(tuple(s) for s in item_sids)
    n = len(item_sids)
    return sum(cnt for cnt in counts.values() if cnt > 1) / n


def unordered_collision_rate(item_sids: list[list[int]]) -> float:
    """无序碰撞率：top-K 特征集合（排序后）相同的物品比例。"""
    counts = Counter(tuple(sorted(s)) for s in item_sids)
    n = len(item_sids)
    return sum(cnt for cnt in counts.values() if cnt > 1) / n


def dedup(
    item_sids: list[list[int]],
    topk_indices: torch.Tensor,
    k: int,
    max_iters: int = 30,
) -> list[list[int]]:
    """last-token 替换去重算法（全局占用集合版），返回去重后的副本。"""
    sids = [list(s) for s in item_sids]
    fetch_k = topk_indices.shape[1]

    for _ in range(max_iters):
        sid_to_items: dict[str, list[int]] = {}
        for item_id, sid in enumerate(sids):
            sid_to_items.setdefault(str(sid), []).append(item_id)

        groups = [g for g in sid_to_items.values() if len(g) > 1]
        if not groups:
            break

        global_used: set[tuple] = {tuple(s) for s in sids}

        for group in groups:
            for item_id in group[1:]:
                old_sid = tuple(sids[item_id])
                prefix = sids[item_id][:-1]
                for col in range(k, fetch_k):
                    candidate = int(topk_indices[item_id, col].item())
                    new_sid = tuple(prefix + [candidate])
                    if new_sid not in global_used:
                        global_used.discard(old_sid)
                        sids[item_id][-1] = candidate
                        global_used.add(new_sid)
                        break

    return sids


def dedup_unordered(
    item_sids: list[list[int]],
    topk_indices: torch.Tensor,
    k: int,
    max_iters: int = 30,
) -> list[list[int]]:
    """无序集合去重算法（全局占用集合版）。

    用 frozenset 判断碰撞，替换集合中激活值最弱的特征（topk 中排名最靠后的那个）。
    由于比较的是集合而非有序列表，搜索空间与有序去重相同，但碰撞判定更宽松。
    """
    sids = [list(s) for s in item_sids]
    fetch_k = topk_indices.shape[1]

    for _ in range(max_iters):
        set_to_items: dict[frozenset, list[int]] = {}
        for item_id, sid in enumerate(sids):
            key = frozenset(sid)
            set_to_items.setdefault(key, []).append(item_id)

        groups = [g for g in set_to_items.values() if len(g) > 1]
        if not groups:
            break

        global_used: set[frozenset] = {frozenset(s) for s in sids}

        for group in groups:
            for item_id in group[1:]:
                old_set = frozenset(sids[item_id])
                # 找出当前 SID 中在 topk_indices 中排名最靠后的特征（最弱），替换它
                sid_set = set(sids[item_id])
                weakest_pos = max(
                    range(k), key=lambda pos: next(
                        col for col in range(fetch_k)
                        if int(topk_indices[item_id, col].item()) == sids[item_id][pos]
                    )
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


def run_expansion(ef: int, feature_acts: torch.Tensor, topk_cache: dict) -> list[dict]:
    """给定已编码的 feature_acts，计算该 ef 下所有 K 的指标。

    topk_cache: 缓存不同 fetch_k 的 topk_indices，避免重复计算。
    """
    rows = []
    for k in K_VALUES:
        fetch_k = min(k + 512, feature_acts.shape[1])
        if fetch_k not in topk_cache:
            _, topk_indices = torch.topk(feature_acts, k=fetch_k, dim=1)
            topk_cache[fetch_k] = topk_indices
        topk_indices = topk_cache[fetch_k]

        raw_sids = topk_indices[:, :k].tolist()

        pre = ordered_collision_rate(raw_sids)
        unord_pre = unordered_collision_rate(raw_sids)
        dedup_sids_ord = dedup(raw_sids, topk_indices, k, SAE_BASE_CONFIG["max_dedup_iters"])
        post = ordered_collision_rate(dedup_sids_ord)
        dedup_sids_unord = dedup_unordered(raw_sids, topk_indices, k, SAE_BASE_CONFIG["max_dedup_iters"])
        unord_post = unordered_collision_rate(dedup_sids_unord)

        rows.append(
            {
                "ef": ef,
                "k": k,
                "ordered_pre_pct": pre * 100,
                "ordered_post_pct": post * 100,
                "unordered_pre_pct": unord_pre * 100,
                "unordered_post_pct": unord_post * 100,
            }
        )
        print(
            f"  ef={ef:2d}x  K={k}:  "
            f"ord_pre={pre*100:6.2f}%  ord_post={post*100:6.2f}%  "
            f"unord_pre={unord_pre*100:6.2f}%  unord_post={unord_post*100:6.2f}%"
        )
    return rows


def encode_all(sae, embeddings: np.ndarray, batch_size: int = 512) -> torch.Tensor:
    device = next(sae.parameters()).device
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.from_numpy(embeddings[i : i + batch_size]).float().to(device)
            all_acts.append(sae.encode(batch).cpu())
    return torch.cat(all_acts, dim=0)


def make_table(all_rows: list[dict], metric_key: str, fmt: str = ".2f") -> list[str]:
    """生成 ef × K 的二维 Markdown 表格。"""
    efs = sorted({r["ef"] for r in all_rows})
    ks = sorted({r["k"] for r in all_rows})

    lookup = {(r["ef"], r["k"]): r[metric_key] for r in all_rows}

    header = "| ef \\ K | " + " | ".join(str(k) for k in ks) + " |"
    sep = "|--------|" + "|".join([":------:"] * len(ks)) + "|"
    lines = [header, sep]
    for ef in efs:
        cells = []
        for k in ks:
            v = lookup.get((ef, k))
            cells.append(f"{v:{fmt}}%" if v is not None else "N/A")
        lines.append(f"| {ef}x | " + " | ".join(cells) + " |")
    return lines


def generate_report(all_rows: list[dict]) -> str:
    lines = ["# SAE SID 扩展系数 × K 值对比报告\n"]
    lines.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 实验配置\n")
    lines.append("| 参数 | 值 |")
    lines.append("|------|----|")
    lines.append(f"| 数据集 | Beauty (Amazon 2015, 全量) |")
    lines.append(f"| 嵌入模型 | Qwen3-Embedding-0.6B |")
    lines.append(f"| l1_coefficient | {SAE_BASE_CONFIG['l1_coefficient']} |")
    lines.append(f"| 训练样本数 | {SAE_BASE_CONFIG['total_training_samples']:,} |")
    lines.append(f"| 扩展系数 | {EXPANSION_FACTORS} |")
    lines.append(f"| K 范围 | {min(K_VALUES)}~{max(K_VALUES)} |")
    lines.append("")

    for title, key in [
        ("有序碰撞率（去重前）", "ordered_pre_pct"),
        ("有序碰撞率（去重后）", "ordered_post_pct"),
        ("无序集合碰撞率（去重前）", "unordered_pre_pct"),
        ("无序集合碰撞率（去重后）", "unordered_post_pct"),
    ]:
        lines.append(f"## {title}\n")
        lines.extend(make_table(all_rows, key))
        lines.append("")

    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw = np.load(EMBEDDING_PATH).astype(np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    n_items = raw.shape[0]
    print(f"Loaded embeddings: {raw.shape}")

    # 先检查/复用 ef=8 的已有 checkpoint（路径不同，需映射）
    EF_CHECKPOINT_MAP = {
        8: os.path.join(CHECKPOINT_BASE, "Beauty"),  # 已有
    }

    all_rows = []

    for ef in EXPANSION_FACTORS:
        checkpoint = EF_CHECKPOINT_MAP.get(ef, os.path.join(CHECKPOINT_BASE, f"ef{ef}"))

        if not os.path.exists(os.path.join(checkpoint, "sae_weights.safetensors")):
            print(f"\n{'='*60}")
            print(f"  Training GatedSAE (expansion_factor={ef}x)")
            print(f"{'='*60}")
            from SAEGenRec.sid_builder.gated_sae import gated_sae_train

            gated_sae_train(
                embedding_path=EMBEDDING_PATH,
                output_dir=checkpoint,
                expansion_factor=ef,
                k=max(K_VALUES),
                **SAE_BASE_CONFIG,
            )
        else:
            print(f"\n[Skip] Checkpoint exists: {checkpoint}")

        # 检查 training_config 中的 l1_coefficient 是否匹配
        cfg_path = os.path.join(checkpoint, "training_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            if abs(cfg.get("l1_coefficient", 0) - SAE_BASE_CONFIG["l1_coefficient"]) > 1e-6:
                print(
                    f"  ⚠️  Checkpoint l1={cfg['l1_coefficient']} != current {SAE_BASE_CONFIG['l1_coefficient']}, "
                    "results may be inconsistent."
                )

        from sae_lens import SAE

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        sae = SAE.load_from_disk(checkpoint, device=device)
        sae.eval()
        print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae} (ef={sae.cfg.d_sae // sae.cfg.d_in}x)")

        print(f"  Encoding {n_items} items...")
        feature_acts = encode_all(sae, raw)

        topk_cache: dict[int, torch.Tensor] = {}
        rows = run_expansion(ef, feature_acts, topk_cache)
        all_rows.extend(rows)

        del sae, feature_acts
        torch.cuda.empty_cache()

    # 保存原始数据
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(all_rows, f, indent=2)

    report = generate_report(all_rows)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved: {REPORT_PATH}")
    print("\n" + report)


if __name__ == "__main__":
    main()
