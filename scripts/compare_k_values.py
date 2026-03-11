"""SAE SID K值对比实验脚本。

训练一个 GatedSAE，然后用不同 K 值生成 SID 并统计指标，最终输出 Markdown 报告。
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ────────────────────────── 配置 ──────────────────────────
EMBEDDING_PATH = "data/interim/Beauty.emb-Qwen3-Embedding-0.6B-text.npy"
SAE_CHECKPOINT = "models/gated_sae_compare/Beauty"
OUTPUT_DIR = "results/k_compare"
REPORT_PATH = "results/k_compare/report.md"

K_VALUES = [4, 6, 8, 10, 12, 16, 20]

# SAE 训练参数（固定）
SAE_CONFIG = dict(
    expansion_factor=8,
    l1_coefficient=0.04,
    lr=3e-4,
    total_training_samples=2_000_000,
    train_batch_size=4096,
    device="cuda:0",
    seed=42,
    max_dedup_iters=30,
)
# ──────────────────────────────────────────────────────────


def compute_sid_metrics(index_path: str, k: int, n_items: int, d_sae: int) -> dict:
    """从 .index.json 计算 SID 质量指标。"""
    with open(index_path) as f:
        index = json.load(f)

    sids = [tuple(index[str(i)]) for i in range(n_items) if str(i) in index]
    n_valid = len(sids)

    # 碰撞统计（collision_rate = 参与碰撞的物品比例，即没有唯一SID的物品占比）
    from collections import Counter
    sid_counts = Counter(sids)
    n_unique = len(sid_counts)
    colliding_items = sum(cnt for cnt in sid_counts.values() if cnt > 1)
    collision_rate = colliding_items / n_valid

    # 覆盖的特征索引（词表利用率）
    all_feat_indices = set()
    for sid in sids:
        for token in sid:
            # token 格式: "[f_123]"
            idx = int(token.split("_")[1].rstrip("]"))
            all_feat_indices.add(idx)
    vocab_coverage = len(all_feat_indices) / d_sae

    # 碰撞组统计
    collision_groups = {s: c for s, c in sid_counts.items() if c > 1}
    max_collision = max(sid_counts.values())

    # 读取去重前统计（由 generate_sae_indices 保存的 meta 文件）
    meta_path = index_path + ".meta.json"
    pre_collision_rate_pct = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        pre_collision_rate_pct = meta.get("pre_dedup_collision_rate", 0) * 100

    return {
        "k": k,
        "n_items": n_valid,
        "n_unique": n_unique,
        "collision_rate": collision_rate,
        "collision_rate_pct": collision_rate * 100,
        "pre_collision_rate_pct": pre_collision_rate_pct,
        "colliding_items": colliding_items,
        "n_collision_groups": len(collision_groups),
        "max_items_per_sid": max_collision,
        "unique_features_used": len(all_feat_indices),
        "vocab_coverage_pct": vocab_coverage * 100,
    }


def generate_for_k(checkpoint: str, embedding_path: str, k: int, output_dir: str, max_dedup_iters: int) -> dict:
    """为指定 K 值生成 SID 并返回指标。"""
    from SAEGenRec.sid_builder.generate_sae_indices import generate_sae_indices
    from sae_lens import SAE

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"Beauty.sae_k{k}.index.json")

    print(f"\n{'='*50}")
    print(f"  Generating SIDs with K={k}")
    print(f"{'='*50}")

    t0 = time.time()
    generate_sae_indices(
        checkpoint=checkpoint,
        embedding_path=embedding_path,
        k=k,
        output_path=output_path,
        max_dedup_iters=max_dedup_iters,
        device="cuda:0",
    )
    elapsed = time.time() - t0

    # 获取 d_sae
    sae = SAE.load_from_disk(checkpoint, device="cpu")
    d_sae = sae.cfg.d_sae
    n_items = np.load(embedding_path).shape[0]

    metrics = compute_sid_metrics(output_path, k, n_items, d_sae)
    metrics["generate_time_s"] = elapsed
    return metrics


def generate_markdown_report(all_metrics: list[dict], sae_config: dict, embedding_path: str) -> str:
    """生成 Markdown 格式报告。"""
    emb = np.load(embedding_path)
    n_items, d_in = emb.shape
    d_sae = d_in * sae_config["expansion_factor"]

    lines = []
    lines.append("# SAE SID K值对比实验报告\n")
    lines.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 实验配置\n")
    lines.append(f"| 参数 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| 数据集 | Beauty (Amazon 2015, 全量) |")
    lines.append(f"| 物品数 | {n_items:,} |")
    lines.append(f"| 嵌入模型 | Qwen3-Embedding-0.6B |")
    lines.append(f"| 嵌入维度 d_in | {d_in} |")
    lines.append(f"| expansion_factor | {sae_config['expansion_factor']} |")
    lines.append(f"| 字典大小 d_sae | {d_sae:,} |")
    lines.append(f"| l1_coefficient | {sae_config['l1_coefficient']} |")
    lines.append(f"| 学习率 | {sae_config['lr']} |")
    lines.append(f"| 训练样本数 | {sae_config['total_training_samples']:,} |")
    lines.append(f"| 最大去重迭代 | {sae_config['max_dedup_iters']} |")
    lines.append("")

    lines.append("## K值对比结果\n")
    lines.append("| K | 唯一SID数 | 去重前碰撞率 | 去重后碰撞率 | 碰撞组数 | 最大碰撞 | 使用特征数 | 词表覆盖率 |")
    lines.append("|---|-----------|-------------|-------------|----------|----------|------------|------------|")
    for m in all_metrics:
        collision_flag = " ⚠️" if m["collision_rate_pct"] > 5 else (" ✅" if m["collision_rate_pct"] < 1 else "")
        pre_str = (
            f"{m['pre_collision_rate_pct']:.2f}%"
            if m["pre_collision_rate_pct"] is not None
            else "N/A"
        )
        lines.append(
            f"| {m['k']} "
            f"| {m['n_unique']:,} "
            f"| {pre_str} "
            f"| {m['collision_rate_pct']:.2f}%{collision_flag} "
            f"| {m['n_collision_groups']} "
            f"| {m['max_items_per_sid']} "
            f"| {m['unique_features_used']:,} "
            f"| {m['vocab_coverage_pct']:.1f}% |"
        )
    lines.append("")

    lines.append("## 分析\n")

    # 找最佳 K（碰撞率 < 1% 的最小 K）
    best_k = None
    for m in all_metrics:
        if m["collision_rate_pct"] < 1.0:
            best_k = m["k"]
            break

    if best_k:
        lines.append(f"### 推荐 K 值: **K={best_k}**\n")
        lines.append(f"K={best_k} 是碰撞率首次低于 1% 的最小 K 值，在 SID 区分度和序列长度之间取得平衡。\n")
    else:
        lines.append("### ⚠️ 所有 K 值碰撞率均超过 1%\n")
        lines.append("建议增大 `expansion_factor` 或 `max_dedup_iters`。\n")

    lines.append("### 碰撞率趋势\n")
    lines.append("随 K 增大，每个物品 SID 包含更多特征，两个物品的完整 top-K 集合完全相同的概率降低，碰撞率下降。\n")

    lines.append("### 词表覆盖率趋势\n")
    lines.append("K 增大时，使用到的特征索引更多，SAE 字典的利用率提升。覆盖率越高，说明 SAE 的表达能力被更充分地利用。\n")

    lines.append("### 序列长度代价\n")
    lines.append("| K | 单物品 token 数 | 历史长度10时总 token 数 |")
    lines.append("|---|-----------------|-------------------------|")
    for m in all_metrics:
        lines.append(f"| {m['k']} | {m['k']} | {m['k'] * 10} |")
    lines.append("")
    lines.append("K 值越大，LLM 需处理的 SID 序列越长，训练难度和推理开销随之增加。\n")

    lines.append("## 结论\n")
    lines.append("- **碰撞率**：K ↑ → 碰撞率 ↓（更好的物品区分度）")
    lines.append("- **词表覆盖**：K ↑ → 覆盖率 ↑（SAE 字典利用更充分）")
    lines.append("- **序列开销**：K ↑ → LLM 输入序列更长（训练和推理代价更大）")
    lines.append("- **最优权衡**：选取碰撞率 < 1% 的最小 K 值，避免不必要的序列膨胀\n")

    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: 训练 SAE ──
    if not os.path.exists(os.path.join(SAE_CHECKPOINT, "sae_weights.safetensors")):
        print("\n" + "="*60)
        print("  Step 1: Training GatedSAE")
        print("="*60)
        from SAEGenRec.sid_builder.gated_sae import gated_sae_train
        gated_sae_train(
            embedding_path=EMBEDDING_PATH,
            output_dir=SAE_CHECKPOINT,
            k=max(K_VALUES),   # 记录最大 K（训练本身不用 K）
            **SAE_CONFIG,
        )
    else:
        print(f"\n[Skip] SAE checkpoint already exists: {SAE_CHECKPOINT}")

    # ── Step 2: 用不同 K 生成 SID ──
    print("\n" + "="*60)
    print("  Step 2: Generating SIDs for each K value")
    print("="*60)

    all_metrics = []
    for k in K_VALUES:
        index_path = os.path.join(OUTPUT_DIR, f"Beauty.sae_k{k}.index.json")
        if os.path.exists(index_path):
            print(f"\n[Skip] K={k} index already exists, loading metrics...")
            from sae_lens import SAE
            sae = SAE.load_from_disk(SAE_CHECKPOINT, device="cpu")
            d_sae = sae.cfg.d_sae
            n_items = np.load(EMBEDDING_PATH).shape[0]
            metrics = compute_sid_metrics(index_path, k, n_items, d_sae)
            metrics["generate_time_s"] = 0
        else:
            metrics = generate_for_k(
                checkpoint=SAE_CHECKPOINT,
                embedding_path=EMBEDDING_PATH,
                k=k,
                output_dir=OUTPUT_DIR,
                max_dedup_iters=SAE_CONFIG["max_dedup_iters"],
            )
        all_metrics.append(metrics)
        print(f"  K={k}: collision={metrics['collision_rate_pct']:.2f}%, unique={metrics['n_unique']}")

    # ── Step 3: 生成报告 ──
    print("\n" + "="*60)
    print("  Step 3: Generating report")
    print("="*60)

    report = generate_markdown_report(all_metrics, SAE_CONFIG, EMBEDDING_PATH)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved: {REPORT_PATH}")
    print("\n" + report)


if __name__ == "__main__":
    main()
