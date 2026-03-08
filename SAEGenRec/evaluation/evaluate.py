"""评估入口：约束波束搜索 + HR@K/NDCG@K 计算。

Ported from references/MiniOneRec/evaluate.py
"""

import json
import os
from typing import List, Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from SAEGenRec.config import CATEGORY_MAP
from SAEGenRec.datasets.eval_datasets import EvalSidDataset
from SAEGenRec.evaluation.logit_processor import ConstrainedLogitsProcessor, build_prefix_tree
from SAEGenRec.evaluation.metrics import compute_hr_ndcg


def evaluate(
    model_path: str = "",
    test_csv: str = "",
    info_file: str = "",
    category: str = "",
    output_dir: str = "results",
    batch_size: int = 4,
    num_beams: int = 50,
    max_new_tokens: int = 256,
    length_penalty: float = 0.0,
    k_values: str = "1,3,5,10,20",
    seed: int = 42,
    save_predictions: bool = True,
) -> dict:
    """使用约束波束搜索评估模型，计算 HR@K 和 NDCG@K。

    Args:
        model_path: 模型检查点路径
        test_csv: 测试 CSV 路径
        info_file: info TXT 路径（用于构建前缀树）
        num_beams: 波束搜索宽度
        k_values: 逗号分隔的 K 值列表
        save_predictions: 是否保存预测结果到 JSON 文件

    Returns:
        metrics_dict: {"HR@1": ..., "NDCG@1": ..., ...}
    """
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(k_values, (list, tuple)):
        ks = [int(k) for k in k_values]
    else:
        ks = [int(k.strip()) for k in str(k_values).split(",")]

    logger.info(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build prefix tree
    logger.info(f"Building prefix tree from: {info_file}")
    hash_dict = build_prefix_tree(info_file, tokenizer)

    def prefix_fn(batch_id: int, hash_key: list) -> list:
        return hash_dict.get(tuple(hash_key), [])

    # Build item dict from info file (for metric computation)
    item_dict = {}
    with open(info_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                sid = parts[0].strip()
                if sid not in item_dict:
                    item_dict[sid] = []
                try:
                    item_dict[sid].append(int(parts[-1].strip()))
                except (ValueError, IndexError):
                    pass

    # Load dataset
    cat_name = CATEGORY_MAP.get(category, category)
    dataset = EvalSidDataset(
        train_file=test_csv,
        tokenizer=tokenizer,
        test=True,
        seed=seed,
        category=cat_name,
    )
    logger.info(f"Test samples: {len(dataset)}")

    # Collect predictions
    all_predictions = []
    all_targets = []

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    import pandas as pd

    test_df = pd.read_csv(test_csv)

    constrained_processor = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=prefix_fn,
        num_beams=num_beams,
        base_model=model_path,
        eos_token_id=tokenizer.eos_token_id,
    )
    logits_processor_list = LogitsProcessorList([constrained_processor])

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Each item in batch is a dict with input_ids / attention_mask
            # Pad to same length (left-pad since tokenizer.padding_side="left")
            max_len = max(len(item["input_ids"]) for item in batch)
            pad_id = tokenizer.pad_token_id
            padded_ids = []
            padded_masks = []
            for item in batch:
                pad_len = max_len - len(item["input_ids"])
                padded_ids.append([pad_id] * pad_len + list(item["input_ids"]))
                padded_masks.append([0] * pad_len + list(item["attention_mask"]))
            input_ids = torch.tensor(padded_ids, dtype=torch.long).to(device)
            attention_mask = torch.tensor(padded_masks, dtype=torch.long).to(device)

            # Reset processor state for each batch
            constrained_processor.count = 0

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
                logits_processor=logits_processor_list,
                early_stopping=True,
            )

            # Decode outputs
            input_len = input_ids.shape[1]
            for i in range(len(batch)):
                sample_preds = []
                for j in range(num_beams):
                    gen = outputs[i * num_beams + j][input_len:]
                    decoded = tokenizer.decode(gen, skip_special_tokens=True).strip()
                    sample_preds.append(decoded)
                all_predictions.append(sample_preds)

            # Collect targets
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test_df))
            batch_targets = test_df.iloc[start_idx:end_idx]["target_item_sid"].tolist()
            all_targets.extend(batch_targets)

            if batch_idx % 10 == 0:
                logger.info(f"Processed {(batch_idx + 1) * batch_size} samples")

    # Compute metrics
    metrics = compute_hr_ndcg(all_predictions, all_targets, k_values=ks)
    logger.info("Evaluation results:")
    for k in ks:
        logger.info(f"  HR@{k}: {metrics.get(f'HR@{k}', 0):.4f}  NDCG@{k}: {metrics.get(f'NDCG@{k}', 0):.4f}")

    # Save predictions
    if save_predictions:
        os.makedirs(output_dir, exist_ok=True)
        result_data = [
            {"predict": preds, "output": target}
            for preds, target in zip(all_predictions, all_targets)
        ]
        result_path = os.path.join(output_dir, "predictions.json")
        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2)
        logger.info(f"Saved predictions: {result_path}")

        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics: {metrics_path}")

    return metrics
