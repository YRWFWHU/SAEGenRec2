"""评估入口：约束波束搜索 + HR@K/NDCG@K 计算。

Ported from references/MiniOneRec/evaluate.py
"""

import json
import os
from typing import List

from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from SAEGenRec.config import CATEGORY_MAP
from SAEGenRec.datasets.eval_datasets import EvalSidDataset
from SAEGenRec.evaluation.logit_processor import ConstrainedLogitsProcessor, build_prefix_tree
from SAEGenRec.evaluation.metrics import compute_hr_ndcg


def evaluate_subset(
    model,
    tokenizer,
    test_csv: str,
    info_file: str,
    num_beams: int = 20,
    n_samples: int = -1,
    k_values: List[int] = None,
    device: str = "auto",
    batch_size: int = 4,
    max_new_tokens: int = 256,
    length_penalty: float = 0.0,
    seed: int = 42,
    category: str = "",
) -> dict:
    """核心评估逻辑，可程序化调用（用于训练回调）。

    Args:
        model: 已加载的 causal LM 模型（torch.nn.Module）
        tokenizer: 已加载的 tokenizer
        test_csv: 测试 CSV 路径
        info_file: info TXT 路径（用于构建前缀树）
        num_beams: 波束搜索宽度
        n_samples: 评估样本数（-1 表示全部）
        k_values: K 值列表，默认 [1, 5, 10]
        device: 设备（"auto" 时使用模型所在设备）
        batch_size: 批大小
        max_new_tokens: 最大生成 token 数
        seed: 随机种子

    Returns:
        metrics_dict: {"HR@1": ..., "NDCG@1": ..., ...}
    """
    import pandas as pd
    from torch.utils.data import DataLoader

    if k_values is None:
        k_values = [1, 5, 10]

    torch.manual_seed(seed)
    if device == "auto":
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    # Build prefix tree
    hash_dict = build_prefix_tree(info_file, tokenizer)

    def prefix_fn(batch_id: int, hash_key: list) -> list:
        return hash_dict.get(tuple(hash_key), [])

    # Load dataset
    cat_name = CATEGORY_MAP.get(category, category)
    dataset = EvalSidDataset(
        train_file=test_csv,
        tokenizer=tokenizer,
        test=True,
        seed=seed,
        category=cat_name,
    )

    test_df = pd.read_csv(test_csv)
    if n_samples > 0:
        import random
        random.seed(seed)
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        dataset = torch.utils.data.Subset(dataset, indices)
        test_df = test_df.iloc[indices].reset_index(drop=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    constrained_processor = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=prefix_fn,
        num_beams=num_beams,
        base_model=None,
        eos_token_id=tokenizer.eos_token_id,
    )
    logits_processor_list = LogitsProcessorList([constrained_processor])

    all_predictions = []
    all_targets = []
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            max_len = max(len(item["input_ids"]) for item in batch)
            pad_id = tokenizer.pad_token_id
            padded_ids, padded_masks = [], []
            for item in batch:
                pad_len = max_len - len(item["input_ids"])
                padded_ids.append([pad_id] * pad_len + list(item["input_ids"]))
                padded_masks.append([0] * pad_len + list(item["attention_mask"]))
            input_ids = torch.tensor(padded_ids, dtype=torch.long).to(dev)
            attention_mask = torch.tensor(padded_masks, dtype=torch.long).to(dev)

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

            input_len = input_ids.shape[1]
            for i in range(len(batch)):
                sample_preds = []
                for j in range(num_beams):
                    gen = outputs[i * num_beams + j][input_len:]
                    decoded = tokenizer.decode(gen, skip_special_tokens=True).strip()
                    sample_preds.append(decoded)
                all_predictions.append(sample_preds)

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test_df))
            batch_targets = test_df.iloc[start_idx:end_idx]["target_item_sid"].tolist()
            all_targets.extend(batch_targets)

    return compute_hr_ndcg(all_predictions, all_targets, k_values=k_values)


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

    metrics = evaluate_subset(
        model=model,
        tokenizer=tokenizer,
        test_csv=test_csv,
        info_file=info_file,
        num_beams=num_beams,
        k_values=ks,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
        seed=seed,
        category=category,
    )

    logger.info("Evaluation results:")
    for k in ks:
        logger.info(f"  HR@{k}: {metrics.get(f'HR@{k}', 0):.4f}  NDCG@{k}: {metrics.get(f'NDCG@{k}', 0):.4f}")

    # Save predictions (need to regenerate for saving)
    if save_predictions:
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics: {metrics_path}")

    return metrics
