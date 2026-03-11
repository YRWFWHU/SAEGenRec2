"""prepare_sft：将 CSV + index.json 预构建为 prompt-completion JSONL 数据。"""

from datetime import datetime
import json
import os
import random

from loguru import logger


def prepare_sft(
    category: str = "",
    sid_type: str = "rqvae",
    task: str = "sid_seq",
    dataset: str = "Amazon",
    prompt_template: str = "",
    data_dir: str = "data/processed",
    interim_dir: str = "data/interim",
    overwrite: bool = False,
    tasks: str = "",
    task_weights: str = "",
) -> str:
    """将 CSV + index.json 预构建为 JSONL 格式的 SFT 数据。

    Args:
        category: 商品类别名
        sid_type: SID 方法名（rqvae / rqkmeans / gated_sae）
        task: SFT 任务类型（sid_seq / item_feat / fusion / sid_to_title）
        dataset: 数据集名（默认 Amazon）
        prompt_template: 自定义 prompt 模板路径（空字符串时按任务自动查找）
        data_dir: 输出基路径
        interim_dir: 中间数据目录（含 CSV、index.json、item.json）
        overwrite: 是否覆盖已存在数据
        tasks: 多任务混合（如 'sid_seq+item_feat'）
        task_weights: 混合权重（如 '0.7,0.3'）

    Returns:
        output_dir: JSONL 输出目录路径
    """
    import pandas as pd

    from SAEGenRec.datasets.task_registry import get_sft_task
    from SAEGenRec.datasets.template_utils import load_template, validate_placeholders

    # 多任务混合模式
    if tasks:
        return _prepare_mixed(
            category=category,
            sid_type=sid_type,
            tasks=tasks,
            task_weights=task_weights,
            dataset=dataset,
            data_dir=data_dir,
            interim_dir=interim_dir,
            overwrite=overwrite,
        )

    # 获取任务实例
    task_obj = get_sft_task(task)

    # 推断输出目录
    output_dir = os.path.join(data_dir, sid_type, dataset, category, task)
    if os.path.exists(output_dir) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Use --overwrite=True to overwrite."
        )
    os.makedirs(output_dir, exist_ok=True)

    # 加载模板
    if not prompt_template:
        prompt_template = task_obj.default_template
    template = load_template(prompt_template)
    validate_placeholders(template, task_obj.required_placeholders)

    # 加载输入数据
    index_json = None
    item_json = None

    index_path = os.path.join(interim_dir, f"{category}.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index_json = json.load(f)

    item_path = os.path.join(interim_dir, f"{category}.item.json")
    if os.path.exists(item_path):
        with open(item_path) as f:
            item_json = json.load(f)

    # 处理每个 split
    split_counts = {}
    for split in ["train", "valid", "test"]:
        csv_path = os.path.join(interim_dir, f"{category}.{split}.csv")
        if not os.path.exists(csv_path):
            # 在 processed 目录下查找
            csv_path = os.path.join(
                os.path.dirname(data_dir), "processed", f"{category}.{split}.csv"
            )
        if not os.path.exists(csv_path):
            logger.warning(f"CSV not found for split '{split}': {csv_path}. Skipping.")
            continue

        csv_data = pd.read_csv(csv_path)
        examples = task_obj.build_examples(
            csv_data=csv_data,
            index_json=index_json,
            item_json=item_json,
            template=template,
            category=category,
        )

        # 过滤空 completion
        filtered = [e for e in examples if e.get("completion", "").strip()]
        skipped = len(examples) - len(filtered)
        if skipped:
            logger.warning(f"Skipped {skipped} examples with empty completion in split '{split}'")

        jsonl_path = os.path.join(output_dir, f"{split}.jsonl")
        with open(jsonl_path, "w") as f:
            for ex in filtered:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        split_counts[split] = len(filtered)
        logger.info(f"Written {len(filtered)} examples to {jsonl_path}")

    # 写入 meta.json
    meta = {
        "sid_type": sid_type,
        "dataset": dataset,
        "category": category,
        "task": task,
        "template": prompt_template,
        "created_at": datetime.now().isoformat(),
        **{f"n_{k}": v for k, v in split_counts.items()},
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved meta.json: {meta_path}")

    return output_dir


def _prepare_mixed(
    category: str,
    sid_type: str,
    tasks: str,
    task_weights: str,
    dataset: str,
    data_dir: str,
    interim_dir: str,
    overwrite: bool,
) -> str:
    """多任务混合模式：按权重采样合并 JSONL。"""
    task_list = [t.strip() for t in tasks.split("+") if t.strip()]
    if task_weights:
        weights = [float(w.strip()) for w in task_weights.split(",")]
    else:
        weights = [1.0 / len(task_list)] * len(task_list)

    assert len(task_list) == len(weights), "tasks 和 task_weights 数量不匹配"

    # 先生成各任务数据
    task_dirs = []
    for t in task_list:
        d = prepare_sft(
            category=category,
            sid_type=sid_type,
            task=t,
            dataset=dataset,
            data_dir=data_dir,
            interim_dir=interim_dir,
            overwrite=overwrite,
        )
        task_dirs.append(d)

    # 混合输出目录
    mixed_dir = os.path.join(data_dir, sid_type, dataset, category, "mixed")
    os.makedirs(mixed_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        all_examples = []
        for task_dir, weight in zip(task_dirs, weights):
            jsonl_path = os.path.join(task_dir, f"{split}.jsonl")
            if not os.path.exists(jsonl_path):
                continue
            with open(jsonl_path) as f:
                examples = [json.loads(ln) for ln in f if ln.strip()]
            # 按权重采样
            n = int(len(examples) * weight)
            if n > 0 and n < len(examples):
                examples = random.sample(examples, n)
            all_examples.extend(examples)

        random.shuffle(all_examples)
        out_path = os.path.join(mixed_dir, f"{split}.jsonl")
        with open(out_path, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info(f"Mixed {split}: {len(all_examples)} examples → {out_path}")

    meta = {
        "sid_type": sid_type,
        "dataset": dataset,
        "category": category,
        "task": "mixed",
        "tasks": task_list,
        "task_weights": weights,
        "created_at": datetime.now().isoformat(),
    }
    with open(os.path.join(mixed_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return mixed_dir


def list_sft_tasks() -> None:
    """列出所有已注册的 SFT 任务类型。"""
    from SAEGenRec.datasets.task_registry import _SFT_TASK_REGISTRY

    print("Available SFT Tasks:")
    for name, cls in _SFT_TASK_REGISTRY.items():
        instance = cls()
        print(
            f"  {name:<15} - template: {instance.default_template}, "
            f"required_inputs: {instance.required_inputs}"
        )
