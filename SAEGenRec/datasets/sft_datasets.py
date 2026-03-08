"""SFT 训练数据集：SidSFTDataset、SidItemFeatDataset、FusionSeqRecDataset。

Ported from references/MiniOneRec/data.py
"""

import copy
import json
import re
import random
from typing import List, Optional

import pandas as pd

from SAEGenRec.datasets.base import BaseDataset, CSVBaseDataset


def _parse_sid_sequence(value: str) -> List[str]:
    """解析 history_item_sid 列为 item-level SID 列表。

    存储格式（convert_dataset 生成）：
        '[a_1][b_2][c_3] [a_4][b_5][c_6]'  → ['[a_1][b_2][c_3]', '[a_4][b_5][c_6]']

    也支持旧格式（Python list string）：
        "['[a_1][b_2][c_3]', '[a_4][b_5][c_6]']" → ['[a_1][b_2][c_3]', '[a_4][b_5][c_6]']
    """
    value = str(value).strip()
    # 空格分隔的多个 SID（每个 SID 是 [a_x][b_y]... 的连续串）
    parts = value.split(" ")
    result = [p.strip() for p in parts if re.match(r'(\[[a-z]_\d+\])+', p.strip())]
    if result:
        return result
    # Python list format fallback
    try:
        parsed = eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [value] if value else []


def _parse_title_sequence(value: str) -> List[str]:
    """解析 history_item_title 列为 title 列表（逗号分隔）。"""
    value = str(value).strip()
    # Python list string fallback
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = eval(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return [t.strip() for t in value.split(",") if t.strip()]


class SidSFTDataset(CSVBaseDataset):
    """SID 序列推荐 SFT 数据集：输入历史 SID 序列，预测下一个 SID。

    Ported from references/MiniOneRec/data.py SidSFTDataset
    """

    def __init__(
        self,
        train_file: str,
        tokenizer,
        max_len: int = 2048,
        sample: int = -1,
        test: bool = False,
        seed: int = 0,
        category: str = "",
        dedup: bool = False,
    ):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)
        self.get_inputs()

    def get_history(self, row):
        history_sids = _parse_sid_sequence(row["history_item_sid"])
        history = ""
        history_str = ", ".join(history_sids)
        for i, sid in enumerate(history_sids):
            if i == 0:
                history += sid
            else:
                history += ", " + sid
        target_item = str(row["target_item_sid"])
        last_history_sid = history_sids[-1] if history_sids else None
        return {
            "input": f"The user has interacted with items {history} in chronological order. "
                     f"Can you predict the next possible item that the user may expect?",
            "output": target_item + "\n",
            "history_str": history_str,
            "dedup": target_item == last_history_sid,
        }

    def pre(self, idx):
        instruction = (
            "Below is an instruction that describes a task, paired with an input that provides "
            "further context. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nCan you predict the next possible item that the user may expect?\n\n"
        )
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history = self.get_history(self.data.iloc[idx])
        target_item = history["output"]
        history["output"] = ""

        prompt = self.generate_prompt(history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)

        if self.test:
            return {"input_ids": tokens, "attention_mask": attention_mask}

        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        return {
            "input_ids": tokens[-self.max_len :],
            "attention_mask": attention_mask[-self.max_len :],
            "labels": labels[-self.max_len :],
        }


class SidItemFeatDataset(BaseDataset):
    """SID ↔ 标题双向映射数据集。

    Ported from references/MiniOneRec/data.py SidItemFeatDataset
    """

    def __init__(
        self,
        item_file: str,
        index_file: str,
        tokenizer=None,
        max_len: int = 2048,
        sample: int = -1,
        test: bool = False,
        seed: int = 0,
        category: str = "",
    ):
        super().__init__(tokenizer, max_len, test, category, dedup=False, seed=seed)

        with open(item_file) as f:
            self.item_feat = json.load(f)
        with open(index_file) as f:
            self.indices = json.load(f)

        self.sid2title = {}
        self.title2sid = {}

        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids) >= 2:
                title = self.item_feat[item_id]["title"]
                combined_sid = "".join(sids)
                self.sid2title[combined_sid] = title
                self.title2sid[title] = combined_sid

        self.data = []
        for sid, title in self.sid2title.items():
            self.data.append({"task": "sid2title", "input": sid, "output": title})
        for title, sid in self.title2sid.items():
            self.data.append({"task": "title2sid", "input": title, "output": sid})

        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)

        if self.tokenizer is not None:
            self.get_inputs()

    def generate_prompt(self, data_point):
        if data_point["task"] == "title2sid":
            prompt = f"Which item has the title: {data_point['input']}?"
        else:
            prompt = f'What is the title of item "{data_point["input"]}"?'
        return f"### User Input:\n{prompt}\n\n### Response:\n"

    def pre(self, idx):
        if self.tokenizer is None:
            return self.data[idx]

        instruction = (
            "Below is an instruction that describes a task, paired with an input that provides "
            "further context. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nAnswer the question about item identification.\n\n"
        )
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        data_point = self.data[idx]
        prompt = self.generate_prompt(data_point)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)

        if self.test:
            return {"input_ids": tokens, "attention_mask": attention_mask}

        target = data_point["output"] + "\n"
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        return {
            "input_ids": tokens[-self.max_len :],
            "attention_mask": attention_mask[-self.max_len :],
            "labels": labels[-self.max_len :],
        }


class FusionSeqRecDataset(BaseDataset):
    """融合序列推荐数据集：历史 SID 序列 → 目标 item title。

    Ported from references/MiniOneRec/data.py FusionSeqRecDataset
    """

    def __init__(
        self,
        train_file: str,
        item_file: str,
        index_file: str,
        tokenizer,
        max_len: int = 2048,
        sample: int = -1,
        test: bool = False,
        seed: int = 0,
        category: str = "",
        dedup: bool = False,
    ):
        super().__init__(tokenizer, max_len, test, category, dedup, seed)

        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)

        with open(item_file) as f:
            self.item_feat = json.load(f)
        with open(index_file) as f:
            self.indices = json.load(f)

        self.sid2title = {}
        self.sid2description = {}

        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids) >= 2:
                title = self.item_feat[item_id]["title"]
                desc = self.item_feat[item_id].get("description", "")
                processed_desc = self._process_description(desc, title)
                combined_sid = "".join(sids)
                self.sid2title[combined_sid] = title
                self.sid2description[combined_sid] = processed_desc

        self.get_inputs()

    def _process_description(self, description, title: str) -> str:
        if not description or description == "":
            return title
        if isinstance(description, list):
            desc_list = description
        elif isinstance(description, str) and description.startswith("[") and description.endswith("]"):
            try:
                desc_list = eval(description)
            except Exception:
                return description.strip() or title
        else:
            return description.strip() or title

        non_empty = [d for d in desc_list if d and str(d).strip()]
        if non_empty:
            return max(non_empty, key=len)
        return title

    def generate_prompt_title(self, history: str) -> str:
        return (
            f"The user has sequentially interacted with items {history}. "
            f"Can you recommend the next item for him? Tell me the title of the item"
        )

    def get_history(self, row):
        history_sids = _parse_sid_sequence(row["history_item_sid"])
        history_str = ", ".join(history_sids)
        target_sid = row["target_item_sid"]
        target_title = self.sid2title.get(target_sid, target_sid)
        last_sid = history_sids[-1] if history_sids else None
        return {
            "history_str": history_str,
            "target_title": target_title,
            "target_sid": target_sid,
            "dedup": target_sid == last_sid,
        }

    def generate_formatted_prompt(self, prompt: str, response: str) -> str:
        return f"### User Input:\n{prompt}\n\n### Response:\n"

    def pre(self, idx):
        instruction = (
            "Below is an instruction that describes a task, paired with an input that provides "
            "further context. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nCan you recommend the next item for the user based on their "
            "interaction history?\n\n"
        )
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        history_data = self.get_history(self.data.iloc[idx])

        if self.dedup and history_data["dedup"]:
            return None

        prompt = self.generate_prompt_title(history_data["history_str"])
        target = history_data["target_title"] + "\n"
        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        tokens = tokens + self.tokenizer.encode(formatted_prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)

        if self.test:
            return {"input_ids": tokens, "attention_mask": attention_mask}

        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        return {
            "input_ids": tokens[-self.max_len :],
            "attention_mask": attention_mask[-self.max_len :],
            "labels": labels[-self.max_len :],
        }
