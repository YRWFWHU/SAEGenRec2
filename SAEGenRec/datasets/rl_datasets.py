"""RL 训练数据集：SidDataset、RLTitle2SidDataset、RLSeqTitle2SidDataset。

Ported from references/MiniOneRec/data.py
"""

import json
import random

import pandas as pd

from SAEGenRec.datasets.base import BaseDataset, CSVBaseDataset
from SAEGenRec.datasets.sft_datasets import _parse_sid_sequence, _parse_title_sequence


class SidDataset(CSVBaseDataset):
    """SID 序列推荐 RL 数据集（prompt-completion 格式）。

    Ported from references/MiniOneRec/data.py SidDataset
    """

    def __init__(
        self,
        train_file: str,
        max_len: int = 2048,
        sample: int = -1,
        seed: int = 0,
        category: str = "",
        dedup: bool = False,
    ):
        super().__init__(
            train_file, sample, seed, max_len, category, dedup, tokenizer=None, test=False
        )
        self.prompt2history = {}
        self.history2target = {}
        self.get_inputs()

    def get_history(self, row):
        history_sids = _parse_sid_sequence(row["history_item_sid"])
        history = ""
        history_str = "::".join(history_sids)
        for i, sid in enumerate(history_sids):
            if i == 0:
                history += sid
            else:
                history += ", " + sid
        target_item = str(row["target_item_sid"])
        last_sid = history_sids[-1] if history_sids else None
        return {
            "input": f"The user has interacted with items {history} in chronological order. "
                     f"Can you predict the next possible item that the user may expect?",
            "output": target_item + "\n",
            "history_str": history_str,
            "dedup": target_item == last_sid,
        }

    def generate_prompt(self, data_point):
        return f"### User Input:\n{data_point['input']}\n\n### Response:\n{data_point['output']}"

    def pre(self, idx):
        history = self.get_history(self.data.iloc[idx])
        target_item = history["output"]
        history["output"] = ""
        prompt = self.generate_prompt(history)
        self.prompt2history[prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target_item
        return {"prompt": prompt, "completion": target_item}


class RLTitle2SidDataset(BaseDataset):
    """RL 数据集：title/description → SID。

    Ported from references/MiniOneRec/data.py RLTitle2SidDataset
    """

    def __init__(
        self,
        item_file: str,
        index_file: str,
        sample: int = -1,
        seed: int = 0,
        category: str = "",
        dedup: bool = False,
    ):
        super().__init__(tokenizer=None, max_len=1024, test=False, category=category, dedup=dedup, seed=seed)
        self.prompt2history = {}
        self.history2target = {}

        with open(item_file) as f:
            self.item_feat = json.load(f)
        with open(index_file) as f:
            self.indices = json.load(f)

        self.title2sid = {}
        self.description2sid = {}

        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids) >= 2:
                title = self.item_feat[item_id]["title"]
                desc = self.item_feat[item_id].get("description", "")
                if isinstance(desc, list):
                    desc = desc[0] if desc else ""
                combined_sid = "".join(sids)
                self.title2sid[title] = combined_sid
                if desc:
                    self.description2sid[desc] = combined_sid

        self.data = []
        for title, sid in self.title2sid.items():
            self.data.append({"task": "title2sid", "input": title, "output": sid})
        for desc, sid in self.description2sid.items():
            self.data.append({"task": "description2sid", "input": desc, "output": sid})

        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)

        self.get_inputs()

    def generate_prompt(self, data_point):
        if data_point["task"] == "title2sid":
            prompt = f"Which item has the title: {data_point['input']}?"
        else:
            prompt = f'An item can be described as follows: "{data_point["input"]}". Which item is it describing?'
        return f"### User Input:\n{prompt}\n\n### Response:\n"

    def pre(self, idx):
        data_point = self.data[idx]
        prompt = self.generate_prompt(data_point)
        target_item = data_point["output"] + "\n"
        self.prompt2history[prompt] = data_point["input"]
        self.history2target[data_point["input"]] = target_item
        return {"prompt": prompt, "completion": target_item}


class RLSeqTitle2SidDataset(CSVBaseDataset):
    """RL 数据集：用户历史 title 序列 → 目标 SID。

    Ported from references/MiniOneRec/data.py RLSeqTitle2SidDataset
    """

    def __init__(
        self,
        train_file: str,
        sample: int = -1,
        seed: int = 0,
        category: str = "",
        dedup: bool = False,
    ):
        super().__init__(
            train_file, sample, seed, max_len=1024, category=category, dedup=dedup,
            tokenizer=None, test=False
        )
        self.prompt2history = {}
        self.history2target = {}
        self.get_inputs()

    def generate_prompt(self, inter_titles: str) -> str:
        return (
            f"Given the title sequence of user historical interactive items: {inter_titles}, "
            f"can you recommend a suitable next item for the user?"
        )

    def get_history(self, row):
        history_titles = _parse_title_sequence(row["history_item_title"])
        inter_titles = ", ".join(f'"{t}"' for t in history_titles)
        target_sid = row["target_item_sid"]
        return {
            "inter_titles": inter_titles,
            "target_sid": target_sid,
            "dedup": False,
            "history_str": "::".join(history_titles),
        }

    def generate_formatted_prompt(self, prompt: str, response: str) -> str:
        return f"### User Input:\n{prompt}\n\n### Response:\n"

    def pre(self, idx):
        history_data = self.get_history(self.data.iloc[idx])
        if self.dedup and history_data["dedup"]:
            return None
        prompt = self.generate_prompt(history_data["inter_titles"])
        target = history_data["target_sid"] + "\n"
        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        self.prompt2history[formatted_prompt] = history_data["history_str"]
        self.history2target[history_data["history_str"]] = target
        return {"prompt": formatted_prompt, "completion": target}
