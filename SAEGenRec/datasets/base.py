"""基础数据集类：BaseDataset、CSVBaseDataset。

Ported from references/MiniOneRec/data.py
"""

import random
from typing import List, Optional

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert isinstance(s, str)
        t = self.tokenizer.encode(s)
        while t and t[0] == self.bos_id:
            t = t[1:]
        while t and t[-1] == self.eos_id:
            t = t[:-1]
        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


class BaseDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        max_len: int = 2048,
        test: bool = False,
        category: str = "",
        dedup: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.data = None
        self.inputs = None

        if tokenizer is not None:
            self.tokenizer = Tokenizer(tokenizer)
        if seed is not None:
            random.seed(seed)

        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup

    def __len__(self):
        return len(self.data)

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
        self.inputs = inputs

    def __getitem__(self, idx):
        return self.inputs[idx]

    def pre(self, idx):
        raise NotImplementedError

    def get_history(self, row):
        return {}

    def generate_prompt(self, data_point):
        return f"""### User Input:
{data_point["input"]}

### Response:\n{data_point["output"]}"""


class CSVBaseDataset(BaseDataset):
    def __init__(
        self,
        train_file: str,
        sample: int = -1,
        seed: int = 0,
        max_len: int = 2048,
        category: str = "",
        dedup: bool = False,
        tokenizer=None,
        test: bool = False,
    ):
        super().__init__(tokenizer, max_len, test, category, dedup, seed)
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
