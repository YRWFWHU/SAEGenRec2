"""评估数据集：EvalSidDataset。

Ported from references/MiniOneRec/data.py EvalSidDataset
"""

from SAEGenRec.datasets.base import CSVBaseDataset
from SAEGenRec.datasets.sft_datasets import _parse_sid_sequence


class EvalSidDataset(CSVBaseDataset):
    """SID 推荐评估数据集：输入历史 SID 序列，生成预测 SID。

    Ported from references/MiniOneRec/data.py EvalSidDataset
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
        for i, sid in enumerate(history_sids):
            if i == 0:
                history += sid
            else:
                history += ", " + sid
        target_item = str(row["target_item_sid"])
        last_sid = history_sids[-1] if history_sids else None
        return {
            "input": f"Can you predict the next possible item the user may expect, "
                     f"given the following chronological interaction history: {history}",
            "output": target_item + "\n",
            "dedup": target_item == last_sid,
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
