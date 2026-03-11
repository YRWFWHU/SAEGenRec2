"""SFT 任务注册表：4 种内置任务 + 用户扩展机制。"""

from abc import ABC, abstractmethod
import re
from typing import Dict, List, Optional, Type

# SID 边界特殊 token
SID_START = "<sid_token>"
SID_END = "</sid_token>"
SID_SPECIAL_TOKENS = [SID_START, SID_END]


def _wrap_sid(sid: str) -> str:
    """用边界 token 包裹单个物品 SID。"""
    return f"{SID_START}{sid}{SID_END}"

_SFT_TASK_REGISTRY: Dict[str, Type["SFTTask"]] = {}


class SFTTask(ABC):
    """SFT 任务抽象基类。

    子类需实现 build_examples()，并设置类属性：
    - name: 任务名
    - default_template: 默认模板文件路径
    - required_inputs: 所需输入类型列表
    - required_placeholders: 模板必须包含的占位符
    """

    name: str = ""
    default_template: str = ""
    required_inputs: List[str] = []
    required_placeholders: List[str] = []

    @abstractmethod
    def build_examples(
        self,
        csv_data,
        index_json: Optional[dict],
        item_json: Optional[dict],
        template: str,
        **kwargs,
    ) -> List[dict]:
        """构建 prompt-completion 示例列表。

        Args:
            csv_data: pandas DataFrame（train/valid/test CSV）
            index_json: item_asin → SID token 列表的字典（可能为 None）
            item_json: item_id → meta dict 的字典（可能为 None）
            template: 已加载的模板字符串（含占位符）
            **kwargs: 额外参数（category 等）

        Returns:
            List[{"prompt": str, "completion": str}]，completion 为空的会被跳过
        """
        ...


def register_sft_task(name: str):
    """注册 SFTTask 子类的装饰器。"""
    def decorator(cls: Type[SFTTask]) -> Type[SFTTask]:
        _SFT_TASK_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def get_sft_task(name: str) -> SFTTask:
    """通过名称获取 SFTTask 实例。"""
    if name not in _SFT_TASK_REGISTRY:
        available = list(_SFT_TASK_REGISTRY.keys())
        raise ValueError(f"Unknown SFT task: '{name}'. Available tasks: {available}")
    return _SFT_TASK_REGISTRY[name]()


def list_sft_tasks() -> List[str]:
    """列出所有已注册的 SFT 任务名。"""
    return list(_SFT_TASK_REGISTRY.keys())


def _parse_sid_sequence(value: str) -> List[str]:
    """解析 history_item_sid 列。"""
    value = str(value).strip()
    parts = value.split(" ")
    result = [p.strip() for p in parts if re.match(r"(\[[a-z]_\d+\])+", p.strip())]
    if result:
        return result
    try:
        parsed = eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [value] if value else []


# ---- 内置任务实现 ----

@register_sft_task("sid_seq")
class SidSeqTask(SFTTask):
    """序列推荐任务：历史 SID 序列 → 目标 SID。"""

    name = "sid_seq"
    default_template = "templates/sid_seq.txt"
    required_inputs = ["csv", "index_json"]
    required_placeholders = ["history"]

    def build_examples(self, csv_data, index_json, item_json, template, **kwargs):
        from SAEGenRec.datasets.template_utils import render_template

        examples = []
        for _, row in csv_data.iterrows():
            history_sids = _parse_sid_sequence(row["history_item_sid"])
            history = ", ".join(_wrap_sid(s) for s in history_sids)
            target = str(row["target_item_sid"]).strip()
            if not target:
                continue
            prompt = render_template(template, history=history)
            examples.append({"prompt": prompt, "completion": _wrap_sid(target)})
        return examples


@register_sft_task("item_feat")
class ItemFeatTask(SFTTask):
    """Item 特征描述任务：item title → SID。"""

    name = "item_feat"
    default_template = "templates/item_feat.txt"
    required_inputs = ["index_json", "item_json"]
    required_placeholders = ["title"]

    def build_examples(self, csv_data, index_json, item_json, template, **kwargs):
        from SAEGenRec.datasets.template_utils import render_template

        if not index_json or not item_json:
            return []

        examples = []
        for item_id, sids in index_json.items():
            if item_id not in item_json:
                continue
            title = item_json[item_id].get("title", "")
            if not title:
                continue
            combined_sid = "".join(sids)
            if not combined_sid:
                continue
            prompt = render_template(template, title=title)
            examples.append({"prompt": prompt, "completion": _wrap_sid(combined_sid)})
        return examples


@register_sft_task("fusion")
class FusionTask(SFTTask):
    """融合推荐任务：历史 title + SID → 目标 SID。"""

    name = "fusion"
    default_template = "templates/fusion.txt"
    required_inputs = ["csv", "index_json", "item_json"]
    required_placeholders = ["history", "titles"]

    def build_examples(self, csv_data, index_json, item_json, template, **kwargs):
        from SAEGenRec.datasets.template_utils import render_template

        if not index_json or not item_json:
            return []

        # 构建 sid → title 映射
        sid2title = {}
        for item_id, sids in index_json.items():
            if item_id in item_json:
                title = item_json[item_id].get("title", "")
                combined_sid = "".join(sids)
                if combined_sid and title:
                    sid2title[combined_sid] = title

        examples = []
        for _, row in csv_data.iterrows():
            history_sids = _parse_sid_sequence(row["history_item_sid"])
            history = ", ".join(_wrap_sid(s) for s in history_sids)
            titles = ", ".join(sid2title.get(s, s) for s in history_sids)
            target = str(row["target_item_sid"]).strip()
            if not target:
                continue
            prompt = render_template(template, history=history, titles=titles)
            examples.append({"prompt": prompt, "completion": _wrap_sid(target)})
        return examples


@register_sft_task("sid_to_title")
class SidToTitleTask(SFTTask):
    """反向映射任务：SID → item title。"""

    name = "sid_to_title"
    default_template = "templates/sid_to_title.txt"
    required_inputs = ["index_json", "item_json"]
    required_placeholders = ["sid"]

    def build_examples(self, csv_data, index_json, item_json, template, **kwargs):
        from SAEGenRec.datasets.template_utils import render_template

        if not index_json or not item_json:
            return []

        examples = []
        for item_id, sids in index_json.items():
            if item_id not in item_json:
                continue
            title = item_json[item_id].get("title", "")
            if not title:
                continue
            combined_sid = "".join(sids)
            if not combined_sid:
                continue
            prompt = render_template(template, sid=_wrap_sid(combined_sid))
            examples.append({"prompt": prompt, "completion": title})
        return examples
