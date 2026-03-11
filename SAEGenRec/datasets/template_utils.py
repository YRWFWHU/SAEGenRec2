"""Prompt 模板加载和渲染工具。"""

from collections import defaultdict


def load_template(path: str) -> str:
    """从文件加载 prompt 模板。

    Args:
        path: 模板文件路径

    Returns:
        模板字符串
    """
    with open(path) as f:
        return f.read()


def render_template(template: str, **kwargs) -> str:
    """使用 str.format_map() 渲染模板，缺失键用空字符串填充。

    Args:
        template: 含占位符的模板字符串（如 `{history}`、`{title}`）
        **kwargs: 占位符值

    Returns:
        渲染后的字符串
    """
    mapping = defaultdict(str, kwargs)
    return template.format_map(mapping)


def validate_placeholders(template: str, required: list[str]) -> None:
    """验证模板包含所有必需的占位符。

    Args:
        template: 模板字符串
        required: 必需的占位符列表（如 ['history', 'target']）

    Raises:
        ValueError: 若模板缺少必需占位符
    """
    missing = [p for p in required if "{" + p + "}" not in template]
    if missing:
        raise ValueError(
            f"Template is missing required placeholders: {missing}. "
            f"Found template content:\n{template[:200]}..."
        )
