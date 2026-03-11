"""SID 方法注册表：管理 SIDMethod 实现的注册和查找。"""

from typing import Dict, List, Type

from SAEGenRec.sid_builder.base import SIDMethod

SID_METHODS: Dict[str, Type[SIDMethod]] = {}


def register_sid_method(name: str):
    """注册 SIDMethod 子类的装饰器。

    Usage:
        @register_sid_method("rqvae")
        class RQVAEMethod(SIDMethod):
            ...
    """
    def decorator(cls: Type[SIDMethod]) -> Type[SIDMethod]:
        SID_METHODS[name] = cls
        cls.name = name
        return cls
    return decorator


def get_sid_method(name: str) -> SIDMethod:
    """通过名称获取 SIDMethod 实例。

    Args:
        name: 方法名（如 'rqvae', 'rqkmeans', 'gated_sae'）

    Returns:
        SIDMethod 实例

    Raises:
        ValueError: 若方法名未注册
    """
    if name not in SID_METHODS:
        available = list(SID_METHODS.keys())
        raise ValueError(
            f"Unknown SID method: '{name}'. Available methods: {available}"
        )
    return SID_METHODS[name]()


def list_sid_methods() -> List[str]:
    """列出所有已注册的 SID 方法名。"""
    return list(SID_METHODS.keys())
