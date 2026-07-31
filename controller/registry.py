"""控制器插件注册表。"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from .base import Controller

C = TypeVar("C", bound=type)

_REGISTRY: dict[str, Callable[..., Controller]] = {}


def register(name: str) -> Callable[[C], C]:
    """
    装饰器：把控制器类注册为插件。

    用法:
        @register("pid")
        class CartPolePID:
            def __init__(self, dt: float, **kwargs): ...
            def reset(self) -> None: ...
            def compute(self, obs) -> float: ...
    """

    key = name.strip().lower()
    if not key:
        raise ValueError("控制器插件名不能为空")

    def decorator(cls: C) -> C:
        if key in _REGISTRY:
            raise ValueError(f"控制器插件名重复: {key!r}")
        _REGISTRY[key] = cls
        setattr(cls, "plugin_name", key)
        return cls

    return decorator


def list_controllers() -> list[str]:
    return sorted(_REGISTRY)


def make_controller(name: str, dt: float, **kwargs: Any) -> Controller:
    """按插件名构造控制器；所有插件统一接收 dt=仿真步长。"""
    key = name.strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(list_controllers()) or "(无)"
        raise KeyError(f"未知控制器 {name!r}，可选: {available}")
    return _REGISTRY[key](dt=dt, **kwargs)
