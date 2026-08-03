"""
控制器插件包。

新增控制器:
  1. 在 controller/ 下新建模块（如 lqr.py）
  2. 实现 reset(self) / compute(self, obs) -> float
  3. 用 @register(\"名字\") 装饰类
  4. 在下方 _load_plugins() 中 import 该模块
  5. main.py 里设 CONTROLLER = \"名字\"
"""

from __future__ import annotations

from .base import Controller
from .registry import list_controllers, make_controller, register


def _load_plugins() -> None:
    """导入内置插件以触发 @register。"""
    from . import lqr as _lqr  # noqa: F401
    from . import mpc as _mpc  # noqa: F401
    from . import pid as _pid  # noqa: F401
    from . import smc as _smc  # noqa: F401


_load_plugins()

__all__ = [
    "Controller",
    "register",
    "make_controller",
    "list_controllers",
]
