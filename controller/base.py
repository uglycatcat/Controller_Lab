"""控制器插件接口。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Controller(Protocol):
    """
    所有控制器插件需满足的接口。

    约定（与 main.py 一致）:
      obs shape=(4,): [cart_pos, cart_vel, pole_angle, pole_angvel]
      返回 u: 小车水平力 (N)
    """

    def reset(self) -> None:
        """重置内部状态（积分项、滑模面记忆等）。"""
        ...

    def compute(self, obs: np.ndarray) -> float:
        """由观测计算控制量。"""
        ...
