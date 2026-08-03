"""
MPC 插件占位。

实现 reset / compute 后即可被 main 通过 CONTROLLER = "mpc" 选用。
"""

from __future__ import annotations

import numpy as np

from .registry import register


@register("mpc")
class CartPoleMPC:
    """Cart-Pole MPC 控制器（待实现）。"""

    def __init__(self, dt: float = 0.002, **kwargs) -> None:
        self.dt = float(dt)
        _ = kwargs

    def reset(self) -> None:
        pass

    def compute(self, obs: np.ndarray) -> float:
        """TODO: 实现 MPC 控制律。当前开环 u=0。"""
        _ = obs
        return 0.0
