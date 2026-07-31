"""
滑模控制（SMC）插件占位。

实现 reset / compute 后即可被 main 通过 CONTROLLER = "smc" 选用。
"""

from __future__ import annotations

import numpy as np

from .registry import register


@register("smc")
class CartPoleSMC:
    """Cart-Pole 滑模控制器（待实现）。"""

    def __init__(self, dt: float = 0.002, **kwargs) -> None:
        self.dt = float(dt)
        # 可在此读取 kwargs 中的滑模参数
        _ = kwargs

    def reset(self) -> None:
        pass

    def compute(self, obs: np.ndarray) -> float:
        """TODO: 实现滑模控制律。当前开环 u=0。"""
        _ = obs
        return 0.0
