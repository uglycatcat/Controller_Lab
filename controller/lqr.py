"""
LQR 插件占位。

实现 reset / compute 后即可被 main 通过 CONTROLLER = "lqr" 选用。
"""

from __future__ import annotations

import numpy as np

from .registry import register
from scipy.linalg import solve_continuous_are

@register("lqr")
class CartPoleLQR:
    """Cart-Pole LQR 控制器（待实现）。"""

    def __init__(self, dt: float = 0.002, **kwargs) -> None:
        self.dt = float(dt)
        _ = kwargs

    def reset(self) -> None:
        pass

    def compute(self, obs: np.ndarray) -> float:
        """TODO: 实现 LQR 控制律。"""
        # --- 与 mjcf/robot_description.xml 对齐的名义参数 ---
        M = 1.0          # 小车质量 (kg)
        m = 0.1          # 摆杆质量 (kg)
        L = 0.6          # 杆几何长度 (m)
        l = L / 2.0      # 质心到铰链距离 (均匀杆)
        I = m * L**2 / 12.0  # 绕质心转动惯量
        b_f = 0.1        # 滑轨阻尼，对应 joint damping
        g = 9.81
        M_mat = np.array([
            [M+m,m*l],
            [m*l,I+m*l**2]
        ])
        det_M = np.linalg.det(M_mat)
        A = np.array([
            [0,1,0,0],
            [0,-(I+m*l**2)*b_f/det_M,-m**2*g*l**2/det_M,0],
            [0,0,0,1],
            [0,(m*l*b_f)/det_M,((M+m)*l*g*m)/det_M,0]
        ])
        B = np.array([
            [0],
            [(I+m*l**2)/det_M],
            [0],
            [-(m*l)/det_M]
        ])
        Q = np.diag([500.0, 1.0, 100.0, 10.0])  # [x, dx, θ, dθ]
        R = np.diag([0.5])
        P = solve_continuous_are(A,B,Q,R)
        K = np.linalg.inv(R) @ B.T @ P
        u = - K @ obs
        return np.clip(u.item(),-50,50)
