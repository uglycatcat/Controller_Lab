"""
Cart-Pole 级联 PID 示例控制器。

结构（外环位置 → 内环角度 → 力）:
  1. 位置环：根据小车偏离原点，给出期望摆角（先倾后移，非最小相位）
  2. 角度环：PID 跟踪期望摆角，输出小车水平力

约定与 main.py / 本仓库 MJCF 一致:
  obs = [cart_pos, cart_vel, pole_angle, pole_angvel]
  θ = 0 为竖直向上；u > 0 沿 +x 推小车。
  实测：θ>0 时需 u>0 才能把杆扶正（绕 +y 铰链）。
"""

from __future__ import annotations

import numpy as np

from .registry import register


class PID:
    """标准位置式 PID，带积分限幅与输出限幅。"""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        *,
        integral_limit: float | None = None,
        output_limit: float | None = None,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit

        self._integral = 0.0
        self._prev_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def __call__(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            raise ValueError(f"dt 必须为正，得到 {dt}")

        self._integral += error * dt
        if self.integral_limit is not None:
            self._integral = float(
                np.clip(self._integral, -self.integral_limit, self.integral_limit)
            )

        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        u = self.kp * error + self.ki * self._integral + self.kd * derivative
        if self.output_limit is not None:
            u = float(np.clip(u, -self.output_limit, self.output_limit))
        return u


@register("pid")
class CartPolePID:
    """
    级联 PID：外环稳车，内环稳杆。

    默认增益在 mujoco_env 下对本仓库 MJCF 整定，适合近直立起控；
    大角度倒下后输出 0，需重置仿真。
    """

    FALLEN_ANGLE = 0.6  # rad ≈ 34°

    def __init__(
        self,
        dt: float = 0.002,
        *,
        # 外环：位置误差 → 期望摆角
        kp_pos: float = 0.08,
        ki_pos: float = 0.0,
        kd_pos: float = 0.12,
        # 内环：角度误差 → 力（本模型下 u ∝ θ）
        kp_ang: float = 160.0,
        ki_ang: float = 0.0,
        kd_ang: float = 22.0,
        # 角速度显式阻尼（与角度环 D 互补）
        k_angvel: float = 12.0,
        max_lean: float = 0.30,
        max_force: float = 50.0,
    ) -> None:
        self.dt = dt
        self.k_angvel = k_angvel
        self.max_force = max_force

        # x>0 → θ_des<0：先向 -θ 倾，再把车拉回原点
        self.pos_pid = PID(
            kp_pos,
            ki_pos,
            kd_pos,
            integral_limit=0.5,
            output_limit=max_lean,
        )
        # 对 (θ - θ_des) 做 PID；θ>0 时输出正力
        self.ang_pid = PID(
            kp_ang,
            ki_ang,
            kd_ang,
            integral_limit=2.0,
            output_limit=max_force,
        )

    def reset(self) -> None:
        self.pos_pid.reset()
        self.ang_pid.reset()

    def compute(self, obs: np.ndarray) -> float:
        cart_pos, _cart_vel, pole_angle, pole_angvel = (float(v) for v in obs)

        if abs(pole_angle) > self.FALLEN_ANGLE:
            self.ang_pid.reset()
            return 0.0

        theta_des = self.pos_pid(-cart_pos, self.dt)
        u = self.ang_pid(pole_angle - theta_des, self.dt)
        u += self.k_angvel * pole_angvel
        return float(np.clip(u, -self.max_force, self.max_force))
