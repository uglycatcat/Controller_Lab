"""
二维一阶小车倒立摆（Cart-Pole）MuJoCo 仿真入口。

本文件只负责：加载场景、推进物理、暴露控制 I/O。
不包含任何控制器实现——请在下方标注处接入你的控制律。
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# ---------------------------------------------------------------------------
# 路径与仿真参数
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SCENE_XML = ROOT / "mjcf" / "scene.xml"

SIM_DURATION = None  # None = 一直跑；或设为秒数
RENDER = True
REALTIME = True  # True=按物理步长墙钟同步，便于手动拖动观察

# 初始状态：小车位置 (m)、摆角 (rad，0=竖直向上，π=竖直向下)、速度
INIT_CART_POS = 0.0
INIT_POLE_ANGLE = 0.02  # 近直立微扰动；起摆测试可改为 np.pi
INIT_CART_VEL = 0.0
INIT_POLE_ANGVEL = 0.0


# ---------------------------------------------------------------------------
# 控制场景 I/O
# ---------------------------------------------------------------------------
# 状态维度约定（观测输出）:
#   obs[0] = cart_pos      小车位置 x          (m)
#   obs[1] = cart_vel      小车速度 dx/dt      (m/s)
#   obs[2] = pole_angle    摆角 θ（0=直立向上）(rad)
#   obs[3] = pole_angvel   摆角速度 dθ/dt      (rad/s)
#
# 控制输入约定:
#   u ∈ R^1  作用在小车上的水平力 (N)，被 actuator 限制在 ctrlrange 内
# ---------------------------------------------------------------------------

OBS_DIM = 4
CTRL_DIM = 1


def get_observation(data: mujoco.MjData) -> np.ndarray:
    """读取当前状态（控制场景输出）。"""
    return np.array(
        [
            data.qpos[0],  # cart_pos
            data.qvel[0],  # cart_vel
            data.qpos[1],  # pole_angle
            data.qvel[1],  # pole_angvel
        ],
        dtype=np.float64,
    )


def set_control(data: mujoco.MjData, u: float | np.ndarray) -> None:
    """写入控制输入（控制场景输入）。u 为标量或 shape=(1,) 的数组。"""
    data.ctrl[0] = float(np.asarray(u).reshape(-1)[0])


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """重置到指定初始状态，返回初始观测。"""
    mujoco.mj_resetData(model, data)
    data.qpos[0] = INIT_CART_POS
    data.qpos[1] = INIT_POLE_ANGLE
    data.qvel[0] = INIT_CART_VEL
    data.qvel[1] = INIT_POLE_ANGVEL
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return get_observation(data)


def step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    u: float | np.ndarray,
) -> np.ndarray:
    """施加控制并推进一个物理步，返回新观测。"""
    set_control(data, u)
    mujoco.mj_step(model, data)
    return get_observation(data)


# ===========================================================================
# TODO: 在此接入你的控制器
#   输入: obs (np.ndarray, shape=(4,))
#   输出: u   (float 或 shape=(1,) 的数组)  — 小车水平力
# ===========================================================================
def compute_control(obs: np.ndarray) -> float:
    """占位：无控制（开环 u=0）。请替换为你的控制律。"""
    _ = obs
    return 0.0


def main() -> None:
    if not SCENE_XML.is_file():
        raise FileNotFoundError(f"找不到场景文件: {SCENE_XML}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    # 校验 DOF / 执行器维度，避免模型改动后接口 silently 错位
    assert model.nq == 2 and model.nv == 2, (
        f"期望 qpos/qvel 维数为 2，实际 nq={model.nq}, nv={model.nv}"
    )
    assert model.nu == CTRL_DIM, f"期望控制维数 {CTRL_DIM}，实际 nu={model.nu}"

    obs = reset(model, data)
    print("Cart-Pole 已加载")
    print(f"  timestep = {model.opt.timestep} s")
    print(f"  观测输出 obs = [cart_pos, cart_vel, pole_angle, pole_angvel]")
    print(f"  控制输入 u   = cart_force (N), range={model.actuator_ctrlrange[0]}")
    print(f"  初始观测     = {obs}")

    def run_loop(viewer=None) -> None:
        nonlocal obs
        while viewer is None or viewer.is_running():
            if SIM_DURATION is not None and data.time >= SIM_DURATION:
                break

            step_start = time.perf_counter()

            # ---- 控制 I/O 边界 ----
            # 输出: obs  |  输入: u
            u = compute_control(obs)
            obs = step(model, data, u)
            # ----------------------

            if viewer is not None:
                viewer.sync()

            # 墙钟同步：否则仿真远快于实时，杆会“瞬间”倒下且拖不动
            if REALTIME and viewer is not None:
                leftover = model.opt.timestep - (time.perf_counter() - step_start)
                if leftover > 0:
                    time.sleep(leftover)

    if RENDER:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 侧视，便于观察二维运动
            viewer.cam.lookat[:] = [0.0, 0.0, 0.95]
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -10.0
            print("提示: 双击选中小车后拖动可施加扰动；仿真已按实时速率运行")
            run_loop(viewer)
    else:
        run_loop(None)

    print(f"仿真结束 t={data.time:.3f}s, 最终观测={get_observation(data)}")


if __name__ == "__main__":
    main()
