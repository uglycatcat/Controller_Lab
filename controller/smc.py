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
        # 参数表
        # M为小车质量
        # m为杆质量
        # l为杆长度
        # I为杆绕质心转动惯量
        # b_f为摩擦系数
        # g为重力加速度
        # k为滑模增益
        # eta为鲁棒项增益
        # k1,k2为权重系数
        M,m,l,I,b_f,g,k,eta,k1,k2=1,0.1,0.6,0.003,0.01,9.8,1,0.1,1,1
        # weight_array = np.array([k1,k2,k3,k4])
        # smc_s = weight_array * obs        
        # 对于该cart-pole系统，根据拉格朗日列出其动力学方程
        # x = np.array([ddot_x,ddot_theta])
        # A = np.array([M+m,m*l*cos(obs[2])],[m*l*cos(obs[2]),m*l*l+I])
        # b = np.array([u-b*obs[1]+m*sin(obs[2])*obs[3]*obs[3],-m*g*l*sin(obs[2])])
        # 存在A * x = b
        # 对smc_s求导可得到 dot_smc_s = k1*dot_x+k3*dot_theta+k2*ddot_x+k4*ddot_theta
        # 为了控制dot_smc_s的正负，需要引入u来代替ddot_x和ddot_theta
        # 由克莱姆定理知 
        # ddot_x = b[0]*A[1][1]-b[1]*A[0][1]/det(A)
        # ddot_theta = b[1]*A[0][0]-b[0]*A[1][0]/det(A)
        # 综上可得 dot_smc_s 关于 u的表达式。
        # 由于只靠当个u去同时控制x和theta要列的式子太麻烦了。接下来只考虑控制杆向上（即theta=0，dot_theta=0）的情况
        # 此时 smc_s_pole = k1 * theta + k2 * dot_theta
        # 即 dot_smc_s_pole = k1* dot_theta + k2 * ddot_theta
        # 与之前的思路类似的给出u
        _ = obs
        weight_array = np.array([k1,k2])
        smc_s_pole = np.dot(weight_array,[obs[2],obs[3]])
        A = np.array([[M+m,m*l*np.cos(obs[2])],[m*l*np.cos(obs[2]),m*l*l+I]])
        # 暂时引入u=0,用于计算b，但是b[0]并不会被用到
        u = 0
        b = np.array([u-b_f*obs[1]+m*l*np.sin(obs[2])*obs[3]*obs[3],-m*g*l*np.sin(obs[2])])
        # 此时 dot_smc_s_pole = k1*dot_theta + A[0][0]*b[1]*k2/det(A)-A[1][0]*k2*u/det(A)+A[1][0]*k2/det(A)*(b*dot_x-m*l*sin(theta)*dot_theta*dot_theta)
        # 设计控制律 u = (det(A)/A[1][0]*k2)*(k1*dot_theta + A[0][0]*b[1]*k2/det(A)+A[1][0]*k2/det(A)*(b*dot_x-m*l*sin(theta)*dot_theta*dot_theta-smc_s_pole*k-eta*sign(smc_s_pole)))
        a00 = M+m
        a01 = m*l*np.cos(obs[2])
        a10 = m*l*np.cos(obs[2])
        a11 = m*l*l+I
        b1 = m*g*l*np.sin(obs[2])
        det_A = np.linalg.det(A)
        theta = obs[2]
        dot_theta = obs[3]
        u = (det_A/(a10*k2))*(k1*dot_theta + (a00*b1*k2)/det_A+((a10*k2)/det_A)*(b_f*obs[1]-m*l*np.sin(theta)*obs[3]*obs[3])+smc_s_pole*k+eta*np.sign(smc_s_pole))
        return u

    def _compute(self, obs: np.ndarray) -> float:
        """
        标准 Cart-Pole 滑模控制示例（稳杆 + 小车回中）。

        约定与 main / MJCF 一致:
          obs = [x, dx, θ, dθ]，θ=0 竖直向上；u>0 沿 +x 推车。

        设计要点:
          1) 全状态滑模面 s = cᵀ z，同时约束小车与摆杆（欠驱动单输入只能有一个面）
          2) 用名义模型写 ṡ = α + β u，取到达律 ṡ = -k s - η sat(s/φ)
          3) u = -(α + k s + η sat)/β，边界层 sat 抑制抖振
        """
        x, dx, theta, dtheta = (float(v) for v in obs)

        # --- 与 mjcf/robot_description.xml 对齐的名义参数 ---
        M = 1.0          # 小车质量 (kg)
        m = 0.1          # 摆杆质量 (kg)
        L = 0.6          # 杆几何长度 (m)
        l = L / 2.0      # 质心到铰链距离 (均匀杆)
        I = m * L**2 / 12.0  # 绕质心转动惯量
        b_f = 0.1        # 滑轨阻尼，对应 joint damping
        g = 9.81
        u_max = 50.0     # 与 actuator ctrlrange 一致

        # --- 滑模面 s = c1 x + c2 ẋ + c3 θ + c4 θ̇ ---
        # c4 相对 c2 足够大，保证直立附近 β<0，且 θ>0 时 u>0（与本仓库实测一致）
        c1, c2, c3, c4 = 1.0, 2.2, 18.0, 4.0
        s = c1 * x + c2 * dx + c3 * theta + c4 * dtheta

        # --- 动力学: A [ẍ, θ̈]ᵀ = [u + φ, ψ]ᵀ，θ=0 为向上不稳定平衡 ---
        st, ct = np.sin(theta), np.cos(theta)
        A = np.array(
            [
                [M + m, m * l * ct],
                [m * l * ct, I + m * l * l],
            ],
            dtype=np.float64,
        )
        phi = -b_f * dx + m * l * st * dtheta**2
        psi = m * g * l * st

        A_inv = np.linalg.inv(A)
        f = A_inv @ np.array([phi, psi], dtype=np.float64)       # u=0 时的加速度
        b_u = A_inv @ np.array([1.0, 0.0], dtype=np.float64)     # u 对 [ẍ, θ̈] 的增益

        # ṡ = α + β u
        alpha = c1 * dx + c2 * f[0] + c3 * dtheta + c4 * f[1]
        beta = c2 * b_u[0] + c4 * b_u[1]
        if abs(beta) < 1e-8:
            return 0.0

        # 到达律 ṡ = -k s - η sat(s/φ_b)
        k_reach = 8.0
        eta = 6.0
        phi_b = 0.08
        sat = float(np.clip(s / phi_b, -1.0, 1.0))

        u = -(alpha + k_reach * s + eta * sat) / beta
        return float(np.clip(u, -u_max, u_max))
