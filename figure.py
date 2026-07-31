"""
控制信号实时可视化：5 路滚动曲线（窗口 10 s）。

绘图在独立进程中运行，避免拖慢 MuJoCo 主循环帧率。
主线程只做降采样投递（默认 50 Hz），示波器约 15 Hz 刷新。

通道:
  0–3: obs = [cart_pos, cart_vel, pole_angle, pole_angvel]
  4:   u   = cart_force
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import deque
from typing import Deque

import numpy as np

_DEFAULT_SAMPLE_HZ = 50.0
_DEFAULT_DRAW_HZ = 15.0


def _scope_worker(
    queue: mp.Queue,
    stop_event: mp.synchronize.Event,
    closed_event: mp.synchronize.Event,
    window_s: float,
    draw_hz: float,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    channels = (
        ("cart_pos", "m"),
        ("cart_vel", "m/s"),
        ("pole_angle", "rad"),
        ("pole_angvel", "rad/s"),
        ("u (force)", "N"),
    )
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
    maxlen = max(64, int(window_s * 100) + 8)

    t_buf: Deque[float] = deque(maxlen=maxlen)
    y_buf: list[Deque[float]] = [deque(maxlen=maxlen) for _ in channels]

    plt.ion()
    fig, axes = plt.subplots(len(channels), 1, sharex=True, figsize=(9, 8))
    fig.subplots_adjust(hspace=0.08, left=0.12, right=0.98, top=0.94, bottom=0.06)
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        manager.set_window_title(title)
    fig.suptitle(f"{title}  (window = {window_s:.0f} s)")

    lines = []
    for ax, (name, unit), color in zip(axes, channels, colors):
        (line,) = ax.plot([], [], color=color, lw=1.0)
        ax.set_ylabel(f"{name}\n[{unit}]", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, window_s)
        ax.set_ylim(-1.0, 1.0)
        lines.append(line)
    axes[-1].set_xlabel("time [s]")
    fig.canvas.draw()
    fig.canvas.flush_events()

    draw_interval = 1.0 / draw_hz
    next_draw = time.perf_counter()
    ylim_countdown = 0

    try:
        while not stop_event.is_set():
            while True:
                try:
                    item = queue.get_nowait()
                except Exception:
                    break
                if item is None:
                    stop_event.set()
                    break
                t, *vals = item
                t_buf.append(float(t))
                for i, v in enumerate(vals):
                    y_buf[i].append(float(v))

            if not plt.fignum_exists(fig.number):
                break

            now = time.perf_counter()
            if now < next_draw or len(t_buf) < 2:
                fig.canvas.flush_events()
                time.sleep(0.001)
                continue
            next_draw = now + draw_interval

            t = np.asarray(t_buf, dtype=np.float64)
            t_max = float(t[-1])
            t_min = t_max - window_s
            mask = t >= t_min
            t_view = t[mask]
            x0 = t_min if t_max >= window_s else 0.0
            x1 = x0 + window_s

            ylim_countdown -= 1
            do_ylim = ylim_countdown <= 0

            for ax, line, buf in zip(axes, lines, y_buf):
                y = np.asarray(buf, dtype=np.float64)[mask]
                line.set_data(t_view, y)
                ax.set_xlim(x0, x1)
                if do_ylim and y.size:
                    y_min = float(y.min())
                    y_max = float(y.max())
                    if y_min == y_max:
                        pad = 0.1 if y_min == 0.0 else 0.1 * abs(y_min)
                    else:
                        pad = 0.12 * (y_max - y_min)
                    ax.set_ylim(y_min - pad, y_max + pad)

            if do_ylim:
                ylim_countdown = 5

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    finally:
        closed_event.set()
        try:
            plt.close(fig)
        except Exception:
            pass


class LiveScope:
    """五个子图的滚动示波器（独立进程，不阻塞仿真）。"""

    def __init__(
        self,
        dt: float,
        *,
        window_s: float = 10.0,
        sample_hz: float = _DEFAULT_SAMPLE_HZ,
        draw_hz: float = _DEFAULT_DRAW_HZ,
        title: str = "Cart-Pole Scope",
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt 必须为正，得到 {dt}")

        self.dt = float(dt)
        self.window_s = float(window_s)
        self._sample_every = max(1, int(round(1.0 / (sample_hz * self.dt))))
        self._step = 0

        # Linux 下用 fork：在创建 MuJoCo viewer 之前启动，避免 spawn 重导入主模块。
        # macOS/Windows 回退 spawn（调用方需保证 if __name__ == "__main__"）。
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")
        self._queue: mp.Queue = ctx.Queue(maxsize=256)
        self._stop = ctx.Event()
        self._closed = ctx.Event()
        self._proc = ctx.Process(
            target=_scope_worker,
            args=(
                self._queue,
                self._stop,
                self._closed,
                self.window_s,
                float(draw_hz),
                title,
            ),
            daemon=True,
        )
        self._proc.start()

    @property
    def is_open(self) -> bool:
        return self._proc.is_alive() and not self._closed.is_set()

    def push(self, t: float, obs: np.ndarray, u: float) -> None:
        """主循环调用：降采样投递，队列满则丢旧，绝不阻塞。"""
        if not self.is_open:
            return

        self._step += 1
        if self._step % self._sample_every != 0:
            return

        obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
        sample = (
            float(t),
            float(obs_arr[0]),
            float(obs_arr[1]),
            float(obs_arr[2]),
            float(obs_arr[3]),
            float(u),
        )
        try:
            self._queue.put_nowait(sample)
        except Exception:
            try:
                self._queue.get_nowait()
            except Exception:
                pass
            try:
                self._queue.put_nowait(sample)
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.join(timeout=1.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=0.5)
