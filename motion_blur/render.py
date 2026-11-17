"""工作进程入口：每个任务对应一个曝光时刻，独立求值并光栅化。"""

from __future__ import annotations

from dataclasses import dataclass

from . import scene
from .raster import Image


@dataclass(frozen=True)
class RenderTask:
    index: int
    time: float
    weight: float
    width: int
    height: int
    fail_once: bool = False       # 模拟该时刻首次任务失败，供重试链路验证
    duplicate_once: bool = False  # 模拟重复投递同一时刻，供去重验证


@dataclass(frozen=True)
class RenderResult:
    index: int
    time: float
    weight: float
    rgb: bytes  # 紧密排列的 8 位 RGB 数据，长度 w*h*3
    width: int
    height: int
    attempt: int
    duplicated_delivery: bool


def _render_to_rgb(time: float, w: int, h: int) -> bytes:
    img: Image = scene.render_state(time, w, h)
    return img.to_bytes()


def run_task(task: RenderTask, attempt: int) -> RenderResult:
    """在工作进程中执行。attempt 从 1 开始；注入的失败只在首次触发。"""
    if task.fail_once and attempt == 1:
        raise RuntimeError(
            f"注入故障：时刻样本 #{task.index} (t={task.time:.6f}s) 首次渲染失败"
        )
    rgb = _render_to_rgb(task.time, task.width, task.height)
    return RenderResult(
        index=task.index,
        time=task.time,
        weight=task.weight,
        rgb=rgb,
        width=task.width,
        height=task.height,
        attempt=attempt,
        duplicated_delivery=task.duplicate_once and attempt == 1,
    )
