"""快门曲线与曝光时间采样计划。

所有采样时刻都在快门开启区间 [t_open, t_close] 内按分层中点生成，
样本权重取自快门曲线在该时刻的取值，整体归一化到和为 1。
帧率或快门角度改变时，区间长度与采样时刻随之改变。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List


def _box(_u: float) -> float:
    return 1.0


def _trapezoid(u: float) -> float:
    ramp = 0.12
    if u < ramp:
        return u / ramp
    if u > 1.0 - ramp:
        return (1.0 - u) / ramp
    return 1.0


def _gaussian(u: float) -> float:
    sigma = 0.2
    return math.exp(-0.5 * ((u - 0.5) / sigma) ** 2)


def _double_pulse(u: float) -> float:
    # 两个高斯瓣，中间存在低权重区，便于观察非矩形快门。
    return math.exp(-0.5 * ((u - 0.25) / 0.1) ** 2) + math.exp(
        -0.5 * ((u - 0.75) / 0.1) ** 2
    )


SHUTTER_CURVES: Dict[str, Callable[[float], float]] = {
    "box": _box,
    "trapezoid": _trapezoid,
    "gaussian": _gaussian,
    "double_pulse": _double_pulse,
}

CURVE_LABELS: Dict[str, str] = {
    "box": "矩形（均匀权重）",
    "trapezoid": "梯形（软快门）",
    "gaussian": "高斯（中心权重高）",
    "double_pulse": "双脉冲（两段曝光）",
}


@dataclass(frozen=True)
class Sample:
    index: int
    time: float          # 相对帧起点的绝对时刻（秒）
    weight: float        # 归一化权重
    raw_weight: float    # 快门曲线原始取值
    position: float      # 归一化快门位置 u ∈ (0,1)


@dataclass(frozen=True)
class Schedule:
    samples: List[Sample]
    fps: float
    shutter_angle: float
    frame_period: float  # 帧周期（秒）
    t_open: float        # 快门开启时刻（相对帧起点）
    t_close: float       # 快门关闭时刻
    exposure: float      # 曝光区间长度（秒）
    curve: str

    @property
    def times(self) -> List[float]:
        return [s.time for s in self.samples]

    @property
    def weights(self) -> List[float]:
        return [s.weight for s in self.samples]


def build_schedule(
    fps: float,
    shutter_angle: float,
    sample_count: int,
    curve: str = "box",
    t_open: float = 0.0,
) -> Schedule:
    """根据帧率、快门角度和样本数构造确定性分层采样计划。"""
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError(f"帧率必须为正数，收到: {fps!r}")
    if not isinstance(shutter_angle, (int, float)):
        raise ValueError(f"快门角度必须是数值，收到: {shutter_angle!r}")
    if not (0.0 < shutter_angle <= 360.0):
        raise ValueError(f"快门角度必须在 (0, 360] 度内，收到: {shutter_angle}")
    if not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError(f"采样数必须是 >=1 的整数，收到: {sample_count!r}")
    if curve not in SHUTTER_CURVES:
        raise ValueError(
            f"未知快门曲线 {curve!r}，可选: {sorted(SHUTTER_CURVES)}"
        )

    frame_period = 1.0 / float(fps)
    exposure = frame_period * float(shutter_angle) / 360.0
    close = t_open + exposure
    fn = SHUTTER_CURVES[curve]

    positions = [(i + 0.5) / sample_count for i in range(sample_count)]
    raw = [fn(u) for u in positions]
    total = sum(raw)
    if total <= 0.0:
        raise ValueError("快门曲线在采样区间上的权重总和为 0，无法积分")
    weights = [r / total for r in raw]

    samples = [
        Sample(
            index=i,
            time=t_open + u * exposure,
            weight=weights[i],
            raw_weight=raw[i],
            position=u,
        )
        for i, u in enumerate(positions)
    ]
    return Schedule(
        samples=samples,
        fps=float(fps),
        shutter_angle=float(shutter_angle),
        frame_period=frame_period,
        t_open=float(t_open),
        t_close=close,
        exposure=exposure,
        curve=curve,
    )
