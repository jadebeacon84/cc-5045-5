"""快门曲线与曝光时间采样计划。

所有采样时刻都在快门开启区间 [t_open, t_close] 内按分层中点生成，
样本权重取自快门曲线在该时刻的取值，整体归一化到和为 1。
帧率或快门角度改变时，区间长度与采样时刻随之改变。

时间映射分两层，绝不允许互相混用：

- ``t_open_offset``：快门在**本帧内**的开启偏移（相对帧起点，秒）；
- ``frame_index``：帧号（非负整数）。帧起点的绝对时刻为
  ``frame_start = frame_index * frame_period``；
- 每个样本的绝对时刻 ``time = frame_start + t_open_offset + u * exposure``，
  连续时间场景只对该绝对时刻求值。

跨帧边界（非对称快门使 t_close 越过本帧末尾）时，绝不能把绝对时刻
对帧周期取模回绕、也不能丢弃帧号——否则匀速/旋转轨迹的物体会在错误
相位被取样。``build_schedule`` 会校验快门区间必须闭合在本帧周期内，
把越界作为错误显式抛出，而不是静默回绕。
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
    time: float          # 绝对采样时刻（秒）：frame_start + 帧内时刻
    frame_time: float    # 相对本帧起点的帧内时刻（秒）
    weight: float        # 归一化权重
    raw_weight: float    # 快门曲线原始取值
    position: float      # 归一化快门位置 u ∈ (0,1)


@dataclass(frozen=True)
class Schedule:
    samples: List[Sample]
    fps: float
    shutter_angle: float
    frame_index: int
    frame_period: float  # 帧周期（秒）
    frame_start: float   # 本帧起点的绝对时刻（秒）
    t_open: float        # 快门开启的绝对时刻（秒）
    t_close: float       # 快门关闭的绝对时刻（秒）
    t_open_offset: float  # 快门开启相对本帧起点的偏移（秒）
    exposure: float      # 曝光区间长度（秒）
    curve: str

    @property
    def times(self) -> List[float]:
        return [s.time for s in self.samples]

    @property
    def frame_times(self) -> List[float]:
        return [s.frame_time for s in self.samples]

    @property
    def weights(self) -> List[float]:
        return [s.weight for s in self.samples]


def build_schedule(
    fps: float,
    shutter_angle: float,
    sample_count: int,
    curve: str = "box",
    t_open_offset: float = 0.0,
    frame_index: int = 0,
) -> Schedule:
    """根据帧率、快门角度、帧号和帧内开启偏移构造确定性分层采样计划。

    ``frame_index`` 给出帧号，``t_open_offset`` 给出快门在本帧内的开启
    偏移（相对帧起点）。样本绝对时刻由 ``frame_start + t_open_offset +
    u*exposure`` 得到，连续时间场景只对绝对时刻求值，不做任何帧周期回绕。
    快门区间必须闭合在本帧周期内，越界（含跨帧）会显式报错。
    """
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
    if not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError(f"帧号必须是 >=0 的整数，收到: {frame_index!r}")
    if not isinstance(t_open_offset, (int, float)) or t_open_offset < 0:
        raise ValueError(
            f"帧内快门开启偏移必须为非负数值，收到: {t_open_offset!r}"
        )

    frame_period = 1.0 / float(fps)
    exposure = frame_period * float(shutter_angle) / 360.0
    offset = float(t_open_offset)
    # 快门区间必须落在本帧内：跨帧边界属于非法时间映射，显式拒绝，
    # 绝不静默地把绝对时刻对帧周期取模回绕。
    if offset + exposure > frame_period + 1e-12:
        raise ValueError(
            "快门区间越过本帧边界：t_open_offset={:.6f}s + exposure={:.6f}s "
            "= {:.6f}s > 帧周期 {:.6f}s；跨帧取样会把运动物体定位到错误相位，"
            "请减小快门角度或开启偏移。".format(
                offset, exposure, offset + exposure, frame_period
            )
        )

    frame_start = frame_period * frame_index
    open_abs = frame_start + offset
    close_abs = open_abs + exposure
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
            time=open_abs + u * exposure,
            frame_time=offset + u * exposure,
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
        frame_index=frame_index,
        frame_period=frame_period,
        frame_start=frame_start,
        t_open=open_abs,
        t_close=close_abs,
        t_open_offset=offset,
        exposure=exposure,
        curve=curve,
    )
