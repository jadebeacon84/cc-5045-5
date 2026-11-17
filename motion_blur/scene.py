"""连续时间场景求值。

所有物体的位姿与光照都由绝对时间 t 直接求值，不存在“先渲染一张锐利
图再施加方向模糊”的捷径。每个时间样本都会得到该时刻独立的场景状态，
遮挡关系、透明物体与光照强度因此随时间保持一致。

世界坐标 x,y ∈ [-1, 1]，通过 view 映射到像素。
深度 z 仅用于同一样本内的 back-to-front 绘制排序（画家算法）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .raster import Image, RGB

Point = Tuple[float, float]

ROTOR_SPEED_RPS = 2.0      # 叶片每秒转数
ROTOR_CENTER = (-0.25, 0.05)
BALL_FREQ_HZ = 3.2         # 小球水平往复频率（保证一次曝光内穿过遮挡条）
BALL_BOUNCE_HZ = 1.8       # 小球竖直起伏频率
LIGHT_FLICKER_HZ = 8.0     # 光照闪烁频率
LIGHT_DIRECTION_SPEED = 1.5  # 光照方向旋转速度（转/秒）


@dataclass(frozen=True)
class View:
    w: int
    h: int
    margin: float = 0.08

    def to_px(self, x: float, y: float) -> Point:
        # 以较短边为基准统一缩放，保证世界坐标在 x/y 方向等比例，
        # 圆形不会因画幅宽高比被拉伸，场景居中显示。
        scale = min(self.w, self.h) * (1.0 - 2.0 * self.margin) / 2.0
        px = self.w / 2.0 + x * scale
        py = self.h / 2.0 - y * scale
        return px, py


@dataclass(frozen=True)
class Primitive:
    kind: str             # "polygon" | "circle" | "rect"
    points: List[Point]
    rgb: RGB
    alpha: float
    z: float
    label: str


def light_state(t: float) -> Tuple[float, float, float]:
    """返回 (闪烁增益, 光源方向角, 环境强度)，全部随时间变化。"""
    gain = 0.95 + 0.25 * math.sin(2.0 * math.pi * LIGHT_FLICKER_HZ * t)
    angle = 2.0 * math.pi * LIGHT_DIRECTION_SPEED * t
    ambient = 0.72 + 0.08 * math.sin(2.0 * math.pi * 0.5 * t)
    return gain, angle, ambient


def _shade(base: RGB, t: float, normal_angle: float) -> RGB:
    gain, light_angle, ambient = light_state(t)
    diffuse = max(0.0, math.cos(normal_angle - light_angle))
    factor = (ambient + (1.0 - ambient) * diffuse) * gain
    return (
        min(1.0, base[0] * factor),
        min(1.0, base[1] * factor),
        min(1.0, base[2] * factor),
    )


def _blade_polygon(angle: float) -> List[Point]:
    r_in, r_out = 0.1, 0.52
    half_in, half_out = 0.36, 0.14
    a0, a1 = angle - half_in, angle + half_in
    b0, b1 = angle - half_out, angle + half_out
    cx, cy = ROTOR_CENTER
    return [
        (cx + r_in * math.cos(a0), cy + r_in * math.sin(a0)),
        (cx + r_out * math.cos(b0), cy + r_out * math.sin(b0)),
        (cx + r_out * math.cos(b1), cy + r_out * math.sin(b1)),
        (cx + r_in * math.cos(a1), cy + r_in * math.sin(a1)),
    ]


def evaluate(t: float) -> List[Primitive]:
    """求时刻 t 的完整场景状态（不含背景），按深度从远到近返回。"""
    prims: List[Primitive] = []

    # 旋转叶片（中景）：四叶转子，角度由连续时间积分得到。
    rotor_angle = 2.0 * math.pi * ROTOR_SPEED_RPS * t
    blade_colors: List[RGB] = [
        (0.62, 0.80, 1.0),
        (0.95, 0.78, 0.55),
        (0.62, 0.80, 1.0),
        (0.95, 0.78, 0.55),
    ]
    for k in range(4):
        a = rotor_angle + k * math.pi / 2.0
        normal = a
        prims.append(
            Primitive(
                kind="polygon",
                points=_blade_polygon(a),
                rgb=_shade(blade_colors[k], t, normal),
                alpha=1.0,
                z=0.5,
                label=f"blade_{k}",
            )
        )

    # 转子轮毂（与叶片同深度层，视觉锚点）。
    prims.append(
        Primitive(
            kind="circle",
            points=[(ROTOR_CENTER[0], ROTOR_CENTER[1], 0.075)],
            rgb=_shade((0.82, 0.86, 0.92), t, 0.0),
            alpha=1.0,
            z=0.49,
            label="hub",
        )
    )

    # 快速移动小球（位于叶片之后），水平行程在一次曝光内穿过右侧半透明遮挡条。
    phase = 0.7
    bx = 0.5 * math.sin(2.0 * math.pi * BALL_FREQ_HZ * t + phase) + 0.3
    by = 0.3 + 0.12 * math.sin(2.0 * math.pi * BALL_BOUNCE_HZ * t + 0.8)
    prims.append(
        Primitive(
            kind="circle",
            points=[(bx, by, 0.085)],  # (cx, cy, radius)
            rgb=_shade((0.95, 0.45, 0.14), t, 0.0),
            alpha=1.0,
            z=0.65,
            label="ball",
        )
    )

    # 前景半透明遮挡条：物体从其后穿过，演示轮廓重叠与透明合成。
    prims.append(
        Primitive(
            kind="rect",
            points=[(-0.72, -1.0), (-0.52, 1.0)],
            rgb=(0.65, 0.82, 0.90),
            alpha=0.32,
            z=0.15,
            label="glass_bar_left",
        )
    )
    prims.append(
        Primitive(
            kind="rect",
            points=[(0.62, -1.0), (0.8, 1.0)],
            rgb=(0.92, 0.80, 0.60),
            alpha=0.4,
            z=0.1,
            label="glass_bar_right",
        )
    )

    prims.sort(key=lambda p: -p.z)  # z 大者更远，先画
    return prims


def _draw_background(img: Image, view: View, t: float) -> None:
    bands = 48
    gain = light_state(t)[0]
    for i in range(bands):
        y0 = int(i * img.h / bands)
        y1 = max(y0 + 1, int((i + 1) * img.h / bands))
        f = i / (bands - 1)
        # 背景也受时变光照影响，保证全场景时间一致性。
        img.fill_rect(
            0, y0, img.w, y1,
            (
                min(1.0, (0.09 + 0.10 * f) * gain),
                min(1.0, (0.12 + 0.14 * f) * gain),
                min(1.0, (0.18 + 0.20 * f) * gain),
            ),
        )


def render_state(t: float, w: int, h: int) -> Image:
    """渲染单个时刻 t 的锐利场景状态。"""
    view = View(w, h)
    img = Image(w, h, bg=(0.05, 0.07, 0.11, 1.0))
    _draw_background(img, view, t)
    for p in evaluate(t):
        if p.kind == "polygon":
            pts = [view.to_px(x, y) for x, y in p.points]
            img.fill_polygon(pts, p.rgb, p.alpha)
        elif p.kind == "circle":
            cx, cy, radius = p.points[0]
            px, py = view.to_px(cx, cy)
            scale = min(view.w, view.h) * (1.0 - 2.0 * view.margin) / 2.0
            img.fill_circle(px, py, radius * scale, p.rgb, p.alpha)
        elif p.kind == "rect":
            (x0, y0), (x1, y1) = p.points
            xa, ya = view.to_px(x0, y0)
            xb, yb = view.to_px(x1, y1)
            img.fill_rect(int(xa), int(yb), int(xb), int(ya), p.rgb, p.alpha)
    return img
