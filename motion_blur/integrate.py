"""按快门权重把各时刻样本合成为运动模糊积分图。

合并以样本 index（唯一对应一个曝光时刻）为键：重复投递的结果不会被
二次累加，遗漏的时刻会被显式报告，接受的权重重新归一化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .raster import Image
from .render import RenderResult


@dataclass
class MergeReport:
    accepted: List[int] = field(default_factory=list)
    duplicates_rejected: List[int] = field(default_factory=list)
    missing: List[int] = field(default_factory=list)
    accepted_weight_sum: float = 0.0
    planned_weight_sum: float = 0.0
    attempts: Dict[int, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "duplicates_rejected": self.duplicates_rejected,
            "missing": self.missing,
            "accepted_weight_sum": self.accepted_weight_sum,
            "planned_weight_sum": self.planned_weight_sum,
            "attempts": {str(k): v for k, v in sorted(self.attempts.items())},
        }


def merge_results(
    results: List[RenderResult],
    width: int,
    height: int,
    planned_weights: List[float],
) -> Tuple[Image, MergeReport]:
    """去重合并实际接受的时间样本，返回积分图与合并记录。"""
    n = len(planned_weights)
    report = MergeReport(planned_weight_sum=sum(planned_weights))
    acc = [0.0] * (width * height * 3)
    seen: Dict[int, RenderResult] = {}

    for res in results:
        report.attempts[res.index] = max(
            report.attempts.get(res.index, 0), res.attempt
        )
        if res.index in seen:
            # 同一时刻的重复贡献直接丢弃，绝不二次累加。
            report.duplicates_rejected.append(res.index)
            continue
        seen[res.index] = res

    for idx in range(n):
        if idx not in seen:
            report.missing.append(idx)

    for idx, res in seen.items():
        report.accepted.append(idx)
        report.accepted_weight_sum += res.weight
    report.accepted.sort()

    weight_scale = (
        1.0 / report.accepted_weight_sum if report.accepted_weight_sum > 0 else 0.0
    )
    for idx in report.accepted:
        res = seen[idx]
        w = res.weight * weight_scale
        rgb = res.rgb
        for o in range(len(acc)):
            acc[o] += w * (rgb[o] / 255.0)

    img = Image(width, height)
    for i in range(width * height):
        o = i * 3
        img.px[i * 4] = min(1.0, max(0.0, acc[o]))
        img.px[i * 4 + 1] = min(1.0, max(0.0, acc[o + 1]))
        img.px[i * 4 + 2] = min(1.0, max(0.0, acc[o + 2]))
        img.px[i * 4 + 3] = 1.0
    return img, report


def pixel_error(a: Image, b: Image) -> Tuple[float, float, List[List[float]]]:
    """逐像素 RGB 平均绝对误差，返回 (平均误差, 最大误差, 误差矩阵)。"""
    if a.w != b.w or a.h != b.h:
        raise ValueError("对比图像尺寸不一致")
    matrix: List[List[float]] = [[0.0] * a.w for _ in range(a.h)]
    total = 0.0
    worst = 0.0
    count = a.w * a.h * 3
    for y in range(a.h):
        for x in range(a.w):
            o = (y * a.w + x) * 4
            e = (
                abs(a.px[o] - b.px[o])
                + abs(a.px[o + 1] - b.px[o + 1])
                + abs(a.px[o + 2] - b.px[o + 2])
            ) / 3.0
            matrix[y][x] = e
            total += (
                abs(a.px[o] - b.px[o])
                + abs(a.px[o + 1] - b.px[o + 1])
                + abs(a.px[o + 2] - b.px[o + 2])
            )
            worst = max(worst, e)
    return total / count, worst, matrix
