"""软件光栅化与 PNG 编码（仅使用标准库）。

像素以浮点 RGBA 存储，值范围 [0, 1]。绘制顺序由调用方保证
（远 -> 近），透明度通过标准 source-over 合成处理。
"""

from __future__ import annotations

import math
import struct
import zlib
from typing import List, Sequence, Tuple

RGB = Tuple[float, float, float]


class Image:
    __slots__ = ("w", "h", "px")

    def __init__(self, w: int, h: int, bg: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        if w <= 0 or h <= 0:
            raise ValueError("图像尺寸必须为正整数")
        self.w = w
        self.h = h
        self.px: List[float] = [0.0] * (w * h * 4)
        r, g, b, a = bg
        for i in range(w * h):
            o = i * 4
            self.px[o] = r
            self.px[o + 1] = g
            self.px[o + 2] = b
            self.px[o + 3] = a

    def composite_span(self, x0: int, x1: int, y: int, rgb: RGB, alpha: float) -> None:
        x0 = max(0, x0)
        x1 = min(self.w, x1)
        if x1 <= x0:
            return
        sr, sg, sb = rgb
        for x in range(x0, x1):
            o = (y * self.w + x) * 4
            da = self.px[o + 3]
            out_a = alpha + da * (1.0 - alpha)
            if out_a <= 0.0:
                continue
            self.px[o] = (sr * alpha + self.px[o] * da * (1.0 - alpha)) / out_a
            self.px[o + 1] = (sg * alpha + self.px[o + 1] * da * (1.0 - alpha)) / out_a
            self.px[o + 2] = (sb * alpha + self.px[o + 2] * da * (1.0 - alpha)) / out_a
            self.px[o + 3] = out_a

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, rgb: RGB, alpha: float = 1.0) -> None:
        for y in range(max(0, y0), min(self.h, y1)):
            self.composite_span(x0, x1, y, rgb, alpha)

    def fill_circle(self, cx: float, cy: float, radius: float, rgb: RGB, alpha: float = 1.0) -> None:
        if radius <= 0:
            return
        y_lo = max(0, int(math.floor(cy - radius)))
        y_hi = min(self.h - 1, int(math.ceil(cy + radius)))
        r2 = radius * radius
        for y in range(y_lo, y_hi + 1):
            dy = y + 0.5 - cy
            half = math.sqrt(max(0.0, r2 - dy * dy))
            x0 = int(math.ceil(cx - half - 0.5))
            x1 = int(math.floor(cx + half - 0.5)) + 1
            self.composite_span(x0, x1, y, rgb, alpha)

    def fill_polygon(
        self, points: Sequence[Tuple[float, float]], rgb: RGB, alpha: float = 1.0
    ) -> None:
        """扫描线多边形填充，支持凹多边形（每扫描行做奇偶配对）。"""
        if len(points) < 3:
            return
        y_min = max(0, int(math.floor(min(p[1] for p in points))) )
        y_max = min(self.h - 1, int(math.ceil(max(p[1] for p in points))) - 1)
        for y in range(y_min, y_max + 1):
            yc = y + 0.5
            xs: List[float] = []
            for i in range(len(points)):
                ax, ay = points[i]
                bx, by = points[(i + 1) % len(points)]
                if ay == by:
                    continue
                if min(ay, by) <= yc < max(ay, by):
                    xs.append(ax + (yc - ay) * (bx - ax) / (by - ay))
            xs.sort()
            for k in range(0, len(xs) - 1, 2):
                x0 = int(math.ceil(xs[k] - 0.5))
                x1 = int(math.floor(xs[k + 1] - 0.5)) + 1
                self.composite_span(x0, x1, y, rgb, alpha)

    def to_bytes(self) -> bytes:
        out = bytearray()
        for i in range(self.w * self.h):
            o = i * 4
            for c in range(3):  # 背景始终不透明，丢弃 alpha 通道
                v = self.px[o + c]
                out.append(0 if v <= 0 else 255 if v >= 1 else int(round(v * 255)))
        return bytes(out)


def write_png(path: str, img: Image) -> None:
    """将 Image 编码为 8 位 RGB PNG（无第三方依赖）。"""
    raw = bytearray()
    data = img.to_bytes()
    stride = img.w * 3
    for y in range(img.h):
        raw.append(0)  # filter type 0: None
        raw.extend(data[y * stride:(y + 1) * stride])
    compressed = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", img.w, img.h, 8, 2, 0, 0, 0)  # 8-bit RGB
    with open(path, "wb") as f:
        f.write(header)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))


def grayscale_error_png(path: str, error: List[List[float]], gain: float = 4.0) -> None:
    """把逐像素平均误差矩阵编码为放大显示的灰度 PNG。"""
    h = len(error)
    w = len(error[0]) if h else 0
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            v = min(1.0, error[y][x] * gain)
            raw.append(int(round(v * 255)))
    compressed = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))
