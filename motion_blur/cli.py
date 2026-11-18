"""命令行辅助入口。

图形出图面板（webui 模块）是快门采样的主要入口；CLI 仅用于脚本化
快门采样与无界面环境下的验证。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .pipeline import JobConfig, run_job
from .shutter import CURVE_LABELS, SHUTTER_CURVES, build_schedule


def _cmd_schedule(args: argparse.Namespace) -> int:
    sched = build_schedule(
        args.fps,
        args.angle,
        args.samples,
        args.curve,
        t_open_offset=args.t_open_offset,
        frame_index=args.frame,
    )
    payload = {
        "fps": sched.fps,
        "shutter_angle": sched.shutter_angle,
        "curve": sched.curve,
        "curve_label": CURVE_LABELS[sched.curve],
        "frame_index": sched.frame_index,
        "frame_period": sched.frame_period,
        "frame_start": sched.frame_start,
        "t_open": sched.t_open,
        "t_close": sched.t_close,
        "t_open_offset": sched.t_open_offset,
        "exposure": sched.exposure,
        "samples": [
            {
                "index": s.index,
                "time": s.time,
                "frame_time": s.frame_time,
                "position": s.position,
                "weight": s.weight,
                "raw_weight": s.raw_weight,
            }
            for s in sched.samples
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    cfg = JobConfig(
        fps=args.fps,
        shutter_angle=args.angle,
        samples=args.samples,
        curve=args.curve,
        width=args.width,
        height=args.height,
        workers=args.workers,
        max_retries=args.max_retries,
        reference_samples=args.reference_samples,
        frame_index=args.frame,
        t_open_offset=args.t_open_offset,
        fail_indices=_parse_indices(args.fail),
        duplicate_indices=_parse_indices(args.duplicate),
    )
    meta = run_job(cfg, args.out)
    print(f"任务完成，产物目录: {os.path.abspath(args.out)}")
    print(json.dumps(
        {
            "exposure": meta["schedule"]["exposure"],
            "accepted": len(meta["merge"]["accepted"]),
            "duplicates_rejected": meta["merge"]["duplicates_rejected"],
            "missing": meta["merge"]["missing"],
            "mean_abs_error": meta["error"]["mean_abs"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def _parse_indices(raw: str):
    if not raw:
        return ()
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motion-blur",
        description="快速机械运动离线时间采样渲染器（CLI 为辅助入口，图形面板见 python -m motion_blur.webui）",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("schedule", help="只计算快门采样计划，不渲染")
    p1.add_argument("--fps", type=float, default=24.0)
    p1.add_argument("--angle", type=float, default=180.0, help="快门角度（度）")
    p1.add_argument("--samples", type=int, default=24)
    p1.add_argument("--curve", choices=sorted(SHUTTER_CURVES), default="trapezoid")
    p1.add_argument("--frame", type=int, default=0, help="帧号（>=0 的整数）")
    p1.add_argument(
        "--t-open-offset",
        type=float,
        default=0.0,
        help="快门在本帧内的开启偏移（秒，相对帧起点）",
    )
    p1.set_defaults(func=_cmd_schedule)

    p2 = sub.add_parser("render", help="执行一次完整离线渲染任务")
    p2.add_argument("--out", default="job_out", help="产物目录")
    p2.add_argument("--fps", type=float, default=24.0)
    p2.add_argument("--angle", type=float, default=180.0)
    p2.add_argument("--samples", type=int, default=24)
    p2.add_argument("--curve", choices=sorted(SHUTTER_CURVES), default="trapezoid")
    p2.add_argument("--width", type=int, default=400)
    p2.add_argument("--height", type=int, default=300)
    p2.add_argument("--workers", type=int, default=4)
    p2.add_argument("--max-retries", type=int, default=2)
    p2.add_argument("--reference-samples", type=int, default=256)
    p2.add_argument("--frame", type=int, default=0, help="帧号（>=0 的整数）")
    p2.add_argument(
        "--t-open-offset",
        type=float,
        default=0.0,
        help="快门在本帧内的开启偏移（秒，相对帧起点）",
    )
    p2.add_argument("--fail", default="", help="注入首次故障的样本序号，逗号分隔，如 3,7")
    p2.add_argument("--duplicate", default="", help="注入重复投递的样本序号，逗号分隔")
    p2.set_defaults(func=_cmd_render)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"渲染失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
