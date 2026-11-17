"""端到端管线：采样计划 -> 多进程工作进程 -> 重试 -> 去重合并 -> 参考对照。

任务以样本 index（即曝光时刻）为身份。重试与重复投递都不会改变合并
语义：同一时刻的贡献最多被接受一次，所有实际接受/拒绝/遗漏的样本
都会写入任务记录。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

from . import integrate, render
from .raster import Image, grayscale_error_png, write_png
from .render import RenderResult, RenderTask
from .shutter import Schedule, build_schedule


def _worker(payload):
    """模块级函数，可被 multiprocessing pickle。payload=(task, attempt)。

    异常被捕获后作为返回值回传，由调度循环决定重试，而不是让整批
    pool.map 失败、丢失同一批其他时刻的结果。
    """
    task, attempt = payload
    try:
        return render.run_task(task, attempt)
    except Exception as exc:
        return exc


@dataclass
class JobConfig:
    fps: float = 24.0
    shutter_angle: float = 180.0
    samples: int = 24
    curve: str = "trapezoid"
    width: int = 400
    height: int = 300
    workers: int = 4
    max_retries: int = 2
    reference_samples: int = 256
    fail_indices: Sequence[int] = ()      # 首次执行注入故障的样本
    duplicate_indices: Sequence[int] = ()  # 额外重复投递的样本


def _build_tasks(schedule: Schedule, cfg: JobConfig) -> List[RenderTask]:
    tasks: List[RenderTask] = []
    fail_set = set(cfg.fail_indices)
    dup_set = set(cfg.duplicate_indices)
    for s in schedule.samples:
        # 每个时刻的“正本”任务：可能注入首次故障，但不携带重复标记。
        tasks.append(
            RenderTask(
                index=s.index,
                time=s.time,
                weight=s.weight,
                width=cfg.width,
                height=cfg.height,
                fail_once=s.index in fail_set,
                duplicate_once=False,
            )
        )
    # 同一时刻的额外副本：不注入故障，仅标记为重复投递，用于验证去重。
    for idx in cfg.duplicate_indices:
        if idx not in dup_set:
            continue
        s = schedule.samples[idx]
        tasks.append(
            RenderTask(
                index=s.index,
                time=s.time,
                weight=s.weight,
                width=cfg.width,
                height=cfg.height,
                fail_once=False,
                duplicate_once=True,
            )
        )
    return tasks


def _execute(
    tasks: List[RenderTask], workers: int, max_retries: int
) -> List[RenderResult]:
    # 以任务对象（而非时刻 index）为粒度跟踪尝试次数，同一时刻的正本与
    # 副本互不干扰；时刻级别的去重交给合并阶段。
    attempts: Dict[int, int] = {id(t): 0 for t in tasks}
    pending = list(tasks)
    results: List[RenderResult] = []
    errors: List[str] = []

    ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context("fork")
    with ctx.Pool(processes=max(1, workers)) as pool:
        while pending:
            payloads = []
            for t in pending:
                attempts[id(t)] += 1
                payloads.append((t, attempts[id(t)]))
            still_pending: List[RenderTask] = []
            for payload, res in zip(payloads, pool.map(_worker, payloads)):
                task = payload[0]
                if isinstance(res, Exception):
                    if attempts[id(task)] <= max_retries:
                        still_pending.append(task)
                    else:
                        errors.append(str(res))
                else:
                    results.append(res)
            pending = still_pending

    if errors:
        raise RuntimeError("样本在重试上限后仍失败: " + "; ".join(errors))
    return results


def run_job(cfg: JobConfig, out_dir: str) -> dict:
    """执行完整出图任务，写出全部产物并返回元数据字典。"""
    os.makedirs(out_dir, exist_ok=True)
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    schedule = build_schedule(
        cfg.fps, cfg.shutter_angle, cfg.samples, cfg.curve
    )
    ref_schedule = build_schedule(
        cfg.fps, cfg.shutter_angle, cfg.reference_samples, cfg.curve
    )

    tasks = _build_tasks(schedule, cfg)
    results = _execute(tasks, cfg.workers, cfg.max_retries)

    # 保存每个实际接受时刻的锐利状态图，供面板单独查看曝光内状态。
    saved: set = set()
    for res in results:
        if res.index in saved:
            continue
        saved.add(res.index)
        state = Image(cfg.width, cfg.height)
        for i in range(cfg.width * cfg.height):
            o = i * 3
            state.px[i * 4] = res.rgb[o] / 255.0
            state.px[i * 4 + 1] = res.rgb[o + 1] / 255.0
            state.px[i * 4 + 2] = res.rgb[o + 2] / 255.0
            state.px[i * 4 + 3] = 1.0
        write_png(os.path.join(samples_dir, f"sample_{res.index:04d}.png"), state)

    integral, report = integrate.merge_results(
        results, cfg.width, cfg.height, schedule.weights
    )
    write_png(os.path.join(out_dir, "integral.png"), integral)

    # 高密度时间参考：同一条快门曲线、相同区间，更多时刻。
    ref_tasks = [
        RenderTask(
            index=s.index,
            time=s.time,
            weight=s.weight,
            width=cfg.width,
            height=cfg.height,
        )
        for s in ref_schedule.samples
    ]
    ref_results = _execute(ref_tasks, cfg.workers, max_retries=0)
    ref_img, ref_report = integrate.merge_results(
        ref_results, cfg.width, cfg.height, ref_schedule.weights
    )
    write_png(os.path.join(out_dir, "reference.png"), ref_img)

    mean_err, max_err, err_matrix = integrate.pixel_error(integral, ref_img)
    grayscale_error_png(os.path.join(out_dir, "error.png"), err_matrix)

    meta = {
        "config": {
            "fps": cfg.fps,
            "shutter_angle": cfg.shutter_angle,
            "samples": cfg.samples,
            "curve": cfg.curve,
            "width": cfg.width,
            "height": cfg.height,
            "workers": cfg.workers,
            "max_retries": cfg.max_retries,
            "reference_samples": cfg.reference_samples,
            "fail_indices": list(cfg.fail_indices),
            "duplicate_indices": list(cfg.duplicate_indices),
        },
        "schedule": {
            "frame_period": schedule.frame_period,
            "t_open": schedule.t_open,
            "t_close": schedule.t_close,
            "exposure": schedule.exposure,
            "times": schedule.times,
            "weights": schedule.weights,
            "positions": [s.position for s in schedule.samples],
        },
        "reference": {
            "samples": cfg.reference_samples,
            "exposure": ref_schedule.exposure,
        },
        "merge": report.as_dict(),
        "error": {
            "mean_abs": mean_err,
            "max_abs": max_err,
            "gain": 4.0,
        },
        "files": {
            "integral": "integral.png",
            "reference": "reference.png",
            "error": "error.png",
            "sample_dir": "samples",
        },
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta
