"""离线出图任务面板的服务端（仅标准库）。

图形入口：python -m motion_blur.webui，浏览器打开 http://127.0.0.1:8000/
面板负责快门区间调整与出图；渲染在后台线程内通过多进程工作进程执行，
参数或渲染异常以 JSON 返回，由采样界面提示，不吞掉时间一致性问题。
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict
from urllib.parse import unquote, urlparse

from .pipeline import JobConfig, run_job
from .shutter import CURVE_LABELS, SHUTTER_CURVES, build_schedule

JOBS_ROOT = Path(__file__).resolve().parent.parent / "jobs"
STATIC_ROOT = Path(__file__).resolve().parent / "static"

_JOBS: Dict[str, dict] = {}
_LOCK = threading.Lock()


def _job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def _run_job_thread(job_id: str, cfg: JobConfig) -> None:
    try:
        meta = run_job(cfg, str(_job_dir(job_id)))
        with _LOCK:
            _JOBS[job_id].update(status="done", meta=meta)
    except Exception as exc:  # 面板需要看到真实失败原因
        with _LOCK:
            _JOBS[job_id].update(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )


def _validate(payload: dict) -> JobConfig:
    required = ("fps", "shutter_angle", "samples", "curve", "width", "height", "workers")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"缺少参数: {', '.join(missing)}")
    if not isinstance(payload["fps"], (int, float)) or payload["fps"] <= 0:
        raise ValueError(f"帧率必须为正数，收到: {payload['fps']!r}")
    if not (0 < float(payload["shutter_angle"]) <= 360):
        raise ValueError(
            f"快门角度必须在 (0, 360] 度内，收到: {payload['shutter_angle']}"
        )
    if int(payload["samples"]) < 1:
        raise ValueError(f"时间样本数必须 >= 1，收到: {payload['samples']}")
    if int(payload["workers"]) < 1:
        raise ValueError(f"工作进程数必须 >= 1，收到: {payload['workers']}")
    if int(payload["width"]) < 1 or int(payload["height"]) < 1:
        raise ValueError("图像宽高必须为正整数")
    if payload["curve"] not in SHUTTER_CURVES:
        raise ValueError(f"未知快门曲线: {payload['curve']!r}")
    if int(payload["width"]) * int(payload["height"]) > 1_600_000:
        raise ValueError("分辨率过大（上限约 160 万像素），请调小后重试")

    def _indices(name):
        raw = payload.get(name, [])
        if isinstance(raw, str):
            raw = [x.strip() for x in raw.split(",") if x.strip()]
        out = [int(x) for x in raw]
        n = int(payload["samples"])
        bad = [x for x in out if x < 0 or x >= n]
        if bad:
            raise ValueError(f"{name} 中的样本序号超出 [0, {n - 1}]: {bad}")
        return out

    return JobConfig(
        fps=float(payload["fps"]),
        shutter_angle=float(payload["shutter_angle"]),
        samples=int(payload["samples"]),
        curve=str(payload["curve"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        workers=int(payload["workers"]),
        max_retries=int(payload.get("max_retries", 2)),
        reference_samples=int(payload.get("reference_samples", 256)),
        fail_indices=tuple(_indices("fail_indices")),
        duplicate_indices=tuple(_indices("duplicate_indices")),
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "MotionBlurPanel/1.0"

    def log_message(self, fmt, *args):  # 安静日志
        pass

    def _send_json(self, obj: dict, code: int = HTTPStatus.OK) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, ctype: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            return self._send_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
        if path == "/app.js":
            return self._send_file(STATIC_ROOT / "app.js", "application/javascript; charset=utf-8")
        if path == "/styles.css":
            return self._send_file(STATIC_ROOT / "styles.css", "text/css; charset=utf-8")
        if path == "/api/curves":
            return self._send_json({
                "curves": {k: CURVE_LABELS[k] for k in sorted(SHUTTER_CURVES)}
            })
        if path == "/api/schedule":
            q = self._query(parsed.query)
            try:
                sched = build_schedule(
                    float(q.get("fps", "24")),
                    float(q.get("angle", "180")),
                    int(q.get("samples", "24")),
                    q.get("curve", "trapezoid"),
                )
                return self._send_json({
                    "frame_period": sched.frame_period,
                    "t_open": sched.t_open,
                    "t_close": sched.t_close,
                    "exposure": sched.exposure,
                    "times": sched.times,
                    "weights": sched.weights,
                    "positions": [s.position for s in sched.samples],
                })
            except ValueError as exc:
                return self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        if path.startswith("/api/jobs/"):
            job_id = path.split("/")[3]
            with _LOCK:
                job = _JOBS.get(job_id)
                if job is None:
                    return self._send_json({"error": "任务不存在"}, HTTPStatus.NOT_FOUND)
                if path.endswith("/status"):
                    return self._send_json({"status": job["status"], "error": job.get("error")})
                if path.endswith("/meta") and job["status"] == "done":
                    return self._send_json(job["meta"])
                return self._send_json(
                    {"error": "任务尚未完成", "status": job["status"]},
                    HTTPStatus.CONFLICT,
                )

        if path == "/jobs" or path.startswith("/jobs/"):
            rel = unquote(path[len("/jobs/"):] if path.startswith("/jobs/") else "")
            target = (JOBS_ROOT / rel).resolve()
            root = JOBS_ROOT.resolve()
            if target != root and root not in target.parents:
                return self.send_error(HTTPStatus.FORBIDDEN)
            if not target.is_file():
                return self.send_error(HTTPStatus.NOT_FOUND)
            ctype = "image/png" if target.suffix == ".png" else "application/json"
            return self._send_file(target, ctype)

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/render":
            return self.send_error(HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            cfg = _validate(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            return self._send_json({"error": f"参数错误: {exc}"}, HTTPStatus.BAD_REQUEST)

        job_id = uuid.uuid4().hex[:12]
        with _LOCK:
            _JOBS[job_id] = {"status": "running", "error": None, "meta": None}
        threading.Thread(
            target=_run_job_thread, args=(job_id, cfg), daemon=True
        ).start()
        self._send_json({"job_id": job_id, "status": "running"})

    @staticmethod
    def _query(query: str) -> dict:
        out = {}
        for pair in query.split("&"):
            if not pair:
                continue
            if "=" in pair:
                k, v = pair.split("=", 1)
                out[k] = v
        return out


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"快门采样出图面板: http://{host}:{port}/  (Ctrl+C 退出)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
