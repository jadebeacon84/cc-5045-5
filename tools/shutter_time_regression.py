"""快门跨帧时间映射回归（可复跑，纯标准库）。

用解析匀速直线与解析匀速旋转轨迹，核验采样时刻 -> 物体位置/相位的时间
映射。脚本同时实现三种映射并对照解析积分：

- absolute：修补后的正确映射，t = frame*T + t_open_offset + u*exposure；
- wrap：    错误映射，把绝对时刻对帧周期取模回绕（帧号换算 bug）；
- drop：    错误映射，丢弃帧号只留帧内时刻（默认值遮错）。

复跑：python3 tools/shutter_time_regression.py
退出码 0 表示正确映射在所有用例上达标，且两种错误映射都被证伪（失败证据）。
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_blur.shutter import build_schedule

V_LINEAR = 7.5          # 解析匀速直线速度（世界单位/秒）
OMEGA = 2 * math.pi * 2  # 解析匀速旋转角速度（弧度/秒）


def exact_linear(a: float, b: float) -> float:
    # 1/(b-a) int_a^b v*t dt
    return V_LINEAR * (a + b) / 2.0


def exact_cos(a: float, b: float) -> float:
    # 1/(b-a) int_a^b cos(omega*t) dt
    return (math.sin(OMEGA * b) - math.sin(OMEGA * a)) / (OMEGA * (b - a))


def sampled(times, weights):
    linear = sum(V_LINEAR * t * w for t, w in zip(times, weights))
    rot = sum(math.cos(OMEGA * t) * w for t, w in zip(times, weights))
    return linear, rot


def run_case(name, fps, angle, frame_index, t_open_offset, n=64, curve="box"):
    T = 1.0 / fps
    sched = build_schedule(
        fps, angle, n, curve,
        t_open_offset=t_open_offset, frame_index=frame_index,
    )
    correct = sched.times
    # 两种典型错误映射，仅用于留证，不会进入渲染链路。
    wrapped = [t % T for t in correct]
    dropped = sched.frame_times

    a, b = sched.t_open, sched.t_close
    xl_exact, cos_exact = exact_linear(a, b), exact_cos(a, b)

    rows = []
    for label, ts in (("absolute", correct), ("wrap", wrapped), ("drop", dropped)):
        xl, cr = sampled(ts, sched.weights)
        rows.append((label, abs(xl - xl_exact), abs(cr - cos_exact)))
    return name, T, sched, rows, xl_exact, cos_exact


def main() -> int:
    cases = [
        # 非对称快门贴近帧末尾（帧内合法），非零帧号
        ("asymmetric-near-boundary", 24.0, 90.0, 3, 0.7 / 24.0),
        # 非对称快门恰好闭合在帧末尾
        ("asymmetric-closes-at-edge", 24.0, 180.0, 5, 0.5 / 24.0),
        # 不同帧率与软快门曲线
        ("trapezoid-offset", 30.0, 144.0, 2, 0.2 / 30.0, 64, "trapezoid"),
        ("gaussian-offset", 60.0, 120.0, 7, 0.25 / 60.0, 64, "gaussian"),
    ]

    ok = True
    for case in cases:
        name, T, sched, rows, xl_exact, cos_exact = run_case(*case)
        print(f"\n[{name}] fps 周期 T={T:.6f}s frame=#{sched.frame_index} "
              f"窗口=[{sched.t_open:.6f},{sched.t_close:.6f}]s")
        for label, e_lin, e_rot in rows:
            mark = "PASS" if label == "absolute" else "BUG-EVIDENCE"
            print(f"  {label:9s} 线性位置误差={e_lin:.3e}  旋转相位误差={e_rot:.3e}  {mark}")
        abs_lin, abs_rot = rows[0][1], rows[0][2]
        wrap_lin, wrap_rot = rows[1][1], rows[1][2]
        drop_lin, drop_rot = rows[2][1], rows[2][2]
        # 正确映射必须收敛到解析积分
        if not (abs_lin < 1e-9 and abs_rot < 1e-3):
            ok = False
            print("  !! 正确映射未达标")
        # 错误映射必须显著偏离，作为失败证据保留（不能被默认值遮掉）
        if not (wrap_lin > abs_lin * 100 and drop_lin > abs_lin * 100):
            ok = False
            print("  !! 错误映射未被证伪，回归失效")
        if not (wrap_rot > abs_rot * 10 and drop_rot > abs_rot * 10):
            ok = False
            print("  !! 旋转相位错误证据不足")

    # 跨帧窗口必须被显式拒绝，而不是静默回绕
    print("\n[cross-frame rejection]")
    try:
        build_schedule(24.0, 180.0, 8, "box", t_open_offset=0.6 / 24.0)
    except ValueError as exc:
        print("  已拒绝越帧窗口:", str(exc).split("：", 1)[-1][:48], "...")
    else:
        ok = False
        print("  !! 跨帧窗口未被拒绝")

    print("\n结果:", "全部通过（正确映射达标，错误映射留证）" if ok else "存在回归失败")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
