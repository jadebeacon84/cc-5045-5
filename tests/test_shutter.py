import math
import unittest

from motion_blur.shutter import SHUTTER_CURVES, build_schedule


class ShutterTests(unittest.TestCase):
    def test_weights_normalized_and_distinct_times(self):
        sched = build_schedule(fps=24, shutter_angle=180, sample_count=32, curve="gaussian")
        self.assertAlmostEqual(sum(sched.weights), 1.0, places=12)
        self.assertEqual(len(set(sched.times)), 32)
        for t in sched.times:
            self.assertGreaterEqual(t, sched.t_open)
            self.assertLessEqual(t, sched.t_close)

    def test_frame_rate_and_angle_change_range(self):
        base = build_schedule(24, 180, 16)
        wide = build_schedule(24, 360, 16)
        slow = build_schedule(12, 180, 16)
        # 快门角度翻倍 -> 曝光区间翻倍
        self.assertAlmostEqual(wide.exposure, 2 * base.exposure, places=12)
        # 帧率减半 -> 帧周期与曝光均翻倍，采样时刻整体平移
        self.assertAlmostEqual(slow.exposure, 2 * base.exposure, places=12)
        self.assertNotEqual(wide.times, base.times)
        self.assertNotEqual(slow.times, base.times)
        # 360 度快门覆盖整个帧周期
        self.assertAlmostEqual(wide.t_close, wide.frame_period, places=12)

    def test_stratified_positions_inside_open_interval(self):
        sched = build_schedule(30, 90, 8)
        for s in sched.samples:
            self.assertGreater(s.position, 0.0)
            self.assertLess(s.position, 1.0)

    def test_all_curves_supported(self):
        for name in SHUTTER_CURVES:
            sched = build_schedule(24, 180, 20, curve=name)
            self.assertAlmostEqual(sum(sched.weights), 1.0, places=12)

    def test_invalid_arguments_rejected(self):
        with self.assertRaises(ValueError):
            build_schedule(0, 180, 10)
        with self.assertRaises(ValueError):
            build_schedule(24, 0, 10)
        with self.assertRaises(ValueError):
            build_schedule(24, 400, 10)
        with self.assertRaises(ValueError):
            build_schedule(24, 180, 0)
        with self.assertRaises(ValueError):
            build_schedule(24, 180, 10, curve="nope")

    def test_nonuniform_curve_changes_weights(self):
        box = build_schedule(24, 180, 16, "box")
        gauss = build_schedule(24, 180, 16, "gaussian")
        # 同一时刻集合，不同权重
        self.assertEqual(box.times, gauss.times)
        self.assertNotAlmostEqual(box.weights[0], gauss.weights[0], places=8)
        self.assertGreater(gauss.weights[8], gauss.weights[0])


class FrameTimeMappingTests(unittest.TestCase):
    """跨帧边界时间映射回归：采样时刻与最终像素同时核验。"""

    def test_frame_index_shifts_absolute_time_keeps_frame_time(self):
        # 非对称快门：帧内偏移 0.4T、180 度曝光；改帧号只平移绝对时刻。
        fps, angle, n = 24.0, 180.0, 12
        T = 1.0 / fps
        offset = 0.4 * T
        f0 = build_schedule(fps, angle, n, "box", t_open_offset=offset, frame_index=0)
        f3 = build_schedule(fps, angle, n, "box", t_open_offset=offset, frame_index=3)
        self.assertEqual(f0.frame_times, f3.frame_times)
        for a, b in zip(f0.times, f3.times):
            self.assertAlmostEqual(b - a, 3 * T, places=15)
        # 绝对时刻 = 帧起点 + 帧内时刻，绝不是帧内时刻本身
        for s in f3.samples:
            self.assertAlmostEqual(s.time, f3.frame_start + s.frame_time, places=15)
            self.assertAlmostEqual(s.time, 3 * T + offset + s.position * f3.exposure, places=15)

    def test_no_mod_wrapping_at_frame_boundary(self):
        # 快门贴近帧末尾：任何样本都不得被取模回绕到帧起点附近。
        fps, n = 24.0, 16
        T = 1.0 / fps
        # 90 度 -> exposure=0.25T，偏移 0.7T，闭合于 0.95T（帧内、贴近边界）
        sched = build_schedule(fps, 90.0, n, "box", t_open_offset=0.7 * T, frame_index=2)
        frame_start, frame_end = 2 * T, 3 * T
        for s in sched.samples:
            self.assertGreaterEqual(s.time, frame_start)
            self.assertLess(s.time, frame_end)
            # 若错误地 mod T，绝对时刻会跌回 < frame_start
            self.assertGreaterEqual(s.time % T, 0.7 * T - 1e-12)

    def test_cross_frame_window_is_rejected_not_silently_wrapped(self):
        T = 1.0 / 24.0
        # 偏移 + 曝光越过帧周期：必须显式报错，禁止静默回绕遮住偏差
        with self.assertRaises(ValueError):
            build_schedule(24.0, 180.0, 8, "box", t_open_offset=0.6 * T)
        with self.assertRaises(ValueError):
            build_schedule(24.0, 360.0, 8, "box", t_open_offset=0.01)
        # 合法的非对称快门（恰好闭合在帧末尾）不被误杀
        ok = build_schedule(24.0, 180.0, 8, "box", t_open_offset=0.5 * T)
        self.assertAlmostEqual(ok.t_close, ok.frame_period, places=12)

    def test_invalid_frame_index_and_offset_rejected(self):
        with self.assertRaises(ValueError):
            build_schedule(24, 180, 8, frame_index=-1)
        with self.assertRaises(ValueError):
            build_schedule(24, 180, 8, t_open_offset=-0.01)

    def test_analytic_uniform_motion_integral_matches_exact(self):
        # 解析匀速直线 x=v*t：加权采样积分必须等于解析积分 v*(a+b)/2。
        fps, n = 24.0, 64
        T = 1.0 / fps
        offset, frame = 0.3 * T, 5
        sched = build_schedule(fps, 144.0, n, "trapezoid",
                               t_open_offset=offset, frame_index=frame)
        v = 7.5
        a, b = sched.t_open, sched.t_close
        sampled = sum(v * s.time * s.weight for s in sched.samples)
        exact = v * (a + b) / 2.0
        self.assertAlmostEqual(sampled, exact, places=10)
        # 对照：丢弃帧号会产生与帧号成比例的系统性偏差，回归必须捕获
        dropped = sum(v * s.frame_time * s.weight for s in sched.samples)
        self.assertGreater(abs(dropped - exact), v * frame * T * 0.9)

    def test_analytic_rotation_integral_converges_to_exact(self):
        # 解析匀速旋转 theta=omega*t：cos 的采样平均随密度收敛到解析积分。
        fps = 24.0
        T = 1.0 / fps
        omega = 2.0 * math.pi * 2.0
        offset, frame = 0.2 * T, 4
        a = frame * T + offset

        def exact_cos(exposure):
            b = a + exposure
            return (math.sin(omega * b) - math.sin(omega * a)) / (omega * (b - a))

        coarse = build_schedule(fps, 120.0, 16, "box",
                                t_open_offset=offset, frame_index=frame)
        fine = build_schedule(fps, 120.0, 512, "box",
                              t_open_offset=offset, frame_index=frame)
        exact = exact_cos(coarse.exposure)
        err_coarse = abs(sum(math.cos(omega * s.time) * s.weight
                             for s in coarse.samples) - exact)
        err_fine = abs(sum(math.cos(omega * s.time) * s.weight
                           for s in fine.samples) - exact)
        self.assertLess(err_fine, err_coarse)
        self.assertLess(err_fine, 1e-5)

    def test_exposure_centroid_matches_curve_centroid(self):
        # 曝光重心（加权平均时刻）必须与快门曲线的解析重心一致，
        # 且随帧号整体平移、不改变帧内重心。
        for curve in SHUTTER_CURVES:
            sched = build_schedule(24, 200, 4096, curve,
                                   t_open_offset=0.1 / 24, frame_index=7)
            centroid = sum(s.time * s.weight for s in sched.samples)
            frame_centroid = sum(s.frame_time * s.weight for s in sched.samples)
            # 对称曲线的帧内重心落在曝光区间中点
            self.assertAlmostEqual(
                frame_centroid,
                sched.t_open_offset + sched.exposure / 2.0,
                places=9,
            )
            self.assertAlmostEqual(centroid, sched.frame_start + frame_centroid, places=9)


class PixelTimeRegressionTests(unittest.TestCase):
    """最终像素回归：同一帧内时刻的渲染结果必须只由绝对时间决定。"""

    def test_rendered_pixel_uses_absolute_time(self):
        from motion_blur.scene import render_state
        # 帧 k 内的帧内时刻 ft，其绝对时间为 k*T+ft；渲染该绝对时间必须
        # 与直接渲染同一绝对时间一致（证明场景拿到的是绝对时间）。
        fps = 24.0
        T = 1.0 / fps
        frame, offset = 3, 0.35 * T
        sched = build_schedule(fps, 180.0, 8, "gaussian",
                               t_open_offset=offset, frame_index=frame)
        for s in sched.samples:
            via_schedule = render_state(s.time, 48, 36)
            direct = render_state(frame * T + s.frame_time, 48, 36)
            # 两条路径数学上同一绝对时刻，仅允许末位浮点舍入差异
            for a, b in zip(via_schedule.px, direct.px):
                self.assertAlmostEqual(a, b, places=12)
            # 负向证据：用帧内时刻（丢帧号）渲染，跨帧后必然是不同状态
            wrong = render_state(s.frame_time, 48, 36)
            self.assertNotEqual(wrong.px, direct.px)


if __name__ == "__main__":
    unittest.main()
