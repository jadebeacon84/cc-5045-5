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


if __name__ == "__main__":
    unittest.main()
