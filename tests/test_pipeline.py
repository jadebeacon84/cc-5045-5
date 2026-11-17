import tempfile
import unittest

from motion_blur.pipeline import JobConfig, run_job
from motion_blur.scene import light_state, render_state
from motion_blur.raster import Image


class SceneConsistencyTests(unittest.TestCase):
    def test_time_actually_changes_state(self):
        a = render_state(0.0, 48, 36)
        b = render_state(0.05, 48, 36)
        changed_pixels = 0
        for i in range(48 * 36):
            o = i * 4
            if any(abs(a.px[o + c] - b.px[o + c]) > 0.02 for c in range(3)):
                changed_pixels += 1
        # 相当比例的像素随时间变化，证明是逐时刻真实求值而非静态图复用
        self.assertGreater(changed_pixels, 48 * 36 * 0.25)

    def test_light_evolves_with_time(self):
        g0 = light_state(0.0)[0]
        g1 = light_state(1.0 / 8.0 / 4.0)[0]  # 8 Hz 正弦的 1/4 周期为峰值
        self.assertNotAlmostEqual(g0, g1)

    def test_transparency_blends_background(self):
        # 玻璃条区域像素不应是纯遮挡色，证明 alpha 合成在起作用。
        img = render_state(0.0, 64, 64)
        # 左侧玻璃条 x 世界坐标约 -0.42，映射到像素左侧区域
        o = (32 * 64 + 8) * 4
        self.assertGreater(img.px[o + 3], 0.99)  # 结果不透明
        # 通道值处于混合区间，而非玻璃条本色
        self.assertLess(img.px[o + 2], 0.95)


class PipelineTests(unittest.TestCase):
    def _base_cfg(self, **kw):
        cfg = JobConfig(
            fps=24, shutter_angle=180, samples=12, curve="trapezoid",
            width=96, height=72, workers=2, max_retries=2, reference_samples=64,
        )
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg

    def test_full_job_writes_products_and_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            meta = run_job(self._base_cfg(), d)
            import os
            for rel in ("integral.png", "reference.png", "error.png", "metadata.json"):
                self.assertTrue(os.path.exists(os.path.join(d, rel)), rel)
            self.assertEqual(len(meta["merge"]["accepted"]), 12)
            self.assertEqual(meta["merge"]["missing"], [])
            self.assertGreaterEqual(meta["error"]["mean_abs"], 0.0)
            samples = os.listdir(os.path.join(d, "samples"))
            self.assertEqual(len(samples), 12)

    def test_retry_succeeds_and_records_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._base_cfg(fail_indices=(2, 5))
            meta = run_job(cfg, d)
            self.assertEqual(meta["merge"]["missing"], [])
            self.assertEqual(meta["merge"]["attempts"]["2"], 2)
            self.assertEqual(meta["merge"]["attempts"]["5"], 2)
            self.assertEqual(meta["merge"]["attempts"]["0"], 1)

    def test_duplicate_delivery_rejected_not_accumulated(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._base_cfg(duplicate_indices=(4,))
            meta = run_job(cfg, d)
            self.assertEqual(meta["merge"]["duplicates_rejected"], [4])
            self.assertEqual(len(meta["merge"]["accepted"]), 12)

    def test_fail_and_duplicate_same_index(self):
        # 同一时刻既注入首次故障又重复投递：正本重试后接受，副本被拒绝，
        # 该时刻的贡献只能被累加一次。
        with tempfile.TemporaryDirectory() as d:
            cfg = self._base_cfg(fail_indices=(4,), duplicate_indices=(4,))
            meta = run_job(cfg, d)
            self.assertEqual(meta["merge"]["missing"], [])
            self.assertEqual(meta["merge"]["duplicates_rejected"], [4])
            self.assertEqual(len(meta["merge"]["accepted"]), 12)
            self.assertEqual(meta["merge"]["attempts"]["4"], 2)

    def test_exhausted_retries_surfaces_error(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._base_cfg(max_retries=0, fail_indices=(1,))
            with self.assertRaises(RuntimeError):
                run_job(cfg, d)

    def test_denser_samples_reduce_error(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            coarse = run_job(self._base_cfg(samples=8), d1)
            fine = run_job(self._base_cfg(samples=48), d2)
            self.assertLess(fine["error"]["mean_abs"], coarse["error"]["mean_abs"])


if __name__ == "__main__":
    unittest.main()
