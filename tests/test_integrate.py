import unittest

from motion_blur.integrate import merge_results, pixel_error
from motion_blur.render import RenderResult


def make_result(index, weight, value=128, w=4, h=3, attempt=1, dup=False):
    rgb = bytes([value, value, value]) * (w * h)
    return RenderResult(index, index * 0.01, weight, rgb, w, h, attempt, dup)


class MergeTests(unittest.TestCase):
    def test_weighted_integral(self):
        weights = [0.25, 0.25, 0.25, 0.25]
        results = [
            make_result(0, 0.25, 0),
            make_result(1, 0.25, 100),
            make_result(2, 0.25, 200),
            make_result(3, 0.25, 255),
        ]
        img, report = merge_results(results, 4, 3, weights)
        expected = (0 + 100 + 200 + 255) / 4 / 255.0
        self.assertAlmostEqual(img.px[0], expected, places=5)
        self.assertEqual(report.missing, [])
        self.assertEqual(report.duplicates_rejected, [])
        self.assertEqual(sorted(report.accepted), [0, 1, 2, 3])

    def test_duplicate_time_not_accumulated(self):
        weights = [0.5, 0.5]
        results = [
            make_result(0, 0.5, 0),
            make_result(1, 0.5, 255),
            make_result(1, 0.5, 255),  # 同一时刻重复投递
        ]
        img, report = merge_results(results, 4, 3, weights)
        self.assertEqual(report.duplicates_rejected, [1])
        self.assertEqual(sorted(report.accepted), [0, 1])
        self.assertAlmostEqual(img.px[0], 0.5, places=5)  # 而非被重复累加

    def test_missing_sample_detected_and_renormalized(self):
        weights = [0.5, 0.5]
        results = [make_result(0, 0.5, 100)]
        img, report = merge_results(results, 4, 3, weights)
        self.assertEqual(report.missing, [1])
        self.assertFalse(report.complete)
        # 仅有的样本重新归一化为全权重
        self.assertAlmostEqual(img.px[0], 100 / 255.0, places=5)

    def test_attempts_recorded(self):
        weights = [1.0]
        results = [make_result(0, 1.0, 50, attempt=3)]
        _, report = merge_results(results, 4, 3, weights)
        self.assertEqual(report.attempts, {0: 3})

    def test_pixel_error_zero_for_identical(self):
        r1 = make_result(0, 1.0, 90, w=5, h=4)
        r2 = make_result(0, 1.0, 90, w=5, h=4)
        from motion_blur.raster import Image

        def to_image(res):
            img = Image(res.width, res.height)
            for i in range(res.width * res.height):
                o = i * 3
                img.px[i * 4] = res.rgb[o] / 255
                img.px[i * 4 + 1] = res.rgb[o + 1] / 255
                img.px[i * 4 + 2] = res.rgb[o + 2] / 255
                img.px[i * 4 + 3] = 1
            return img

        mean_err, worst, matrix = pixel_error(to_image(r1), to_image(r2))
        self.assertEqual(mean_err, 0.0)
        self.assertEqual(worst, 0.0)
        self.assertEqual(len(matrix), 4)


if __name__ == "__main__":
    unittest.main()
