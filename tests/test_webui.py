import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from motion_blur.webui import Handler


class WebPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path):
        return urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10)

    def _post(self, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/render",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_curves_and_static(self):
        with self._get("/api/curves") as resp:
            curves = json.loads(resp.read())["curves"]
        self.assertIn("trapezoid", curves)
        with self._get("/") as resp:
            self.assertEqual(resp.status, 200)

    def test_schedule_reflects_angle_and_fps(self):
        with self._get("/api/schedule?fps=60&angle=90&samples=8&curve=box") as resp:
            data = json.loads(resp.read())
        self.assertAlmostEqual(data["exposure"], 1.0 / 60 * 0.25, places=12)
        self.assertEqual(len(data["times"]), 8)

    def test_schedule_absolute_time_uses_frame_index(self):
        T = 1.0 / 24.0
        url = (
            "/api/schedule?fps=24&angle=90&samples=8&curve=box"
            f"&frame_index=3&t_open_offset={0.7 * T}"
        )
        with self._get(url) as resp:
            data = json.loads(resp.read())
        self.assertEqual(data["frame_index"], 3)
        self.assertAlmostEqual(data["frame_start"], 3 * T, places=12)
        self.assertAlmostEqual(data["t_open"], 3 * T + 0.7 * T, places=12)
        self.assertGreater(data["times"][0], 3 * T)
        self.assertEqual(len(data["frame_times"]), 8)

    def test_schedule_cross_frame_rejected_with_400(self):
        T = 1.0 / 24.0
        url = (
            "/api/schedule?fps=24&angle=180&samples=8&curve=box"
            f"&t_open_offset={0.6 * T}"
        )
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get(url)
        self.assertEqual(ctx.exception.code, 400)
        self.assertIn("帧边界", json.loads(ctx.exception.read())["error"])

    def test_render_accepts_frame_mapping(self):
        T = 1.0 / 24.0
        code, body = self._post({
            "fps": 24, "shutter_angle": 90, "samples": 6, "curve": "box",
            "width": 80, "height": 60, "workers": 1,
            "frame_index": 2, "t_open_offset": 0.5 * T,
        })
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "running")

    def test_render_negative_frame_rejected(self):
        code, body = self._post({
            "fps": 24, "shutter_angle": 180, "samples": 8, "curve": "box",
            "width": 80, "height": 60, "workers": 1, "frame_index": -1,
        })
        self.assertEqual(code, 400)
        self.assertIn("帧号", body["error"])

    def test_invalid_render_payload_rejected(self):
        code, body = self._post({
            "fps": -1, "shutter_angle": 180, "samples": 8, "curve": "box",
            "width": 80, "height": 60, "workers": 1,
        })
        self.assertEqual(code, 400)
        self.assertIn("帧率", body["error"])

    def test_unknown_curve_rejected(self):
        code, body = self._post({
            "fps": 24, "shutter_angle": 180, "samples": 8, "curve": "nope",
            "width": 80, "height": 60, "workers": 1,
        })
        self.assertEqual(code, 400)

    def test_traversal_blocked_and_missing_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/jobs/..%2f..%2fetc%2fpasswd")
        self.assertEqual(ctx.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/jobs/does-not-exist/x.png")
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
