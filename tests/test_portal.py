"""Run with: uv run python -m unittest discover -s tests -v."""

import json
import subprocess
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app import Lessons, ROOT, make_server


class PortalRoutes(unittest.TestCase):
    def setUp(self):
        self.lessons = MagicMock()
        self.server = make_server(0, self.lessons)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_home_assets_and_catalog(self):
        for path, mime in [("/", "text/html"), ("/portal.js", "text/javascript"), ("/portal.css", "text/css")]:
            with urlopen(self.origin + path) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers.get_content_type(), mime)
        with urlopen(self.origin + "/api/chapters") as response:
            chapters = json.load(response)
        self.assertEqual([c["mode"] for c in chapters], ["edit", "run", "run", "run", "edit"])
        for chapter in chapters:
            if chapter["mode"] in ("edit", "run"):
                self.assertTrue((ROOT / chapter["id"] / "notebook.py").is_file())

    def test_start_returns_url_and_reports_failure(self):
        self.lessons.open.return_value = "http://127.0.0.1:9000/"
        request = Request(self.origin + "/api/chapters/01-python", method="POST", headers={"Origin": self.origin})
        with urlopen(request) as response:
            self.assertEqual(json.load(response)["url"], self.lessons.open.return_value)
        self.lessons.open.assert_called_once_with("01-python")
        for error, code in [(KeyError("unknown"), 404), (RuntimeError("起動失敗"), 503)]:
            self.lessons.open.side_effect = error
            with self.assertRaises(HTTPError) as caught:
                urlopen(request)
            self.assertEqual(caught.exception.code, code)
            caught.exception.close()

    def test_external_origins_and_arbitrary_files_are_rejected(self):
        for headers in [{}, {"Origin": "http://example.com"}]:
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(self.origin + "/api/chapters/01-python", method="POST", headers=headers))
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()
        self.lessons.open.assert_not_called()
        for path in ["/app.py", "/../pyproject.toml", "/api/chapters/01-python", "/06-ml-landscape/index.html"]:
            with self.assertRaises(HTTPError) as caught:
                urlopen(self.origin + path)
            self.assertEqual(caught.exception.code, 404)
            caught.exception.close()


class LessonLifetime(unittest.TestCase):
    @patch("app.urlopen")
    @patch("app.subprocess.Popen")
    def test_pytorch_opens_real_editor_with_initial_outputs(self, start, request):
        from marimo._config.manager import ScriptConfigManager

        start.return_value.poll.return_value = None
        request.return_value.__enter__.return_value.status = 200
        lessons = Lessons()
        try:
            lessons.open("05-pytorch")
            self.assertEqual(start.call_args.args[0][3], "edit")
            self.assertEqual(start.call_args.kwargs["env"]["_MARIMO_CONFIG_OVERLOAD_RUNTIME_AUTO_INSTANTIATE"], "true")
            config = ScriptConfigManager(str(ROOT / "05-pytorch" / "notebook.py")).get_config()
            self.assertEqual(config["runtime"]["on_cell_change"], "lazy")
        finally:
            lessons.close()

    def test_only_available_chapters_can_start(self):
        lessons = Lessons()
        with patch("app.subprocess.Popen") as start:
            self.assertEqual(lessons.children, {})
            for chapter_id in ["06-ml-landscape", "missing", "../app.py"]:
                with self.assertRaises(KeyError):
                    lessons.open(chapter_id)
            start.assert_not_called()
        lessons.close()

    @patch("app.urlopen")
    @patch("app.subprocess.Popen")
    def test_concurrent_opens_reuse_process_and_shutdown_stops_it(self, start, request):
        process = start.return_value
        process.poll.return_value = None
        request.return_value.__enter__.return_value.status = 200
        lessons = Lessons()
        with ThreadPoolExecutor(max_workers=3) as pool:
            urls = list(pool.map(lessons.open, ["01-python"] * 3))
        self.assertEqual(len(set(urls)), 1)
        start.assert_called_once()
        self.assertEqual(start.call_args.args[0][3], "edit")
        log = lessons.children["01-python"][1]
        lessons.close()
        process.terminate.assert_called_once()
        self.assertTrue(log.closed)
        with self.assertRaises(RuntimeError):
            lessons.open("01-python")

    @patch("app.urlopen")
    @patch("app.subprocess.Popen")
    def test_failed_start_can_retry_and_run_mode_is_preserved(self, start, request):
        process = start.return_value
        process.poll.return_value = 1
        lessons = Lessons()
        with self.assertRaises(RuntimeError):
            lessons.open("02-regression")
        self.assertEqual(lessons.children, {})
        process.poll.return_value = None
        request.return_value.__enter__.return_value.status = 200
        lessons.open("02-regression")
        self.assertEqual(start.call_args.args[0][3], "run")
        process.wait.side_effect = [subprocess.TimeoutExpired("marimo", 5), None]
        lessons.close()
        process.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
