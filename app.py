"""Start the local course portal with: uv run python app.py."""

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
CHAPTERS = [
    dict(id="01-python", number="01", title="Python入門", description="短いコードを読み、値を変えて実行する。", mode="edit"),
    dict(id="02-regression", number="02", title="回帰とモデルの選び方", description="予測と誤差から、モデルの選び方へ。", mode="run"),
    dict(id="03-classification", number="03", title="線形分類と勾配降下法", description="分類のしくみと、重みを学習する流れ。", mode="run"),
    dict(id="04-neural-networks", number="04", title="ニューラルネットワークと逆伝播", description="小さなネットワークで、予測から更新まで。", mode="run"),
    dict(id="05-pytorch", number="05", title="PyTorchによる学習と評価", description="コードを実行し、学習した重みで数字を読む。", mode="edit"),
    dict(id="06-feature-space", number="06", title="特徴空間と次元削減（任意）", description="モデルの中間表現をPCA・Isomap・UMAPで見る。", mode="run"),
]


class Lessons:
    """Lazily start each notebook once and own its lifetime."""

    def __init__(self):
        self.lock = threading.Lock()
        self.children = {}
        self.token = secrets.token_urlsafe(32)
        self.closed = False

    def open(self, chapter_id):
        chapter = next((c for c in CHAPTERS if c["id"] == chapter_id and c["mode"]), None)
        if chapter is None:
            raise KeyError(chapter_id)
        with self.lock:
            if self.closed:
                raise RuntimeError("教材を終了しています。入口を起動し直してください。")
            child = self.children.get(chapter_id)
            if child and child[0].poll() is None:
                return child[2]
            if child:
                child[1].close()
                del self.children[chapter_id]
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            url = f"http://127.0.0.1:{port}/?access_token={self.token}"
            log = tempfile.TemporaryFile()
            command = [
                sys.executable, "-m", "marimo", chapter["mode"],
                str(ROOT / chapter_id / "notebook.py"),
                "--headless", "--host", "127.0.0.1", "--port", str(port),
                "--token", "--token-password", self.token,
            ]
            if chapter["mode"] == "edit":
                command.append("--skip-update-check")
            try:
                options = {}
                if chapter["mode"] == "edit":
                    # Run every cell on opening. Otherwise marimo leaves cells unrun and marks each
                    # with a "not yet run" icon. In 05, later edits still use lazy execution.
                    options["env"] = {**os.environ, "_MARIMO_CONFIG_OVERLOAD_RUNTIME_AUTO_INSTANTIATE": "true"}
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, **options)
            except OSError:
                log.close()
                raise
            self.children[chapter_id] = (process, log, url)
            try:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        log.seek(0)
                        detail = log.read().decode(errors="replace").replace(self.token, "…")
                        print(detail, file=sys.stderr)
                        raise RuntimeError("教材を起動できませんでした。ターミナルのエラーを確認してください。")
                    try:
                        with urlopen(url, timeout=1) as response:
                            if response.status == 200:
                                return url
                    except (URLError, TimeoutError, OSError):
                        time.sleep(0.15)
                raise RuntimeError("起動に時間がかかっています。もう一度お試しください。")
            except Exception:
                self._stop(process)
                log.close()
                del self.children[chapter_id]
                raise

    @staticmethod
    def _stop(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def close(self):
        with self.lock:
            self.closed = True
            for process, log, _ in self.children.values():
                self._stop(process)
                log.close()
            self.children.clear()


def make_server(port, lessons):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, body, content_type="application/json; charset=utf-8", headers=None):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # Browsers cancel requests when navigating or seeking in media.
                pass

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/api/chapters":
                self.respond(200, CHAPTERS)
            elif path in ("/", "/portal.css", "/portal.js"):
                name, mime = {
                    "/": ("index.html", "text/html"),
                    "/portal.css": ("portal.css", "text/css"),
                    "/portal.js": ("portal.js", "text/javascript"),
                }[path]
                self.respond(200, (ROOT / "portal" / name).read_bytes(), mime + "; charset=utf-8")
            else:
                self.respond(404, {"error": "ページが見つかりません。"})

        def do_POST(self):
            # Only our own page may start a local editor process.
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if self.headers.get("Origin") != origin:
                self.respond(403, {"error": "教材の入口から開いてください。"})
                return
            prefix = "/api/chapters/"
            if not self.path.startswith(prefix):
                self.respond(404, {"error": "ページが見つかりません。"})
                return
            try:
                self.respond(200, {"url": lessons.open(self.path[len(prefix):])})
            except KeyError:
                self.respond(404, {"error": "この教材はまだ公開されていません。"})
            except (RuntimeError, OSError) as error:
                self.respond(503, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description="機械学習入門の教材を開きます。")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    lessons = Lessons()
    try:
        server = make_server(args.port, lessons)
    except OSError as error:
        parser.exit(1, f"起動できませんでした: {error}\n別のポートを使う場合: uv run python app.py --port 8001\n")
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"\n機械学習入門: {url}\n終了するには Ctrl+C を押してください。\n", flush=True)
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        lessons.close()


if __name__ == "__main__":
    main()
