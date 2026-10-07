"""Local stand-in for Vercel: serves the site from web/ and forwards the API paths, like vercel.json's rewrites.

    python web/dev_server.py                                  -> forwards to the local API (http://127.0.0.1:8000)
    API=http://3.106.107.237 ORIGIN_SECRET=… python web/dev_server.py   -> the live AWS server (it refuses requests
                                                              without Vercel's secret header; value in the server's deploy/.env)
    then open http://localhost:3000

Standard library only. For local testing — Vercel does this job in production.
"""
import os, urllib.error, urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = os.environ.get("API", "http://127.0.0.1:8000").rstrip("/")
FORWARD = ("/predict", "/explain", "/xai/status", "/complaint/", "/options/", "/model-info", "/health", "/docs", "/openapi.json")
CLEAN = {"/real": "/real.html", "/how": "/how.html"}           # vercel.json "cleanUrls"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def _forwarded(self):
        return self.path.startswith(FORWARD)

    def _proxy(self):
        body = None
        if self.command == "POST":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        if os.environ.get("ORIGIN_SECRET"):                 # what Vercel adds on every forwarded request
            headers["x-origin-secret"] = os.environ["ORIGIN_SECRET"]
        req = urllib.request.Request(API + self.path, data=body, method=self.command, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                status, ctype, data = r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:                   # 4xx / 5xx from the API: pass them through
            status, ctype, data = e.code, e.headers.get("Content-Type", ""), e.read()
        except urllib.error.URLError:
            status, ctype, data = 502, "application/json", b'{"detail": "API not reachable"}'
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self._forwarded():
            return self._proxy()
        self.path = CLEAN.get(self.path.split("?")[0], self.path)
        return super().do_GET()

    def do_POST(self):
        if self._forwarded():
            return self._proxy()
        self.send_error(405)


if __name__ == "__main__":
    print(f"site: http://localhost:3000   ·   API forwarded to {API}")
    ThreadingHTTPServer(("127.0.0.1", 3000), Handler).serve_forever()
