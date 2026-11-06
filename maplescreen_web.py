"""HTTP endpoints for the loopback monitor and its live event stream."""

import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit


def _encode_json(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _event(name: str, value: dict) -> bytes:
    return f"event: {name}\ndata: ".encode("ascii") + _encode_json(value) + b"\n\n"


def make_handler(state, html: bytes, keyword_script: bytes | None = None) -> type[BaseHTTPRequestHandler]:
    """Bind a monitor state to a handler for a loopback ThreadingHTTPServer.

    State exposes snapshots, the latest preview bytes and a revision-based
    wait method. On shutdown it sets its closed event and wakes waiters.
    """

    class Handler(BaseHTTPRequestHandler):
        def _headers(self, content_type: str) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")

        def _reply(self, status: int, content_type: str, body: bytes, *, location: str | None = None) -> None:
            self.send_response(status)
            self._headers(content_type)
            self.send_header("Content-Length", str(len(body)))
            if location is not None:
                self.send_header("Location", location)
            self.end_headers()
            self.wfile.write(body)

        def _events(self) -> None:
            # A stalled tab must not keep this server thread indefinitely.
            self.connection.settimeout(10)
            self.close_connection = True
            self.send_response(200)
            self._headers("text/event-stream; charset=utf-8")
            self.end_headers()
            state_version = preview_version = -1
            while not state.closed.is_set():
                state_version, preview_version, snapshot, preview = state.wait_for_update(
                    state_version, preview_version, 10
                )
                payload = b""
                if snapshot is not None:
                    payload += _event("state", snapshot)
                if preview is not None:
                    payload += _event("preview", preview)
                self.wfile.write(payload or b": heartbeat\n\n")
                self.wfile.flush()

        def do_GET(self) -> None:
            try:
                self._get()
            except (ConnectionError, TimeoutError):
                # A tab may navigate away during a preview or stream write.
                pass

        def _get(self) -> None:
            port = self.server.server_port
            if self.headers.get("Host", "").lower() not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                self._reply(403, "text/plain; charset=utf-8", b"Forbidden")
                return

            url = urlsplit(self.path)
            path = url.path
            if path == "/" and not url.query:
                self._reply(302, "text/plain; charset=utf-8", b"", location="/?live=1")
            elif path in ("/", "/demo.html"):
                self._reply(200, "text/html; charset=utf-8", html)
            elif path == "/assets/maplescreen_alerts.js" and keyword_script is not None:
                self._reply(200, "text/javascript; charset=utf-8", keyword_script)
            elif path == "/api/messages":
                self._reply(200, "application/json; charset=utf-8", _encode_json(state.snapshot()))
            elif path == "/api/preview.png":
                if (preview := state.preview_bytes()) is not None:
                    self._reply(200, "image/png", preview)
                else:
                    self._reply(404, "text/plain; charset=utf-8", b"Preview unavailable")
            elif path == "/api/events":
                self._events()
            else:
                self._reply(404, "text/plain; charset=utf-8", b"Not found")

        def log_message(self, *_):
            pass  # Keep preview and event traffic out of the terminal.

    return Handler
