"""Monitor a selected screen region and serve a local chat viewer."""

import argparse
import json
import sqlite3
import sys
from collections import deque
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Condition, Event, Thread
from time import monotonic

from maplescreen_store import ChatStore
from maplescreen_web import make_handler


_ROOT = Path(__file__).resolve().parent


class MonitorState:
    def __init__(self, region, session_id: int, recent: list[dict]):
        self.pending = Queue(maxsize=128)
        self.messages = deque(recent, maxlen=500)
        self.region = {"x": region.x, "y": region.y, "width": region.width, "height": region.height}
        self.session_id = session_id
        self.active = True
        self.gaps = 0
        self.last_error = None
        self.closed = Event()
        self._changed = Condition()
        self._queue_depth = 0
        self._state_version = 0
        self._preview_version = 0
        self._preview = None

    def _state_changed(self) -> None:
        self._state_version += 1
        self._changed.notify_all()

    def update_queue(self) -> None:
        with self._changed:
            depth = self.pending.qsize()
            if depth != self._queue_depth:
                self._queue_depth = depth
                self._state_changed()

    def publish(self, messages: list[dict]) -> None:
        with self._changed:
            self.messages.extend(messages)
            self._state_changed()

    def publish_preview(self, png: bytes) -> None:
        with self._changed:
            self._preview = png
            self._preview_version += 1
            self._changed.notify_all()

    def mark_gap(self) -> None:
        with self._changed:
            self.gaps += 1
            self._state_changed()

    def stop(self, error: str | None = None) -> None:
        with self._changed:
            if self.active or (error is not None and error != self.last_error):
                self.active = False
                if error is not None:
                    self.last_error = error
                self._state_changed()

    def close(self) -> None:
        with self._changed:
            self.closed.set()
            self._changed.notify_all()

    def _snapshot(self) -> dict:
        return {
            "messages": list(self.messages),
            "active": self.active,
            "queue_depth": self._queue_depth,
            "gaps": self.gaps,
            "last_error": self.last_error,
            "session_id": self.session_id,
        }

    def snapshot(self) -> dict:
        with self._changed:
            return self._snapshot()

    def preview_bytes(self) -> bytes | None:
        with self._changed:
            return self._preview

    def wait_for_update(self, state_version: int, preview_version: int, timeout: float):
        with self._changed:
            self._changed.wait_for(
                lambda: self.closed.is_set() or self._state_version != state_version
                or self._preview_version != preview_version,
                timeout=timeout,
            )
            snapshot = self._snapshot() if self._state_version != state_version else None
            preview = None
            if self._preview_version != preview_version:
                preview = {"revision": self._preview_version if self._preview is not None else None,
                           "region": self.region}
            return self._state_version, self._preview_version, snapshot, preview


def _recognize(state: MonitorState, stop: Event, store: ChatStore, sink) -> None:
    try:
        from maplescreen_ocr import recognize_new_rows

        while not stop.is_set() or not state.pending.empty():
            try:
                observed_at, strips = state.pending.get(timeout=0.1)
            except Empty:
                continue
            state.update_queue()
            messages = recognize_new_rows(strips)
            if not messages:
                continue
            committed = store.append(state.session_id, observed_at, messages)
            state.publish(committed)
            for message in messages:
                sink.write(json.dumps(message, ensure_ascii=False) + "\n")
    except Exception as exc:
        state.stop(f"识别或写入失败：{exc}")
        stop.set()


def _capture(capture, region, tracker, state: MonitorState, stop: Event, fps: int) -> None:
    import cv2
    import numpy as np

    frame_interval = 1 / fps
    next_preview = 0.0
    previous_preview = None
    while not stop.is_set():
        started = monotonic()
        frame = capture.grab(region)
        update = tracker.update(frame)
        if update.gap:
            state.mark_gap()
        if update.bands:
            strips = [frame[top:bottom].copy() for top, bottom in update.bands]
            observed_at = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            try:
                state.pending.put_nowait((observed_at, strips))
                state.update_queue()
            except Full:
                state.mark_gap()
        if started >= next_preview:
            next_preview = started + 0.2
            height, width = frame.shape[:2]
            scale = min(1, 1400 / width, 1000 / height)
            preview = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                                 interpolation=cv2.INTER_AREA) if scale < 1 else frame
            if previous_preview is None or not np.array_equal(previous_preview, preview):
                ok, png = cv2.imencode(".png", preview, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if ok:
                    previous_preview = preview
                    state.publish_preview(png.tobytes())
        stop.wait(max(0, frame_interval - (monotonic() - started)))


def run_monitor(capture, region, output: Path, database: Path, *, fps: int, port: int,
                include_existing: bool) -> None:
    from maplescreen_tracking import ChatTracker

    stop = Event()
    tracker = ChatTracker(include_existing=include_existing)
    database.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    store = ChatStore(database)
    session = store.start_session((region.x, region.y, region.width, region.height))
    try:
        state = MonitorState(region, session, store.recent())
        handler = make_handler(state, (_ROOT / "demo.html").read_bytes())
        with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
            with output.open("a", encoding="utf-8", newline="\n", buffering=1) as sink:
                web = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
                worker = Thread(target=_recognize, args=(state, stop, store, sink), daemon=True)
                print(f"在浏览器打开 http://127.0.0.1:{server.server_port}/", flush=True)
                print(f"SQLite 聊天记录写入 {database.resolve()}", flush=True)
                print("保持选中的聊天区可见。按 Ctrl+C 停止采集。", flush=True)
                web.start()
                worker.start()
                try:
                    _capture(capture, region, tracker, state, stop, fps)
                except KeyboardInterrupt:
                    print("采集已停止，正在完成已入队的识别。", flush=True)
                except (OSError, RuntimeError, ValueError) as exc:
                    state.stop(f"采集失败：{exc}")
                finally:
                    state.stop()
                    stop.set()
                    try:
                        worker.join()
                    finally:
                        state.close()
                        server.shutdown()
                        web.join()
    finally:
        store.finish_session(session)
    if error := state.snapshot()["last_error"]:
        raise RuntimeError(error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Screen rectangle in desktop pixels. Omit to select with the mouse.")
    parser.add_argument("--fps", type=int, default=30, help="Target screen capture rate (1–60, default 30)")
    parser.add_argument("--port", type=int, default=8765, help="Local viewer port (default 8765)")
    parser.add_argument("--include-existing", action="store_true", help="Read already visible lines on startup")
    parser.add_argument("--database", type=Path, default=_ROOT / "outputs" / "maplescreen.sqlite3",
                        help="SQLite database for chat messages and capture sessions")
    parser.add_argument("-o", "--output", type=Path, default=_ROOT / "outputs" / "live_messages.jsonl",
                        help="Append recognized messages to this UTF-8 JSON Lines file")
    args = parser.parse_args()
    if not 1 <= args.fps <= 60 or not 1 <= args.port <= 65535:
        parser.error("--fps must be 1–60 and --port must be 1–65535")

    deps = _ROOT / ".deps"
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    try:
        from maplescreen_capture import ScreenCapture, ScreenRegion
        from maplescreen_ocr import warm_up

        if len({args.output.resolve(), args.database.resolve(), (_ROOT / "demo.html").resolve()}) != 3:
            raise ValueError("SQLite、JSONL 和 demo.html 必须使用不同的文件。")
        with ScreenCapture() as capture:
            print("正在加载本地识别模型。", flush=True)
            warm_up()
            region = ScreenRegion(*args.roi) if args.roi else capture.select_region()
            capture.validate(region)
            print(f"监控区域 x={region.x} y={region.y} 宽={region.width} 高={region.height}", flush=True)
            run_monitor(capture, region, args.output, args.database, fps=args.fps, port=args.port,
                        include_existing=args.include_existing)
    except (ImportError, OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
