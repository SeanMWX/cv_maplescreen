"""Select and capture a region in Windows virtual-desktop coordinates."""

import sys
from dataclasses import dataclass
from math import ceil, floor

import cv2
import numpy as np
from mss import MSS, ScreenShotError


@dataclass(frozen=True, slots=True)
class ScreenRegion:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("The screen region must have a positive width and height.")

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height


class ScreenCapture:
    """Capture pixels on the thread that owns this context."""

    def __enter__(self) -> "ScreenCapture":
        try:
            self._capture = MSS()
        except ScreenShotError as exc:
            raise RuntimeError(f"Could not start screen capture: {exc}") from exc
        try:
            desktop = self._capture.monitors[0]
            self.desktop = ScreenRegion(desktop["left"], desktop["top"], desktop["width"], desktop["height"])
        except (ScreenShotError, IndexError, ValueError) as exc:
            self._capture.close()
            raise RuntimeError(f"Could not locate the desktop for screen capture: {exc}") from exc
        return self

    def __exit__(self, *_) -> None:
        self._capture.close()

    def validate(self, region: ScreenRegion) -> None:
        left, top, right, bottom = region.bounds
        min_x, min_y, max_x, max_y = self.desktop.bounds
        if not (min_x <= left < right <= max_x and min_y <= top < bottom <= max_y):
            raise ValueError(f"The screen region must be inside desktop bounds {self.desktop.bounds}.")

    def grab(self, region: ScreenRegion) -> np.ndarray:
        self.validate(region)
        try:
            pixels = np.asarray(self._capture.grab(region.bounds))
        except ScreenShotError as exc:
            raise RuntimeError(f"Could not capture the selected screen region: {exc}") from exc
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)

    def select_region(self) -> ScreenRegion:
        image = self.grab(self.desktop)
        height, width = image.shape[:2]
        scale = min(1, 1200 / width, 720 / height)
        size = max(1, round(width * scale)), max(1, round(height * scale))
        preview = image if scale == 1 else cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        first, second = _select_corners(preview)
        return _desktop_region(self.desktop, size, first, second)


def _desktop_region(desktop, preview_size, first, second):
    preview_width, preview_height = preview_size
    x1, x2 = sorted((first[0], second[0]))
    y1, y2 = sorted((first[1], second[1]))
    left = floor(x1 * desktop.width / preview_width)
    right = ceil(x2 * desktop.width / preview_width)
    top = floor(y1 * desktop.height / preview_height)
    bottom = ceil(y2 * desktop.height / preview_height)
    return ScreenRegion(desktop.x + left, desktop.y + top, right - left, bottom - top)


def _select_corners(preview):
    title = "MapleScreen - drag chat area, Enter to confirm, Esc to cancel"
    print("在预览中拖选聊天区，按 Enter 确认，按 Esc 取消。", file=sys.stderr)
    height, width = preview.shape[:2]
    first = second = None
    dragging = False

    def on_mouse(event, x, y, *_):
        nonlocal first, second, dragging
        point = (min(max(x, 0), width), min(max(y, 0), height))
        if event == cv2.EVENT_LBUTTONDOWN:
            first = second = point
            dragging = True
        elif dragging and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
            second = point
            dragging = event == cv2.EVENT_MOUSEMOVE

    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    try:
        cv2.imshow(title, preview)
        cv2.resizeWindow(title, width, height)
        cv2.setMouseCallback(title, on_mouse)
        while True:
            try:
                visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                break  # Some OpenCV backends raise when the window closes.
            if visible <= 0:
                break
            if first is not None and second is not None:
                marked = preview.copy()
                cv2.rectangle(marked, first, second, (55, 225, 255), 2)
                cv2.imshow(title, marked)
            key = cv2.waitKey(30) & 0xff
            if key in (27, ord("c"), ord("C")):
                break
            if key in (10, 13, 32) and first and second and first[0] != second[0] and first[1] != second[1]:
                return first, second
    finally:
        try:
            cv2.destroyWindow(title)
        except cv2.error:
            pass  # The user may already have closed the window.
    raise ValueError("Chat area selection cancelled.")
