"""Follow newly visible chat lines in successive crops of the same screen area."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameUpdate:
    """New physical line bounds, in order, and a possible loss of continuity."""

    bands: list[tuple[int, int]]
    gap: bool = False


@dataclass(frozen=True)
class _Row:
    top: int
    bottom: int
    center: float
    pixels: np.ndarray


def _ink_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    active = np.count_nonzero(mask, axis=1) >= 3
    edges = np.flatnonzero(np.diff(np.pad(active.astype(np.int8), 1)))
    return [(int(top), int(bottom)) for top, bottom in zip(edges[::2], edges[1::2])
            if 3 <= bottom - top <= 34]


def text_mask(frame: np.ndarray) -> np.ndarray:
    """Isolate likely chat ink for line tracking and bounds within a BGR crop."""
    # MapleWings' plum chat text and cyan, yellow and lime system styles.
    ink = cv2.inRange(frame, (55, 0, 100), (145, 90, 175))
    ink |= cv2.inRange(frame, (247, 195, 94), (255, 212, 110))
    ink |= cv2.inRange(frame, (0, 248, 248), (8, 255, 255))
    lime = cv2.inRange(frame, (45, 198, 147), (57, 210, 159))
    lime_rows = np.count_nonzero(lime, axis=1) > frame.shape[1] * 0.20
    ink |= cv2.inRange(frame, (0, 0, 0), (45, 45, 45)) * lime_rows[:, None]

    # Preserve the channel digits as a visual cue when the same text repeats.
    badge = cv2.inRange(frame, (235, 217, 251), (242, 227, 255))
    near_badge = cv2.dilate(badge, np.ones((3, 3), dtype=np.uint8))
    ink |= near_badge & cv2.inRange(frame, (0, 0, 160), (75, 75, 255))

    # Contrast fallback for text outside the known chat colors. The short
    # horizontal bands reject large continuous regions of game artwork.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast = cv2.inRange(cv2.absdiff(gray, cv2.medianBlur(gray, 5)), 49, 255)
    contrast &= cv2.inRange(gray, 0, 90) | cv2.inRange(gray, 165, 255)
    for top, bottom in _ink_runs(ink):
        contrast[max(0, top - 1):bottom + 1] = 0
    for top, bottom in _ink_runs(contrast):
        ink[top:bottom] |= contrast[top:bottom]
    return ink


def _rows(mask: np.ndarray) -> list[_Row]:
    height, width = mask.shape
    runs = _ink_runs(mask)
    rows = []
    for index, (top, bottom) in enumerate(runs):
        before = (runs[index - 1][1] + top) // 2 if index else 0
        after = (bottom + runs[index + 1][0]) // 2 if index + 1 < len(runs) else height
        # Bounds include ascenders and the badge outline, while the signature
        # uses the narrower ink extent to tolerate vertical motion.
        sample = cv2.resize(mask[top:bottom], (min(width, 220), 8), interpolation=cv2.INTER_AREA)
        pixels = (sample >= 35).ravel().astype(np.float32)
        rows.append(_Row(max(before, top - 2), min(after, bottom + 2), (top + bottom) / 2, pixels))
    return rows


def _last_match(previous: list[_Row], current: list[_Row]) -> int:
    old = np.stack([row.pixels for row in previous])
    new = np.stack([row.pixels for row in current])
    common = old @ new.T
    union = old.sum(axis=1)[:, None] + new.sum(axis=1)[None, :] - common
    similarity = common / np.maximum(union, 1)
    motion = np.array([row.center for row in previous])[:, None] - np.array([row.center for row in current])
    likely = (similarity >= 0.78) & (motion >= -2)
    # Estimate one shared vertical displacement. Matching the same row at
    # the displaced position preserves repeated messages as separate events.
    best_score = (0, 0.0, float('-inf'))
    last_match = -1
    for displacement in np.unique(np.rint(motion[likely])):
        pairs = likely & (np.abs(motion - displacement) <= 2)
        _, indices = np.nonzero(pairs)
        score = (len(indices), float(similarity[pairs].sum()), -abs(displacement))
        if score > best_score:
            best_score = score
            last_match = int(indices.max())
    return last_match


class ChatTracker:
    """Track downward append order as older messages move up the chat box."""

    def __init__(self, *, include_existing: bool = False) -> None:
        self.include_existing = include_existing
        self._shape: tuple[int, ...] | None = None
        self._previous: list[_Row] = []
        self._mask: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> FrameUpdate:
        mask = text_mask(frame)
        if self._mask is not None and np.array_equal(mask, self._mask):
            return FrameUpdate([])
        rows = _rows(mask)
        self._mask = mask
        if self._shape != frame.shape:
            initial = self._shape is None
            self._shape, self._previous = frame.shape, rows
            bands = [(row.top, row.bottom) for row in rows] if initial and self.include_existing else []
            return FrameUpdate(bands, gap=not initial)
        if not rows:
            # A short occlusion should not turn the still-visible history into
            # new records when the chat panel becomes readable again.
            return FrameUpdate([])
        previous, self._previous = self._previous, rows
        last = _last_match(previous, rows) if previous else -1
        return FrameUpdate([(row.top, row.bottom) for row in rows[last + 1:]], gap=bool(previous) and last < 0)
