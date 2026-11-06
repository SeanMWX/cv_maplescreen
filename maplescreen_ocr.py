"""Read player and system messages from a cropped MapleStory screenshot."""

import re
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from maplescreen_records import ChatLine, ChatMessage, parse_messages


_OCR_SCALE = 2
_BRACKETED_MEDAL = re.compile(r"^[<＜〈《≤“]([^>＞〉》]+)[>＞〉》]\s*(.*)$")
_MEDAL_AND_NAME = re.compile(r"^[<＜〈《≤“]?(?P<medal>[^>＞〉》]+?勋章)\s*[>＞〉》?？]?\s*(?P<name>.*)$")
_SYSTEM_COLORS = ((255, 204, 102), (0, 255, 255), (51, 204, 153))  # BGR


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class Fragment:
    box: Box
    text: str


@lru_cache(maxsize=1)
def _engine() -> RapidOCR:
    return RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1, use_cls=False)


def _enlarge(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, None, fx=_OCR_SCALE, fy=_OCR_SCALE, interpolation=cv2.INTER_CUBIC)


def _detect_rows(engine: RapidOCR, image: np.ndarray) -> list[list[Fragment]]:
    result, _ = engine(_enlarge(image), use_cls=False)
    fragments = []
    for points, text, _ in result or []:
        bounds = cv2.boundingRect(np.asarray(points, dtype=np.float32) / _OCR_SCALE)
        fragments.append(Fragment(Box(*bounds), text))

    rows: list[list[Fragment]] = []
    for fragment in sorted(fragments, key=lambda part: part.box.center_y):
        if rows:
            anchor = rows[-1][0].box
            distance = abs(anchor.center_y - fragment.box.center_y)
            if distance <= 0.45 * min(anchor.height, fragment.box.height):
                rows[-1].append(fragment)
                continue
        rows.append([fragment])
    return [sorted(row, key=lambda part: part.box.x) for row in rows]


def _channel_badges(image: np.ndarray) -> list[Box]:
    # The game's pale pink badge interior touches its white outline.
    pale = np.all(image == (238, 221, 255), axis=2)
    white = np.all(image == 255, axis=2)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((pale | white).astype(np.uint8))
    badges = []
    for label, (x, y, width, height, area) in enumerate(stats[1:], start=1):
        if not (8 <= height <= 40 and 1.7 <= width / height <= 3.5 and area > width * height / 2):
            continue
        region = np.s_[y:y + height, x:x + width]
        if np.count_nonzero(pale[region] & (labels[region] == label)) > area / 4:
            badges.append(Box(int(x), int(y), int(width), int(height)))
    return sorted(badges, key=lambda badge: badge.y)


def _channel_digits(image: np.ndarray, badge: Box) -> np.ndarray:
    left = badge.x + round(badge.height * 11 / 12)
    right = min(badge.right, badge.x + round(badge.height * 22 / 12))
    crop = image[badge.y:badge.y + badge.height, left:right].astype(np.int16)
    blue, green, red = cv2.split(crop)
    ink = (red > green + 80) & (red > blue + 80)
    digits = cv2.cvtColor(np.where(ink, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    padded = cv2.copyMakeBorder(digits, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return cv2.resize(padded, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


def _split_name(header: str) -> tuple[str | None, str | None]:
    header = header.strip()
    if match := _BRACKETED_MEDAL.fullmatch(header) or _MEDAL_AND_NAME.fullmatch(header):
        return match[1].strip(), match[2].strip() or None
    return None, header or None


def _badge_crops(image: np.ndarray, badge: Box, right: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top = max(0, badge.y - 1)
    bottom = badge.y + badge.height
    if right <= badge.right:
        right = image.shape[1]
    header = image[top:bottom, :badge.x]
    body = image[top:bottom, badge.right:right]
    blank = np.full((badge.height, 8, 3), 255, dtype=np.uint8)
    return (
        _enlarge(header if header.size else blank),
        _channel_digits(image, badge),
        _enlarge(body if body.size else blank),
    )


def _badge_line(header: str, digits: str, body: str) -> ChatLine:
    medal, player = _split_name(header)
    digits = digits.strip()
    channel = f"CH{digits}" if re.fullmatch(r"[0-9]{2}", digits) else None
    body = re.sub(r"^[:：]\s*", "", body.strip())
    return ChatLine(
        body, medal=medal, player_name=player, channel=channel,
        layout="megaphone", body_only=True,
    )


def _badge_lines(
    engine: RapidOCR, image: np.ndarray, badges: list[Box], rows: list[list[Fragment]],
) -> list[ChatLine]:
    if not badges:
        return []

    crops = []
    for badge, row in zip(badges, rows):
        right = max((part.box.right + 3 for part in row), default=0)
        crops.extend(_badge_crops(image, badge, right))

    recognized, _ = engine.text_rec(crops)
    lines = []
    for index in range(0, len(recognized), 3):
        (header, _), (digits, _), (body, _) = recognized[index:index + 3]
        lines.append(_badge_line(header, digits, body))
    return lines


def _system_color(image: np.ndarray, row: list[Fragment]) -> tuple[int, int, int] | None:
    top = min(part.box.y for part in row)
    bottom = max(part.box.y + part.box.height for part in row)
    region = image[top:bottom, row[0].box.x:max(part.box.right for part in row)]
    for color in _SYSTEM_COLORS:
        if np.count_nonzero(np.all(region == color, axis=2)) > region.shape[0] * region.shape[1] * 0.07:
            return color
    return None


def _local_chat_crop(image: np.ndarray) -> tuple[int, int, np.ndarray] | None:
    from maplescreen_tracking import local_text_mask

    white = local_text_mask(image)
    left, top, width, height = cv2.boundingRect(white)
    if height < 4 or cv2.countNonZero(white) < 8:
        return None
    right = left + width
    gray = cv2.cvtColor(image[top:top + height, max(0, left - 2):right + 2], cv2.COLOR_BGR2GRAY)
    if np.median(gray) >= 160:
        return None
    # Use white pixels for bounds, then keep dimmer glyph strokes and colons.
    ink = cv2.subtract(gray, 120)
    black_on_white = cv2.bitwise_not(cv2.add(ink, ink))
    return left, right, _enlarge(cv2.cvtColor(black_on_white, cv2.COLOR_GRAY2BGR))


def warm_up() -> None:
    """Load the local models before the first live message reaches the worker."""
    _engine()


def recognize_new_rows(strips: list[np.ndarray]) -> list[ChatMessage]:
    """OCR newly captured physical lines without detecting text on the whole frame."""
    from maplescreen_tracking import text_mask

    rows = []
    text_crops = []
    digit_crops = []
    for strip in strips:
        height, width = strip.shape[:2]
        if height < 6:
            continue  # A sliver at the edge of the capture cannot contain a readable line.
        columns = np.flatnonzero(np.any(text_mask(strip), axis=0))
        badges = _channel_badges(strip)
        if not len(columns) and not badges:
            continue
        left, right = (int(columns[0]), int(columns[-1]) + 1) if len(columns) else (0, width)
        crop = strip[:, max(0, left - 3):min(width, right + 3)]
        color = _system_color(strip, [Fragment(Box(left, 0, right - left, height), "")])
        local = _local_chat_crop(strip) if not badges and color is None else None
        if badges:
            header, digits, body = _badge_crops(strip, badges[0], right + 3)
            text_crops.extend((header, body))
            digit_crops.append(digits)
        elif local is not None:
            left, right, enhanced = local
            text_crops.append(enhanced)
        else:
            text_crops.append(_enlarge(crop))
        rows.append((left, right, height, width, color, bool(badges), local is not None))

    if not text_crops:
        return []
    engine = _engine()
    texts, _ = engine.text_rec(text_crops)
    digits, _ = engine.text_rec(digit_crops) if digit_crops else ([], 0)
    text_iter = iter(texts)
    digit_iter = iter(digits)
    lines = []
    left_edge = min(row[0] for row in rows)
    previous_style = None
    previous_right = None
    for left, right, height, width, color, has_badge, is_local in rows:
        text, _ = next(text_iter)
        if has_badge:
            body, _ = next(text_iter)
            channel, _ = next(digit_iter)
            line = _badge_line(text, channel, body)
            style = None
        else:
            style = "local" if is_local else color
            same_style = bool(lines) and style == previous_style
            indented = left - left_edge > height * 0.6 and not re.search(r"[:：]", text)
            wraps_system = color is not None and previous_right is not None and (
                width - previous_right <= 3 * height and not text.startswith(("[", "【"))
            )
            if same_style and (indented or wraps_system):
                line = ChatLine(text, layout="continuation")
            else:
                line = ChatLine(text, layout="player" if is_local else "system" if color else "unknown")
        lines.append(line)
        previous_style, previous_right = style, right
    return parse_messages(lines)


def extract_messages(image: np.ndarray) -> list[ChatMessage]:
    """Extract messages in screen order from an OpenCV BGR chat crop."""
    engine = _engine()
    rows = _detect_rows(engine, image)
    badges = _channel_badges(image)
    badge_rows: list[list[Fragment]] = [[] for _ in badges]
    other_rows = []
    for row in rows:
        center_y = float(np.median([part.box.center_y for part in row]))
        matched = next((i for i, badge in enumerate(badges)
                        if abs(badge.center_y - center_y) < badge.height / 2), None)
        if matched is not None:
            badge_rows[matched].extend(row)
        else:
            other_rows.append((center_y, row))

    items = [(badge.center_y, line, None) for badge, line in
             zip(badges, _badge_lines(engine, image, badges, badge_rows))]
    local_rows = []
    for index, (center_y, row) in enumerate(other_rows):
        text = "".join(part.text for part in row)
        color = _system_color(image, row)
        top = min(part.box.y for part in row)
        bottom = max(part.box.y + part.box.height for part in row)
        if index + 1 < len(other_rows):
            next_top = min(part.box.y for part in other_rows[index + 1][1])
            if center_y < next_top < bottom:
                bottom = next_top  # Detector rectangles can overlap the next text row.
        local = _local_chat_crop(image[top:bottom]) if color is None else None
        if local is not None:
            local_rows.append((center_y, row, text, local[2]))
            continue
        items.append((center_y, ChatLine(text, layout="system" if color else "unknown"), (row, color)))
    if local_rows:
        recognized, _ = engine.text_rec([crop for _, _, _, crop in local_rows])
        for (center_y, row, raw, _), (text, _) in zip(local_rows, recognized):
            if not text.strip() or (re.search(r"[:：]", raw) and not re.search(r"[:：]", text)):
                text = raw
            items.append((center_y, ChatLine(text, layout="player"), (row, None)))

    lines = []
    previous_style = None
    previous_right = None
    previous_y = None
    left_edge = min((part.box.x for row in rows for part in row), default=0)
    right_edge = max((part.box.right for row in rows for part in row), default=0)
    for center_y, line, detected in sorted(items, key=lambda item: item[0]):
        style = None
        right = None
        if detected:
            row, color = detected
            style = "local" if line.layout == "player" else color
            right = max(part.box.right for part in row)
            height = row[0].box.height
            near_previous = previous_y is not None and center_y - previous_y < height * 1.6
            indented = row[0].box.x - left_edge > height * 0.6
            same_system_style = (
                color is not None and style == previous_style and previous_right is not None
                and right_edge - previous_right <= 3 * height and not line.text.startswith(("[", "【"))
            )
            plain_continuation = indented and style == previous_style and not re.search(r"[:：]", line.text)
            if near_previous and (same_system_style or plain_continuation):
                line = ChatLine(line.text, layout="continuation")
        lines.append(line)
        previous_style, previous_right, previous_y = style, right, center_y
    return parse_messages(lines)
