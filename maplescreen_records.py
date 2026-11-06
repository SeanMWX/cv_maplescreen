"""Turn physical OCR lines into structured chat messages."""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypedDict


_MEDAL = re.compile(r"^[<＜〈《]\s*([^<>＜＞〈〉《》]+?)\s*[>＞〉》]")
_DELIMITER = re.compile(r"[:：]")
_CHANNEL = re.compile(
    r"(?:[\[【(（]\s*)?[CＣ][HＨ]\s*([0-9０-９](?:\s*[0-9０-９]){0,2})"
    r"\s*(?:[\]】)）])?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChatLine:
    """One line in reading order, optionally with fields read from separate crops.

    Set body_only if text is the cropped body, so its own colons are retained.
    Continuation lines contain only body text and extend the preceding message.
    """

    text: str
    medal: str | None = None
    player_name: str | None = None
    channel: str | None = None
    layout: Literal["unknown", "player", "megaphone", "system", "continuation"] = "unknown"
    body_only: bool = False


class ChatMessage(TypedDict):
    medal: str | None
    player_name: str | None
    channel: str | None
    text: str
    message_type: Literal["local", "megaphone", "system", "unknown"]


def _normalize_channel(match: re.Match[str]) -> str:
    digits = unicodedata.normalize("NFKC", "".join(match[1].split()))
    return f"CH{digits.zfill(2)}"


def _parse_line(line: ChatLine) -> ChatMessage:
    body = line.text.strip()
    medal = player = channel = None
    if line.layout == "system":
        return ChatMessage(
            medal=None, player_name=None, channel=None, text=body, message_type="system"
        )

    if not line.body_only and (delimiter := _DELIMITER.search(body)):
        header = body[:delimiter.start()].strip()
        if match := _MEDAL.match(header):
            medal = f"<{match[1]}>"
            header = header[match.end():].strip()
        if line.layout != "player" and (match := _CHANNEL.search(header)):
            prefix = header[:match.start()]
            if (
                medal or line.channel or not prefix or prefix[-1].isspace()
                or match[0][0] in "[【(（"
            ):
                channel = _normalize_channel(match)
                header = prefix.strip()
        if header and not any(char in header for char in "<>＜＞〈〉《》"):
            if medal or channel or line.layout == "player" or len(header.split()) == 1:
                player = header
        if medal or player or channel:
            body = body[delimiter.end():].lstrip()

    if line.medal and (name := line.medal.strip(" \t<>＜＞〈〉《》")):
        medal = f"<{name}>"
    if line.player_name:
        player = line.player_name.strip() or player
    if line.channel and (match := _CHANNEL.fullmatch(line.channel.strip())):
        channel = _normalize_channel(match)

    message_type = "unknown"
    if channel or line.layout == "megaphone":
        message_type = "megaphone"
    elif player:
        message_type = "local"
    return ChatMessage(
        medal=medal, player_name=player, channel=channel, text=body, message_type=message_type
    )


def parse_messages(lines: Iterable[ChatLine]) -> list[ChatMessage]:
    """Keep message order and join screen-wrapped lines into one body."""
    messages: list[ChatMessage] = []
    for line in lines:
        if not any((line.text.strip(), line.medal, line.player_name, line.channel)):
            continue
        if line.layout == "continuation" and messages:
            messages[-1]["text"] += line.text.strip()
        else:
            messages.append(_parse_line(line))
    return messages
