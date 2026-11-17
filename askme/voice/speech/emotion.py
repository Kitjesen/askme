"""Map a reply emoji to a Xiaozhi device emotion and build the llm payload.

The map and the scan order follow ``get_emotion`` in
xinnan-tech/xiaozhi-esp32-server ``core/utils/textUtils.py``. The first mapped
emoji wins. Anything else, including emoji that are not in the map, is ignored.
No mapped emoji means 🙂 / ``happy``, including when the product prompt forbids
emoji entirely.

Sending the payload does not change TTS text. Emoji stripping stays in the
existing speech cleanup.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Official keys. 😂 is "funny"; 😆 is "laughing". Do not collapse them.
EMOJI_MAP: dict[str, str] = {
    "😂": "funny",
    "😭": "crying",
    "😠": "angry",
    "😔": "sad",
    "😍": "loving",
    "😲": "surprised",
    "😱": "shocked",
    "🤔": "thinking",
    "😌": "relaxed",
    "😴": "sleepy",
    "😜": "silly",
    "🙄": "confused",
    "😶": "neutral",
    "🙂": "happy",
    "😆": "laughing",
    "😳": "embarrassed",
    "😉": "winking",
    "😎": "cool",
    "🤤": "delicious",
    "😘": "kissy",
    "😏": "confident",
}

DEFAULT_EMOJI = "🙂"
DEFAULT_EMOTION = "happy"

EmotionSender = Callable[[str], Any]


@dataclass(frozen=True, slots=True)
class Emotion:
    """Emoji shown on the device and the emotion name from the official map."""

    emoji: str
    emotion: str


def extract_emotion(text: str) -> Emotion:
    """Return the first mapped emoji in ``text``, or the happy default."""
    for char in text or "":
        mapped = EMOJI_MAP.get(char)
        if mapped is not None:
            return Emotion(emoji=char, emotion=mapped)
    return Emotion(emoji=DEFAULT_EMOJI, emotion=DEFAULT_EMOTION)


def build_llm_emotion_message(text: str, session_id: str = "") -> dict[str, str]:
    """Build the Xiaozhi WebSocket object for one reply.

    Shape: ``{"type": "llm", "text": emoji, "emotion": name, "session_id": id}``.
    """
    found = extract_emotion(text)
    return {
        "type": "llm",
        "text": found.emoji,
        "emotion": found.emotion,
        "session_id": "" if session_id is None else str(session_id),
    }


def dumps_llm_emotion_message(message: dict[str, str]) -> str:
    """Serialize a message the way the official server sends it."""
    return json.dumps(
        {
            "type": message.get("type", "llm"),
            "text": message.get("text", DEFAULT_EMOJI),
            "emotion": message.get("emotion", DEFAULT_EMOTION),
            "session_id": message.get("session_id", ""),
        }
    )


def push_device_emotion(
    text: str,
    session_id: str = "",
    sender: EmotionSender | Any | None = None,
) -> dict[str, str]:
    """Build the emotion message and hand the JSON string to ``sender``.

    ``sender`` may be a callable or any object with a ``send`` method, such as
    a Xiaozhi WebSocket. There is no device socket in this process; pass one
    in from the board integration. Send errors are logged and swallowed so a
    display failure cannot abort speech. The returned dict is the payload even
    when no sender is configured.
    """
    message = build_llm_emotion_message(text, session_id)
    if sender is None:
        return message
    raw = dumps_llm_emotion_message(message)
    try:
        _deliver(sender, raw)
    except Exception as exc:
        logger.warning("发送情绪表情失败，错误:%s", exc)
    return message


def _deliver(sender: EmotionSender | Any, raw: str) -> None:
    send = getattr(sender, "send", None)
    if callable(send) and not isinstance(sender, type):
        send(raw)
        return
    if callable(sender):
        sender(raw)
        return
    raise TypeError("device emotion sender must be callable or provide send()")
