from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def record_external_turn(pipeline: Any, user_text: str, assistant_text: str, *, source: str = "external") -> None:
    """Best-effort recording for turns handled outside BrainPipeline.process()."""
    if not assistant_text:
        return

    conversation = getattr(pipeline, "_conversation", None)
    if conversation is not None:
        add_user_message = getattr(conversation, "add_user_message", None)
        add_assistant_message = getattr(conversation, "add_assistant_message", None)
        if callable(add_user_message):
            add_user_message(user_text)
        if callable(add_assistant_message):
            add_assistant_message(assistant_text)

    episodic = getattr(pipeline, "_episodic", None)
    if episodic is not None:
        log = getattr(episodic, "log", None)
        should_reflect = getattr(episodic, "should_reflect", None)
        reflect = getattr(episodic, "reflect", None)
        if callable(log):
            log("command", f"用户说: {user_text}")
            log("outcome", f"{source}回复: {assistant_text[:100]}")
        if callable(should_reflect) and callable(reflect) and should_reflect():
            try:
                task = asyncio.create_task(reflect())
                task.add_done_callback(
                    lambda t: logger.error("[Episodic] Reflection failed: %s", t.exception())
                    if not t.cancelled() and t.exception() else None
                )
            except RuntimeError:
                logger.debug("No running loop for external-turn reflection")
