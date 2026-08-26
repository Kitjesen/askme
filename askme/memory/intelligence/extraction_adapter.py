"""LLM-based fact extraction adapter for qp_memory.

After each conversation turn, extracts structured facts (anomalies,
observations, location mentions) and writes them into qp_memory.

This is the askme-side adapter that implements qp_memory's
ExtractionCallback protocol using the LLM client.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from collections.abc import Awaitable
from typing import Any

from askme.llm.core.contracts import LLMCallContext

logger = logging.getLogger(__name__)


async def _resolve_awaitable(value: Awaitable[Any]) -> Any:
    """Normalize an arbitrary awaitable into a coroutine for ``asyncio.run``."""

    return await value


# Extraction prompt — asks LLM to identify facts from a conversation turn
_EXTRACT_PROMPT = """From this conversation turn, extract any factual observations.
Return a JSON array of facts. Each fact has: type, location, text.

Types: "anomaly" (problem/issue), "observation" (normal status), "visit" (arrived somewhere)
Location: the place mentioned, or "general" if none.

Rules:
- Only extract if there are real facts, not greetings or questions.
- Return [] if nothing worth remembering.
- Max 3 facts per turn.

User: {user_text}
Assistant: {assistant_text}

Return ONLY valid JSON array, no explanation:"""


class ExtractionAdapter:
    """Askme-side adapter: uses LLM to extract facts from conversation turns.

    Implements qp_memory.ExtractionCallback protocol.

    Usage::

        adapter = ExtractionAdapter(llm_client, model="memory-compact")
        mem.set_extraction_callback(adapter)
        # Now mem.process_turn(user, assistant) auto-extracts facts
    """

    def __init__(self, llm_client: Any | None, model: str = "memory-compact") -> None:
        self._llm = llm_client
        self._model = model
        self._enabled = True
        # Rate limit: max 1 extraction per 5 seconds to avoid LLM spam
        self._last_extract: float = 0.0
        self._cooldown: float = 5.0

    def extract(self, user_text: str, assistant_text: str) -> list[dict]:
        """Extract facts from a conversation turn via LLM.

        Returns list of dicts: [{"type": "anomaly", "location": "仓库A", "text": "..."}]
        """
        import time

        if not self._enabled:
            return []

        # Rate limit
        now = time.time()
        if now - self._last_extract < self._cooldown:
            return []
        self._last_extract = now

        # Skip trivial turns
        if len(user_text) < 4 or len(assistant_text) < 4:
            return []

        # Skip greetings/commands
        skip_words = ["几点", "你好", "再见", "停", "音量", "静音", "闭嘴"]
        if any(w in user_text for w in skip_words):
            return []

        try:
            prompt = _EXTRACT_PROMPT.format(
                user_text=user_text[:200],
                assistant_text=assistant_text[:200],
            )

            text = self._complete(prompt).strip()
            if not text:
                return []

            # Parse JSON from response (handle markdown fences)
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            facts = json.loads(text)
            if not isinstance(facts, list):
                return []

            # Validate and clean
            valid = []
            for f in facts[:3]:  # max 3
                if isinstance(f, dict) and "type" in f and "text" in f:
                    valid.append(
                        {
                            "type": f.get("type", "observation"),
                            "location": f.get("location", "general"),
                            "text": str(f["text"])[:100],
                        }
                    )
            if valid:
                logger.info("Extracted %d facts from turn", len(valid))
            return valid

        except Exception as exc:
            logger.debug("Extraction failed: %s", exc)
            return []

    def _complete(self, prompt: str) -> str:
        """Call only the injected central LLM boundary, or fail closed."""

        if self._llm is None:
            return ""
        chat = getattr(self._llm, "chat", None)
        if not callable(chat):
            return ""

        result = chat(
            [{"role": "user", "content": prompt}],
            model=self._model,
            temperature=0.1,
            context=LLMCallContext(
                call_id=uuid.uuid4().hex,
                purpose="memory_compact",
                channel="background",
                request_class="memory",
                privacy_class="sensitive",
                allow_cache=False,
            ),
        )
        if not inspect.isawaitable(result):
            return str(result or "")

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return str(asyncio.run(_resolve_awaitable(result)) or "")

        close = getattr(result, "close", None)
        if callable(close):
            close()
        logger.debug(
            "Extraction skipped: synchronous callback cannot await central LLM on event-loop thread"
        )
        return ""
