"""LLM-based fact extraction adapter for qp_memory.

After each conversation turn, extracts structured facts (anomalies,
observations, location mentions) and writes them into qp_memory.

This is the askme-side adapter that implements qp_memory's
ExtractionCallback protocol using the LLM client.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

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

        adapter = ExtractionAdapter(llm_client, model="qwen-turbo")
        mem.set_extraction_callback(adapter)
        # Now mem.process_turn(user, assistant) auto-extracts facts
    """

    def __init__(self, llm_client: Any, model: str = "qwen-turbo") -> None:
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

            # Synchronous LLM call (fire-and-forget, not on hot path)
            import os

            import httpx

            key = os.environ.get("DASHSCOPE_API_KEY", "")
            if not key:
                return []

            resp = httpx.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.1,
                },
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                timeout=5,
            )
            text = resp.json()["choices"][0]["message"]["content"].strip()

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
                    valid.append({
                        "type": f.get("type", "observation"),
                        "location": f.get("location", "general"),
                        "text": str(f["text"])[:100],
                    })
            if valid:
                logger.info("Extracted %d facts from turn", len(valid))
            return valid

        except Exception as e:
            logger.debug("Extraction failed: %s", e)
            return []
