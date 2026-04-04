"""
Episodic memory — experience logging, decay, reflection, and knowledge consolidation.

Inspired by:
  - Park et al. 2023 (Generative Agents): importance scoring + retrieval + reflection
  - OpenClaw: two-layer separation (daily log vs curated knowledge)
  - ACT-R cognitive architecture: base-level activation with power-law decay
  - Ebbinghaus forgetting curve: exponential decay with rehearsal strengthening
  - Letta/MemGPT: sleep-time consolidation

Architecture:
  L1 Episode Buffer  →  L2 Episode Digest  →  L3 World Knowledge
  (raw events with      (reflected summaries    (categorized facts:
   importance + decay)    as .md files)           environment, entities,
                                                  routines, etc.)

Reflection triggers (Park 2023 hybrid):
  - Cumulative importance exceeds threshold (primary)
  - Buffer count exceeds minimum (fallback)
  - Cooldown prevents excessive reflection

Usage::

    from askme.memory.episodic_memory import EpisodicMemory

    mem = EpisodicMemory(llm=llm_client)
    mem.log("perception", "检测到一个人站在门口",
            context={"label": "person", "confidence": 0.92})
    mem.log("action", "执行巡逻路径 A", context={"path": "A"})
    mem.log("outcome", "巡逻完成，未发现异常")

    # Triggered by importance accumulation or periodically
    await mem.reflect()
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from askme.memory.admission import MemoryAdmissionControl
from askme.memory.trend_analyzer import TrendAnalyzer
from askme.memory.episode import (
    Episode,
    score_importance,
    DEFAULT_STABILITY_S,
    STABILITY_GROWTH_FACTOR,
    MAX_STABILITY_S,
    WEIGHT_RECENCY,
    WEIGHT_IMPORTANCE,
    WEIGHT_RELEVANCE,
    IMPORTANCE_RULES,
    IMPORTANCE_BOOSTS,
)
from askme.config import get_config, project_root

if TYPE_CHECKING:
    from askme.llm.client import LLMClient

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────

# L1: Episode buffer limits
MAX_BUFFER_SIZE = 200       # Max events in memory buffer
FLUSH_THRESHOLD = 100       # Flush to disk when buffer reaches this
EPISODE_RETENTION_HOURS = 24  # Keep raw episodes for this long

# L2: Reflection triggers
REFLECT_MIN_EVENTS = 10     # Minimum events before reflection makes sense
REFLECT_COOLDOWN_S = 300    # At least 5 min between reflections
IMPORTANCE_THRESHOLD = 15.0  # Park 2023: cumulative importance to trigger reflection

# L3: Knowledge categories for a robot agent
KNOWLEDGE_CATEGORIES = {
    "environment":    "环境布局、空间结构、地标、路径",
    "entities":       "识别的人、动物、物体及其特征",
    "routines":       "时间规律、日程、周期性事件",
    "interactions":   "交互经验、命令响应、沟通模式",
    "self_knowledge": "自身能力、限制、校准、性能",
}

# ── Prompts ────────────────────────────────────────────────

REFLECT_PROMPT = """\
你是一个机器人的记忆反思系统。以下是最近发生的事件记录（带重要性评分）。
请像一个聪明的观察者一样，从这些经历中提取有价值的认知。

事件记录:
{episodes}

当前已知知识:
{existing_knowledge}

请完成以下任务:
1. **归纳总结**: 用 2-3 句话概括这段时间发生了什么
2. **新发现**: 列出从这些事件中学到的新信息（如果有），并分类
3. **模式识别**: 是否发现重复出现的规律？（如果有）
4. **知识更新**: 需要更新或修正的已有认知（如果有）

知识分类说明:
- environment: 环境布局、空间、地标
- entities: 人、动物、物体
- routines: 时间规律、日程
- interactions: 交互经验、命令响应
- self_knowledge: 自身能力、限制

用 JSON 格式回复:
{{
  "summary": "这段时间的概要",
  "new_facts": [{{"fact": "新发现内容", "category": "分类名"}}],
  "patterns": [{{"pattern": "规律描述", "category": "分类名", "confidence": 0.8}}],
  "updates": [{{"old": "旧认知", "new": "新认知", "category": "分类名"}}],
  "importance": "low|medium|high"
}}"""


_RE_NORMALIZE = re.compile(r"[\s\-\—\·、，。！？：；""''（）()\[\]「」.,!?:;]+")


def _normalize_for_dedup(text: str) -> str:
    """Strip whitespace and punctuation for fuzzy dedup comparison."""
    return _RE_NORMALIZE.sub("", text).lower()


def _is_duplicate(new_text: str, existing_texts: list[str], threshold: float = 0.8) -> bool:
    """Check if new_text is a near-duplicate of any existing text.

    Uses character-set overlap ratio: if >threshold of the shorter text's
    characters appear in the longer text, it's considered a duplicate.
    """
    norm_new = _normalize_for_dedup(new_text)
    if not norm_new:
        return False
    new_chars = set(norm_new)
    for existing in existing_texts:
        norm_existing = _normalize_for_dedup(existing)
        if not norm_existing:
            continue
        # Exact match after normalization
        if norm_new == norm_existing:
            return True
        # Character overlap ratio
        existing_chars = set(norm_existing)
        intersection = new_chars & existing_chars
        union = new_chars | existing_chars
        if union and len(intersection) / len(union) > threshold:
            return True
    return False


class EpisodicMemory:
    """Three-layer episodic memory for embodied robot agents.

    Layer 1: Episode buffer (in-memory ring buffer + JSONL on disk)
    Layer 2: Episode digests (reflected summaries as .md files)
    Layer 3: World knowledge (categorized .md files per KNOWLEDGE_CATEGORIES)

    Reflection trigger: cumulative importance > IMPORTANCE_THRESHOLD (Park 2023)
    """

    def __init__(self, *, llm: LLMClient | None = None, vector_store: Any = None) -> None:
        cfg = get_config()
        data_dir = cfg.get("app", {}).get("data_dir", "data")
        episodic_cfg = cfg.get("memory", {}).get("episodic", {})
        resolved = Path(data_dir)
        if not resolved.is_absolute():
            resolved = project_root() / resolved

        self._data_dir = resolved / "memory"
        self._episodes_dir = self._data_dir / "episodes"
        self._digests_dir = self._data_dir / "digests"
        self._knowledge_dir = self._data_dir / "knowledge"
        self._active_file = self._episodes_dir / "_active.jsonl"

        for d in (self._episodes_dir, self._digests_dir, self._knowledge_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._llm = llm
        self._flush_threshold = int(episodic_cfg.get("flush_threshold", FLUSH_THRESHOLD))
        self._episode_retention_hours = int(
            episodic_cfg.get("episode_retention_hours", EPISODE_RETENTION_HOURS)
        )
        self._reflect_min_events = int(
            episodic_cfg.get("reflect_min_events", REFLECT_MIN_EVENTS)
        )
        self._reflect_cooldown_s = float(
            episodic_cfg.get("reflect_cooldown_seconds", REFLECT_COOLDOWN_S)
        )
        self._importance_threshold = float(
            episodic_cfg.get("importance_threshold", IMPORTANCE_THRESHOLD)
        )
        self._knowledge_context_chars = int(
            episodic_cfg.get("knowledge_context_chars", 1000)
        )
        self._digest_context_chars = int(
            episodic_cfg.get("digest_context_chars", 600)
        )
        self._relevant_context_chars = int(
            episodic_cfg.get("relevant_context_chars", 600)
        )
        self._relevant_top_k = int(episodic_cfg.get("relevant_top_k", 5))

        # L1: In-memory episode buffer
        self._buffer: deque[Episode] = deque(maxlen=MAX_BUFFER_SIZE)
        self._last_reflect_time: float = 0.0
        self._total_logged: int = 0
        self._cumulative_importance: float = 0.0  # Park 2023 trigger
        self._reflecting: bool = False
        self._reflect_lock: asyncio.Lock | None = None  # lazy-init in async context
        # Cache for _load_all_knowledge — invalidated by _update_knowledge
        self._knowledge_cache: str | None = None
        self._knowledge_cache_time: float = 0.0
        # Admission control gate — filters trivial events before buffering
        admission_threshold = float(episodic_cfg.get("admission_threshold", 0.1))
        self._admission = MemoryAdmissionControl(threshold=admission_threshold)
        # Barrier capabilities (optional, injected by MemorySystem or caller)
        self._trend_analyzer = TrendAnalyzer()
        self._vector_store = vector_store  # VectorStore or None

        self._restore_active_buffer()

    # ── L1: Episode Logging ────────────────────────────────

    def log(
        self,
        event_type: str,
        description: str,
        context: dict[str, Any] | None = None,
        importance: float | None = None,
    ) -> Episode:
        """Log an event to the episode buffer.

        Args:
            event_type: One of perception, action, outcome, command, error, system
            description: Human-readable description of the event
            context: Optional structured data (detection labels, coordinates, etc.)
            importance: Override importance score (0-1). If None, auto-scored by rules.

        Returns:
            The created Episode for further use.
        """
        if importance is None:
            importance = score_importance(event_type, description, context)

        # Admission control — skip trivial/duplicate events
        admitted, _score = self._admission.should_admit(event_type, description, importance)
        if not admitted:
            logger.debug("Admission rejected [%s] imp=%.2f: %s", event_type, importance, description[:60])
            # Return a throwaway episode so callers don't need None checks
            return Episode(event_type, description, context, importance=importance)

        episode = Episode(event_type, description, context, importance=importance)
        self._buffer.append(episode)
        self._total_logged += 1
        self._cumulative_importance += importance
        self._append_to_active_journal(episode)

        # Periodic flush to disk
        if len(self._buffer) >= self._flush_threshold:
            self._flush_to_disk()

        logger.debug("Episode [%s] imp=%.2f: %s", event_type, importance, description[:60])
        return episode

    def get_recent(self, n: int = 20) -> list[Episode]:
        """Get the N most recent episodes."""
        return list(self._buffer)[-n:]

    def retrieve(self, query: str, top_k: int = 10) -> list[Episode]:
        """Retrieve episodes most relevant to query (Park 2023 style scoring).

        Scores each episode by: activation (recency+decay) + importance + keyword relevance
        """
        keywords = set(query.lower().split())
        now = time.time()
        scored = [
            (ep, ep.retrieval_score(keywords, now))
            for ep in self._buffer
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Mark retrieved episodes as accessed (Hebbian strengthening)
        results = []
        for ep, _ in scored[:top_k]:
            ep.access()
            results.append(ep)
        return results

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def cumulative_importance(self) -> float:
        return self._cumulative_importance

    # ── L2: Reflection ─────────────────────────────────────

    async def reflect(self, force: bool = False) -> str | None:
        """Trigger a reflection cycle: summarize recent episodes and extract knowledge.

        Returns the reflection summary, or None if skipped.
        """
        if not self._llm:
            return None
        if self._reflect_lock is None:
            self._reflect_lock = asyncio.Lock()
        async with self._reflect_lock:
            if self._reflecting:
                logger.debug("Reflection skipped: already running")
                return None

            # Check cooldown
            now = time.time()
            if not force and (now - self._last_reflect_time) < self._reflect_cooldown_s:
                logger.debug("Reflection skipped: cooldown (%.0fs remaining)",
                             self._reflect_cooldown_s - (now - self._last_reflect_time))
                return None

            # Check minimum events
            if not force and len(self._buffer) < self._reflect_min_events:
                logger.debug("Reflection skipped: only %d events", len(self._buffer))
                return None

            self._reflecting = True
        buffer_snapshot = list(self._buffer)  # snapshot before async LLM call
        episodes_text = "\n".join(ep.to_log_line() for ep in buffer_snapshot)
        existing_knowledge = self._load_all_knowledge()

        prompt = REFLECT_PROMPT.format(
            episodes=episodes_text,
            existing_knowledge=existing_knowledge or "暂无已知知识。",
        )

        try:
            response = await asyncio.wait_for(
                self._llm.chat([
                    {"role": "system", "content": "你是一个机器人的认知反思系统。用中文回答，直接输出JSON，不要思考过程。"},
                    {"role": "user", "content": prompt},
                ]),
                timeout=30.0,
            )

            reflection = self._parse_reflection(response)
            if reflection:
                self._save_digest(reflection)
                self._update_knowledge(reflection)
                self._last_reflect_time = time.time()
                logger.info("[EpisodicMemory] Reflection complete: %s",
                            reflection.get("summary", "")[:80])

                # Clear only the episodes we reflected on (preserve items added during await).
                # Use identity-based drain: if maxlen eviction occurred during the 15s LLM
                # call, the deque's left end may no longer contain the snapshot items.
                self._flush_to_disk()
                snapshot_ids = {id(ep) for ep in buffer_snapshot}
                # Use snapshot (not current buffer) — episodes may have been evicted
                # by maxlen during the LLM await, making id() lookups unreliable.
                importance_reflected = sum(ep.importance for ep in buffer_snapshot)
                self._buffer = deque(
                    (ep for ep in self._buffer if id(ep) not in snapshot_ids),
                    maxlen=MAX_BUFFER_SIZE,
                )
                self._cumulative_importance = max(
                    0.0, self._cumulative_importance - importance_reflected
                )
                # Re-write events that arrived during the LLM await so they
                # survive a restart (the old journal file contains them too,
                # but _restore_active_buffer only reads _active.jsonl).
                self._clear_active_journal()
                for ep in self._buffer:
                    self._append_to_active_journal(ep)

                # Run trend analysis and persist to vector store
                if self._trend_analyzer and self._vector_store:
                    try:
                        trends = self._trend_analyzer.analyze(buffer_snapshot)
                        for trend in trends:
                            self._vector_store.add(
                                trend.description_zh,
                                {"type": "trend", "ts": trend.window_end},
                            )
                    except Exception as _te:
                        logger.debug("Trend analysis after reflect failed: %s", _te)

                return reflection.get("summary", "")

        except asyncio.CancelledError:
            # Re-raise so asyncio.Task.cancel() propagates correctly.
            # Without this, the task appears "successfully completed" and
            # _pending_tasks callbacks never see the cancelled state.
            raise
        except Exception as exc:
            logger.warning("[EpisodicMemory] Reflection failed: %s", exc)
        finally:
            self._reflecting = False

        return None

    def should_reflect(self) -> bool:
        """Check if reflection should be triggered.

        Uses Park 2023 hybrid approach:
          1. Primary: cumulative importance exceeds threshold
          2. Fallback: buffer count exceeds minimum (for low-importance event streams)
          3. Cooldown prevents excessive reflection
        """
        if self._reflecting:
            return False
        now = time.time()
        if (now - self._last_reflect_time) < self._reflect_cooldown_s:
            return False
        # Primary: importance-based trigger
        if self._cumulative_importance >= self._importance_threshold:
            return True
        # Fallback: count-based
        return len(self._buffer) >= self._reflect_min_events

    # ── L3: World Knowledge ────────────────────────────────

    def get_knowledge_context(self, max_chars: int | None = None) -> str:
        """Load world knowledge for system prompt injection."""
        if max_chars is None:
            max_chars = self._knowledge_context_chars
        knowledge = self._load_all_knowledge()
        if not knowledge:
            return ""
        if len(knowledge) > max_chars:
            knowledge = knowledge[:max_chars] + "\n..."
        return f"世界知识:\n{knowledge}"

    def get_recent_digest(self, n: int = 3, max_chars: int | None = None) -> str:
        """Load recent episode digests for system prompt."""
        if max_chars is None:
            max_chars = self._digest_context_chars
        digest_files = sorted(self._digests_dir.glob("*.md"), reverse=True)
        if not digest_files:
            return ""

        entries = []
        total_chars = 0
        for f in digest_files[:n]:
            try:
                content = f.read_text(encoding="utf-8").strip()
                if content:
                    if total_chars + len(content) > max_chars:
                        remaining = max_chars - total_chars
                        if remaining <= 0:
                            break
                        entries.append(content[:remaining] + "...")
                        total_chars = max_chars
                        break
                    entries.append(content)
                    total_chars += len(content)
            except Exception:
                continue

        if not entries:
            return ""
        return "近期经历:\n" + "\n---\n".join(entries)

    # ── Internal: Disk operations ──────────────────────────

    def get_relevant_context(
        self,
        query: str,
        *,
        top_k: int | None = None,
        max_chars: int | None = None,
    ) -> str:
        """Return decayed/relevance-ranked episodic context for the current turn."""
        if not query.strip() or not self._buffer:
            return ""
        if top_k is None:
            top_k = self._relevant_top_k
        if max_chars is None:
            max_chars = self._relevant_context_chars

        entries: list[str] = []
        total_chars = 0
        for episode in self.retrieve(query, top_k=top_k):
            line = episode.to_log_line()
            if total_chars + len(line) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 0:
                    entries.append(line[:remaining] + "...")
                break
            entries.append(line)
            total_chars += len(line)

        if not entries:
            return ""
        return "Relevant episodes:\n" + "\n".join(entries)

    def _flush_to_disk(self) -> None:
        """Write current buffer to a timestamped JSONL file."""
        if not self._buffer:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filepath = self._episodes_dir / f"{timestamp}.jsonl"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for ep in self._buffer:
                    f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")
            logger.debug("Flushed %d episodes to %s", len(self._buffer), filepath.name)
        except Exception as exc:
            logger.warning("Episode flush failed: %s", exc)

    def _append_to_active_journal(self, episode: Episode) -> None:
        """Persist the live, unreflected buffer so restarts can restore it."""
        try:
            with open(self._active_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Episode journal append failed: %s", exc)

    def _restore_active_buffer(self) -> None:
        """Replay the live journal into the in-memory buffer on startup."""
        if not self._active_file.exists():
            return
        cutoff = time.time() - self._episode_retention_hours * 3600
        skipped_expired = 0
        skipped_corrupt = 0
        try:
            with open(self._active_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Parse each line independently — a truncated/corrupt line
                    # from an unclean shutdown must not discard the valid ones.
                    try:
                        payload = json.loads(line)
                        ep = Episode.from_dict(payload)
                    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                        logger.warning(
                            "Episode journal: skipping corrupt line: %s", exc
                        )
                        skipped_corrupt += 1
                        continue
                    # Skip episodes older than retention window — restoring
                    # them after a long shutdown pollutes the reflection context
                    # and causes immediate spurious should_reflect() triggers.
                    if ep.timestamp < cutoff:
                        skipped_expired += 1
                        continue
                    self._buffer.append(ep)
            self._total_logged = len(self._buffer)
            self._cumulative_importance = sum(ep.importance for ep in self._buffer)
            if self._buffer or skipped_corrupt or skipped_expired:
                logger.info(
                    "[EpisodicMemory] Restored %d live episodes from journal "
                    "(skipped: %d expired, %d corrupt)",
                    len(self._buffer), skipped_expired, skipped_corrupt,
                )
        except Exception as exc:
            logger.warning("Episode journal restore failed: %s", exc)
            self._buffer.clear()
            self._total_logged = 0
            self._cumulative_importance = 0.0

    def _clear_active_journal(self) -> None:
        """Clear the live journal after a successful reflection."""
        try:
            self._active_file.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Episode journal cleanup failed: %s", exc)

    def _save_digest(self, reflection: dict[str, Any]) -> None:
        """Save a reflection digest as a dated .md file."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        summary = reflection.get("summary", "")
        new_facts = reflection.get("new_facts", [])
        patterns = reflection.get("patterns", [])
        importance = reflection.get("importance", "medium")

        lines = [f"## {datetime.now().strftime('%Y-%m-%d %H:%M')} [{importance}]"]
        lines.append(summary)
        if new_facts:
            lines.append("\n新发现:")
            for item in new_facts:
                if isinstance(item, dict):
                    lines.append(f"- [{item.get('category', '?')}] {item.get('fact', item)}")
                else:
                    lines.append(f"- {item}")
        if patterns:
            lines.append("\n规律:")
            for item in patterns:
                if isinstance(item, dict):
                    conf = item.get("confidence", "?")
                    lines.append(f"- [{item.get('category', '?')}] {item.get('pattern', item)} (conf={conf})")
                else:
                    lines.append(f"- {item}")

        filepath = self._digests_dir / f"{timestamp}.md"
        filepath.write_text("\n".join(lines), encoding="utf-8")

    def _update_knowledge(self, reflection: dict[str, Any]) -> None:
        """Update categorized world knowledge files based on reflection output."""
        new_facts = reflection.get("new_facts", [])
        updates = reflection.get("updates", [])
        patterns = reflection.get("patterns", [])

        if not new_facts and not updates and not patterns:
            return
        # Invalidate knowledge cache so next _load_all_knowledge() re-reads disk
        self._knowledge_cache = None

        # Group by category
        categorized: dict[str, list[str]] = {}
        update_map: dict[str, list[dict[str, str]]] = {}

        for item in new_facts:
            if isinstance(item, dict):
                cat = item.get("category", "general")
                fact = item.get("fact", "")
            else:
                cat, fact = "general", str(item)
            if cat not in KNOWLEDGE_CATEGORIES:
                cat = "general"
            categorized.setdefault(cat, []).append(fact)

        for item in patterns:
            if isinstance(item, dict):
                cat = item.get("category", "routines")
                pattern = item.get("pattern", "")
                conf = item.get("confidence", "?")
                text = f"{pattern} (conf={conf})"
            else:
                cat, text = "routines", str(item)
            if cat not in KNOWLEDGE_CATEGORIES:
                cat = "routines"
            categorized.setdefault(cat, []).append(text)

        for item in updates:
            cat = item.get("category", "general")
            if cat not in KNOWLEDGE_CATEGORIES:
                cat = "general"
            update_map.setdefault(cat, []).append(item)

        # Write to category files
        all_cats = set(categorized.keys()) | set(update_map.keys())
        for cat in all_cats:
            self._write_category_knowledge(
                cat,
                categorized.get(cat, []),
                update_map.get(cat, []),
            )

    def _write_category_knowledge(
        self,
        category: str,
        new_facts: list[str],
        updates: list[dict[str, str]],
    ) -> None:
        """Write knowledge to a category-specific file."""
        knowledge_file = self._knowledge_dir / f"{category}.md"
        try:
            existing = ""
            if knowledge_file.exists():
                existing = knowledge_file.read_text(encoding="utf-8")

            cat_label = KNOWLEDGE_CATEGORIES.get(category, category)
            header = f"# {category}: {cat_label}"
            lines = existing.rstrip().split("\n") if existing.strip() else [header]

            # Handle updates (replace old facts)
            for update in updates:
                old_fact = update.get("old", "")
                new_fact = update.get("new", "")
                if old_fact and new_fact:
                    for i, line in enumerate(lines):
                        if old_fact in line:
                            lines[i] = f"- {new_fact}"
                            logger.info("[Knowledge/%s] Updated: %s -> %s",
                                        category, old_fact[:30], new_fact[:30])
                            break

            # Append new facts (fuzzy dedup — normalized 80% overlap)
            for fact in new_facts:
                if not fact:
                    continue
                fact_line = f"- {fact}"
                if not _is_duplicate(fact_line, lines):
                    lines.append(fact_line)

            knowledge_file.write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            # Disk full, permission error, Windows file lock, etc.
            # Log and return — do NOT propagate. The caller (_update_knowledge)
            # must still reach the buffer-drain step; an unhandled exception
            # here would leave the buffer un-cleared while the cooldown timer
            # has already been advanced, causing duplicate knowledge entries.
            logger.warning("[Knowledge/%s] Write failed, skipping: %s", category, exc)

    def _load_all_knowledge(self) -> str:
        """Load all knowledge .md files into a single string (cached 10s)."""
        import time as _time
        now = _time.monotonic()
        if (
            self._knowledge_cache is not None
            and (now - self._knowledge_cache_time) < 10.0
        ):
            return self._knowledge_cache
        parts = []
        for f in sorted(self._knowledge_dir.glob("*.md")):
            try:
                content = f.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(content)
            except Exception:
                continue
        result = "\n\n".join(parts)
        self._knowledge_cache: str | None = result
        self._knowledge_cache_time: float = now
        return result

    def _parse_reflection(self, response: str) -> dict[str, Any] | None:
        """Extract JSON from LLM reflection response using balanced brace matching."""
        # Strip <think>...</think> blocks (MiniMax thinking mode)
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        response = cleaned
        start = response.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(response[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(response[start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        logger.warning(
                            "Failed to parse reflection JSON: %s", response[:100]
                        )
                        return None
        logger.warning("Failed to parse reflection JSON: %s", response[:100])
        return None

    # ── Cleanup ────────────────────────────────────────────

    def cleanup_old_episodes(self) -> int:
        """Remove episode files older than EPISODE_RETENTION_HOURS. Returns count removed."""
        cutoff = time.time() - (self._episode_retention_hours * 3600)
        removed = 0
        for f in self._episodes_dir.glob("*.jsonl"):
            if f.name == self._active_file.name:
                continue
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except Exception as _e:
                logger.debug("skip cleanup %s: %s", f, _e)
                continue
        if removed:
            logger.info("[EpisodicMemory] Cleaned up %d old episode files", removed)
        return removed
