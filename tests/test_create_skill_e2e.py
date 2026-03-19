"""
End-to-end integration test for dynamic skill creation via LLM tool call.

Tests the full chain:
  user message
    → BrainPipeline._stream_with_tools()
      → MockLLM returns create_skill tool call
        → ToolRegistry executes CreateSkillTool
          → SKILL.md written to disk
            → SkillManager.hot_reload()
              → IntentRouter.update_voice_triggers()
                → new voice trigger routes correctly

No real LLM or TTS needed — the mock LLM provides a canned response that
includes a create_skill tool call, which exercises the real ToolRegistry,
SkillManager, and IntentRouter code paths.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock LLM that returns a create_skill tool call on first call,
# then a text confirmation on second call (after tool result).
# ---------------------------------------------------------------------------


def _make_tool_call_chunk(name: str, args: dict[str, Any]) -> Any:
    """Build a fake streaming chunk that looks like a tool_use message."""
    tc = types.SimpleNamespace(
        index=0,
        id="call_test_001",
        function=types.SimpleNamespace(
            name=name,
            arguments=json.dumps(args, ensure_ascii=False),
        ),
    )
    delta = types.SimpleNamespace(tool_calls=[tc], content=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


def _make_text_chunk(text: str) -> Any:
    delta = types.SimpleNamespace(tool_calls=None, content=text)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


class _MockLLM:
    """Fake LLM client.

    Call 1: returns a create_skill tool call for skill 'greet_visitor'.
    Call 2: returns a text confirmation after receiving the tool result.
    """

    def __init__(self) -> None:
        self._call_count = 0

    async def chat_stream(self, messages, **kwargs):
        self._call_count += 1
        if self._call_count == 1:
            # First call: emit a create_skill tool call
            yield _make_tool_call_chunk(
                "create_skill",
                {
                    "name": "greet_visitor",
                    "description": "当有访客时向他们打招呼",
                    "voice_trigger": "打招呼,欢迎访客,来客人了",
                    "prompt": (
                        "有访客到来，用热情的中文问候语迎接他们。\n\n"
                        "用户指令：{{user_input}}"
                    ),
                },
            )
        else:
            # Second call (after tool result): confirm in text
            yield _make_text_chunk("好的，技能已经创建好了，")
            yield _make_text_chunk('以后说"打招呼"就能触发这个技能了。')


# ---------------------------------------------------------------------------
# Minimal stubs for dependencies not under test
# ---------------------------------------------------------------------------


class _SilentAudio:
    """AudioAgent stub — captures speak() calls, ignores playback."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self._muted = False
        self._is_playing = False
        self.tts = MagicMock()
        self.tts.is_active.return_value = False

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def drain_buffers(self) -> None:
        pass

    def start_playback(self) -> None:
        pass

    def stop_playback(self) -> None:
        pass

    def wait_speaking_done(self) -> None:
        pass

    def is_busy(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_creates_skill_via_tool_call(tmp_path: Path) -> None:
    """LLM issues a create_skill tool call → skill written → router updated."""
    from askme.brain.conversation import ConversationManager
    from askme.brain.intent_router import IntentRouter, IntentType
    from askme.brain.memory_bridge import MemoryBridge
    from askme.pipeline.brain_pipeline import BrainPipeline
    from askme.skills.skill_executor import SkillExecutor
    from askme.skills.skill_manager import SkillManager
    from askme.tools.builtin_tools import register_builtin_tools
    from askme.tools.skill_tools import register_skill_tools
    from askme.tools.tool_registry import ToolRegistry
    from askme.voice.stream_splitter import StreamSplitter

    # ── Point generated_skills_dir at a temp directory ───────────────────────
    # Monkeypatch the property so CreateSkillTool writes to tmp_path.
    # Save and restore the original descriptor so this mutation doesn't leak
    # into subsequent tests (plain `del` would remove the property entirely).
    _orig_prop = SkillManager.__dict__["generated_skills_dir"]
    SkillManager.generated_skills_dir = property(lambda self: tmp_path / "skills")  # type: ignore[assignment]

    try:
        # ── Build dependencies ────────────────────────────────────────────────
        skill_mgr = SkillManager()
        skill_mgr.load()

        router = IntentRouter(voice_triggers=skill_mgr.get_voice_triggers())

        registry = ToolRegistry()
        register_builtin_tools(registry, production_mode=True)
        register_skill_tools(registry, skill_mgr, router)

        audio = _SilentAudio()
        splitter = StreamSplitter()
        conversation = ConversationManager()
        memory = MemoryBridge()
        skill_executor = SkillExecutor(MagicMock(), registry)

        pipeline = BrainPipeline(
            llm=_MockLLM(),
            conversation=conversation,
            memory=memory,
            tools=registry,
            skill_manager=skill_mgr,
            skill_executor=skill_executor,
            audio=audio,
            splitter=splitter,
            system_prompt="你是一个有用的助手，可以创建新技能。",
            general_tool_max_safety_level="normal",
        )

        # ── Step 1: user asks for something that triggers skill creation ──────
        response = await pipeline.process("帮我创建一个打招呼的技能，叫 greet_visitor")
        print(f"\nLLM response: {response!r}")

        # ── Step 2: verify skill was written to disk ──────────────────────────
        skill_file = tmp_path / "skills" / "greet_visitor" / "SKILL.md"
        assert skill_file.exists(), f"SKILL.md not found at {skill_file}"
        content = skill_file.read_text(encoding="utf-8")
        assert "greet_visitor" in content
        assert "打招呼" in content
        print(f"SKILL.md created at: {skill_file}")

        # ── Step 3: verify router was hot-reloaded with new triggers ──────────
        intent = router.route("打招呼")
        assert intent.type == IntentType.VOICE_TRIGGER, (
            f"Expected VOICE_TRIGGER, got {intent.type}"
        )
        assert intent.skill_name == "greet_visitor", (
            f"Expected greet_visitor, got {intent.skill_name}"
        )
        print(f"Router updated: '打招呼' -> {intent.skill_name}")

        intent2 = router.route("来客人了")
        assert intent2.skill_name == "greet_visitor"
        print(f"Router updated: '来客人了' -> {intent2.skill_name}")

        # ── Step 4: skill appears in skill manager ────────────────────────────
        skill = skill_mgr.get("greet_visitor")
        assert skill is not None
        assert skill.enabled
        print(f"Skill in manager: {skill.name} v{skill.version}")

        # ── Step 5: LLM was called exactly twice (tool call + follow-up) ──────
        assert pipeline._llm._call_count == 2, (  # type: ignore[attr-defined]
            f"Expected 2 LLM calls, got {pipeline._llm._call_count}"  # type: ignore[attr-defined]
        )

        print("\n✓ All assertions passed — create_skill tool call works end-to-end")

    finally:
        # Restore original property descriptor (not del — that would remove it entirely)
        SkillManager.generated_skills_dir = _orig_prop  # type: ignore[assignment]
