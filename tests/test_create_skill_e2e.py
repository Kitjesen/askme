"""End-to-end generated skill creation through BrainPipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _tool_call_chunk(name: str, arguments: dict) -> SimpleNamespace:
    tool_call = SimpleNamespace(
        index=0,
        id="call-create-skill",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="", tool_calls=[tool_call]),
                finish_reason=None,
            )
        ]
    )


def _text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=[]),
                finish_reason=None,
            )
        ]
    )


class _MockLLM:
    def __init__(self) -> None:
        self._call_count = 0

    async def chat_stream(self, messages, **kwargs):
        self._call_count += 1
        if self._call_count == 1:
            yield _tool_call_chunk(
                "create_skill",
                {
                    "name": "greet_visitor",
                    "description": "Greet visitors when they arrive",
                    "voice_trigger": "greet-visitor-trigger,visitor-arrived-trigger",
                    "prompt": "Greet the visitor in short Chinese. User input: {{user_input}}",
                },
            )
        else:
            yield _text_chunk("好的，技能草稿已经创建，等待管理员审批。")


class _SilentAudio:
    def __init__(self) -> None:
        self.spoken: list[str] = []
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


@pytest.mark.asyncio
async def test_llm_creates_generated_skill_draft_then_operator_approves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import askme.skills.skill_manager as skill_manager_module
    from askme.memory.bridge import MemoryBridge
    from askme.pipeline.brain_pipeline import BrainPipeline
    from askme.skills.audit import SkillAuditLog
    from askme.skills.skill_executor import SkillExecutor
    from askme.skills.skill_manager import SkillManager
    from askme.tools.builtin_tools import register_builtin_tools
    from askme.tools.skill_tools import register_skill_tools
    from askme.tools.tool_registry import ToolRegistry
    from askme.voice.stream_splitter import StreamSplitter

    from askme.llm.conversation import ConversationManager
    from askme.robot_interaction import IntentRouter, IntentType

    original_generated_dir = SkillManager.__dict__["generated_skills_dir"]
    SkillManager.generated_skills_dir = property(lambda self: tmp_path / "skills")  # type: ignore[assignment]
    monkeypatch.setattr(skill_manager_module, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        skill_manager_module,
        "_SETTINGS_FILE",
        tmp_path / "skills_settings.json",
    )
    monkeypatch.setattr(
        skill_manager_module,
        "SkillAuditLog",
        lambda: SkillAuditLog(tmp_path / "skill-audit.jsonl"),
    )

    try:
        skill_manager = SkillManager(project_dir=tmp_path)
        skill_manager.load()
        router = IntentRouter(voice_triggers=skill_manager.get_voice_triggers())

        registry = ToolRegistry()
        register_builtin_tools(registry, production_mode=True)
        register_skill_tools(registry, skill_manager, router)

        pipeline = BrainPipeline(
            llm=_MockLLM(),
            conversation=ConversationManager(),
            memory=MemoryBridge(),
            tools=registry,
            skill_manager=skill_manager,
            skill_executor=SkillExecutor(MagicMock(), registry),
            audio=_SilentAudio(),
            splitter=StreamSplitter(),
            system_prompt="You can create generated skill drafts.",
            general_tool_max_safety_level="normal",
        )

        response = await pipeline.process("Create a greet_visitor skill")

        skill_file = tmp_path / "skills" / "greet_visitor" / "SKILL.md"
        assert skill_file.exists()
        assert "greet_visitor" in skill_file.read_text(encoding="utf-8")
        assert "等待管理员审批" in response

        skill = skill_manager.get("greet_visitor")
        assert skill is not None
        assert skill.enabled is False
        assert router.route("greet-visitor-trigger").type != IntentType.VOICE_TRIGGER

        queue = skill_manager.get_generated_skill_governance()
        record = next(item for item in queue["records"] if item["skill_name"] == "greet_visitor")
        assert record["status"] == "pending_approval"
        assert record["enabled"] is False

        approved = skill_manager.review_generated_skill(
            "greet_visitor",
            action="approve",
            operator_id="test.operator",
            router=router,
        )

        assert approved["ok"] is True
        assert approved["enabled"] is True
        intent = router.route("greet-visitor-trigger")
        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "greet_visitor"
        assert pipeline._llm._call_count == 2  # type: ignore[attr-defined]
    finally:
        SkillManager.generated_skills_dir = original_generated_dir  # type: ignore[assignment]
