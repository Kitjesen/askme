from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from askme.pipeline.voice_loop import VoiceLoop
from askme.skills.skill_manager import SkillManager

from askme.robot_interaction import IntentRouter, IntentType

VOICE_TEXT = "\u770b\u770b\u6587\u4ef6"
EXPECTED_SKILL = "list_directory"


class _Audio:
    awaiting_confirmation = False

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self._index = 0
        self.ack_count = 0
        self.spoken: list[str] = []

    def listen_loop(self) -> str:
        text = self._texts[self._index]
        self._index += 1
        return text

    def acknowledge(self) -> None:
        self.ack_count += 1

    def drain_buffers(self) -> None:
        return None

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def start_playback(self) -> None:
        return None

    def wait_speaking_done(self) -> None:
        return None

    def stop_playback(self) -> None:
        return None

    async def speak_and_wait(self, text: str) -> None:
        self.spoken.append(text)

    @property
    def is_muted(self) -> bool:
        return False


class _Pipeline:
    last_spoken_text = ""

    def __init__(self) -> None:
        self.process_calls: list[str] = []
        self.execute_skill_calls: list[tuple[str, str]] = []

    def has_pending_tool_approval(self) -> bool:
        return False

    def start_idle_reflection(self):
        return None

    def start_memory_prefetch(self, user_text: str):
        return asyncio.create_task(asyncio.sleep(0, result=""))

    async def handle_pending_tool_response(self, user_text: str):
        return None

    async def process(self, user_text: str, *, memory_task=None):
        self.process_calls.append(user_text)
        return "general"

    async def execute_skill(self, skill_name: str, user_text: str):
        self.execute_skill_calls.append((skill_name, user_text))
        return "skill"

    def handle_estop(self) -> None:
        return None


class _Dispatcher:
    def __init__(self, skill_manager: SkillManager) -> None:
        self._skill_manager = skill_manager
        self.dispatch_calls: list[tuple[str, str, str]] = []
        self.general_calls: list[tuple[str, str]] = []
        self.cancel_calls = 0

    @property
    def has_active_agent_task(self) -> bool:
        return False

    @property
    def current_mission(self):
        return None

    def get_skill(self, name: str):
        return self._skill_manager.get(name)

    async def dispatch(self, skill_name: str, user_text: str, *, source: str = "") -> None:
        self.dispatch_calls.append((skill_name, user_text, source))

    async def handle_general(self, user_text: str, *, source: str = "", memory_task=None) -> None:
        self.general_calls.append((user_text, source))

    def cancel_active_agent_task(self) -> bool:
        self.cancel_calls += 1
        return False


@pytest.fixture()
def skill_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SkillManager:
    import askme.skills.skill_manager as skill_manager_module

    data_dir = tmp_path / "data"
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    generated_dir = data_dir / "skills"
    home_dir.mkdir()
    project_dir.mkdir()
    generated_dir.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("USERPROFILE", str(home_dir))
    monkeypatch.setattr(skill_manager_module, "_DATA_DIR", data_dir)
    monkeypatch.setattr(
        skill_manager_module,
        "_SETTINGS_FILE",
        data_dir / "skills_settings.json",
    )

    manager = SkillManager(project_dir=project_dir)
    manager.load()
    return manager


@pytest.mark.asyncio
async def test_builtin_voice_trigger_dispatches_from_voice_loop_without_general_fallback(
    skill_manager: SkillManager,
) -> None:
    router = IntentRouter(voice_triggers=skill_manager.get_voice_triggers())
    intent = router.route(VOICE_TEXT)
    assert intent.type == IntentType.VOICE_TRIGGER
    assert intent.skill_name == EXPECTED_SKILL

    pipeline = _Pipeline()
    dispatcher = _Dispatcher(skill_manager)
    loop = VoiceLoop(
        router=router,
        pipeline=pipeline,
        audio=_Audio([VOICE_TEXT, "exit"]),
        dispatcher=dispatcher,
    )

    await loop.run()

    assert dispatcher.dispatch_calls == [(EXPECTED_SKILL, VOICE_TEXT, "voice")]
    assert dispatcher.general_calls == []
    assert pipeline.process_calls == []
    assert pipeline.execute_skill_calls == []
