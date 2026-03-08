"""Tests for askme.brain.intent_router module."""

from askme.brain.intent_router import IntentRouter, IntentType


class FakeSafety:
    """Minimal safety checker stub."""
    def is_estop_command(self, text: str) -> bool:
        return "停" in text or "stop" in text.lower()


class TestIntentRouter:
    def _make_router(self, *, safety=True, triggers=None):
        return IntentRouter(
            safety_checker=FakeSafety() if safety else None,
            voice_triggers=triggers or {},
        )

    # ── E-STOP ──
    def test_estop_detected(self):
        router = self._make_router()
        intent = router.route("紧急停止")
        assert intent.type == IntentType.ESTOP

    def test_estop_english(self):
        router = self._make_router()
        intent = router.route("STOP NOW")
        assert intent.type == IntentType.ESTOP

    # ── Built-in commands ──
    def test_quit_command(self):
        router = self._make_router()
        intent = router.route("/quit")
        assert intent.type == IntentType.COMMAND
        assert intent.command == "/quit"

    def test_exit_command(self):
        router = self._make_router()
        intent = router.route("exit")
        assert intent.type == IntentType.COMMAND

    # ── Voice triggers ──
    def test_voice_trigger_match(self):
        router = self._make_router(triggers={"现在几点": "get_time"})
        intent = router.route("现在几点了？")
        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "get_time"

    def test_voice_trigger_no_match(self):
        router = self._make_router(triggers={"现在几点": "get_time"})
        intent = router.route("今天天气怎么样")
        assert intent.type == IntentType.GENERAL

    def test_voice_trigger_longest_match_wins(self):
        router = self._make_router(triggers={
            "移动": "robot_move",
            "移动到原点": "robot_home",
        })
        intent = router.route("移动到原点位置")
        assert intent.skill_name == "robot_home"

    # ── General fallback ──
    def test_general_fallback(self):
        router = self._make_router()
        intent = router.route("你好，请帮我写一段代码")
        assert intent.type == IntentType.GENERAL

    def test_empty_input(self):
        router = self._make_router(safety=False)
        intent = router.route("  ")
        assert intent.type == IntentType.GENERAL

    # ── No safety checker ──
    def test_no_safety_skips_estop(self):
        router = self._make_router(safety=False)
        intent = router.route("停")
        # Without safety checker, "停" is not recognized as e-stop
        assert intent.type == IntentType.GENERAL
