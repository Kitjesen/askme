"""Tests for robot interaction intent routing."""

from askme.robot_interaction import IntentRouter, IntentType, RoutingPolicy, intent_route_payload


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
        assert intent.reason == "safety_checker"

    def test_estop_keyword_is_case_insensitive_without_safety(self):
        router = self._make_router(safety=False)
        intent = router.route("Emergency Stop")
        assert intent.type == IntentType.ESTOP
        assert intent.reason == "estop_keyword"

    # ── Quick replies ──
    def test_quick_reply_has_explicit_reply_text(self):
        router = self._make_router()
        intent = router.route("你好")
        assert intent.type == IntentType.QUICK_REPLY
        assert intent.reply_text == "你好，有什么需要帮忙的？"
        assert intent.skill_name == intent.reply_text  # legacy compatibility
        assert intent.reason == "quick_reply"

    def test_intent_route_payload_is_audit_ready(self):
        router = self._make_router(triggers={"导航到仓库": "navigate"})
        intent = router.route("帮我导航到仓库")

        payload = intent_route_payload(intent, source="voice")

        assert payload == {
            "type": "voice_trigger",
            "reason": "voice_trigger",
            "source": "voice",
            "raw_text_preview": "帮我导航到仓库",
            "skill_name": "navigate",
            "trigger_phrase": "导航到仓库",
        }

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
        # No question mark — treated as a command, trigger fires
        intent = router.route("现在几点了")
        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "get_time"
        assert intent.trigger_phrase == "现在几点"
        assert intent.reason == "voice_trigger"

    def test_voice_trigger_question_mark_allows_query_skill(self):
        router = self._make_router(triggers={"现在几点": "get_time"})
        # ASR may add question punctuation to query skills; keep read-only
        # skill routing active for those cases.
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
        assert intent.reason == "empty_input"

    # ── No safety checker ──
    def test_no_safety_skips_estop(self):
        router = self._make_router(safety=False)
        intent = router.route("停")
        # Without safety checker, "停" is not recognized as e-stop
        assert intent.type == IntentType.GENERAL

    # ── Routing policy ──
    def test_routing_policy_can_override_deterministic_surfaces(self):
        policy = RoutingPolicy(
            builtin_commands={"/status"},
            estop_keywords={"halt"},
            quick_replies={"ping": "pong"},
        )
        router = IntentRouter(policy=policy)

        assert router.route("ping").reply_text == "pong"
        assert router.route("/STATUS").command == "/status"
        assert router.route("HALT").type == IntentType.ESTOP


class TestNegationDetection:
    """Voice triggers preceded by negation words must NOT fire."""

    def setup_method(self):
        self.router = IntentRouter(
            voice_triggers={
                "停下": "stop_speaking",   # 2-char trigger (meets MIN_TRIGGER_LENGTH)
                "导航": "navigate",
                "说话": "talking_skill",
                "导航到仓库": "navigate",  # longer trigger for priority
            }
        )

    def test_bu_yao_stop(self):
        intent = self.router.route("不要停下")
        assert intent.type == IntentType.GENERAL

    def test_bu_stop(self):
        intent = self.router.route("不停下")
        assert intent.type == IntentType.GENERAL

    def test_bie_stop(self):
        intent = self.router.route("别停下")
        assert intent.type == IntentType.GENERAL

    def test_bu_yao_navigate(self):
        intent = self.router.route("不要导航")
        assert intent.type == IntentType.GENERAL

    def test_bu_yong_talking(self):
        intent = self.router.route("不用说话了")
        assert intent.type == IntentType.GENERAL

    def test_bie_zai_stop(self):
        intent = self.router.route("别再停下来了")
        assert intent.type == IntentType.GENERAL

    def test_positive_stop_still_fires(self):
        """Non-negated trigger still fires the skill."""
        intent = self.router.route("帮我停下来")
        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "stop_speaking"

    def test_positive_navigate_still_fires(self):
        intent = self.router.route("导航到仓库")
        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "navigate"

    def test_qing_bu_yao_stop(self):
        """Negation with leading polite prefix."""
        intent = self.router.route("请不要停下")
        assert intent.type == IntentType.GENERAL

    def test_mei_you_stop(self):
        """没有 also negates."""
        intent = self.router.route("没有停下")
        assert intent.type == IntentType.GENERAL


class TestQuestionContext:
    """Voice triggers inside question phrases must NOT fire."""

    def setup_method(self):
        self.router = IntentRouter(
            voice_triggers={
                "导航": "navigate",
                "导航到仓库": "navigate",
                "停止播放": "stop_speaking",
                "环境报告": "environment_report",
            }
        )

    def test_question_ending_ma(self):
        """'导航会失败吗' ends with 吗 → GENERAL."""
        intent = self.router.route("导航会失败吗")
        assert intent.type == IntentType.GENERAL

    def test_question_ending_me(self):
        intent = self.router.route("导航到底有没有用么")
        assert intent.type == IntentType.GENERAL

    def test_question_ending_ne(self):
        intent = self.router.route("你能帮我导航呢")
        assert intent.type == IntentType.GENERAL

    def test_question_ending_ma2(self):
        intent = self.router.route("停止播放好用吗")
        assert intent.type == IntentType.GENERAL

    def test_question_mark_fullwidth(self):
        intent = self.router.route("导航到仓库？")
        assert intent.type == IntentType.GENERAL

    def test_question_mark_ascii(self):
        intent = self.router.route("导航到仓库?")
        assert intent.type == IntentType.GENERAL

    def test_command_still_fires(self):
        """Non-question command still triggers the skill."""
        intent = self.router.route("帮我导航到仓库")
        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "navigate"

    def test_bare_trigger_fires(self):
        """Single trigger word (no question) still fires."""
        intent = self.router.route("导航")
        assert intent.type == IntentType.VOICE_TRIGGER

    def test_environment_report_fires(self):
        intent = self.router.route("环境报告")
        assert intent.type == IntentType.VOICE_TRIGGER

    def test_environment_report_question(self):
        intent = self.router.route("环境报告准确吗")
        assert intent.type == IntentType.GENERAL
