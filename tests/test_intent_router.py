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

    def test_exact_estop_ignores_trailing_punctuation_without_safety(self):
        router = self._make_router(safety=False)

        assert router.route("\u6025\u505c\uff01").type == IntentType.ESTOP
        assert router.route("estop!").type == IntentType.ESTOP
        assert router.route("\u4e0d\u8981\u6025\u505c\uff01").type == IntentType.GENERAL

    # ── Quick replies ──
    def test_quick_reply_has_explicit_reply_text(self):
        router = self._make_router()
        intent = router.route("你好")
        assert intent.type == IntentType.QUICK_REPLY
        assert intent.reply_text == "你好，有什么需要帮忙的？"
        assert intent.skill_name == intent.reply_text  # legacy compatibility
        assert intent.reason == "quick_reply"

    def test_self_introduction_uses_cached_quick_reply_path(self):
        router = self._make_router()

        intent = router.route("\u4f60\u662f\u8c01\uff1f")

        assert intent.type == IntentType.QUICK_REPLY
        assert intent.fast_path is True
        assert intent.cached_audio_key
        assert "\u5c0f\u7b97" in (intent.reply_text or "")

    def test_location_status_routes_to_read_only_skill(self):
        router = self._make_router()

        intent = router.route("\u5f53\u524d\u4f4d\u7f6e")

        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "nav_query"
        assert intent.reason == "read_only_fast_path"
        assert intent.fast_path is True
        assert intent.preface_text
        assert intent.preface_audio_key

    def test_action_phrase_uses_runtime_task_without_fast_path(self):
        router = self._make_router(triggers={"\u5bfc\u822a": "navigate"})

        intent = router.route("\u5e26\u6211\u53bb\u5927\u5802")

        assert intent.fast_path is False
        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "runtime_task"

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

    def test_task_status_is_a_deterministic_local_control(self):
        router = self._make_router()

        intent = router.route("任务怎么样了？")

        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "task_status"
        assert intent.reason == "task_control"

    def test_task_cancel_is_a_deterministic_local_control(self):
        router = self._make_router()

        intent = router.route("取消当前任务")

        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "task_cancel"
        assert intent.reason == "task_control"

    def test_task_confirmation_is_a_deterministic_local_control(self):
        router = self._make_router()

        intent = router.route("确认执行")

        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "task_confirm"
        assert intent.reason == "task_control"

    def test_task_evidence_is_a_deterministic_local_control(self):
        router = self._make_router()

        intent = router.route("照片呢？")

        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "task_evidence"
        assert intent.reason == "task_control"

    def test_task_cancel_capability_question_does_not_cancel(self):
        router = self._make_router()

        intent = router.route("你可以取消任务吗？")

        assert intent.type == IntentType.GENERAL

    def test_runtime_robot_task_requests_use_dedicated_route(self):
        router = self._make_router()

        for phrase in (
            "导航到北门",
            "去北门",
            "请前往仓库",
            "巡检A区",
            "巡检A区后拍照汇报",
            "生成状态报告",
        ):
            intent = router.route(phrase)
            assert intent.type == IntentType.VOICE_TRIGGER, phrase
            assert intent.skill_name == "runtime_task", phrase
            assert intent.reason == "runtime_task_request", phrase

    def test_explicit_robot_skill_trigger_precedes_runtime_task_fallback(self):
        router = self._make_router(
            triggers={
                "导航": "navigate_generic",
                "导航到仓库": "navigate_warehouse",
            }
        )

        intent = router.route("导航到仓库取货")

        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "navigate_warehouse"
        assert intent.reason == "voice_trigger"

    def test_runtime_task_capability_questions_stay_general(self):
        router = self._make_router()

        for phrase in (
            "你会巡检吗？",
            "你能去北门吗",
            "如何前往北门",
            "状态报告是什么",
        ):
            assert router.route(phrase).type == IntentType.GENERAL, phrase

    def test_runtime_task_fallback_rejects_history_results_and_unpunctuated_questions(self):
        router = self._make_router()

        for phrase in (
            "巡检结束了",
            "刚才巡检A区失败了",
            "昨天巡检了北门",
            "请问巡检A区要多久",
            "巡检A区怎么样",
            "去年巡检A区",
        ):
            intent = router.route(phrase)
            assert intent.type == IntentType.GENERAL, phrase

    def test_runtime_task_fallback_accepts_explicit_polite_commands(self):
        router = self._make_router()

        for phrase in (
            "请巡检A区",
            "巡检过道",
            "麻烦你导航到北门",
            "让机器人巡查仓库",
            "立即生成状态报告",
        ):
            intent = router.route(phrase)
            assert intent.type == IntentType.VOICE_TRIGGER, phrase
            assert intent.skill_name == "runtime_task", phrase

    def test_negated_runtime_task_request_stays_general(self):
        router = self._make_router()

        for phrase in ("不要去北门", "先别巡检A区", "不要生成状态报告"):
            assert router.route(phrase).type == IntentType.GENERAL, phrase

    def test_generic_agent_task_trigger_is_not_reclassified_as_runtime_task(self):
        router = self._make_router(triggers={"帮我写代码": "agent_task"})

        intent = router.route("帮我写代码")

        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "agent_task"
        assert intent.reason == "voice_trigger"

    def test_agent_authoring_about_robot_tasks_is_not_executed_on_the_robot(self):
        router = self._make_router(
            triggers={
                "帮我写代码": "agent_task",
                "帮我整理": "agent_task",
            }
        )

        for phrase in (
            "帮我写代码，做一个巡检系统",
            "帮我整理一份巡检报告",
        ):
            intent = router.route(phrase)
            assert intent.type == IntentType.VOICE_TRIGGER, phrase
            assert intent.skill_name == "agent_task", phrase

    def test_unconfigured_authoring_context_does_not_become_runtime_task(self):
        router = self._make_router()

        for phrase in ("设计一个巡检系统", "写一份巡检报告模板"):
            assert router.route(phrase).type == IntentType.GENERAL, phrase

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
        router = self._make_router(
            triggers={
                "移动": "robot_move",
                "移动到原点": "robot_home",
            }
        )
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
    def test_local_estop_vocabulary_does_not_depend_on_safety_checker(self):
        router = self._make_router(safety=False)

        for phrase in (
            "停",
            "停下",
            "停下来",
            "别动",
            "不要动",
            "立即停止",
            "马上停止",
            "危险",
            "halt",
            "freeze",
        ):
            intent = router.route(phrase)
            assert intent.type == IntentType.ESTOP
            assert intent.reason == "estop_keyword"

    def test_stop_speaking_phrase_is_not_promoted_to_estop(self):
        router = self._make_router(
            safety=False,
            triggers={"停止播放": "stop_speaking"},
        )

        intent = router.route("停止播放")

        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "stop_speaking"

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

    def test_generic_look_word_does_not_override_nonvisual_skill_trigger(self):
        router = self._make_router(triggers={"看看文件": "list_directory"})

        intent = router.route("看看文件")

        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "list_directory"

    def test_visual_environment_request_stays_on_current_camera_pipeline(self):
        router = self._make_router(triggers={"看看周围": "environment_report"})

        intent = router.route("看看周围")

        assert intent.type == IntentType.GENERAL
        assert intent.reason == "visual_query"


class TestNegationDetection:
    """Voice triggers preceded by negation words must NOT fire."""

    def setup_method(self):
        self.router = IntentRouter(
            voice_triggers={
                "停下": "stop_speaking",  # 2-char trigger (meets MIN_TRIGGER_LENGTH)
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
