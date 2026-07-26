from __future__ import annotations

from askme.robot_interaction.routing.fast_voice_intents import (
    FastVoiceIntentKind,
    match_fast_voice_intent,
)
from askme.voice.input.fast_endpoint import FastEndpointAction, FastEndpointController

QUICK_REPLIES = {
    "\u4f60\u597d": "\u4f60\u597d\uff0c\u6709\u4ec0\u4e48\u53ef\u4ee5\u5e2e\u60a8\uff1f",
    "\u4f60\u662f\u8c01": "\u6211\u662f\u5c0f\u7b97\uff0c\u4e00\u53ea\u667a\u80fd\u670d\u52a1\u673a\u5668\u72d7\u3002",
}


def test_fast_intent_matches_exact_quick_reply_with_punctuation() -> None:
    intent = match_fast_voice_intent(" \u4f60\u597d\uff01 ", quick_replies=QUICK_REPLIES)

    assert intent is not None
    assert intent.kind is FastVoiceIntentKind.QUICK_REPLY
    assert intent.reply_text == QUICK_REPLIES["\u4f60\u597d"]
    assert intent.cache_key


def test_fast_intent_does_not_capture_longer_or_action_bearing_phrase() -> None:
    assert (
        match_fast_voice_intent(
            "\u4f60\u597d\uff0c\u8bf7\u5e26\u6211\u53bb\u5927\u5802",
            quick_replies=QUICK_REPLIES,
        )
        is None
    )
    assert (
        match_fast_voice_intent(
            "\u4ecb\u7ecd\u4e00\u4e0b\u8fd9\u4e2a\u8bbe\u5907",
            quick_replies=QUICK_REPLIES,
        )
        is None
    )


def test_location_status_is_read_only_skill_not_robot_action() -> None:
    intent = match_fast_voice_intent("\u5f53\u524d\u4f4d\u7f6e", quick_replies=QUICK_REPLIES)

    assert intent is not None
    assert intent.kind is FastVoiceIntentKind.READ_ONLY_SKILL
    assert intent.skill_name == "nav_query"
    assert intent.preface_text


def test_fast_intent_matches_only_an_exact_estop_utterance() -> None:
    intent = match_fast_voice_intent("\u6025\u505c\uff01", quick_replies=QUICK_REPLIES)

    assert intent is not None
    assert intent.kind is FastVoiceIntentKind.ESTOP
    assert intent.intent_id == "estop"
    assert match_fast_voice_intent("\u4e0d\u8981\u6025\u505c", quick_replies=QUICK_REPLIES) is None
    assert match_fast_voice_intent("\u6025\u505c\u4e00\u4e0b", quick_replies=QUICK_REPLIES) is None


def test_fast_endpoint_requires_quiet_and_stable_partial() -> None:
    controller = FastEndpointController(
        quick_replies=QUICK_REPLIES,
        enabled=True,
        candidate_silence_ms=300,
        stable_partial_ms=150,
    )

    assert controller.observe(partial_text="\u4f60\u597d", quiet=False, now=10.0).action is FastEndpointAction.WAIT
    assert controller.observe(partial_text="\u4f60\u597d", quiet=True, now=10.1).action is FastEndpointAction.WAIT
    assert controller.observe(partial_text="\u4f60\u597d", quiet=True, now=10.39).action is FastEndpointAction.WAIT

    decision = controller.observe(partial_text="\u4f60\u597d", quiet=True, now=10.41)

    assert decision.action is FastEndpointAction.COMMIT
    assert decision.intent is not None
    assert decision.intent.kind is FastVoiceIntentKind.QUICK_REPLY
    assert decision.silence_ms >= 300
    assert decision.stable_text_ms >= 150


def test_fast_endpoint_resets_candidate_when_user_resumes() -> None:
    controller = FastEndpointController(
        quick_replies=QUICK_REPLIES,
        enabled=True,
        candidate_silence_ms=300,
        stable_partial_ms=0,
    )

    controller.observe(partial_text="\u4f60\u597d", quiet=True, now=1.0)
    controller.observe(partial_text="\u4f60\u597d", quiet=False, now=1.2)
    decision = controller.observe(partial_text="\u4f60\u597d", quiet=True, now=1.31)

    assert decision.action is FastEndpointAction.WAIT
    assert decision.silence_ms == 0


def test_fast_endpoint_resets_silence_when_candidate_no_longer_matches() -> None:
    controller = FastEndpointController(
        quick_replies=QUICK_REPLIES,
        enabled=True,
        candidate_silence_ms=300,
        stable_partial_ms=0,
    )

    controller.observe(partial_text="\u4f60\u597d", quiet=True, now=1.0)
    controller.observe(partial_text="\u4f60\u597d\u8bf7\u5e26\u8def", quiet=True, now=1.2)
    decision = controller.observe(partial_text="\u4f60\u597d", quiet=True, now=1.31)

    assert decision.action is FastEndpointAction.WAIT
    assert decision.silence_ms == 0


def test_fast_endpoint_resets_silence_when_candidate_identity_changes() -> None:
    controller = FastEndpointController(
        quick_replies=QUICK_REPLIES,
        enabled=True,
        candidate_silence_ms=300,
        stable_partial_ms=0,
    )

    controller.observe(partial_text="\u4f60\u597d", quiet=True, now=1.0)
    decision = controller.observe(partial_text="\u4f60\u662f\u8c01", quiet=True, now=1.31)

    assert decision.action is FastEndpointAction.WAIT
    assert decision.silence_ms == 0


def test_fast_endpoint_never_commits_an_unknown_or_longer_phrase() -> None:
    controller = FastEndpointController(
        quick_replies=QUICK_REPLIES,
        enabled=True,
        candidate_silence_ms=100,
        stable_partial_ms=0,
    )

    controller.observe(
        partial_text="\u4f60\u597d\uff0c\u8bf7\u8ba9\u4e00\u4e0b",
        quiet=True,
        now=1.0,
    )
    decision = controller.observe(
        partial_text="\u4f60\u597d\uff0c\u8bf7\u8ba9\u4e00\u4e0b",
        quiet=True,
        now=2.0,
    )

    assert decision.action is FastEndpointAction.WAIT


def test_exact_estop_uses_shorter_silence_without_speeding_up_normal_intents() -> None:
    estop = FastEndpointController(
        quick_replies=QUICK_REPLIES,
        enabled=True,
        candidate_silence_ms=300,
        estop_candidate_silence_ms=150,
        stable_partial_ms=0,
    )

    first = estop.observe(partial_text="\u6025\u505c", quiet=True, now=1.0)
    decision = estop.observe(partial_text="\u6025\u505c", quiet=True, now=1.16)

    assert first.action is FastEndpointAction.WAIT
    assert decision.action is FastEndpointAction.COMMIT
    assert decision.intent is not None
    assert decision.intent.kind is FastVoiceIntentKind.ESTOP
    assert 150 <= decision.silence_ms < 300

    ordinary = FastEndpointController(
        quick_replies=QUICK_REPLIES,
        enabled=True,
        candidate_silence_ms=300,
        estop_candidate_silence_ms=150,
        stable_partial_ms=0,
    )
    ordinary.observe(partial_text="\u4f60\u597d", quiet=True, now=1.0)
    ordinary_decision = ordinary.observe(
        partial_text="\u4f60\u597d",
        quiet=True,
        now=1.16,
    )

    assert ordinary_decision.action is FastEndpointAction.WAIT

    read_only = FastEndpointController(
        quick_replies=QUICK_REPLIES,
        enabled=True,
        candidate_silence_ms=300,
        estop_candidate_silence_ms=150,
        stable_partial_ms=0,
    )
    read_only.observe(partial_text="\u5f53\u524d\u4f4d\u7f6e", quiet=True, now=1.0)
    read_only_decision = read_only.observe(
        partial_text="\u5f53\u524d\u4f4d\u7f6e",
        quiet=True,
        now=1.16,
    )

    assert read_only_decision.action is FastEndpointAction.WAIT
