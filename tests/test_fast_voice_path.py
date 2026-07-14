from __future__ import annotations

from askme.voice.input.fast_endpoint import FastEndpointAction, FastEndpointController
from askme.voice.interaction import (
    FastVoiceIntentKind,
    match_fast_voice_intent,
)

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
