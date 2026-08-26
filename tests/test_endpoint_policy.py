from __future__ import annotations

from askme.voice.input.endpoint_policy import EndpointMode, EndpointPolicy


def test_shadow_short_command_reports_without_committing() -> None:
    policy = EndpointPolicy(mode=EndpointMode.SHADOW)

    policy.observe(partial_text="请让一下", quiet=False, confidence=0.96, now=1.0)
    policy.observe(partial_text="请让一下", quiet=True, confidence=0.96, now=1.1)
    decision = policy.observe(
        partial_text="请让一下",
        quiet=True,
        confidence=0.96,
        now=1.51,
    )

    assert decision.would_commit is True
    assert decision.should_commit is False
    assert decision.reason == "short_command_ready"
    assert decision.required_silence_ms == 400.0
    assert decision.observed_silence_ms >= 400.0


def test_active_mode_exposes_commit_signal_after_same_policy_gate() -> None:
    policy = EndpointPolicy(mode="active", stable_partial_ms=0)

    policy.observe(partial_text="停止", quiet=True, confidence=0.9, now=2.0)
    decision = policy.observe(partial_text="停止", quiet=True, confidence=0.9, now=2.41)

    assert decision.would_commit is True
    assert decision.should_commit is True


def test_off_mode_never_reports_or_commits_a_ready_endpoint() -> None:
    policy = EndpointPolicy(mode="off", stable_partial_ms=0)

    policy.observe(partial_text="停止", quiet=True, confidence=0.9, now=3.0)
    decision = policy.observe(partial_text="停止", quiet=True, confidence=0.9, now=3.5)

    assert decision.would_commit is False
    assert decision.should_commit is False
    assert decision.reason == "mode_off"


def test_punctuation_and_complete_sentence_ending_use_complete_threshold() -> None:
    punctuated = EndpointPolicy(stable_partial_ms=0)
    punctuated.observe(partial_text="A区卫生间在一楼。", quiet=True, confidence=0.92, now=4.0)
    punctuated_decision = punctuated.observe(
        partial_text="A区卫生间在一楼。", quiet=True, confidence=0.92, now=4.51
    )

    assert punctuated_decision.would_commit is True
    assert punctuated_decision.reason == "complete_utterance_ready"
    assert punctuated_decision.required_silence_ms == 500.0

    unpunctuated = EndpointPolicy(stable_partial_ms=0)
    unpunctuated.observe(partial_text="A区卫生间在一楼", quiet=True, confidence=0.92, now=5.0)
    unpunctuated_decision = unpunctuated.observe(
        partial_text="A区卫生间在一楼", quiet=True, confidence=0.92, now=5.51
    )

    assert unpunctuated_decision.would_commit is True
    assert unpunctuated_decision.reason == "complete_utterance_ready"
    assert unpunctuated_decision.required_silence_ms == 500.0


def test_mid_pause_after_unfinished_conjunction_does_not_end_turn_early() -> None:
    policy = EndpointPolicy(stable_partial_ms=0)

    policy.observe(partial_text="我想去大堂然后", quiet=True, confidence=0.88, now=6.0)
    decision = policy.observe(partial_text="我想去大堂然后", quiet=True, confidence=0.88, now=6.61)

    assert decision.would_commit is False
    assert decision.should_commit is False
    assert decision.reason == "unfinished_clause_waiting"
    assert decision.required_silence_ms == 1000.0


def test_hesitation_uses_longer_timeout_before_shadow_recommendation() -> None:
    policy = EndpointPolicy(stable_partial_ms=0)

    policy.observe(partial_text="嗯，我想一下", quiet=True, confidence=0.83, now=7.0)
    waiting = policy.observe(partial_text="嗯，我想一下", quiet=True, confidence=0.83, now=8.1)
    ready = policy.observe(partial_text="嗯，我想一下", quiet=True, confidence=0.83, now=8.21)

    assert waiting.would_commit is False
    assert waiting.reason == "hesitation_waiting"
    assert waiting.required_silence_ms == 1200.0
    assert ready.would_commit is True
    assert ready.should_commit is False
    assert ready.reason == "hesitation_timeout"


def test_noise_and_low_confidence_never_create_endpoint_candidate() -> None:
    policy = EndpointPolicy(stable_partial_ms=0)

    empty = policy.observe(partial_text="  ", quiet=True, confidence=0.99, now=9.0)
    policy.observe(partial_text="停止", quiet=True, confidence=0.31, now=9.1)
    low_confidence = policy.observe(partial_text="停止", quiet=True, confidence=0.31, now=14.1)

    assert empty.reason == "empty_partial"
    assert empty.would_commit is False
    assert low_confidence.reason == "low_confidence"
    assert low_confidence.would_commit is False
    assert low_confidence.observed_silence_ms == 0.0


def test_resumed_speech_cancels_candidate_and_restarts_timers() -> None:
    policy = EndpointPolicy(stable_partial_ms=180)

    policy.observe(partial_text="停止", quiet=True, confidence=0.95, now=15.0)
    policy.observe(partial_text="停止", quiet=True, confidence=0.95, now=15.3)
    cancelled = policy.observe(partial_text="停止", quiet=False, confidence=0.95, now=15.35)
    policy.observe(partial_text="停止", quiet=True, confidence=0.95, now=15.4)
    too_soon = policy.observe(partial_text="停止", quiet=True, confidence=0.95, now=15.75)
    ready = policy.observe(partial_text="停止", quiet=True, confidence=0.95, now=15.81)

    assert cancelled.cancelled is True
    assert cancelled.reason == "speech_resumed"
    assert too_soon.would_commit is False
    assert too_soon.observed_silence_ms < 400.0
    assert ready.would_commit is True


def test_partial_change_restarts_silence_and_stability_windows() -> None:
    policy = EndpointPolicy(stable_partial_ms=300)

    policy.observe(partial_text="停", quiet=True, confidence=0.95, now=16.0)
    policy.observe(partial_text="停止", quiet=True, confidence=0.95, now=16.3)
    too_soon = policy.observe(partial_text="停止", quiet=True, confidence=0.95, now=16.61)
    ready = policy.observe(partial_text="停止", quiet=True, confidence=0.95, now=16.71)

    assert too_soon.would_commit is False
    assert too_soon.reason in {"partial_not_stable", "short_command_waiting"}
    assert ready.would_commit is True
