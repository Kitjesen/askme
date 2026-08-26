"""Static browser-contract checks for the operator-only Voice Lab."""

import re
from pathlib import Path

VOICE_LAB_SCRIPT = Path("askme/static/dashboard/voice-lab.js")


def test_voice_lab_browser_uses_server_started_attempts() -> None:
    source = VOICE_LAB_SCRIPT.read_text(encoding="utf-8")

    assert 'mutate("trials/begin"' in source
    assert "state.run.active_trial?.attempt_id" in source
    assert "attempt_id: attemptId" in source
    assert "服务端 attempt_id" in source


def test_voice_lab_executes_the_active_attempt_through_the_server_evidence_api() -> None:
    source = VOICE_LAB_SCRIPT.read_text(encoding="utf-8")

    assert "/api/runtime/voice-turn" not in source
    assert "state.run.active_trial?.attempt_id" in source
    call = re.search(
        r"api\(executePath,\s*\{(?P<options>.*?)\}\)",
        source,
        flags=re.DOTALL,
    )
    assert call is not None
    options = call.group("options")
    assert 'method: "POST"' in options
    assert "version: state.run.version" in options
    assert "key: trialExecutionKey(attemptId)" in options
    assert "body:" not in options
    assert "trials/${encodeURIComponent(attemptId)}/execute" in source


def test_voice_lab_renders_server_owned_evidence_without_promoting_algorithm_telemetry() -> None:
    source = VOICE_LAB_SCRIPT.read_text(encoding="utf-8")

    assert "function renderTurnEvidence" in source
    assert "turn_evidence" in source
    assert "timeline.slice().sort" in source
    assert "fallback.used" in source
    assert "interrupt.dismissed" in source
    assert "interrupt.playback_resumed" in source
    assert "aec.evidence_kind" in source
    assert "evidence.residual_audio" in source
    assert "productGateUsable === true" in source
    assert "算法 AEC 遥测不等于物理门禁" in source
    assert "不可用于产品门禁" in source


def test_voice_lab_keeps_an_unexecuted_attempt_retryable_when_evidence_is_unavailable() -> None:
    source = VOICE_LAB_SCRIPT.read_text(encoding="utf-8")

    assert "trialExecutionFailures: new Map()" in source
    assert "error.status === 503" in source
    assert "服务端证据不可用（503）" in source
    assert "active attempt 已保留" in source
    assert "重试服务端证据" in source

    submit_body = source.split("async function submitTrial", maxsplit=1)[1].split(
        "function renderPaused", maxsplit=1
    )[0]
    assert "window.confirm" in submit_body
    assert "结束这个 active attempt" in submit_body
    assert "已保留，可继续重试服务端证据" in submit_body
    assert "turn_evidence" not in submit_body
    assert "aec_stats" not in submit_body
    assert "product_gate_usable" not in submit_body
