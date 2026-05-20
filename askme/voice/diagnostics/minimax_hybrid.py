"""Readiness model for the domestic MiniMax hybrid voice-brain route.

This module intentionally performs no network or hardware calls.  It describes
and validates the recommended low-latency domestic stack:

    realtime ASR -> MiniMax LLM/tool planning -> askme runtime handoff
    -> MiniMax streaming TTS

The safety invariant is explicit: experimental end-to-end S2S providers may
improve conversational feel, but they must not submit robot control directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

_MINIMAX_MODEL_PREFIX = "minimax"
_MINIMAX_BASE_HINTS = ("minimax", "hailuo")
_SUPPORTED_ASR_PROVIDERS = {
    "dashscope_paraformer",
    "qwen_asr_realtime",
    "tencent_asr_realtime",
}
_DIRECT_CONTROL_RE = re.compile(
    r"\b("
    r"drive_motor_direct|gait_command|motor_command|payload_fire|robot_api|"
    r"dog_control|dog-control|dog_safety|dog-safety|nav_gateway|nav-gateway|"
    r"hardware_dispatch|bypass_safety|ignore_safety"
    r")\b",
    flags=re.IGNORECASE,
)
_PAUSE_RE = re.compile(r"(pause|hold|stop for now|先停|停一下|暂停|等等|等一下)", re.IGNORECASE)
_RESUME_RE = re.compile(r"(resume|continue|go on|继续|恢复)", re.IGNORECASE)
_CANCEL_RE = re.compile(r"(cancel|abort|stop task|取消|终止|别执行|算了)", re.IGNORECASE)
_STATUS_RE = re.compile(r"(status|progress|where are|状态|进度|到哪|执行到哪)", re.IGNORECASE)


@dataclass(frozen=True)
class MinimaxHybridVoiceBrain:
    """Resolved provider choices for the MiniMax hybrid voice brain."""

    enabled: bool
    provider: str = "legacy"
    mode: str = "cascade"
    asr_provider: str = "dashscope_paraformer"
    llm_provider: str = "minimax_m27_highspeed"
    tts_provider: str = "minimax_speech_28_turbo"
    realtime_s2s_provider: str = ""
    realtime_s2s_enabled: bool = False
    task_handoff_required: bool = True
    safety_preflight_required: bool = True
    runtime_arbiter_required: bool = True
    safety_bypass_allowed: bool = False
    hardware_dispatch_allowed: bool = False
    s2s_task_control_allowed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MinimaxVoiceBrainIngress:
    """One normalized transcript ingress from the MiniMax hybrid path."""

    text: str
    source: str
    provider: str = "minimax_hybrid"
    route: str = "chat_transcript"
    transcript_id: str = ""
    confidence: float | None = None
    is_final: bool = True
    experimental_realtime_s2s: bool = False
    runtime_control_intent: str | None = None
    runtime_submit_allowed: bool = False
    task_state_mutation_allowed: bool = False
    hardware_dispatch: bool = False
    safety_bypass_allowed: bool = False
    rejected: bool = False
    rejection_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_minimax_hybrid_voice_brain(
    config: dict[str, Any] | None,
) -> MinimaxHybridVoiceBrain:
    """Resolve the ``voice_brain`` config section into a stable plan.

    ``voice_brain`` is top-level on purpose: it describes the cross-cutting
    ASR/LLM/TTS route, while ``voice`` keeps owning device and audio settings.
    """

    cfg = config or {}
    raw = cfg.get("voice_brain", {}) or {}
    provider = str(raw.get("provider", raw.get("name", "legacy"))).strip() or "legacy"
    enabled = bool(raw.get("enabled", provider == "minimax_hybrid"))

    if not enabled:
        return MinimaxHybridVoiceBrain(enabled=False, provider=provider)

    mode = str(raw.get("mode", "cascade")).strip() or "cascade"
    return MinimaxHybridVoiceBrain(
        enabled=True,
        provider=provider,
        mode=mode,
        asr_provider=str(raw.get("asr_provider", "dashscope_paraformer")).strip()
        or "dashscope_paraformer",
        llm_provider=str(raw.get("llm_provider", "minimax_m27_highspeed")).strip()
        or "minimax_m27_highspeed",
        tts_provider=str(raw.get("tts_provider", "minimax_speech_28_turbo")).strip()
        or "minimax_speech_28_turbo",
        realtime_s2s_provider=str(raw.get("realtime_s2s_provider", "")).strip(),
        realtime_s2s_enabled=bool(
            raw.get("realtime_s2s_enabled", mode == "s2s_experiment")
        ),
        task_handoff_required=bool(raw.get("task_handoff_required", True)),
        safety_preflight_required=bool(raw.get("safety_preflight_required", True)),
        runtime_arbiter_required=bool(raw.get("runtime_arbiter_required", True)),
        safety_bypass_allowed=bool(raw.get("safety_bypass_allowed", False)),
        hardware_dispatch_allowed=bool(raw.get("hardware_dispatch_allowed", False)),
        s2s_task_control_allowed=bool(raw.get("s2s_task_control_allowed", False)),
        metadata={
            "domestic_low_latency": True,
            "robot_control_boundary": "task_handoff_safety_preflight_runtime_arbiter",
            "experimental_realtime_s2s": bool(
                raw.get("realtime_s2s_enabled", mode == "s2s_experiment")
            ),
        },
    )


def build_minimax_voice_brain_ingress(
    text: str,
    *,
    source: str = "asr_final",
    transcript_id: str = "",
    confidence: float | None = None,
    is_final: bool = True,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize MiniMax ASR/S2S output into a safe askme ingress payload.

    The returned payload never authorizes runtime submission or hardware
    dispatch.  Task-changing text must still pass through cognition planning,
    operator confirmation, SafetyPreflight, and the runtime arbiter.
    """

    plan = resolve_minimax_hybrid_voice_brain(config)
    recognized = str(text or "").strip()
    src = str(source or "asr_final").strip() or "asr_final"
    experimental_s2s = src in {"s2s", "s2s_transcript", "realtime_s2s"} or bool(
        plan.realtime_s2s_enabled
    )
    runtime_intent = _runtime_control_intent(recognized)
    rejected, reason = _direct_control_rejection(recognized)
    route = "runtime_voice_turn" if runtime_intent else "chat_transcript"
    if rejected:
        route = "rejected_transcript"

    return MinimaxVoiceBrainIngress(
        text=recognized,
        source=src,
        provider=plan.provider if plan.enabled else "minimax_hybrid",
        route=route,
        transcript_id=str(transcript_id or ""),
        confidence=confidence,
        is_final=bool(is_final),
        experimental_realtime_s2s=experimental_s2s,
        runtime_control_intent=runtime_intent,
        runtime_submit_allowed=False,
        task_state_mutation_allowed=False,
        hardware_dispatch=False,
        safety_bypass_allowed=False,
        rejected=rejected,
        rejection_reason=reason,
        metadata={
            "task_handoff_required": True,
            "safety_preflight_required": True,
            "runtime_arbiter_required": True,
            "s2s_task_control_allowed": False,
            "embedded_tool_json_rejected": reason == "embedded_tool_json",
            "voice_trace": {
                "voice_turn_id": str(transcript_id or ""),
                "asr_provider": plan.asr_provider,
                "asr_transcript_id": str(transcript_id or ""),
                "asr_final_confidence": confidence,
                "planner_provider": "minimax",
                "planner_model": "MiniMax-M2.7-highspeed",
                "tts_provider": plan.tts_provider,
                "experimental_realtime_s2s": experimental_s2s,
                "latency": {
                    "asr_first_partial_ms": None,
                    "asr_final_ms": None,
                    "planner_ms": None,
                    "preflight_ms": None,
                    "tts_first_audio_ms": None,
                },
            },
        },
    ).to_dict()


def check_minimax_hybrid_voice_brain(
    config: dict[str, Any] | None,
    *,
    deps: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Return a no-network readiness report for the MiniMax hybrid route."""

    cfg = config or {}
    dep_snapshot = deps or {}
    plan = resolve_minimax_hybrid_voice_brain(cfg)
    errors: list[str] = []
    warnings: list[str] = []

    if not plan.enabled:
        return {
            "status": "skipped",
            "ok": True,
            "enabled": False,
            "provider": plan.provider,
            "plan": plan.to_dict(),
            "checks": {},
            "errors": [],
            "warnings": [],
        }

    if plan.provider != "minimax_hybrid":
        errors.append(
            f"voice_brain.provider must be 'minimax_hybrid' when enabled, got {plan.provider!r}"
        )

    brain_cfg = cfg.get("brain", {}) or {}
    voice_cfg = cfg.get("voice", {}) or {}
    cloud_asr_cfg = voice_cfg.get("cloud_asr", {}) or {}
    tts_cfg = voice_cfg.get("tts", {}) or {}

    asr_check = _check_realtime_asr(plan, cloud_asr_cfg, dep_snapshot)
    llm_check = _check_minimax_llm(brain_cfg)
    tts_check = _check_minimax_tts(tts_cfg)
    safety_check = _check_runtime_safety(plan)

    for check in (asr_check, llm_check, tts_check, safety_check):
        errors.extend(check.get("errors", []))
        warnings.extend(check.get("warnings", []))

    ok = not errors
    return {
        "status": "ok" if ok else "degraded",
        "ok": ok,
        "enabled": True,
        "provider": plan.provider,
        "plan": plan.to_dict(),
        "checks": {
            "asr": asr_check,
            "llm": llm_check,
            "tts": tts_check,
            "runtime_safety": safety_check,
        },
        "errors": errors,
        "warnings": warnings,
    }


def _check_realtime_asr(
    plan: MinimaxHybridVoiceBrain,
    cloud_asr_cfg: dict[str, Any],
    deps: dict[str, bool],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if plan.asr_provider not in _SUPPORTED_ASR_PROVIDERS:
        warnings.append(
            f"voice_brain.asr_provider={plan.asr_provider!r} is not a known domestic realtime ASR provider"
        )
    enabled = bool(cloud_asr_cfg.get("enabled", False))
    api_key = str(cloud_asr_cfg.get("api_key", "")).strip()
    websocket_ok = bool(deps.get("websocket_client", False))
    if not enabled:
        errors.append("voice.cloud_asr.enabled must be true for minimax_hybrid")
    if not api_key:
        errors.append("voice.cloud_asr.api_key is empty")
    if not websocket_ok:
        errors.append("Cloud ASR dependency missing: websocket-client")
    return {
        "ok": not errors,
        "provider": plan.asr_provider,
        "cloud_asr_enabled": enabled,
        "api_key_configured": bool(api_key),
        "websocket_client": websocket_ok,
        "errors": errors,
        "warnings": warnings,
    }


def _check_minimax_llm(brain_cfg: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    api_key = str(brain_cfg.get("api_key", "")).strip()
    model = str(brain_cfg.get("model", "")).strip()
    base_url = str(brain_cfg.get("base_url", "")).strip()
    if not api_key:
        errors.append("brain.api_key is empty for MiniMax LLM")
    if not model:
        errors.append("brain.model is empty")
    elif not model.lower().startswith(_MINIMAX_MODEL_PREFIX):
        warnings.append(
            f"brain.model={model!r} is not a MiniMax model; latency/provider assumptions may not hold"
        )
    if not base_url:
        errors.append("brain.base_url is empty")
    elif not any(hint in base_url.lower() for hint in _MINIMAX_BASE_HINTS):
        warnings.append(
            f"brain.base_url={base_url!r} does not look like a MiniMax endpoint"
        )
    return {
        "ok": not errors,
        "api_key_configured": bool(api_key),
        "model": model,
        "base_url": base_url,
        "errors": errors,
        "warnings": warnings,
    }


def _check_minimax_tts(tts_cfg: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    backend = str(tts_cfg.get("backend", "")).strip().lower()
    api_key = str(tts_cfg.get("minimax_api_key", "")).strip()
    model = str(tts_cfg.get("minimax_tts_model", "")).strip()
    if backend != "minimax":
        errors.append("voice.tts.backend must be 'minimax' for minimax_hybrid")
    if not api_key:
        errors.append("voice.tts.minimax_api_key is empty")
    if model and not model.lower().startswith("speech-"):
        warnings.append(
            f"voice.tts.minimax_tts_model={model!r} does not look like a MiniMax Speech model"
        )
    return {
        "ok": not errors,
        "backend": backend,
        "api_key_configured": bool(api_key),
        "model": model or "speech-2.8-hd",
        "errors": errors,
        "warnings": warnings,
    }


def _check_runtime_safety(plan: MinimaxHybridVoiceBrain) -> dict[str, Any]:
    errors: list[str] = []
    if not plan.task_handoff_required:
        errors.append("voice_brain.task_handoff_required must stay true")
    if not plan.safety_preflight_required:
        errors.append("voice_brain.safety_preflight_required must stay true")
    if not plan.runtime_arbiter_required:
        errors.append("voice_brain.runtime_arbiter_required must stay true")
    if plan.safety_bypass_allowed:
        errors.append("voice_brain.safety_bypass_allowed must stay false")
    if plan.hardware_dispatch_allowed:
        errors.append("voice_brain.hardware_dispatch_allowed must stay false in askme")
    if plan.realtime_s2s_enabled and plan.s2s_task_control_allowed:
        errors.append(
            "MiniMax realtime/S2S experiment must not control robot tasks directly; "
            "route tasks through TaskHandoff and SafetyPreflight"
        )
    return {
        "ok": not errors,
        "task_handoff_required": plan.task_handoff_required,
        "safety_preflight_required": plan.safety_preflight_required,
        "runtime_arbiter_required": plan.runtime_arbiter_required,
        "safety_bypass_allowed": plan.safety_bypass_allowed,
        "hardware_dispatch_allowed": plan.hardware_dispatch_allowed,
        "realtime_s2s_enabled": plan.realtime_s2s_enabled,
        "s2s_task_control_allowed": plan.s2s_task_control_allowed,
        "errors": errors,
        "warnings": [],
    }


def _runtime_control_intent(text: str) -> str | None:
    if not text:
        return None
    if _CANCEL_RE.search(text):
        return "cancel"
    if _PAUSE_RE.search(text):
        return "pause"
    if _RESUME_RE.search(text):
        return "resume"
    if _STATUS_RE.search(text):
        return "status"
    return None


def _direct_control_rejection(text: str) -> tuple[bool, str]:
    if not text:
        return False, ""
    if _contains_embedded_tool_json(text):
        return True, "embedded_tool_json"
    if _DIRECT_CONTROL_RE.search(text):
        return True, "direct_control_reference"
    return False, ""


def _contains_embedded_tool_json(text: str) -> bool:
    if "{" not in text or "}" not in text:
        return False
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return False
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return bool(re.search(r'["\']?(tool|cmd|command)["\']?\s*:', text, re.IGNORECASE))
    if not isinstance(payload, dict):
        return False
    keys = {str(key).lower() for key in payload}
    values = {str(value).lower() for value in payload.values() if isinstance(value, str)}
    suspicious_keys = {"tool", "cmd", "command", "function", "action"}
    suspicious_values = {
        "robot_api",
        "move",
        "drive_motor_direct",
        "gait_command",
        "dog_control",
        "hardware_dispatch",
    }
    return bool(keys & suspicious_keys and values & suspicious_values)
