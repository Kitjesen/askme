"""Voice profile API response contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VoiceProfileCatalogResponse(BaseModel):
    """Selectable voice profiles and local sound cues for customer-facing speech."""

    model_config = ConfigDict(extra="allow")

    profiles: list[dict[str, Any]] = Field(default_factory=list)
    active_profile: str = ""
    active_profile_settings: dict[str, Any] = Field(default_factory=dict)
    default_profile: str = ""
    available_sound_cues: list[str] = Field(default_factory=list)
    sound_cues_enabled: bool | None = None
    persistence_status: str = ""
    profile_count: int | None = None


class VoiceProfileUpdateResponse(BaseModel):
    """Result of changing the active TTS voice profile."""

    model_config = ConfigDict(extra="allow")

    updated: bool = False
    reason: str = ""
    requested_profile: str = ""
    resolved_profile: str = ""
    active_profile: str = ""
    profile: dict[str, Any] = Field(default_factory=dict)
    applied_settings: dict[str, Any] = Field(default_factory=dict)
    persistence_status: str = ""
    sound_cue: dict[str, Any] = Field(default_factory=dict)
    available: list[str] = Field(default_factory=list)


class VoiceSystemControlResponse(BaseModel):
    """Non-secret runtime state exposed to the voice-system console."""

    model_config = ConfigDict(extra="allow")

    status: str = "unknown"
    runtime: dict[str, Any] = Field(default_factory=dict)
    catalog: dict[str, Any] = Field(default_factory=dict)
    prompt: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    issues: list[dict[str, Any]] = Field(default_factory=list)


class VoiceSystemUpdateResponse(BaseModel):
    """Result of a live model, provider, or prompt update."""

    model_config = ConfigDict(extra="allow")

    updated: bool = False
    component: str = ""
    state: str = ""
    reason: str = ""
    runtime: dict[str, Any] = Field(default_factory=dict)


class VoiceSpeakRequest(BaseModel):
    """Literal text-to-robot speech; this contract never invokes an LLM."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=500)
    robot_id: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=128)
    site_id: str = Field(default="", max_length=128)
    semantics: str = "verbatim"
    priority: str = "normal"
    queue_policy: str = "enqueue"
    voice_profile_id: str = Field(default="", max_length=128)
    speed: float | None = Field(default=None, ge=0.75, le=1.5)
    pitch: float | None = Field(default=None, ge=-12.0, le=12.0)
    volume: float | None = Field(default=None, ge=0.05, le=1.0)
    ttl_s: float = Field(default=60.0, ge=1.0, le=300.0)


class VoicePlaybackCancelRequest(BaseModel):
    """Operator reason for cancelling one playback job."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="operator_cancelled", min_length=1, max_length=160)


class VoicePlaybackResponse(BaseModel):
    """Public lifecycle view for one speech playback job."""

    model_config = ConfigDict(extra="allow")

    playback_id: str
    state: str
    target: dict[str, Any]
    delivery: str = "playback"
    priority: str = "normal"
    text_chars: int = 0
    idempotency_key: str = ""
    timestamps: dict[str, Any] = Field(default_factory=dict)
    cache_hit: bool = False
    artifact: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    customer_message: str = ""


class VoiceLabDevicesResponse(BaseModel):
    """Audio inventory and evidence capabilities exposed by the operator Voice Lab."""

    model_config = ConfigDict(extra="allow")

    status: str = "unknown"
    platform: str = ""
    devices: list[dict[str, Any]] = Field(default_factory=list)
    hostapis: list[dict[str, Any]] = Field(default_factory=list)
    recommendation: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, bool] = Field(default_factory=dict)
    evidence_policy: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class VoiceLabTimelineEventResponse(BaseModel):
    """One privacy-safe, ordered runtime milestone."""

    model_config = ConfigDict(extra="forbid")

    event: str
    stage: str
    offset_ms: float
    sequence: int


class VoiceLabFallbackEvidenceResponse(BaseModel):
    """Server-observed fallback disposition for the executed attempt."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    used: bool
    from_: str = Field(alias="from")
    to: str
    reason: str


class VoiceLabInterruptEvidenceResponse(BaseModel):
    """Server-observed interrupt lifecycle for the executed attempt."""

    model_config = ConfigDict(extra="forbid")

    detected: bool
    confirmed: bool
    dismissed: bool
    playback_resumed: bool


class VoiceLabAecStatsResponse(BaseModel):
    """Algorithm telemetry that cannot by itself prove physical echo control."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    active: bool
    degraded: bool
    erl_db: float | None = None
    erle_db: float | None = None
    residual_echo_likelihood: float | None = None
    evidence_kind: Literal["algorithm_telemetry"]


class VoiceLabResidualAudioResponse(BaseModel):
    """Optional bounded physical residual measurement metadata (never raw audio)."""

    model_config = ConfigDict(extra="forbid")

    evidence_kind: Literal["physical"]
    measurement_source: str
    clock_domain: str
    dropped_frames: int
    tail_ms: float


class VoiceLabTurnEvidenceResponse(BaseModel):
    """Sanitized server-owned evidence persisted for one active attempt."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    source: Literal["server_runtime"]
    captured_at: str
    timeline: list[VoiceLabTimelineEventResponse] = Field(default_factory=list)
    fallback: VoiceLabFallbackEvidenceResponse
    interrupt: VoiceLabInterruptEvidenceResponse
    configured_full_duplex: bool
    runtime_full_duplex: bool
    echo_control_evidence: dict[str, Any] = Field(default_factory=dict)
    aec_stats: VoiceLabAecStatsResponse
    residual_audio: VoiceLabResidualAudioResponse | None = None


class VoiceLabActiveTrialResponse(BaseModel):
    """The one server-issued attempt that may execute or submit next."""

    model_config = ConfigDict(extra="allow")

    attempt_id: str
    scenario: str
    ordinal: int
    started_at: str
    turn_evidence: VoiceLabTurnEvidenceResponse | None = None


class VoiceLabTrialResponse(BaseModel):
    """Completed operator trial, optionally carrying trusted execution evidence."""

    model_config = ConfigDict(extra="allow")

    trial_id: str
    attempt_id: str
    scenario: str
    ordinal: int
    turn_evidence: VoiceLabTurnEvidenceResponse | None = None
    product_gate_usable: bool = False


class VoiceLabRunResponse(BaseModel):
    """Public state of one versioned, target-hardware Voice Lab run."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = ""
    hardware_report_schema_version: str = ""
    run_id: str
    version: int
    status: str
    operator_id: str = ""
    room: str = ""
    no_ros2: bool = True
    device_binding: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, int] = Field(default_factory=dict)
    capabilities: dict[str, bool] = Field(default_factory=dict)
    product_gate_possible: bool = False
    product_gate_blocked_reasons: list[str] = Field(default_factory=list)
    device_check: dict[str, Any] = Field(default_factory=dict)
    calibration: dict[str, Any] = Field(default_factory=dict)
    trials: list[VoiceLabTrialResponse] = Field(default_factory=list)
    active_trial: VoiceLabActiveTrialResponse | None = None
    invalidated_trials: list[dict[str, Any]] = Field(default_factory=list)
    manual_diagnostic_complete: bool = False
    product_gate: dict[str, Any] = Field(default_factory=dict)
    progress: dict[str, Any] = Field(default_factory=dict)
    next_action: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    aborted_at: str | None = None
    completed_at: str | None = None
