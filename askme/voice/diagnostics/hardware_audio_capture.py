"""Pure helpers for target-hardware acoustic latency capture.

This module intentionally does not open microphone streams, start playback, or
write evidence files.  It provides the side-effect-free pieces needed by a
future semi-automatic 20+20+20 hardware runner:

* lazy PortAudio/sounddevice inventory normalization;
* noise-floor calibration from already captured float32 PCM frames;
* streaming onset/offset detection for microphone, WASAPI loopback, or manual
  capture sources.

Evidence boundaries are deliberately explicit: ``microphone`` is physical
acoustic evidence from the room, while ``wasapi_loopback`` is render-chain
evidence from the operating system.  They must not be mixed in one metric or
reported as interchangeable physical first-sound proof.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

CaptureSource = Literal["microphone", "wasapi_loopback", "manual"]
EvidenceKind = Literal["physical_acoustic", "render_chain", "manual"]
InstrumentedEvidenceKind = Literal["physical_acoustic", "render_chain"]

EVIDENCE_KINDS = frozenset({"physical_acoustic", "render_chain", "manual"})

SOURCE_EVIDENCE_KIND: dict[CaptureSource, EvidenceKind] = {
    "microphone": "physical_acoustic",
    "wasapi_loopback": "render_chain",
    "manual": "manual",
}


class HardwareAudioCaptureError(RuntimeError):
    """Controlled diagnostics error safe to show in readiness output."""


def build_instrumented_trial_evidence(
    *,
    evidence_kind: InstrumentedEvidenceKind,
    method: str,
    capture: Mapping[str, Any],
    reference: Mapping[str, Any],
    reference_timestamp_s: float,
    event_timestamp_s: float | None,
    calibration: Calibration | Mapping[str, Any],
    dropped_frames: int = 0,
    clock_id: str,
) -> dict[str, Any]:
    """Build the explicit provenance envelope required by hardware schema v2.

    This helper only serializes capture facts; it does not decide whether a
    capture is suitable for a particular product gate.  In particular, a room
    microphone is physical acoustic input but is not an isolated speaker-stop
    sensor while a person is talking over the speaker.
    """

    if evidence_kind not in {"physical_acoustic", "render_chain"}:
        raise ValueError("instrumented evidence_kind must be physical_acoustic or render_chain")
    normalized_method = str(method).strip()
    if not normalized_method or normalized_method.lower() in {
        "entry",
        "manual",
        "stopwatch",
    }:
        raise ValueError("instrumented method must identify an automatic capture method")
    normalized_clock_id = str(clock_id).strip()
    if not normalized_clock_id:
        raise ValueError("clock_id is required")
    reference_s = _validate_timestamp(reference_timestamp_s, previous=None)
    event_s = (
        None
        if event_timestamp_s is None
        else _validate_timestamp(event_timestamp_s, previous=reference_s)
    )
    if isinstance(dropped_frames, bool) or not isinstance(dropped_frames, int):
        raise ValueError("dropped_frames must be a non-negative integer")
    if dropped_frames < 0:
        raise ValueError("dropped_frames must be a non-negative integer")

    calibration_payload = (
        calibration.to_dict() if isinstance(calibration, Calibration) else dict(calibration)
    )
    return {
        "evidence_kind": evidence_kind,
        "method": normalized_method,
        "capture": dict(capture),
        "reference": dict(reference),
        "monotonic_timestamps": {
            "clock_id": normalized_clock_id,
            "reference_s": reference_s,
            "event_s": event_s,
        },
        "calibration": calibration_payload,
        "dropped_frames": dropped_frames,
    }


def build_manual_trial_evidence(
    *,
    method: str,
    reference_event: str,
    reference_timestamp_s: float | None = None,
    event_timestamp_s: float | None = None,
    observed_timestamp_s: float | None = None,
) -> dict[str, Any]:
    """Describe operator evidence without ever labelling it instrumented."""

    normalized_method = str(method).strip().lower()
    if normalized_method not in {
        "manual_entry",
        "manual_observation",
        "manual_stopwatch",
    }:
        raise ValueError(
            "manual method must be manual_entry, manual_observation, or manual_stopwatch"
        )
    timestamps: dict[str, float | str | None] = {
        "clock_id": "operator",
        "reference_s": _optional_timestamp(reference_timestamp_s),
        "event_s": _optional_timestamp(event_timestamp_s),
        "observed_s": _optional_timestamp(observed_timestamp_s),
    }
    return {
        "evidence_kind": "manual",
        "method": normalized_method,
        "capture": {"source": "operator_observation", "instrumented": False},
        "reference": {
            "event": str(reference_event).strip(),
            "source": "operator_action",
            "instrumented": False,
        },
        "monotonic_timestamps": timestamps,
        "calibration": {"performed": False},
        "dropped_frames": None,
    }


@dataclass(frozen=True)
class DeviceInfo:
    """Serializable PortAudio device snapshot."""

    index: int
    name: str
    hostapi: int | None
    hostapi_name: str | None
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float | None
    is_input: bool
    is_output: bool
    is_default_input: bool = False
    is_default_output: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Calibration:
    """Noise-floor calibration derived from finite PCM samples.

    ``source_evidence_kind`` prevents accidental evidence mixing: microphone
    calibration applies to physical acoustic measurements; WASAPI loopback
    calibration applies only to render-chain measurements.
    """

    source_label: CaptureSource
    source_evidence_kind: EvidenceKind
    sample_rate_hz: int
    frame_count: int
    valid_frame_count: int
    rms_p50: float
    rms_p95: float
    rms_p99: float
    margin_db: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {"performed": True, **asdict(self)}


@dataclass(frozen=True)
class OnsetResult:
    """Current streaming onset/offset detector state.

    ``onset_timestamp`` is backdated to the first loud frame in the confirmed
    run.  ``onset_confirm_timestamp`` is when the detector had enough
    consecutive frames to accept the onset.  ``source_evidence_kind`` carries
    the source boundary so physical microphone evidence is never conflated with
    WASAPI render-chain evidence.
    """

    source_label: CaptureSource
    source_evidence_kind: EvidenceKind
    threshold: float
    onset_timestamp: float | None = None
    onset_confirm_timestamp: float | None = None
    offset_timestamp: float | None = None
    onset_frame_index: int | None = None
    onset_confirm_frame_index: int | None = None
    offset_frame_index: int | None = None
    above_threshold_frames: int = 0
    below_threshold_frames: int = 0
    processed_frames: int = 0
    active: bool = False

    @property
    def detected(self) -> bool:
        return self.onset_timestamp is not None

    @property
    def stopped(self) -> bool:
        return self.offset_timestamp is not None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["detected"] = self.detected
        payload["stopped"] = self.stopped
        return payload


def list_audio_devices(sounddevice_module: Any | None = None) -> dict[str, Any]:
    """Return serializable input/output device inventory without opening streams."""

    sd = sounddevice_module if sounddevice_module is not None else _load_sounddevice()
    try:
        raw_devices = list(sd.query_devices())
        raw_hostapis = list(sd.query_hostapis())
    except Exception as exc:  # pragma: no cover - depends on PortAudio host
        raise HardwareAudioCaptureError(
            f"sounddevice device query failed: {exc.__class__.__name__}"
        ) from exc

    hostapis = [_clean_hostapi(api, idx) for idx, api in enumerate(raw_hostapis)]
    hostapi_names = {
        int(api.get("index", idx)): str(api.get("name") or "") for idx, api in enumerate(hostapis)
    }
    default_input, default_output = _default_device_pair(getattr(sd, "default", None))
    devices = [
        _clean_device(
            device,
            idx,
            hostapi_names=hostapi_names,
            default_input=default_input,
            default_output=default_output,
        ).to_dict()
        for idx, device in enumerate(raw_devices)
    ]
    return {
        "status": "ok",
        "devices": devices,
        "hostapis": hostapis,
        "default_input_device": default_input,
        "default_output_device": default_output,
    }


def calibrate_noise_floor(
    frames: Iterable[Any],
    *,
    sample_rate_hz: int,
    source_label: CaptureSource = "microphone",
    percentile: float = 95.0,
    margin_db: float = 12.0,
    minimum_threshold: float = 1e-5,
) -> Calibration:
    """Compute an RMS noise threshold from already captured PCM frames."""

    _validate_source_label(source_label)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    if minimum_threshold < 0:
        raise ValueError("minimum_threshold must be >= 0")

    rms_values: list[float] = []
    frame_count = 0
    for frame in frames:
        frame_count += 1
        rms = frame_rms(frame)
        if rms is not None:
            rms_values.append(rms)
    if not rms_values:
        raise HardwareAudioCaptureError("noise calibration requires finite non-empty PCM frames")

    rms_array = np.asarray(rms_values, dtype=np.float64)
    p50 = float(np.percentile(rms_array, 50))
    p95 = float(np.percentile(rms_array, 95))
    p99 = float(np.percentile(rms_array, 99))
    base = float(np.percentile(rms_array, percentile))
    threshold = max(float(minimum_threshold), base * _db_to_ratio(margin_db))
    return Calibration(
        source_label=source_label,
        source_evidence_kind=SOURCE_EVIDENCE_KIND[source_label],
        sample_rate_hz=int(sample_rate_hz),
        frame_count=frame_count,
        valid_frame_count=len(rms_values),
        rms_p50=round(p50, 8),
        rms_p95=round(p95, 8),
        rms_p99=round(p99, 8),
        margin_db=float(margin_db),
        threshold=round(threshold, 8),
    )


def frame_rms(frame: Any) -> float | None:
    """Return finite RMS for a PCM frame, or None for empty/NaN-only frames."""

    samples = np.asarray(frame, dtype=np.float32)
    if samples.size == 0:
        return None
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(finite, dtype=np.float64))))


class StreamingOnsetDetector:
    """Consecutive-frame onset/offset detector for pre-captured PCM chunks."""

    def __init__(
        self,
        *,
        threshold: float,
        source_label: CaptureSource = "microphone",
        consecutive_on_frames: int = 2,
        consecutive_off_frames: int = 3,
        hangover_frames: int = 0,
    ) -> None:
        _validate_source_label(source_label)
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        if consecutive_on_frames <= 0:
            raise ValueError("consecutive_on_frames must be > 0")
        if consecutive_off_frames <= 0:
            raise ValueError("consecutive_off_frames must be > 0")
        if hangover_frames < 0:
            raise ValueError("hangover_frames must be >= 0")
        self._threshold = float(threshold)
        self._source_label = source_label
        self._consecutive_on_frames = int(consecutive_on_frames)
        self._consecutive_off_frames = int(consecutive_off_frames)
        self._hangover_frames = int(hangover_frames)
        self._onset_timestamp: float | None = None
        self._onset_confirm_timestamp: float | None = None
        self._offset_timestamp: float | None = None
        self._onset_frame_index: int | None = None
        self._onset_confirm_frame_index: int | None = None
        self._offset_frame_index: int | None = None
        self._candidate_onset_timestamp: float | None = None
        self._candidate_onset_frame_index: int | None = None
        self._last_timestamp: float | None = None
        self._above_frames = 0
        self._below_frames = 0
        self._processed_frames = 0
        self._active = False

    def process_frame(self, *, timestamp: float, pcm: Any) -> OnsetResult:
        """Process one timestamped PCM frame and return the current state."""

        timestamp = _validate_timestamp(timestamp, previous=self._last_timestamp)
        self._last_timestamp = timestamp
        rms = frame_rms(pcm)
        is_loud = rms is not None and rms >= self._threshold
        frame_index = self._processed_frames
        self._processed_frames += 1

        if is_loud:
            if self._above_frames == 0:
                self._candidate_onset_timestamp = timestamp
                self._candidate_onset_frame_index = frame_index
            self._above_frames += 1
            self._below_frames = 0
            if self._onset_timestamp is None and self._above_frames >= self._consecutive_on_frames:
                self._onset_timestamp = self._candidate_onset_timestamp
                self._onset_confirm_timestamp = timestamp
                self._onset_frame_index = self._candidate_onset_frame_index
                self._onset_confirm_frame_index = frame_index
                self._active = True
        else:
            self._above_frames = 0
            self._candidate_onset_timestamp = None
            self._candidate_onset_frame_index = None
            if self._active:
                self._below_frames += 1
                required = self._consecutive_off_frames + self._hangover_frames
                if self._below_frames >= required and self._offset_timestamp is None:
                    self._offset_timestamp = float(timestamp)
                    self._offset_frame_index = frame_index
                    self._active = False

        return self.result()

    def result(self) -> OnsetResult:
        return OnsetResult(
            source_label=self._source_label,
            source_evidence_kind=SOURCE_EVIDENCE_KIND[self._source_label],
            threshold=self._threshold,
            onset_timestamp=self._onset_timestamp,
            onset_confirm_timestamp=self._onset_confirm_timestamp,
            offset_timestamp=self._offset_timestamp,
            onset_frame_index=self._onset_frame_index,
            onset_confirm_frame_index=self._onset_confirm_frame_index,
            offset_frame_index=self._offset_frame_index,
            above_threshold_frames=self._above_frames,
            below_threshold_frames=self._below_frames,
            processed_frames=self._processed_frames,
            active=self._active,
        )


def _load_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:  # pragma: no cover - host dependent
        raise HardwareAudioCaptureError(
            "sounddevice is not installed; install sounddevice to enumerate audio devices"
        ) from exc
    return sd


def _clean_device(
    device: Any,
    index: int,
    *,
    hostapi_names: Mapping[int, str],
    default_input: int | None,
    default_output: int | None,
) -> DeviceInfo:
    row = dict(device)
    hostapi = _optional_int(row.get("hostapi"))
    max_input = max(0, int(row.get("max_input_channels") or 0))
    max_output = max(0, int(row.get("max_output_channels") or 0))
    return DeviceInfo(
        index=index,
        name=str(row.get("name") or f"device-{index}"),
        hostapi=hostapi,
        hostapi_name=hostapi_names.get(hostapi) if hostapi is not None else None,
        max_input_channels=max_input,
        max_output_channels=max_output,
        default_samplerate=_optional_float(row.get("default_samplerate")),
        is_input=max_input > 0,
        is_output=max_output > 0,
        is_default_input=index == default_input,
        is_default_output=index == default_output,
    )


def _clean_hostapi(hostapi: Any, index: int) -> dict[str, Any]:
    row = dict(hostapi)
    return {
        "index": index,
        "name": str(row.get("name") or f"hostapi-{index}"),
        "device_count": _optional_int(row.get("device_count")),
        "default_input_device": _optional_int(row.get("default_input_device")),
        "default_output_device": _optional_int(row.get("default_output_device")),
    }


def _default_device_pair(default: Any) -> tuple[int | None, int | None]:
    value = getattr(default, "device", None)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _optional_int(value[0]), _optional_int(value[1])
    return None, None


def _optional_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _db_to_ratio(db: float) -> float:
    return float(10 ** (float(db) / 20.0))


def _validate_source_label(source_label: str) -> None:
    if source_label not in {"microphone", "wasapi_loopback", "manual"}:
        raise ValueError("source_label must be microphone, wasapi_loopback, or manual")


def _validate_timestamp(timestamp: float, *, previous: float | None) -> float:
    try:
        value = float(timestamp)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be a finite float") from exc
    if not np.isfinite(value):
        raise ValueError("timestamp must be finite")
    if previous is not None and value < previous:
        raise ValueError("timestamp must be non-decreasing")
    return value


def _optional_timestamp(timestamp: float | None) -> float | None:
    if timestamp is None:
        return None
    return _validate_timestamp(timestamp, previous=None)
