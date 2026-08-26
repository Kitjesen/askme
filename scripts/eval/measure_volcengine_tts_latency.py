"""Measure Volcengine TTS latency at the provider/client software boundary.

The command runs the fixed 20-case Chinese TTS corpus through the provider-only
Volcengine bidirectional WebSocket client.  It never opens an audio device or
plays the returned PCM.  Consequently, its measurements are useful for an
online provider comparison but are *not* physical first-sound measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from askme.config import get_config
from askme.voice.output.volcengine_tts_client import (
    VolcengineTTSClient,
    VolcengineTTSConfig,
)

SCHEMA_VERSION = "askme.voice_latency_experiment.v1"
CORPUS_PATH = Path(__file__).resolve().parent / "corpora" / "tts_zh_20_v1.json"
CORPUS_ID = "askme-tts-zh-20-v1"
EXPECTED_CASE_COUNT = 20
DEFAULT_ENDPOINT = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"

Clock = Callable[[], float]
Sleeper = Callable[[float], None]
ClientFactory = Callable[[Mapping[str, Any]], Any]


class MeasurementError(RuntimeError):
    """Expected collection failure whose message is safe to show to users."""


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    """Load and strictly validate the versioned fixed comparison corpus."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "askme.tts_latency_corpus.v1":
        raise MeasurementError("unsupported TTS corpus schema")
    if payload.get("corpus_id") != CORPUS_ID:
        raise MeasurementError("unexpected TTS corpus_id")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise MeasurementError("TTS corpus must contain exactly 20 cases")

    normalized: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for raw_case in cases:
        if not isinstance(raw_case, Mapping):
            raise MeasurementError("TTS corpus cases must be objects")
        case_id = str(raw_case.get("case_id") or "").strip()
        text = str(raw_case.get("text") or "").strip()
        if not case_id or case_id in case_ids:
            raise MeasurementError("TTS corpus must contain 20 distinct case_id values")
        if not text:
            raise MeasurementError(f"{case_id}: empty text")
        case_ids.add(case_id)
        raw_tags = raw_case.get("tags")
        tags = raw_tags if isinstance(raw_tags, list) else []
        normalized.append(
            {
                "case_id": case_id,
                "text": text,
                "tags": [str(tag) for tag in tags],
            }
        )
    return normalized


def _configured_value(
    config: Mapping[str, Any],
    key: str,
    env_name: str | None = None,
    *,
    default: Any = "",
) -> Any:
    value = config.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        value = os.environ.get(env_name, "") if env_name else default
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("${") and value.endswith("}"):
            return ""
    return value


def build_volcengine_settings(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve provider settings and reject incomplete evidence runs early."""

    root = dict(config or get_config(reload=True))
    voice = root.get("voice") if isinstance(root.get("voice"), Mapping) else {}
    raw_tts = voice.get("tts") if isinstance(voice, Mapping) else {}
    tts = dict(raw_tts) if isinstance(raw_tts, Mapping) else {}

    api_key = str(
        _configured_value(tts, "volcengine_tts_api_key", "VOLCENGINE_TTS_API_KEY")
        or ""
    )
    app_id = str(
        _configured_value(tts, "volcengine_tts_app_id", "VOLCENGINE_TTS_APP_ID")
        or ""
    )
    access_key = str(
        _configured_value(
            tts,
            "volcengine_tts_access_key",
            "VOLCENGINE_TTS_ACCESS_KEY",
        )
        or ""
    )
    resource_id = str(
        _configured_value(
            tts,
            "volcengine_tts_resource_id",
            "VOLCENGINE_TTS_RESOURCE_ID",
        )
        or ""
    )
    speaker = str(
        _configured_value(tts, "volcengine_tts_speaker", "VOLCENGINE_TTS_SPEAKER")
        or ""
    )

    if not api_key and not (app_id and access_key):
        raise MeasurementError(
            "Volcengine TTS credentials are missing; set VOLCENGINE_TTS_API_KEY "
            "or both VOLCENGINE_TTS_APP_ID and VOLCENGINE_TTS_ACCESS_KEY"
        )
    if not resource_id:
        raise MeasurementError(
            "Volcengine TTS resource is missing; set VOLCENGINE_TTS_RESOURCE_ID"
        )
    if not speaker:
        raise MeasurementError(
            "Volcengine TTS speaker is missing; set VOLCENGINE_TTS_SPEAKER"
        )

    audio_format = str(
        _configured_value(
            tts,
            "volcengine_tts_audio_format",
            default="pcm",
        )
        or "pcm"
    ).lower()
    if audio_format != "pcm":
        raise MeasurementError(
            "Volcengine latency collection requires voice.tts.volcengine_tts_audio_format=pcm"
        )
    try:
        sample_rate = int(
            _configured_value(
                tts,
                "volcengine_tts_sample_rate",
                default=24000,
            )
        )
        connect_timeout = float(
            _configured_value(
                tts,
                "volcengine_tts_connect_timeout_seconds",
                default=10.0,
            )
        )
        session_timeout = float(
            _configured_value(
                tts,
                "volcengine_tts_session_timeout_seconds",
                default=30.0,
            )
        )
    except (TypeError, ValueError) as exc:
        raise MeasurementError("Volcengine TTS numeric settings are invalid") from exc
    if sample_rate <= 0:
        raise MeasurementError("Volcengine TTS sample rate must be > 0")
    if connect_timeout <= 0 or session_timeout <= 0:
        raise MeasurementError("Volcengine TTS timeouts must be > 0")

    # In the checked-in YAML the model label intentionally mirrors the
    # account-specific resource-id environment variable.  If config loading is
    # bypassed and that placeholder is still unresolved, record the resource
    # header that actually selects the provider product instead of inventing a
    # default label that could contaminate an A/B report.
    model = str(
        _configured_value(
            tts,
            "volcengine_tts_model",
            default=resource_id,
        )
        or resource_id
    )

    return {
        "endpoint": str(
            _configured_value(
                tts,
                "volcengine_tts_ws_url",
                default=DEFAULT_ENDPOINT,
            )
            or DEFAULT_ENDPOINT
        ),
        "api_key": api_key,
        "app_id": app_id,
        "access_key": access_key,
        "resource_id": resource_id,
        "speaker": speaker,
        "model": model,
        "sample_rate": sample_rate,
        "audio_format": audio_format,
        "connect_timeout": connect_timeout,
        "session_timeout": session_timeout,
        "auth_mode": "api_key" if api_key else "legacy_app_access_key",
    }


def _default_client_factory(settings: Mapping[str, Any]) -> VolcengineTTSClient:
    """Adapt collector settings to the provider client in one isolated place."""

    config_kwargs: dict[str, Any] = {
        "endpoint": str(settings["endpoint"]),
        "api_key": str(settings["api_key"]),
        "app_id": str(settings["app_id"]),
        "access_key": str(settings["access_key"]),
        "resource_id": str(settings["resource_id"]),
        "speaker": str(settings["speaker"]),
        "sample_rate": int(settings["sample_rate"]),
        "audio_format": str(settings["audio_format"]),
    }
    parameters = inspect.signature(VolcengineTTSConfig).parameters
    if "connect_timeout" in parameters:
        config_kwargs["connect_timeout"] = float(settings["connect_timeout"])
    if "session_timeout" in parameters:
        config_kwargs["session_timeout"] = float(settings["session_timeout"])
    if "timeout" in parameters:
        # Compatibility with the original client, where one socket timeout
        # covered both the handshake and the synthesis receive loop.
        config_kwargs["timeout"] = float(settings["session_timeout"])
    config = VolcengineTTSConfig(**config_kwargs)
    return VolcengineTTSClient(config)


def _public_endpoint(value: str) -> str:
    """Strip credentials and query material before writing endpoint metadata."""

    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "redacted-invalid-endpoint"


def provider_metadata(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return only non-secret provider settings for the evidence artifact."""

    return {
        "provider": "volcengine",
        "model": str(settings["model"]),
        "transport": "bidirectional_websocket_v3",
        "endpoint": _public_endpoint(str(settings["endpoint"])),
        "resource_id": str(settings["resource_id"]),
        "speaker": str(settings["speaker"]),
        "provider_sample_rate_hz": int(settings["sample_rate"]),
        "audio_format": str(settings["audio_format"]),
        "auth_mode": str(settings["auth_mode"]),
        "connect_timeout_seconds": float(settings["connect_timeout"]),
        "session_timeout_seconds": float(settings["session_timeout"]),
    }


def _base_sample(case: Mapping[str, Any], *, mode: str, label: str) -> dict[str, Any]:
    text = str(case.get("text") or "")
    return {
        "case_id": str(case.get("case_id") or ""),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "char_count": len(text),
        "tags": list(case.get("tags") or []),
        "connection_mode": mode,
        "connection_label": label,
    }


def _result_status(result: Any) -> str:
    if isinstance(result, Mapping):
        value = result.get("status")
    else:
        value = getattr(result, "status", None)
    return str(value or "unknown").strip().lower()


def _round_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, value), 2)


def measure_case(
    *,
    client: Any,
    case: Mapping[str, Any],
    connection_mode: str,
    connection_label: str,
    clock: Clock = time.perf_counter,
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
    """Measure one provider synthesis, without opening a playback device."""

    started = clock()
    first_provider_pcm_at: float | None = None
    first_buffer_commit_at: float | None = None
    collector_buffer: list[bytes] = []

    def on_audio(chunk: bytes) -> None:
        nonlocal first_provider_pcm_at, first_buffer_commit_at
        payload = bytes(chunk)
        if not payload:
            return
        if first_provider_pcm_at is None:
            first_provider_pcm_at = clock()
        collector_buffer.append(payload)
        if first_buffer_commit_at is None:
            first_buffer_commit_at = clock()

    result: Any = None
    error: BaseException | None = None
    try:
        result = client.synthesize(str(case["text"]), on_audio=on_audio)
    except Exception as exc:  # provider failures are evidence, not collector crashes
        error = exc
    ended = clock()

    audio_bytes = sum(len(chunk) for chunk in collector_buffer)
    raw_provider_status = _result_status(result) if error is None else "exception"
    provider_status = _safe_error(
        MeasurementError(raw_provider_status),
        secrets=secrets,
    )[:80]
    accepted_statuses = {"finished", "completed", "success", "succeeded", "ok"}
    if error is None and raw_provider_status not in accepted_statuses:
        error = MeasurementError(
            f"Volcengine TTS synthesis ended with provider status: {raw_provider_status}"
        )
    if error is None and (
        audio_bytes <= 0
        or first_provider_pcm_at is None
        or first_buffer_commit_at is None
    ):
        error = MeasurementError("Volcengine TTS synthesis completed without PCM audio")

    sample = _base_sample(
        case,
        mode=connection_mode,
        label=connection_label,
    )
    sample.update(
        {
            "status": "failed" if error is not None else "passed",
            "provider_status": provider_status,
            "audio_chunks": len(collector_buffer),
            "audio_bytes": audio_bytes,
            "provider_first_pcm_ms": _round_ms(
                None
                if first_provider_pcm_at is None
                else (first_provider_pcm_at - started) * 1000.0
            ),
            "buffer_commit_ms": _round_ms(
                None
                if first_buffer_commit_at is None
                else (first_buffer_commit_at - started) * 1000.0
            ),
            "total_synthesis_ms": _round_ms((ended - started) * 1000.0),
        }
    )
    if error is not None:
        sample["error_type"] = error.__class__.__name__
        sample["error"] = _safe_error(error, secrets=secrets)
    return sample


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _prewarm_label(client: Any, *, secrets: Sequence[str] = ()) -> str:
    prewarm = getattr(client, "prewarm", None)
    if not callable(prewarm):
        raise MeasurementError("Volcengine TTS client does not expose prewarm()")
    result = prewarm()
    if not isinstance(result, Mapping):
        raise MeasurementError("Volcengine TTS prewarm returned an invalid result")
    status = str(result.get("status") or "unknown")
    if result.get("ok") is not True:
        reason = str(result.get("reason") or status)
        raise MeasurementError(f"Volcengine TTS warm preconnect failed: {reason}")
    safe_status = _safe_error(MeasurementError(status), secrets=secrets)[:80]
    return f"warm_{safe_status}"


def _failure_sample(
    *,
    case: Mapping[str, Any],
    mode: str,
    label: str,
    exc: BaseException,
    secrets: Sequence[str],
) -> dict[str, Any]:
    sample = _base_sample(case, mode=mode, label=label)
    sample.update(
        {
            "status": "failed",
            "provider_status": "not_started",
            "audio_chunks": 0,
            "audio_bytes": 0,
            "provider_first_pcm_ms": None,
            "buffer_commit_ms": None,
            "total_synthesis_ms": None,
            "error_type": exc.__class__.__name__,
            "error": _safe_error(exc, secrets=secrets),
        }
    )
    return sample


def run_measurement(
    *,
    mode: str,
    output_path: Path,
    config: Mapping[str, Any] | None = None,
    clock: Clock = time.perf_counter,
    sleeper: Sleeper = time.sleep,
    client_factory: ClientFactory = _default_client_factory,
    case_delay_ms: float = 0.0,
    overwrite: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Collect all 20 cases and durably write success or failure evidence."""

    if mode not in {"cold", "warm"}:
        raise MeasurementError("mode must be 'cold' or 'warm'")
    if case_delay_ms < 0:
        raise MeasurementError("case_delay_ms must be >= 0")
    if output_path.exists() and not overwrite:
        raise MeasurementError(
            f"output already exists: {output_path}; choose a new --out path or pass --overwrite"
        )

    cases = load_corpus()
    settings = build_volcengine_settings(config)
    secrets = tuple(
        str(settings.get(key) or "")
        for key in ("api_key", "app_id", "access_key")
    )
    samples: list[dict[str, Any]] = []
    shared_client: Any | None = None

    try:
        for index, case in enumerate(cases):
            client: Any | None = shared_client
            label = "cold_new_connection"
            try:
                if client is None:
                    client = client_factory(settings)
                    if mode == "warm":
                        shared_client = client
                if mode == "warm":
                    label = _prewarm_label(client, secrets=secrets)
                sample = measure_case(
                    client=client,
                    case=case,
                    connection_mode=mode,
                    connection_label=label,
                    clock=clock,
                    secrets=secrets,
                )
            except Exception as exc:
                sample = _failure_sample(
                    case=case,
                    mode=mode,
                    label="warm_failed" if mode == "warm" else label,
                    exc=exc,
                    secrets=secrets,
                )
            finally:
                if mode == "cold" and client is not None:
                    _close_client(client)
            samples.append(sample)
            if case_delay_ms > 0 and index < len(cases) - 1:
                sleeper(case_delay_ms / 1000.0)
    finally:
        if shared_client is not None:
            _close_client(shared_client)

    failures = sum(sample["status"] != "passed" for sample in samples)
    metadata = provider_metadata(settings)
    measured_at = datetime.now(UTC)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": (
            f"tts-volcengine-{mode}-{measured_at.strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:8]}"
        ),
        "stage": "tts",
        "provider": "volcengine",
        "model": metadata["model"],
        "transport": metadata["transport"],
        "evidence_type": "measured",
        "corpus_id": CORPUS_ID,
        "sample_count": len(samples),
        "measured_at": measured_at.isoformat(),
        "connection_mode": mode,
        "measurement_boundary": (
            "provider_audio_callback_and_collector_memory_buffer_commit_"
            "no_physical_playback"
        ),
        "case_delay_ms": round(float(case_delay_ms), 3),
        "notes": [
            "This command never opens sounddevice or starts physical speaker playback.",
            "provider_first_pcm_ms is recorded when non-empty PCM enters the provider callback.",
            "buffer_commit_ms is recorded after that PCM is appended to the collector memory buffer.",
            "Neither metric is physical first sound; target-hardware acoustic capture is still required.",
            "Warm mode reuses one provider client/connection; cold mode creates one client per case.",
            "Use --case-delay-ms when provider RPM limits would contaminate cold measurements.",
        ],
        "provider_metadata": metadata,
        "samples": samples,
        "status": "passed" if failures == 0 else "failed",
        "failure_count": failures,
    }
    atomic_write_json(output_path, payload, overwrite=overwrite)
    return payload, failures == 0


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Fsync and atomically publish evidence without accidental replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise MeasurementError(
            f"output already exists: {path}; choose a new --out path or pass --overwrite"
        )
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            tmp.replace(path)
        else:
            try:
                # The temporary file lives in the destination directory, so
                # this hard-link publication is atomic and fails if another
                # collector wins the final filename race.  Unlike replace(),
                # it can never clobber an evidence artifact.
                os.link(tmp, path)
            except FileExistsError as exc:
                raise MeasurementError(
                    f"output already exists: {path}; choose a new --out path "
                    "or pass --overwrite"
                ) from exc
            except OSError as exc:
                raise MeasurementError(
                    "filesystem cannot atomically publish no-clobber evidence; "
                    "choose a unique output path on a hard-link-capable filesystem"
                ) from exc
    finally:
        tmp.unlink(missing_ok=True)


_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(authorization|x-api-key|x-api-app-id|x-api-access-key|"
    r"api[_-]?key|app[_-]?id|access[_-]?(?:key|token)|token|secret)"
    r"\s*['\"]?\s*[:=]\s*['\"]?\s*(?:bearer\s+)?[^\s,'\";}\]]+"
)
_BEARER_VALUE_RE = re.compile(r"(?i)\bbearer\s+[^\s,'\";}\]]+")
_SENSITIVE_QUERY_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?(?:key|token)|token|secret)=)[^&#\s]+"
)


def _safe_error(exc: BaseException, *, secrets: Sequence[str] = ()) -> str:
    message = str(exc)
    for secret in sorted({value for value in secrets if value}, key=len, reverse=True):
        message = message.replace(secret, "[redacted]")
    message = _CREDENTIAL_ASSIGNMENT_RE.sub(r"\1=[redacted]", message)
    message = _BEARER_VALUE_RE.sub("Bearer [redacted]", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1[redacted]", message)
    return message[:300]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Volcengine online TTS provider/software-boundary latency on "
            "the fixed 20-case corpus. This is NOT physical first-sound latency "
            "and never starts speaker playback."
        )
    )
    parser.add_argument("--mode", choices=("cold", "warm"), required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path; defaults to a unique path under artifacts/voice/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing artifact (disabled by default).",
    )
    parser.add_argument(
        "--case-delay-ms",
        type=float,
        default=0.0,
        help=(
            "Sleep between corpus cases for provider RPM limits; excluded from "
            "per-case latency."
        ),
    )
    return parser


def default_output_path(mode: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return Path("artifacts") / "voice" / f"volcengine-tts-{mode}-{stamp}-{suffix}.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = args.out or default_output_path(args.mode)
    try:
        _, ok = run_measurement(
            mode=args.mode,
            output_path=output_path,
            case_delay_ms=float(args.case_delay_ms),
            overwrite=bool(args.overwrite),
        )
    except MeasurementError as exc:
        print(
            f"Volcengine TTS latency measurement failed: {_safe_error(exc)}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            f"Volcengine TTS latency measurement failed: {exc.__class__.__name__}",
            file=sys.stderr,
        )
        return 2
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
