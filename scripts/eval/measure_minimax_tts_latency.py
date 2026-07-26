"""Measure MiniMax TTS provider latency without physical playback.

This command synthesizes the fixed 20-case Chinese TTS corpus through the
MiniMax adapter and observes when decoded provider PCM first reaches the client
and when the adapter first commits playback-buffer PCM.  It intentionally does
not start sounddevice playback, so it is not a physical first-sound benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
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

from askme.config import get_config
from askme.voice.output.tts import TTSEngine

SCHEMA_VERSION = "askme.voice_latency_experiment.v1"
CORPUS_PATH = Path(__file__).resolve().parent / "corpora" / "tts_zh_20_v1.json"
CORPUS_ID = "askme-tts-zh-20-v1"
EXPECTED_CASE_COUNT = 20

Clock = Callable[[], float]


class MeasurementError(RuntimeError):
    """Expected measurement failure that should not expose credentials."""


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "askme.tts_latency_corpus.v1":
        raise MeasurementError("unsupported TTS corpus schema")
    if payload.get("corpus_id") != CORPUS_ID:
        raise MeasurementError("unexpected TTS corpus_id")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise MeasurementError("TTS corpus must contain exactly 20 cases")
    case_ids = [str(case.get("case_id") or "").strip() for case in cases]
    if len(set(case_ids)) != EXPECTED_CASE_COUNT or any(not case_id for case_id in case_ids):
        raise MeasurementError("TTS corpus must contain 20 distinct case_id values")
    normalized: list[dict[str, Any]] = []
    for case in cases:
        text = str(case.get("text") or "").strip()
        if not text:
            raise MeasurementError(f"{case.get('case_id')}: empty text")
        tags = case.get("tags") if isinstance(case.get("tags"), list) else []
        normalized.append(
            {
                "case_id": str(case["case_id"]),
                "text": text,
                "tags": [str(tag) for tag in tags],
            }
        )
    return normalized


def build_minimax_tts_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = dict(config or get_config(reload=True))
    voice = root.get("voice") if isinstance(root.get("voice"), Mapping) else {}
    raw_tts = voice.get("tts") if isinstance(voice, Mapping) else {}
    tts = dict(raw_tts) if isinstance(raw_tts, Mapping) else {}
    tts["backend"] = "minimax"
    tts["phrase_cache_enabled"] = False
    api_key = str(tts.get("minimax_api_key") or os.environ.get("MINIMAX_API_KEY") or "").strip()
    if not api_key:
        raise MeasurementError(
            "MiniMax TTS credentials are missing; set MINIMAX_API_KEY or voice.tts.minimax_api_key"
        )
    tts["minimax_api_key"] = api_key
    return tts


def provider_metadata(engine: Any) -> dict[str, Any]:
    snapshot_fn = getattr(engine, "snapshot", None)
    snapshot = snapshot_fn() if callable(snapshot_fn) else {}
    minimax = snapshot.get("minimax") if isinstance(snapshot, Mapping) else {}
    return {
        "provider": "minimax",
        "model": str(
            getattr(engine, "_minimax_tts_model", None)
            or (minimax or {}).get("model")
            or "unknown"
        ),
        "transport": str(
            getattr(engine, "_minimax_tts_transport", None)
            or (minimax or {}).get("transport")
            or "unknown"
        ),
        "voice_id": str(
            getattr(engine, "_minimax_voice_id", None)
            or (minimax or {}).get("voice_id")
            or "unknown"
        ),
        "sample_rate_hz": int(
            getattr(engine, "_sample_rate", None)
            or snapshot.get("sample_rate", 0)
            or 0
        ),
        "provider_sample_rate_hz": int(
            getattr(engine, "_minimax_sample_rate", None)
            or (minimax or {}).get("sample_rate", 0)
            or 0
        ),
        "audio_format": str(
            getattr(engine, "_minimax_audio_format", None)
            or (minimax or {}).get("format")
            or "unknown"
        ),
    }


def _has_audio_samples(samples: Any) -> bool:
    if samples is None:
        return False
    try:
        return len(samples) > 0
    except TypeError:
        return True


def _buffered_samples(engine: Any) -> int:
    buffer = getattr(engine, "tts_buffer", None)
    if buffer is None:
        return 0
    total = 0
    for chunk in list(buffer):
        try:
            total += len(chunk)
        except TypeError:
            total += 1
    return total


def _clear_buffer(engine: Any) -> None:
    buffer = getattr(engine, "tts_buffer", None)
    if buffer is not None and hasattr(buffer, "clear"):
        buffer.clear()


def _round_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, value), 2)


def measure_case(
    *,
    engine: Any,
    case: Mapping[str, Any],
    connection_mode: str,
    connection_label: str,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Measure one synthesis without calling any playback API."""

    started = clock()
    first_provider_pcm_at: float | None = None
    first_buffer_commit_at: float | None = None
    original_commit = getattr(engine, "_commit_minimax_samples_for_generation")
    original_flush = getattr(engine, "_flush_minimax_pending")
    original_fallback = getattr(engine, "_use_minimax_fallback", None)

    def commit_probe(*args: Any, **kwargs: Any) -> Any:
        nonlocal first_provider_pcm_at
        if first_provider_pcm_at is None and _has_audio_samples(kwargs.get("samples")):
            first_provider_pcm_at = clock()
        return original_commit(*args, **kwargs)

    def flush_probe(*args: Any, **kwargs: Any) -> Any:
        nonlocal first_buffer_commit_at
        result = original_flush(*args, **kwargs)
        if first_buffer_commit_at is None and _buffered_samples(engine) > 0:
            first_buffer_commit_at = clock()
        return result

    def fallback_probe(*args: Any, **kwargs: Any) -> str:
        raise MeasurementError("MiniMax provider failed before measurable PCM; fallback suppressed")

    setattr(engine, "_commit_minimax_samples_for_generation", commit_probe)
    setattr(engine, "_flush_minimax_pending", flush_probe)
    if callable(original_fallback):
        setattr(engine, "_use_minimax_fallback", fallback_probe)
    try:
        generation = engine._get_generation()
        generated_backend = engine._generate_audio(str(case["text"]), generation)
        ended = clock()
        if generated_backend != "minimax":
            raise MeasurementError(f"MiniMax synthesis did not complete via MiniMax backend: {generated_backend}")
        provider_first_pcm_ms = _round_ms(
            None if first_provider_pcm_at is None else (first_provider_pcm_at - started) * 1000.0
        )
        buffer_commit_ms = _round_ms(
            None if first_buffer_commit_at is None else (first_buffer_commit_at - started) * 1000.0
        )
        if provider_first_pcm_ms is None or buffer_commit_ms is None:
            raise MeasurementError("MiniMax synthesis completed without observable PCM commit")
        return {
            "case_id": str(case["case_id"]),
            "text_sha256": hashlib.sha256(str(case["text"]).encode("utf-8")).hexdigest(),
            "char_count": len(str(case["text"])),
            "tags": list(case.get("tags") or []),
            "connection_mode": connection_mode,
            "connection_label": connection_label,
            "status": "passed",
            "provider_first_pcm_ms": provider_first_pcm_ms,
            "buffer_commit_ms": buffer_commit_ms,
            "total_synthesis_ms": _round_ms((ended - started) * 1000.0),
        }
    finally:
        setattr(engine, "_commit_minimax_samples_for_generation", original_commit)
        setattr(engine, "_flush_minimax_pending", original_flush)
        if callable(original_fallback):
            setattr(engine, "_use_minimax_fallback", original_fallback)
        _clear_buffer(engine)


def _shutdown_engine(engine: Any) -> None:
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()


def _prewarm_label(engine: Any) -> str:
    prewarm = getattr(engine, "prewarm_provider_session", None)
    if not callable(prewarm):
        raise MeasurementError("TTSEngine does not expose prewarm_provider_session()")
    result = prewarm()
    status = str(result.get("status") or "unknown")
    if result.get("ok") is not True:
        reason = str(result.get("reason") or status or "unknown")
        raise MeasurementError(f"MiniMax warm preconnect failed: {reason}")
    return f"warm_{status}"


def run_measurement(
    *,
    mode: str,
    output_path: Path,
    config: Mapping[str, Any] | None = None,
    clock: Clock = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    engine_factory: Callable[[dict[str, Any]], Any] = TTSEngine,
    case_delay_ms: float = 0.0,
    overwrite: bool = False,
) -> tuple[dict[str, Any], bool]:
    if mode not in {"cold", "warm"}:
        raise MeasurementError("mode must be 'cold' or 'warm'")
    if case_delay_ms < 0:
        raise MeasurementError("case_delay_ms must be >= 0")
    cases = load_corpus()
    tts_config = build_minimax_tts_config(config)
    samples: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    failures = 0
    shared_engine: Any | None = None

    try:
        if mode == "warm":
            shared_engine = engine_factory(dict(tts_config))
            metadata = provider_metadata(shared_engine)
        for index, case in enumerate(cases):
            engine = shared_engine if shared_engine is not None else engine_factory(dict(tts_config))
            try:
                metadata = metadata or provider_metadata(engine)
                connection_label = "cold_new_session"
                if mode == "warm":
                    connection_label = _prewarm_label(engine)
                sample = measure_case(
                    engine=engine,
                    case=case,
                    connection_mode=mode,
                    connection_label=connection_label,
                    clock=clock,
                )
                samples.append(sample)
            except Exception as exc:
                failures += 1
                samples.append(
                    {
                        "case_id": str(case.get("case_id") or ""),
                        "text_sha256": hashlib.sha256(str(case.get("text") or "").encode("utf-8")).hexdigest(),
                        "char_count": len(str(case.get("text") or "")),
                        "tags": list(case.get("tags") or []),
                        "connection_mode": mode,
                        "connection_label": "warm_failed" if mode == "warm" else "cold_new_session",
                        "status": "failed",
                        "error_type": exc.__class__.__name__,
                        "error": _safe_error(
                            exc,
                            secrets=(str(tts_config.get("minimax_api_key") or ""),),
                        ),
                    }
                )
            finally:
                if shared_engine is None:
                    _shutdown_engine(engine)
                if case_delay_ms > 0 and index < len(cases) - 1:
                    sleeper(case_delay_ms / 1000.0)
    finally:
        if shared_engine is not None:
            _shutdown_engine(shared_engine)

    if metadata is None:
        metadata = {
            "provider": "minimax",
            "model": str(tts_config.get("minimax_tts_model") or "unknown"),
            "transport": str(tts_config.get("minimax_tts_transport") or "unknown"),
            "voice_id": str(tts_config.get("minimax_voice_id") or "unknown"),
            "sample_rate_hz": int(tts_config.get("sample_rate") or 0),
            "provider_sample_rate_hz": int(tts_config.get("minimax_sample_rate") or 0),
            "audio_format": str(tts_config.get("minimax_audio_format") or "unknown"),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": f"tts-minimax-{mode}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "stage": "tts",
        "provider": "minimax",
        "model": metadata["model"],
        "transport": metadata["transport"],
        "evidence_type": "measured",
        "corpus_id": CORPUS_ID,
        "sample_count": len(samples),
        "measured_at": datetime.now(UTC).isoformat(),
        "connection_mode": mode,
        "measurement_boundary": "client_decoded_provider_pcm_and_tts_buffer_commit_no_physical_playback",
        "case_delay_ms": round(float(case_delay_ms), 3),
        "notes": [
            "This command never starts sounddevice or physical speaker playback.",
            "provider_first_pcm_ms is observed when provider audio first decodes in the client.",
            "buffer_commit_ms is observed when the adapter first appends PCM to tts_buffer.",
            "physical_first_nonzero_ms must be measured separately on target hardware.",
            "Use --case-delay-ms when provider RPM limits would otherwise contaminate cold-connection measurements.",
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
    """Durably replace an artifact, refusing accidental evidence loss by default."""

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
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|token|secret)"
    r"\s*['\"]?\s*[:=]\s*['\"]?\s*(?:bearer\s+)?[^\s,'\";}\]]+"
)
_BEARER_VALUE_RE = re.compile(r"(?i)\bbearer\s+[^\s,'\";}\]]+")


def _safe_error(
    exc: BaseException,
    *,
    secrets: Sequence[str] = (),
) -> str:
    message = str(exc)
    for secret in sorted({value for value in secrets if value}, key=len, reverse=True):
        message = message.replace(secret, "[redacted]")
    message = _CREDENTIAL_ASSIGNMENT_RE.sub(r"\1=[redacted]", message)
    message = _BEARER_VALUE_RE.sub("Bearer [redacted]", message)
    return message[:300]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure MiniMax online TTS provider latency on the fixed 20-case corpus. "
            "This is NOT physical first-sound latency and never starts speaker playback."
        )
    )
    parser.add_argument("--mode", choices=("cold", "warm"), required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to a unique timestamped path under "
            "artifacts/voice/."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing output artifact (disabled by default).",
    )
    parser.add_argument(
        "--case-delay-ms",
        type=float,
        default=0.0,
        help=(
            "Sleep between corpus cases to respect provider RPM limits. "
            "This delay is not included in per-case latency measurements."
        ),
    )
    return parser


def default_output_path(mode: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return Path("artifacts") / "voice" / f"minimax-tts-{mode}-{stamp}-{suffix}.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = args.out or default_output_path(args.mode)
    try:
        _, ok = run_measurement(
            mode=args.mode,
            output_path=out,
            case_delay_ms=float(args.case_delay_ms),
            overwrite=bool(args.overwrite),
        )
    except MeasurementError as exc:
        print(f"MiniMax TTS latency measurement failed: {_safe_error(exc)}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"MiniMax TTS latency measurement failed: {exc.__class__.__name__}",
            file=sys.stderr,
        )
        return 2
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
