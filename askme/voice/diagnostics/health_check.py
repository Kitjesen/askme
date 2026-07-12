"""Offline health checks for the askme voice pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

from askme.config import get_config, project_root

logger = logging.getLogger(__name__)

from askme.voice.diagnostics.minimax_hybrid import check_minimax_hybrid_voice_brain

_ASR_DEFAULT_DIR = "models/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
_KWS_DEFAULT_DIR = "models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
_TTS_DEFAULT_DIR = "models/tts/vits-melo-tts-zh_en"
_VAD_DEFAULT_MODEL = "models/vad/silero_vad.onnx"

_ASR_ENCODERS = (
    "encoder.int8.onnx",
    "encoder-epoch-99-avg-1.int8.onnx",
    "encoder.onnx",
    "encoder-epoch-99-avg-1.onnx",
)
_ASR_DECODERS = (
    "decoder.onnx",
    "decoder-epoch-99-avg-1.int8.onnx",
    "decoder-epoch-99-avg-1.onnx",
)
_ASR_JOINERS = (
    "joiner.int8.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
    "joiner.onnx",
    "joiner-epoch-99-avg-1.onnx",
)

_KWS_ENCODERS = (
    "encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
    "encoder-epoch-99-avg-1-chunk-16-left-64.onnx",
)
_KWS_DECODERS = (
    "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
    "decoder-epoch-99-avg-1-chunk-16-left-64.onnx",
)
_KWS_JOINERS = (
    "joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
    "joiner-epoch-99-avg-1-chunk-16-left-64.onnx",
)

_TTS_MODELS = ("model.onnx", "model.int8.onnx", "vits-aishell3.onnx", "vits-aishell3.int8.onnx")
_HEALTH_SNAPSHOT_KEYS = (
    "mode",
    "enabled",
    "input_ready",
    "output_ready",
    "pipeline_ok",
    "asr_available",
    "vad_available",
    "kws_available",
    "wake_word_enabled",
    "woken_up",
    "muted",
    "tts_backend",
    "tts_busy",
    "agent_state",
)


def run_voice_health(
    config: dict[str, Any] | None = None,
    *,
    live: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    """Return an offline health report for voice config, models, and imports."""
    cfg = get_config(reload=True) if config is None else config
    project = Path(root or cfg.get("_project_root") or project_root()).resolve()
    voice_cfg = cfg.get("voice", {})
    errors: list[str] = []
    warnings: list[str] = []

    config_ok = isinstance(voice_cfg, dict) and bool(voice_cfg)
    if not config_ok:
        errors.append("voice config section is missing or empty")
        voice_cfg = {}

    deps = {
        "sherpa_onnx": _dependency_available("sherpa_onnx"),
        "edge_tts": _dependency_available("edge_tts"),
        "sounddevice": _dependency_available("sounddevice"),
        "websocket_client": _websocket_client_available(),
    }

    asr = _check_asr(voice_cfg.get("asr", {}), project, deps)
    vad = _check_vad(voice_cfg.get("vad", {}), project, deps)
    kws = _check_kws(voice_cfg.get("kws", {}), project, deps)
    tts = _check_tts(voice_cfg.get("tts", {}), project, deps)
    bridge = _check_runtime_bridge(cfg.get("runtime", {}).get("voice_bridge", {}))
    audio = _check_audio_devices(voice_cfg, deps)
    voice_brain = check_minimax_hybrid_voice_brain(cfg, deps=deps)

    for check in (asr, vad, kws, tts, bridge, audio, voice_brain):
        errors.extend(check.get("errors", []))
        warnings.extend(check.get("warnings", []))

    health_snapshot = _build_health_snapshot(asr, vad, kws, tts, audio)
    health_snapshot_ok = all(key in health_snapshot for key in _HEALTH_SNAPSHOT_KEYS)
    if not health_snapshot_ok:
        errors.append("offline voice health snapshot is missing required keys")

    models_ok = bool(asr["ok"] and vad["ok"] and kws["ok"] and tts["model_ok"])
    runtime_bridge_ok = bool(bridge["ok"])
    voice_brain_ok = bool(voice_brain.get("ok", True))
    voice_ok = bool(
        config_ok
        and models_ok
        and tts["ok"]
        and runtime_bridge_ok
        and voice_brain_ok
        and health_snapshot_ok
    )

    if live:
        warnings.append(
            "live mode requested: hardware/cloud exercise is intentionally not run by this offline check"
        )

    return {
        "status": "ok" if voice_ok else "degraded",
        "config_ok": config_ok,
        "models_ok": models_ok,
        "asr_ok": bool(asr["ok"]),
        "vad_ok": bool(vad["ok"]),
        "kws_ok": bool(kws["ok"]),
        "tts_ok": bool(tts["ok"]),
        "runtime_bridge_ok": runtime_bridge_ok,
        "voice_brain_ok": voice_brain_ok,
        "health_snapshot_ok": health_snapshot_ok,
        "hardware_required": bool(live),
        "live_requested": bool(live),
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "dependencies": deps,
            "asr": asr,
            "vad": vad,
            "kws": kws,
            "tts": tts,
            "runtime_bridge": bridge,
            "audio": audio,
            "voice_brain": voice_brain,
        },
        "health_snapshot": health_snapshot,
    }


def print_voice_health_summary(payload: dict[str, Any]) -> None:
    """Print a compact human-readable health summary."""
    logger.info(f"voice-health: {payload.get('status', 'unknown')}")
    logger.info(f"  config: {_label(payload.get('config_ok'))}")
    logger.info(f"  models: {_label(payload.get('models_ok'))}")
    logger.info(f"  asr: {_label(payload.get('asr_ok'))}")
    logger.info(f"  vad: {_label(payload.get('vad_ok'))}")
    logger.info(f"  kws: {_label(payload.get('kws_ok'))}")
    logger.info(f"  tts: {_label(payload.get('tts_ok'))}")
    logger.info(f"  runtime_bridge: {_label(payload.get('runtime_bridge_ok'))}")
    logger.info(f"  voice_brain: {_label(payload.get('voice_brain_ok', True))}")
    logger.info(f"  health_snapshot: {_label(payload.get('health_snapshot_ok'))}")
    for warning in payload.get("warnings", []):
        logger.warning(f"  warn: {warning}")
    for error in payload.get("errors", []):
        logger.error(f"  error: {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m askme.voice.health_check",
        description="Offline askme voice health check",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Mark the check as a live preflight; no hardware/cloud calls are made",
    )
    args = parser.parse_args(argv)
    payload = run_voice_health(live=args.live)
    if args.json:
        logger.info(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_voice_health_summary(payload)
    return 0 if payload["status"] == "ok" else 1


def _check_asr(cfg: dict[str, Any], root: Path, deps: dict[str, bool]) -> dict[str, Any]:
    model_dir = _resolve_path(root, cfg.get("model_dir", _ASR_DEFAULT_DIR))
    paths = {
        "model_dir": str(model_dir),
        "tokens": str(model_dir / cfg.get("tokens", "tokens.txt")),
        "encoder": str(_find_model_file(model_dir, cfg.get("encoder"), _ASR_ENCODERS)),
        "decoder": str(_find_model_file(model_dir, cfg.get("decoder"), _ASR_DECODERS)),
        "joiner": str(_find_model_file(model_dir, cfg.get("joiner"), _ASR_JOINERS)),
    }
    missing = _missing_paths(paths)
    errors = [f"ASR missing {name}: {path}" for name, path in missing.items()]
    if not deps["sherpa_onnx"]:
        errors.append("ASR dependency missing: sherpa_onnx")
    return {
        "ok": not errors,
        "dependency_ok": deps["sherpa_onnx"],
        "paths": paths,
        "missing": missing,
        "errors": errors,
        "warnings": [],
    }


def _check_vad(cfg: dict[str, Any], root: Path, deps: dict[str, bool]) -> dict[str, Any]:
    configured_key = "model" if cfg.get("model") else "model_path" if cfg.get("model_path") else "default"
    model = _resolve_path(root, cfg.get("model") or cfg.get("model_path") or _VAD_DEFAULT_MODEL)
    missing = {} if model.is_file() else {"model": str(model)}
    errors = [f"VAD missing {name}: {path}" for name, path in missing.items()]
    warnings: list[str] = []
    if configured_key == "model_path":
        warnings.append("voice.vad.model_path is accepted by voice-health; VADEngine also supports model")
    if not deps["sherpa_onnx"]:
        errors.append("VAD dependency missing: sherpa_onnx")
    return {
        "ok": not errors,
        "dependency_ok": deps["sherpa_onnx"],
        "configured_key": configured_key,
        "paths": {"model": str(model)},
        "missing": missing,
        "errors": errors,
        "warnings": warnings,
    }


def _check_kws(cfg: dict[str, Any], root: Path, deps: dict[str, bool]) -> dict[str, Any]:
    from askme.voice.input.kws import normalize_keyword_line, validate_keyword_lines

    keywords = [str(keyword).strip() for keyword in cfg.get("keywords", []) if str(keyword).strip()]
    enabled = bool(keywords)
    model_dir = _resolve_path(root, cfg.get("model_dir", _KWS_DEFAULT_DIR))
    paths = {
        "model_dir": str(model_dir),
        "tokens": str(model_dir / cfg.get("tokens", "tokens.txt")),
        "encoder": str(_find_model_file(model_dir, cfg.get("encoder"), _KWS_ENCODERS)),
        "decoder": str(_find_model_file(model_dir, cfg.get("decoder"), _KWS_DECODERS)),
        "joiner": str(_find_model_file(model_dir, cfg.get("joiner"), _KWS_JOINERS)),
        "keywords_file": str(model_dir / cfg.get("keywords_file", "keywords.txt")),
    }
    missing = _missing_paths(paths)
    errors: list[str] = []
    warnings: list[str] = []
    if enabled:
        errors = [f"KWS missing {name}: {path}" for name, path in missing.items()]
        if not deps["sherpa_onnx"]:
            errors.append("KWS dependency missing: sherpa_onnx")
        if "tokens" not in missing:
            normalized = [normalize_keyword_line(keyword) for keyword in keywords]
            errors.extend(
                f"KWS keyword configuration invalid: {error}"
                for error in validate_keyword_lines(normalized, paths["tokens"])
            )
    else:
        warnings.append("KWS is disabled because voice.kws.keywords is empty")
    return {
        "ok": not errors,
        "enabled": enabled,
        "dependency_ok": deps["sherpa_onnx"],
        "paths": paths,
        "missing": missing if enabled else {},
        "errors": errors,
        "warnings": warnings,
    }


def _check_tts(cfg: dict[str, Any], root: Path, deps: dict[str, bool]) -> dict[str, Any]:
    requested_backend = str(cfg.get("backend", "local")).strip() or "local"
    model_dir = _resolve_path(root, cfg.get("model_dir", _TTS_DEFAULT_DIR))
    model = _find_model_file(model_dir, cfg.get("model"), _TTS_MODELS)
    local_missing = _missing_paths(
        {
            "model_dir": str(model_dir),
            "model": str(model),
            "tokens": str(model_dir / cfg.get("tokens", "tokens.txt")),
        }
    )
    local_model_ok = not local_missing and deps["sherpa_onnx"]
    minimax_key = str(cfg.get("minimax_api_key", "")).strip()
    fallback_backend = str(cfg.get("fallback_backend", "edge")).strip().lower()
    if fallback_backend not in {"local", "edge"}:
        fallback_backend = "edge"
    backend = requested_backend
    warnings: list[str] = []
    errors: list[str] = []

    if backend == "minimax" and not minimax_key:
        backend = "local" if fallback_backend == "local" and local_model_ok else "edge"
        warnings.append(
            f"MiniMax TTS API key is empty; runtime will fall back to {backend}"
        )
    if backend == "local" and not local_model_ok:
        backend = "edge"
        warnings.append("local TTS model is incomplete; runtime will fall back to edge")

    if backend == "edge" and not deps["edge_tts"]:
        errors.append("TTS dependency missing: edge_tts")
    if backend == "minimax" and not minimax_key:
        errors.append("MiniMax TTS API key is empty")
    if backend == "local" and not local_model_ok:
        for name, path in local_missing.items():
            errors.append(f"TTS missing {name}: {path}")
        if not deps["sherpa_onnx"]:
            errors.append("TTS dependency missing: sherpa_onnx")

    return {
        "ok": not errors,
        "model_ok": local_model_ok or backend in {"edge", "minimax"},
        "requested_backend": requested_backend,
        "effective_backend": backend,
        "fallback_backend": fallback_backend,
        "dependency_ok": {
            "sherpa_onnx": deps["sherpa_onnx"],
            "edge_tts": deps["edge_tts"],
        },
        "paths": {
            "model_dir": str(model_dir),
            "model": str(model),
            "tokens": str(model_dir / cfg.get("tokens", "tokens.txt")),
        },
        "missing": local_missing,
        "errors": errors,
        "warnings": warnings,
    }


def _check_runtime_bridge(cfg: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(cfg.get("enabled", False) or cfg.get("text_enabled", False))
    base_url = str(cfg.get("base_url", "")).strip()
    errors = []
    warnings = []
    if enabled and not base_url:
        errors.append("runtime.voice_bridge is enabled but base_url is empty")
    if not enabled:
        warnings.append("runtime voice bridge is disabled")
    return {
        "ok": not errors,
        "enabled": enabled,
        "base_url": base_url,
        "errors": errors,
        "warnings": warnings,
    }


def _check_audio_devices(cfg: dict[str, Any], deps: dict[str, bool]) -> dict[str, Any]:
    warnings = []
    if not deps["sounddevice"]:
        warnings.append("sounddevice is not installed; live microphone/speaker checks cannot run")
    return {
        "ok": deps["sounddevice"],
        "input_device": cfg.get("input_device"),
        "output_device": cfg.get("tts", {}).get("output_device", cfg.get("output_device")),
        "dependency_ok": deps["sounddevice"],
        "errors": [],
        "warnings": warnings,
    }


def _build_health_snapshot(
    asr: dict[str, Any],
    vad: dict[str, Any],
    kws: dict[str, Any],
    tts: dict[str, Any],
    audio: dict[str, Any],
) -> dict[str, Any]:
    asr_ok = bool(asr["ok"])
    vad_ok = bool(vad["ok"])
    kws_ok = bool(kws["ok"])
    tts_ok = bool(tts["ok"])
    input_ready = bool(audio["ok"] and vad_ok)
    output_ready = bool(audio["ok"] and tts_ok)
    return {
        "mode": "offline",
        "enabled": True,
        "input_ready": input_ready,
        "output_ready": output_ready,
        "pipeline_ok": bool(asr_ok and vad_ok and kws_ok and tts_ok),
        "asr_available": asr_ok,
        "vad_available": vad_ok,
        "kws_available": bool(kws_ok and kws.get("enabled")),
        "wake_word_enabled": bool(kws.get("enabled")),
        "woken_up": not bool(kws.get("enabled")),
        "muted": False,
        "tts_backend": tts.get("effective_backend", "unknown"),
        "tts_busy": False,
        "agent_state": "offline-health",
    }


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return root / path


def _find_model_file(model_dir: Path, override: Any, candidates: tuple[str, ...]) -> Path:
    if override:
        return model_dir / str(override)
    for name in candidates:
        path = model_dir / name
        if path.exists():
            return path
    return model_dir / candidates[0]


def _missing_paths(paths: dict[str, str]) -> dict[str, str]:
    missing = {}
    for name, value in paths.items():
        path = Path(value)
        exists = path.is_dir() if name.endswith("_dir") else path.is_file()
        if not exists:
            missing[name] = value
    return missing


def _dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _websocket_client_available() -> bool:
    """Return True only for the websocket-client package API."""
    if importlib.util.find_spec("websocket") is None:
        return False
    try:
        import websocket  # type: ignore[import-not-found]
    except Exception:
        return False
    return callable(getattr(websocket, "create_connection", None))


def _label(value: object) -> str:
    return "ok" if value else "degraded"


if __name__ == "__main__":
    sys.exit(main())
