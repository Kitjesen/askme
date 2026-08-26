"""Fail-closed startup checks for the containerized edge-robot runtime."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from askme.config import get_config
from askme.voice.diagnostics.health_check import run_voice_health

logger = logging.getLogger(__name__)

_CONFIG_ERROR_EXIT = 78
_SOUNDDEVICE_UNSET = object()


def probe_audio_devices(
    config: Mapping[str, Any],
    *,
    sounddevice_module: Any = _SOUNDDEVICE_UNSET,
) -> dict[str, Any]:
    """Verify that the configured PortAudio input and output are usable."""

    if sounddevice_module is _SOUNDDEVICE_UNSET:
        try:
            import sounddevice as imported_sounddevice
        except Exception as exc:
            return _audio_failure(
                f"audio dependency sounddevice is unavailable ({type(exc).__name__})"
            )
        sounddevice_module = imported_sounddevice

    voice_cfg = config.get("voice", {})
    if not isinstance(voice_cfg, Mapping):
        return _audio_failure("voice config section is missing or invalid")
    tts_cfg = voice_cfg.get("tts", {})
    if not isinstance(tts_cfg, Mapping):
        tts_cfg = {}

    try:
        devices = list(sounddevice_module.query_devices())
        default_input, default_output = _default_device_pair(sounddevice_module)
    except Exception as exc:
        return _audio_failure(f"audio device inventory failed ({type(exc).__name__}: {exc})")

    input_result = _probe_audio_direction(
        sounddevice_module,
        devices,
        direction="input",
        selector=voice_cfg.get("input_device"),
        default_selector=default_input,
    )
    output_result = _probe_audio_direction(
        sounddevice_module,
        devices,
        direction="output",
        selector=tts_cfg.get("output_device", voice_cfg.get("output_device")),
        default_selector=default_output,
    )
    errors = [
        str(error) for result in (input_result, output_result) for error in result.get("errors", [])
    ]
    return {
        "ok": not errors,
        "errors": errors,
        "input": input_result,
        "output": output_result,
    }


def run_edge_robot_preflight(
    config: dict[str, Any] | None = None,
    *,
    root: Path | None = None,
    voice_health_runner: Callable[..., dict[str, Any]] = run_voice_health,
    audio_probe: Callable[[Mapping[str, Any]], dict[str, Any]] = probe_audio_devices,
) -> dict[str, Any]:
    """Check config-derived model files and real audio devices before startup."""

    cfg = get_config(reload=True) if config is None else config
    try:
        voice = voice_health_runner(cfg, root=root)
    except Exception as exc:
        voice = {
            "status": "degraded",
            "models_ok": False,
            "errors": [f"voice model preflight failed ({type(exc).__name__}: {exc})"],
            "warnings": [],
        }
    try:
        audio = audio_probe(cfg)
    except Exception as exc:
        audio = _audio_failure(f"audio device preflight failed ({type(exc).__name__}: {exc})")

    errors = _string_items(voice.get("errors")) + _string_items(audio.get("errors"))
    warnings = _string_items(voice.get("warnings"))
    voice_ok = voice.get("status") == "ok"
    audio_ok = audio.get("ok") is True
    ready = bool(voice_ok and audio_ok and not errors)
    return {
        "status": "ok" if ready else "blocked",
        "ready": ready,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "voice": {
                "status": str(voice.get("status", "unknown")),
                "models_ok": voice.get("models_ok") is True,
                "asr_ok": voice.get("asr_ok") is True,
                "vad_ok": voice.get("vad_ok") is True,
                "kws_ok": voice.get("kws_ok") is True,
                "tts_ok": voice.get("tts_ok") is True,
            },
            "audio": {
                "ok": audio_ok,
                "input": dict(audio.get("input", {})),
                "output": dict(audio.get("output", {})),
            },
        },
    }


def main() -> int:
    """Run the edge-robot startup gate without printing config or secrets."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    payload = run_edge_robot_preflight()
    if payload["ready"]:
        logger.info("edge_robot preflight passed: models and audio devices are ready")
        return 0

    logger.error("edge_robot preflight blocked startup")
    for error in payload["errors"]:
        logger.error("  %s", error)
    logger.error("Mount /app/models and /dev/snd, then verify ASKME_AUDIO_GID before retrying")
    return _CONFIG_ERROR_EXIT


def _probe_audio_direction(
    sounddevice_module: Any,
    devices: Sequence[Any],
    *,
    direction: str,
    selector: Any,
    default_selector: int | None,
) -> dict[str, Any]:
    selected = selector if selector not in {None, ""} else default_selector
    if isinstance(selected, int) and selected < 0:
        selected = None
    if selected is None:
        return {
            "ok": False,
            "index": None,
            "errors": [f"audio {direction} device has no configured or system default route"],
        }

    try:
        info = sounddevice_module.query_devices(selected, direction)
        normalized = dict(info)
        channel_key = "max_input_channels" if direction == "input" else "max_output_channels"
        channels = int(normalized.get(channel_key, 0) or 0)
        if channels < 1:
            return {
                "ok": False,
                "index": selected,
                "errors": [f"audio {direction} device exposes no {direction} channels"],
            }
        sample_rate = float(normalized.get("default_samplerate") or 48_000)
        checker = getattr(
            sounddevice_module,
            "check_input_settings" if direction == "input" else "check_output_settings",
        )
        checker(device=selected, channels=1, samplerate=sample_rate)
        return {
            "ok": True,
            "index": selected,
            "name": str(normalized.get("name") or ""),
            "channels": channels,
            "sample_rate": sample_rate,
            "errors": [],
        }
    except Exception as exc:
        return {
            "ok": False,
            "index": selected,
            "errors": [f"audio {direction} device check failed ({type(exc).__name__}: {exc})"],
        }


def _default_device_pair(sounddevice_module: Any) -> tuple[int | None, int | None]:
    raw = getattr(getattr(sounddevice_module, "default", None), "device", None)
    if raw is None:
        return None, None
    try:
        device_pair = tuple(raw)
    except TypeError:
        return None, None
    if len(device_pair) < 2:
        return None, None
    default_input, default_output = device_pair[:2]
    return _valid_device_index(default_input), _valid_device_index(default_output)


def _valid_device_index(value: Any) -> int | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def _audio_failure(message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "errors": [message],
        "input": {},
        "output": {},
    }


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


if __name__ == "__main__":
    raise SystemExit(main())
