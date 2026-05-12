"""Sunrise-specific audio diagnostics for the MCP01 USB voice path."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from askme.config import get_config
from askme.voice.tts import TTSEngine

MCP01_USB_ID = "17ef:a03b"
DEFAULT_FIRST_TOKEN_GUARD_SECONDS = 1.5
_TTS_CONFIG_KEYS = (
    "backend",
    "sample_rate",
    "output_device",
    "output_transport",
    "usb_audio_binary",
    "usb_audio_source",
    "usb_direct_persistent_stream",
    "usb_direct_trust_persistent_warm_state",
    "usb_direct_stream_start_grace_seconds",
    "usb_direct_stream_drain_grace_seconds",
    "usb_direct_preroll_seconds",
    "usb_direct_background_prewarm",
    "usb_direct_quiet_start",
    "usb_direct_speech_leadin_seconds",
    "usb_direct_speech_warm_leadin_seconds",
    "usb_direct_speech_wake_signal_seconds",
    "usb_direct_speech_wake_signal_gain",
    "usb_direct_speech_wake_noise_gain",
    "usb_direct_speech_wake_signal_hz",
    "usb_direct_speech_wake_gap_seconds",
    "usb_direct_speech_onset_cushion_seconds",
    "usb_direct_speech_onset_cushion_gain",
    "usb_direct_speech_onset_gap_seconds",
    "usb_direct_stream_guard_seconds",
    "usb_direct_coalesce_timeout",
    "minimax_sample_rate",
    "minimax_leading_silence_preserve_seconds",
    "minimax_onset_threshold",
)


@dataclass(frozen=True)
class CommandResult:
    """Serializable subprocess result used by the diagnostic runner."""

    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.skipped

    def to_payload(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "ok": self.ok,
            "skipped": self.skipped,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


CommandRunner = Callable[[list[str], float], CommandResult]


def run_sunrise_audio_doctor(
    config: dict[str, Any] | None = None,
    *,
    command_runner: CommandRunner | None = None,
    include_command_probes: bool = True,
    include_output_probe: bool = True,
    guard_min_seconds: float = DEFAULT_FIRST_TOKEN_GUARD_SECONDS,
) -> dict[str, Any]:
    """Return a hardware-aware Sunrise audio diagnostic report.

    The output probe is intentionally non-playing: it captures the exact float32
    chunk that TTSEngine would send to the MCP01 USB helper, then verifies that
    the real speech begins after enough sacrificial wake audio.
    """
    cfg = get_config(reload=True) if config is None else config
    voice_cfg = cfg.get("voice", {}) if isinstance(cfg, dict) else {}
    tts_cfg = voice_cfg.get("tts", {}) if isinstance(voice_cfg, dict) else {}
    if not isinstance(tts_cfg, dict):
        tts_cfg = {}

    runner = command_runner or _run_command
    usb_probe = (
        _probe_usb(runner)
        if include_command_probes
        else {"checked": False, "errors": ["USB command probes skipped"], "warnings": []}
    )
    asound_probe = (
        _probe_asound()
        if include_command_probes
        else {"checked": False, "errors": [], "warnings": ["ALSA card probe skipped"]}
    )
    output_probe = (
        _probe_usb_output_shape(tts_cfg, guard_min_seconds=guard_min_seconds)
        if include_output_probe
        else {"checked": False, "ok": False, "errors": ["USB output shape probe skipped"], "warnings": []}
    )

    config_probe = _probe_tts_config(tts_cfg)
    errors: list[str] = []
    warnings: list[str] = []
    for section in (usb_probe, asound_probe, config_probe, output_probe):
        errors.extend(section.get("errors", []))
        warnings.extend(section.get("warnings", []))

    return {
        "status": "ok" if not errors else "degraded",
        "target": "sunrise-mcp01-usb-audio",
        "guard_min_seconds": guard_min_seconds,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "usb": usb_probe,
            "asound": asound_probe,
            "tts_config": config_probe,
            "usb_output_shape": output_probe,
        },
    }


def print_sunrise_audio_doctor_summary(payload: dict[str, Any]) -> None:
    """Print a compact human-readable diagnostic summary."""
    checks = payload.get("checks", {})
    usb = checks.get("usb", {})
    asound = checks.get("asound", {})
    config = checks.get("tts_config", {})
    output = checks.get("usb_output_shape", {})

    print(f"sunrise-audio-doctor: {payload.get('status', 'unknown')}")  # noqa: T201
    print(  # noqa: T201
        "  usb: mcp01={mcp01} lsusb-tree-audio={audio}".format(
            mcp01=_label(usb.get("mcp01_visible")),
            audio=_label(usb.get("audio_class_visible")),
        )
    )
    print(  # noqa: T201
        "  alsa: cards={cards} ({detail})".format(
            cards=_label(asound.get("cards_visible")),
            detail=asound.get("detail", "unknown"),
        )
    )
    print(  # noqa: T201
        "  tts: transport={transport} persistent={persistent} trust_warm={trust}".format(
            transport=config.get("output_transport", "unknown"),
            persistent=_label(config.get("usb_direct_persistent_stream")),
            trust=_label(config.get("usb_direct_trust_persistent_warm_state")),
        )
    )
    print(  # noqa: T201
        "  first-token guard: {guard:.3f}s / min {minimum:.3f}s ({ok})".format(
            guard=float(output.get("speech_offset_seconds", 0.0) or 0.0),
            minimum=float(payload.get("guard_min_seconds", 0.0) or 0.0),
            ok=_label(output.get("first_token_guard_ok")),
        )
    )
    print(  # noqa: T201
        "  final write: {samples} samples, shape={ok}".format(
            samples=output.get("final_samples", "unknown"),
            ok=_label(output.get("final_shape_ok")),
        )
    )
    for warning in payload.get("warnings", []):
        print(f"  warn: {warning}")  # noqa: T201
    for error in payload.get("errors", []):
        print(f"  error: {error}")  # noqa: T201


def mcp01_visible(lsusb_output: str) -> bool:
    """Return True when the Lenovo MCP01 USB id is listed."""
    return MCP01_USB_ID in lsusb_output.lower()


def lsusb_tree_has_audio_class(lsusb_tree_output: str) -> bool:
    """Return True when lsusb -t exposes at least one USB Audio interface."""
    return "class=audio" in lsusb_tree_output.lower()


def parse_asound_cards(text: str) -> dict[str, Any]:
    """Parse /proc/asound/cards without requiring ALSA to be healthy."""
    stripped = text.strip()
    if not stripped:
        return {"cards_visible": False, "cards": [], "detail": "empty"}
    if "no soundcards" in stripped.lower():
        return {"cards_visible": False, "cards": [], "detail": "no soundcards"}
    cards = []
    for line in stripped.splitlines():
        match = re.match(r"\s*(\d+)\s+\[([^\]]+)\]\s*:\s*(.+)", line)
        if match:
            cards.append(
                {
                    "index": int(match.group(1)),
                    "id": match.group(2).strip(),
                    "description": match.group(3).strip(),
                }
            )
    return {
        "cards_visible": bool(cards),
        "cards": cards,
        "detail": "cards visible" if cards else "unparsed",
    }


def _probe_usb(runner: CommandRunner) -> dict[str, Any]:
    lsusb = runner(["lsusb"], 5.0)
    tree = runner(["lsusb", "-t"], 5.0)
    visible = mcp01_visible(lsusb.stdout)
    audio_class = lsusb_tree_has_audio_class(tree.stdout)
    errors: list[str] = []
    warnings: list[str] = []
    if not visible:
        errors.append("MCP01 USB device 17ef:a03b is not visible in lsusb")
    if not audio_class:
        errors.append("lsusb -t does not show a Class=Audio USB interface")
    if not lsusb.ok:
        warnings.append("lsusb probe did not complete successfully")
    if not tree.ok:
        warnings.append("lsusb -t probe did not complete successfully")
    return {
        "checked": True,
        "mcp01_visible": visible,
        "audio_class_visible": audio_class,
        "mcp01_audio_tree_ok": visible and audio_class,
        "commands": {
            "lsusb": lsusb.to_payload(),
            "lsusb_tree": tree.to_payload(),
        },
        "errors": errors,
        "warnings": warnings,
    }


def _probe_asound() -> dict[str, Any]:
    path = Path("/proc/asound/cards")
    warnings: list[str] = []
    if not path.exists():
        return {
            "checked": True,
            "cards_visible": False,
            "cards": [],
            "detail": "/proc/asound/cards missing",
            "warnings": ["ALSA card file is missing; USB-direct playback may still work"],
            "errors": [],
        }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {
            "checked": True,
            "cards_visible": False,
            "cards": [],
            "detail": str(exc),
            "warnings": [f"could not read /proc/asound/cards: {exc}"],
            "errors": [],
        }
    parsed = parse_asound_cards(text)
    if not parsed["cards_visible"]:
        warnings.append("ALSA exposes no soundcards; Sunrise must use MCP01 USB direct output")
    return {
        "checked": True,
        "raw": text,
        **parsed,
        "warnings": warnings,
        "errors": [],
    }


def _probe_tts_config(tts_cfg: dict[str, Any]) -> dict[str, Any]:
    snapshot = {key: tts_cfg.get(key) for key in _TTS_CONFIG_KEYS if key in tts_cfg}
    transport = str(tts_cfg.get("output_transport", "auto")).lower()
    persistent = bool(tts_cfg.get("usb_direct_persistent_stream", False))
    trust_warm = bool(tts_cfg.get("usb_direct_trust_persistent_warm_state", False))
    background_prewarm = bool(tts_cfg.get("usb_direct_background_prewarm", False))
    quiet_start = bool(tts_cfg.get("usb_direct_quiet_start", False))

    errors: list[str] = []
    warnings: list[str] = []
    if transport not in {"auto", "usb_direct"}:
        errors.append(
            f"voice.tts.output_transport={transport!r} will not reliably use MCP01 USB direct output"
        )
    if persistent and trust_warm:
        errors.append(
            "usb_direct_trust_persistent_warm_state must stay false until idle keepalive is verified"
        )
    if background_prewarm:
        warnings.append("usb_direct_background_prewarm can go stale before speech starts on Sunrise")
    wake_seconds = _float_config(tts_cfg, "usb_direct_speech_wake_signal_seconds", 0.0)
    if wake_seconds <= 0:
        warnings.append("speech lead-in has no active wake signal; dither alone may not open the speaker gate")
    if quiet_start:
        warnings.append("usb_direct_quiet_start reduces audible artifacts but may clip the first syllable")

    return {
        "checked": True,
        "output_transport": transport,
        "usb_direct_persistent_stream": persistent,
        "usb_direct_trust_persistent_warm_state": trust_warm,
        "snapshot": snapshot,
        "errors": errors,
        "warnings": warnings,
    }


def _probe_usb_output_shape(
    tts_cfg: dict[str, Any],
    *,
    guard_min_seconds: float,
) -> dict[str, Any]:
    probe_cfg = dict(tts_cfg)
    probe_cfg["backend"] = "edge"
    probe_cfg.setdefault("output_transport", "usb_direct")
    try:
        engine = TTSEngine(probe_cfg)
    except Exception as exc:
        return {
            "checked": True,
            "ok": False,
            "errors": [f"could not initialize TTSEngine for USB output probe: {exc}"],
            "warnings": [],
        }
    captured: list[np.ndarray] = []

    def capture(chunk: np.ndarray) -> bool:
        captured.append(np.asarray(chunk, dtype=np.float32).copy())
        return True

    try:
        engine._play_chunk_usb_direct_locked = capture  # type: ignore[method-assign]
        try:
            speech = _synthetic_first_token_chunk(engine._sample_rate)
            expected_speech = engine._apply_usb_direct_speech_gain(speech)
            cold_leadin = engine._usb_direct_speech_leadin_chunk(warm=False)
            warm_leadin = engine._usb_direct_speech_leadin_chunk(warm=True)
            active_leadin = engine._usb_direct_speech_leadin_chunk(
                warm=engine._is_persistent_usb_stream_warm()
            )
            cushion = engine._usb_direct_speech_onset_cushion_chunk(expected_speech)
            play_ok = engine._play_chunk_usb_direct_speech(speech)
        except Exception as exc:
            return {
                "checked": True,
                "ok": False,
                "errors": [f"USB output shape probe failed: {exc}"],
                "warnings": [],
            }
    finally:
        engine.shutdown()

    sample_rate = int(engine._sample_rate)
    if captured:
        final = captured[0]
    else:
        final = np.empty(0, dtype=np.float32)

    expected_samples = len(active_leadin) + len(cushion) + len(expected_speech)
    speech_offset_samples = len(active_leadin) + len(cushion)
    speech_offset_seconds = speech_offset_samples / float(sample_rate)
    final_shape_ok = bool(
        play_ok
        and len(final) == expected_samples
        and _segment_equal(final, 0, active_leadin)
        and _segment_equal(final, len(active_leadin), cushion)
        and _segment_equal(final, speech_offset_samples, expected_speech)
    )
    quiet_start = bool(tts_cfg.get("usb_direct_quiet_start", False))
    first_token_guard_ok = speech_offset_seconds >= guard_min_seconds or quiet_start
    wake_peak = int(float(np.max(np.abs(cold_leadin))) * 32768) if len(cold_leadin) else 0

    errors: list[str] = []
    warnings: list[str] = []
    if not final_shape_ok:
        errors.append("USB speech output shape does not match lead-in + cushion + speech")
    if speech_offset_seconds < guard_min_seconds and quiet_start:
        warnings.append(
            f"quiet start accepts first real speech at {speech_offset_seconds:.3f}s "
            f"below {guard_min_seconds:.3f}s guard to reduce audible pre-speech noise"
        )
    elif not first_token_guard_ok:
        errors.append(
            f"first real speech begins at {speech_offset_seconds:.3f}s, "
            f"below {guard_min_seconds:.3f}s guard"
        )
    if wake_peak < 1000:
        warnings.append("speech lead-in peak is very low; speaker gate may not wake consistently")

    return {
        "checked": True,
        "ok": final_shape_ok and first_token_guard_ok,
        "sample_rate": sample_rate,
        "input_speech_samples": len(speech),
        "cold_leadin_samples": len(cold_leadin),
        "warm_leadin_samples": len(warm_leadin),
        "active_leadin_samples": len(active_leadin),
        "onset_cushion_samples": len(cushion),
        "speech_offset_samples": speech_offset_samples,
        "speech_offset_seconds": speech_offset_seconds,
        "guard_min_seconds": guard_min_seconds,
        "quiet_start": quiet_start,
        "first_token_guard_ok": first_token_guard_ok,
        "expected_final_samples": expected_samples,
        "final_samples": len(final),
        "final_shape_ok": final_shape_ok,
        "cold_leadin_peak": wake_peak,
        "errors": errors,
        "warnings": warnings,
    }


def _synthetic_first_token_chunk(sample_rate: int) -> np.ndarray:
    """Build a short first-token-like utterance for non-playing output probes."""
    sample_rate = max(1000, int(sample_rate))
    soft_samples = int(sample_rate * 0.08)
    body_samples = int(sample_rate * 0.42)
    soft_t = np.arange(soft_samples, dtype=np.float32) / float(sample_rate)
    body_t = np.arange(body_samples, dtype=np.float32) / float(sample_rate)
    soft = 0.04 * np.sin(2.0 * np.pi * 260.0 * soft_t)
    body = 0.20 * np.sin(2.0 * np.pi * 520.0 * body_t)
    return np.concatenate([soft, body]).astype(np.float32)


def _segment_equal(haystack: np.ndarray, start: int, needle: np.ndarray) -> bool:
    if len(needle) == 0:
        return True
    end = start + len(needle)
    if start < 0 or end > len(haystack):
        return False
    return bool(np.allclose(haystack[start:end], needle, atol=1e-7))


def _run_command(command: list[str], timeout: float) -> CommandResult:
    executable = shutil.which(command[0])
    if executable is None:
        return CommandResult(command=command, returncode=127, skipped=True, stderr="command not found")
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=command,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "timeout",
        )
    except OSError as exc:
        return CommandResult(command=command, returncode=126, stderr=str(exc))
    return CommandResult(
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _float_config(config: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(config.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _label(value: Any) -> str:
    if value is True:
        return "ok"
    if value is False:
        return "no"
    return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    parser.add_argument("--json-out", default="", help="Also write the diagnostic JSON to this path")
    parser.add_argument(
        "--guard-min-seconds",
        type=float,
        default=DEFAULT_FIRST_TOKEN_GUARD_SECONDS,
        help="Minimum sacrificial lead-in+cushion before real speech",
    )
    parser.add_argument(
        "--skip-command-probes",
        action="store_true",
        help="Skip lsusb and /proc/asound probes",
    )
    parser.add_argument(
        "--skip-output-probe",
        action="store_true",
        help="Skip non-playing TTSEngine USB output-shape probe",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_sunrise_audio_doctor(
        include_command_probes=not args.skip_command_probes,
        include_output_probe=not args.skip_output_probe,
        guard_min_seconds=args.guard_min_seconds,
    )
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))  # noqa: T201
    else:
        print_sunrise_audio_doctor_summary(payload)
    return 0 if payload.get("status") == "ok" else 1
