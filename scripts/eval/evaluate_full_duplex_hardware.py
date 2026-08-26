"""Interactive target-hardware acceptance for full-duplex robot voice."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from askme.voice.diagnostics.full_duplex_hardware import (  # noqa: E402
    MIN_TRIALS_PER_SCENARIO,
    evaluate_hardware_run,
    preflight_hardware_run,
    runtime_readiness,
    write_hardware_report,
)
from askme.voice.diagnostics.hardware_audio_capture import (  # noqa: E402
    build_manual_trial_evidence,
)

DEFAULT_STATUS_SOURCE = "http://127.0.0.1:8765/health"
DEFAULT_OUTPUT = Path("artifacts/voice/full-duplex-hardware.json")


def main(
    argv: list[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    monotonic: Callable[[], float] = time.perf_counter,
) -> int:
    args = _build_parser().parse_args(argv)
    if args.speaker_only_trials < MIN_TRIALS_PER_SCENARIO:
        raise SystemExit(f"--speaker-only-trials must be >= {MIN_TRIALS_PER_SCENARIO}")
    if args.overlap_trials < MIN_TRIALS_PER_SCENARIO:
        raise SystemExit(f"--overlap-trials must be >= {MIN_TRIALS_PER_SCENARIO}")
    if args.response_trials < MIN_TRIALS_PER_SCENARIO:
        raise SystemExit(f"--response-trials must be >= {MIN_TRIALS_PER_SCENARIO}")

    output_path = Path(args.output)
    config_path = Path(args.config)
    config = _load_config(config_path)
    metadata: dict[str, Any] = {
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "config_path": str(config_path.resolve()),
        "status_source": args.status_source,
        "latency_source": args.latency_mode,
    }
    speaker_trials: list[dict[str, Any]] = []
    overlap_trials: list[dict[str, Any]] = []
    response_trials: list[dict[str, Any]] = []
    aborted_reason: str | None = None

    try:
        initial_status = _read_status(args.status_source, timeout_s=args.status_timeout_s)
    except Exception as exc:
        initial_status = {"status": "error", "error": str(exc)}
        aborted_reason = f"status_source_unavailable: {exc}"

    preflight = preflight_hardware_run(
        config=config,
        runtime_status=initial_status,
    )
    evidence = preflight["echo_control_evidence"]
    runtime_backends = evidence.get("runtime_backends", [])
    metadata["aec_backend"] = runtime_backends[0] if len(runtime_backends) == 1 else ""
    if preflight["status"] != "ready":
        aborted_reason = aborted_reason or "preflight_failed"
        report = _build_session_report(
            config=config,
            metadata=metadata,
            speaker_trials=speaker_trials,
            overlap_trials=overlap_trials,
            response_trials=response_trials,
            preflight=preflight,
            aborted_reason=aborted_reason,
        )
        write_hardware_report(report, output_path)
        _print_result(report, output_path)
        return 1

    try:
        metadata.update(_collect_metadata(args, config=config, input_fn=input_fn))
    except (EOFError, KeyboardInterrupt):
        aborted_reason = "operator_interrupted"
        report = _build_session_report(
            config=config,
            metadata=metadata,
            speaker_trials=speaker_trials,
            overlap_trials=overlap_trials,
            response_trials=response_trials,
            preflight=preflight,
            aborted_reason=aborted_reason,
        )
        write_hardware_report(report, output_path)
        _print_result(report, output_path)
        return 1
    metadata_report = _build_session_report(
        config=config,
        metadata=metadata,
        speaker_trials=speaker_trials,
        overlap_trials=overlap_trials,
        response_trials=response_trials,
        preflight=preflight,
        aborted_reason="hardware_metadata_incomplete",
    )
    if metadata_report["metadata_missing"]:
        write_hardware_report(metadata_report, output_path)
        _print_result(metadata_report, output_path)
        return 1
    print(  # noqa: T201
        "\n开始目标硬件验收。每个试次都必须使用同一设备、驱动、房间和音量配置。"
    )
    print("纯扬声器试次：只播放机器人语音，现场保持安静。")  # noqa: T201

    try:
        for index in range(1, args.speaker_only_trials + 1):
            false_barge_in = _prompt_yes_no(
                input_fn,
                f"[{index}/{args.speaker_only_trials}] 播放标准回复后，是否发生误插话/误停播？ [y/N/q] ",
            )
            if false_barge_in is None:
                aborted_reason = "operator_aborted"
                break
            status = _safe_read_status(args.status_source, args.status_timeout_s)
            speaker_trials.append(
                {
                    "trial": index,
                    "observed_at": datetime.now(UTC).isoformat(),
                    "false_barge_in": false_barge_in,
                    "runtime_status": status,
                    **build_manual_trial_evidence(
                        method="manual_observation",
                        reference_event="speaker_only_false_barge_in",
                        observed_timestamp_s=monotonic(),
                    ),
                }
            )
            _persist_progress(
                output_path=output_path,
                config=config,
                metadata=metadata,
                speaker_trials=speaker_trials,
                overlap_trials=overlap_trials,
                response_trials=response_trials,
                preflight=preflight,
            )
            ready, reason = runtime_readiness(status)
            if not ready:
                aborted_reason = f"runtime_degraded: {reason}"
                break

        if aborted_reason is None:
            print(  # noqa: T201
                "真人插话试次：机器人播放期间，由真人开始说话并记录到扬声器停播的延迟。"
            )
            for index in range(1, args.overlap_trials + 1):
                detected, latency_ms, operator_aborted, evidence = _collect_overlap_trial(
                    index=index,
                    total=args.overlap_trials,
                    latency_mode=args.latency_mode,
                    input_fn=input_fn,
                    monotonic=monotonic,
                )
                if operator_aborted:
                    aborted_reason = "operator_aborted"
                    break
                status = _safe_read_status(args.status_source, args.status_timeout_s)
                overlap_trials.append(
                    {
                        "trial": index,
                        "observed_at": datetime.now(UTC).isoformat(),
                        "detected": detected,
                        "speaker_stop_latency_ms": latency_ms,
                        "latency_source": args.latency_mode,
                        "runtime_status": status,
                        **evidence,
                    }
                )
                _persist_progress(
                    output_path=output_path,
                    config=config,
                    metadata=metadata,
                    speaker_trials=speaker_trials,
                    overlap_trials=overlap_trials,
                    response_trials=response_trials,
                    preflight=preflight,
                )
                ready, reason = runtime_readiness(status)
                if not ready:
                    aborted_reason = f"runtime_degraded: {reason}"
                    break

        if aborted_reason is None:
            print(  # noqa: T201
                "真实首音试次：每次自然说完测试句，记录到扬声器首个可听声音的延迟。"
            )
            for index in range(1, args.response_trials + 1):
                heard, latency_ms, operator_aborted, evidence = _collect_response_trial(
                    index=index,
                    total=args.response_trials,
                    latency_mode=args.latency_mode,
                    input_fn=input_fn,
                    monotonic=monotonic,
                )
                if operator_aborted:
                    aborted_reason = "operator_aborted"
                    break
                status = _safe_read_status(args.status_source, args.status_timeout_s)
                response_trials.append(
                    {
                        "trial": index,
                        "observed_at": datetime.now(UTC).isoformat(),
                        "heard": heard,
                        "speech_end_to_first_sound_ms": latency_ms,
                        "latency_source": args.latency_mode,
                        "runtime_status": status,
                        **evidence,
                    }
                )
                _persist_progress(
                    output_path=output_path,
                    config=config,
                    metadata=metadata,
                    speaker_trials=speaker_trials,
                    overlap_trials=overlap_trials,
                    response_trials=response_trials,
                    preflight=preflight,
                )
                ready, reason = runtime_readiness(status)
                if not ready:
                    aborted_reason = f"runtime_degraded: {reason}"
                    break
    except (EOFError, KeyboardInterrupt):
        aborted_reason = "operator_interrupted"

    report = _build_session_report(
        config=config,
        metadata=metadata,
        speaker_trials=speaker_trials,
        overlap_trials=overlap_trials,
        response_trials=response_trials,
        preflight=preflight,
        aborted_reason=aborted_reason,
    )
    write_hardware_report(report, output_path)
    _print_result(report, output_path)
    return 0 if report["status"] == "passed" else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.board.yaml")
    parser.add_argument("--status-source", default=DEFAULT_STATUS_SOURCE)
    parser.add_argument("--status-timeout-s", type=float, default=3.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--speaker-only-trials", type=int, default=20)
    parser.add_argument("--overlap-trials", type=int, default=20)
    parser.add_argument("--response-trials", type=int, default=20)
    parser.add_argument(
        "--latency-mode",
        choices=("stopwatch", "entry"),
        default="stopwatch",
        help=(
            "diagnostic manual timing only: stopwatch uses Enter-to-Enter timing; "
            "entry accepts operator-entered milliseconds. Neither is product-grade instrumented."
        ),
    )
    parser.add_argument("--operator")
    parser.add_argument("--room")
    parser.add_argument("--audio-device")
    parser.add_argument("--audio-driver")
    parser.add_argument("--input-device-id")
    parser.add_argument("--output-device-id")
    parser.add_argument("--input-sample-rate-hz", type=int)
    parser.add_argument("--output-sample-rate-hz", type=int)
    return parser


def _load_config(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _read_status(source: str, *, timeout_s: float) -> Mapping[str, Any]:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        with urlopen(source, timeout=timeout_s) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("runtime status must be a JSON object")
    return payload


def _safe_read_status(source: str, timeout_s: float) -> Mapping[str, Any]:
    try:
        return _read_status(source, timeout_s=timeout_s)
    except Exception as exc:
        return {"status": "error", "error": f"status_source_unavailable: {exc}"}


def _collect_metadata(
    args: argparse.Namespace,
    *,
    config: Mapping[str, Any],
    input_fn: Callable[[str], str],
) -> dict[str, Any]:
    voice = config.get("voice", {})
    voice_cfg = voice if isinstance(voice, Mapping) else {}
    tts = voice_cfg.get("tts", {})
    tts_cfg = tts if isinstance(tts, Mapping) else {}
    return {
        "operator": _prompt_value(input_fn, "操作员", args.operator),
        "room": _prompt_value(input_fn, "测试房间", args.room),
        "audio_device": _prompt_value(input_fn, "音响/麦克风一体设备型号", args.audio_device),
        "audio_driver": _prompt_value(input_fn, "音频驱动/主机接口", args.audio_driver),
        "input_device_id": _prompt_value(
            input_fn,
            "输入设备 ID",
            args.input_device_id,
            voice_cfg.get("input_device"),
        ),
        "output_device_id": _prompt_value(
            input_fn,
            "输出设备 ID",
            args.output_device_id,
            tts_cfg.get("output_device"),
        ),
        "input_sample_rate_hz": _prompt_positive_int(
            input_fn,
            "输入设备原生采样率 Hz",
            args.input_sample_rate_hz,
            voice_cfg.get("mic_native_rate"),
        ),
        "output_sample_rate_hz": _prompt_positive_int(
            input_fn,
            "输出采样率 Hz",
            args.output_sample_rate_hz,
            tts_cfg.get("sample_rate"),
        ),
    }


def _prompt_value(
    input_fn: Callable[[str], str],
    label: str,
    supplied: Any,
    default: Any = None,
) -> str:
    if supplied is not None and str(supplied).strip():
        return str(supplied).strip()
    suffix = f" [{default}]" if default is not None else ""
    entered = input_fn(f"{label}{suffix}: ").strip()
    if entered:
        return entered
    return "" if default is None else str(default)


def _prompt_positive_int(
    input_fn: Callable[[str], str],
    label: str,
    supplied: int | None,
    default: Any,
) -> int | str:
    if supplied is not None:
        return supplied
    value = _prompt_value(input_fn, label, None, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return value
    return parsed if parsed > 0 else value


def _prompt_yes_no(
    input_fn: Callable[[str], str],
    prompt: str,
) -> bool | None:
    while True:
        answer = input_fn(prompt).strip().lower()
        if answer in {"", "n", "no", "否"}:
            return False
        if answer in {"y", "yes", "是"}:
            return True
        if answer in {"q", "quit", "退出"}:
            return None
        print("请输入 y、n 或 q。")  # noqa: T201


def _collect_overlap_trial(
    *,
    index: int,
    total: int,
    latency_mode: str,
    input_fn: Callable[[str], str],
    monotonic: Callable[[], float],
) -> tuple[bool, float | None, bool, dict[str, Any]]:
    if latency_mode == "entry":
        evidence = build_manual_trial_evidence(
            method="manual_entry",
            reference_event="human_speech_onset",
        )
        while True:
            answer = (
                input_fn(f"[{index}/{total}] 输入实测停播延迟 ms；未检测输入 m；退出 q: ")
                .strip()
                .lower()
            )
            if answer in {"q", "quit", "退出"}:
                return False, None, True, evidence
            if answer in {"m", "miss", "未检测"}:
                return False, None, False, evidence
            try:
                latency_ms = float(answer)
            except ValueError:
                print("请输入非负毫秒数、m 或 q。")  # noqa: T201
                continue
            if latency_ms >= 0:
                return True, latency_ms, False, evidence
            print("延迟不能为负数。")  # noqa: T201

    start_answer = (
        input_fn(f"[{index}/{total}] 机器人播放中，按回车的同时开始说话；输入 q 退出: ")
        .strip()
        .lower()
    )
    if start_answer in {"q", "quit", "退出"}:
        evidence = build_manual_trial_evidence(
            method="manual_stopwatch",
            reference_event="human_speech_onset",
        )
        return False, None, True, evidence
    started = monotonic()
    stop_answer = input_fn("扬声器一停就按回车；若未停播输入 m；退出 q: ").strip().lower()
    stopped = monotonic()
    elapsed_ms = max(0.0, (stopped - started) * 1000.0)
    if stop_answer in {"q", "quit", "退出"}:
        evidence = build_manual_trial_evidence(
            method="manual_stopwatch",
            reference_event="human_speech_onset",
            reference_timestamp_s=started,
        )
        return False, None, True, evidence
    if stop_answer in {"m", "miss", "未检测"}:
        evidence = build_manual_trial_evidence(
            method="manual_stopwatch",
            reference_event="human_speech_onset",
            reference_timestamp_s=started,
        )
        return False, None, False, evidence
    evidence = build_manual_trial_evidence(
        method="manual_stopwatch",
        reference_event="human_speech_onset",
        reference_timestamp_s=started,
        event_timestamp_s=stopped,
    )
    return True, elapsed_ms, False, evidence


def _collect_response_trial(
    *,
    index: int,
    total: int,
    latency_mode: str,
    input_fn: Callable[[str], str],
    monotonic: Callable[[], float],
) -> tuple[bool, float | None, bool, dict[str, Any]]:
    if latency_mode == "entry":
        evidence = build_manual_trial_evidence(
            method="manual_entry",
            reference_event="speech_end",
        )
        while True:
            answer = (
                input_fn(
                    f"[{index}/{total}] 输入说完到真实首音的实测延迟 ms；未出声输入 m；退出 q: "
                )
                .strip()
                .lower()
            )
            if answer in {"q", "quit", "退出"}:
                return False, None, True, evidence
            if answer in {"m", "miss", "未出声"}:
                return False, None, False, evidence
            try:
                latency_ms = float(answer)
            except ValueError:
                print("请输入非负毫秒数、m 或 q。")  # noqa: T201
                continue
            if latency_ms >= 0:
                return True, latency_ms, False, evidence
            print("延迟不能为负数。")  # noqa: T201

    start_answer = (
        input_fn(f"[{index}/{total}] 自然说完测试句后立即按回车；输入 q 退出: ").strip().lower()
    )
    if start_answer in {"q", "quit", "退出"}:
        evidence = build_manual_trial_evidence(
            method="manual_stopwatch",
            reference_event="speech_end",
        )
        return False, None, True, evidence
    started = monotonic()
    sound_answer = (
        input_fn("扬声器首个可听声音一出现就按回车；未出声输入 m；退出 q: ").strip().lower()
    )
    sounded = monotonic()
    elapsed_ms = max(0.0, (sounded - started) * 1000.0)
    if sound_answer in {"q", "quit", "退出"}:
        evidence = build_manual_trial_evidence(
            method="manual_stopwatch",
            reference_event="speech_end",
            reference_timestamp_s=started,
        )
        return False, None, True, evidence
    if sound_answer in {"m", "miss", "未出声"}:
        evidence = build_manual_trial_evidence(
            method="manual_stopwatch",
            reference_event="speech_end",
            reference_timestamp_s=started,
        )
        return False, None, False, evidence
    evidence = build_manual_trial_evidence(
        method="manual_stopwatch",
        reference_event="speech_end",
        reference_timestamp_s=started,
        event_timestamp_s=sounded,
    )
    return True, elapsed_ms, False, evidence


def _persist_progress(
    *,
    output_path: Path,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    speaker_trials: list[dict[str, Any]],
    overlap_trials: list[dict[str, Any]],
    response_trials: list[dict[str, Any]],
    preflight: Mapping[str, Any],
) -> None:
    report = _build_session_report(
        config=config,
        metadata=metadata,
        speaker_trials=speaker_trials,
        overlap_trials=overlap_trials,
        response_trials=response_trials,
        preflight=preflight,
        aborted_reason=None,
    )
    write_hardware_report(report, output_path)


def _build_session_report(
    *,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    speaker_trials: list[dict[str, Any]],
    overlap_trials: list[dict[str, Any]],
    response_trials: list[dict[str, Any]],
    preflight: Mapping[str, Any],
    aborted_reason: str | None,
) -> dict[str, Any]:
    report = evaluate_hardware_run(
        config=config,
        metadata=metadata,
        speaker_only_trials=speaker_trials,
        overlap_trials=overlap_trials,
        response_trials=response_trials,
        require_response_trials=True,
    )
    report["preflight"] = dict(preflight)
    report["aborted_reason"] = aborted_reason
    if aborted_reason is not None:
        report["status"] = "failed"
    return report


def _print_result(report: Mapping[str, Any], output_path: Path) -> None:
    print(  # noqa: T201
        json.dumps(
            {
                "status": report.get("status"),
                "failed_checks": report.get("failed_checks"),
                "aborted_reason": report.get("aborted_reason"),
                "report": str(output_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
