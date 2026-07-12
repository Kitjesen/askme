"""Generated voice capability check.

This bench script avoids physical microphones:
1. Generate TTS audio and capture the PCM from ``TTSEngine.tts_buffer``.
2. Feed that generated audio into ``ASRManager``.
3. Route the recognized text through the real skill voice-trigger index.
4. Run a deterministic large-memory vector search probe.

It is intended for local/Sunrise diagnosis, not normal CI.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TEXT = "\u51e0\u70b9\u4e86"  # ji dian le
DEFAULT_EXPECTED_SKILL = "get_time"


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(samples) <= 1:
        return samples.astype(np.float32, copy=False)
    ratio = target_rate / float(source_rate)
    new_len = max(1, int(round(len(samples) * ratio)))
    src_x = np.arange(len(samples), dtype=np.float64)
    dst_x = np.linspace(0, len(samples) - 1, new_len, dtype=np.float64)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def _to_int16(samples: np.ndarray) -> np.ndarray:
    return (samples * 32768.0).clip(-32768, 32767).astype(np.int16)


def _audio_metrics(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if len(samples) == 0:
        return {
            "sample_rate": sample_rate,
            "samples": 0,
            "duration_s": 0.0,
            "peak": 0,
            "rms": 0.0,
            "signal_ok": False,
        }
    pcm16 = _to_int16(samples)
    return {
        "sample_rate": sample_rate,
        "samples": int(len(samples)),
        "duration_s": round(len(samples) / float(sample_rate or 1), 3),
        "peak": int(np.max(np.abs(pcm16))),
        "rms": round(float(np.sqrt(np.mean(pcm16.astype(np.float64) ** 2))), 2),
        "signal_ok": int(np.max(np.abs(pcm16))) >= 500,
    }


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = _to_int16(samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only PCM16 wav is supported, got sample_width={sample_width}")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32), sample_rate


def generate_tts_audio(text: str, tts_cfg: dict[str, Any]) -> dict[str, Any]:
    from askme.voice.tts import TTSEngine

    cfg = copy.deepcopy(tts_cfg)
    started = time.perf_counter()
    tts = TTSEngine(cfg)
    try:
        generation = tts._get_generation()
        tts._generate_audio(text, generation)
        chunks: list[np.ndarray]
        with tts._buffer_lock:
            chunks = [chunk.copy() for chunk in tts.tts_buffer if len(chunk) > 0]
            tts.tts_buffer.clear()
        samples = np.concatenate(chunks).astype(np.float32) if chunks else np.array([], np.float32)
        sample_rate = int(getattr(tts, "_sample_rate", cfg.get("sample_rate", 24000)))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ok": len(samples) > 0,
            "samples": samples,
            "sample_rate": sample_rate,
            "backend": getattr(tts, "backend", "unknown"),
            "elapsed_ms": round(elapsed_ms, 1),
            "chunks": len(chunks),
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "samples": np.array([], np.float32),
            "sample_rate": int(cfg.get("sample_rate", 24000)),
            "backend": getattr(tts, "backend", "unknown"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "chunks": 0,
            "error": str(exc),
        }
    finally:
        try:
            tts.shutdown()
        except Exception:
            pass


def transcribe_generated_audio(
    samples: np.ndarray,
    sample_rate: int,
    voice_cfg: dict[str, Any],
    *,
    use_cloud: bool,
) -> dict[str, Any]:
    from askme.voice.asr_manager import ASRManager

    cfg = copy.deepcopy(voice_cfg)
    if not use_cloud:
        cfg.setdefault("cloud_asr", {})["enabled"] = False

    asr_rate = int(cfg.get("asr", {}).get("sample_rate", 16000) or 16000)
    audio = _resample_linear(samples, sample_rate, asr_rate)
    pcm16 = _to_int16(audio)

    started = time.perf_counter()
    mgr = ASRManager(cfg)
    try:
        mgr.start_session()
        step = max(1, int(asr_rate * 0.1))
        for pos in range(0, len(audio), step):
            chunk_f32 = audio[pos : pos + step]
            chunk_i16 = pcm16[pos : pos + step]
            mgr.feed_audio(chunk_f32, chunk_i16, asr_rate)
        silence = np.zeros(int(asr_rate * 0.5), dtype=np.float32)
        mgr.feed_audio(silence, _to_int16(silence), asr_rate)
        result = mgr.finish_and_get_result()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result is None:
            return {
                "ok": False,
                "text": "",
                "source": "",
                "is_noise": False,
                "elapsed_ms": round(elapsed_ms, 1),
                "error": "empty ASR result",
            }
        return {
            "ok": bool(result.text) and not result.is_noise,
            "text": result.text,
            "source": result.source,
            "is_noise": bool(result.is_noise),
            "elapsed_ms": round(elapsed_ms, 1),
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "text": "",
            "source": "",
            "is_noise": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "error": str(exc),
        }
    finally:
        try:
            mgr.reset()
        except Exception:
            pass


def route_skill_text(text: str, *, expected_skill: str | None = None) -> dict[str, Any]:
    from askme.skills.skill_manager import SkillManager

    from askme.robot_interaction import IntentRouter, IntentType

    manager = SkillManager(project_dir=ROOT)
    manager.load()
    router = IntentRouter(voice_triggers=manager.get_voice_triggers())
    intent = router.route(text)
    matched = intent.type == IntentType.VOICE_TRIGGER
    expected_ok = True if expected_skill is None else intent.skill_name == expected_skill
    return {
        "ok": bool(matched and expected_ok),
        "text": text,
        "intent": intent.type.value,
        "skill": intent.skill_name,
        "expected_skill": expected_skill,
        "trigger_count": len(manager.get_voice_triggers()),
    }


def memory_scale_probe(items: int, dim: int, top_k: int) -> dict[str, Any]:
    import askme.memory.vector_store as vector_store_mod

    items = max(1, int(items))
    dim = max(4, int(dim))
    top_k = max(1, int(top_k))

    original_check = vector_store_mod._check_st_available
    vector_store_mod._check_st_available = lambda: True
    try:
        store = vector_store_mod.VectorStore()
        rng = np.random.default_rng(20260506)
        embeddings = rng.normal(size=(items, dim)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-6)

        query_vec = np.zeros(dim, dtype=np.float32)
        query_vec[0] = 1.0
        target = items // 2
        embeddings[target] = query_vec

        store._texts = [f"memory item {i}" for i in range(items)]
        store._metadata = [{"id": i} for i in range(items)]
        store._embeddings = embeddings
        store._encode = lambda _texts: np.array([query_vec], dtype=np.float32)

        started = time.perf_counter()
        results = store.search("target memory", top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        vector_store_mod._check_st_available = original_check

    first_id = results[0]["metadata"].get("id") if results else None
    schema_ok = all(
        isinstance(item, dict)
        and "text" in item
        and "score" in item
        and "metadata" in item
        for item in results
    )
    return {
        "ok": len(results) == top_k and first_id == target and schema_ok,
        "items": items,
        "dim": dim,
        "top_k": top_k,
        "first_id": first_id,
        "target_id": target,
        "elapsed_ms": round(elapsed_ms, 1),
        "embedding_bytes": int(embeddings.nbytes),
        "schema_ok": schema_ok,
    }


def _case_status(ok: bool) -> str:
    return "ok" if ok else "fail"


def _case_failure(*parts: str | None) -> str:
    for part in parts:
        if part:
            return part
    return ""


def _build_case_report(
    *,
    run_id: str,
    name: str,
    ok: bool,
    input_audio: dict[str, Any] | None = None,
    asr_text: str = "",
    expected_intent: str = "",
    skill_called: str | None = None,
    memory_assertion: dict[str, Any] | None = None,
    tts_playback: dict[str, Any] | None = None,
    failure_reason: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "case_name": name,
        "input_audio": input_audio or {},
        "asr_text": asr_text,
        "expected_intent": expected_intent,
        "skill_called": skill_called,
        "memory_assertion": memory_assertion or {},
        "tts_playback": tts_playback or {},
        "status": _case_status(ok),
        "failure_reason": failure_reason,
    }


def _load_config(config_path: str | None) -> dict[str, Any]:
    if config_path:
        os.environ["ASKME_CONFIG_PATH"] = str(Path(config_path).resolve())
    from askme.config import get_config

    return get_config(reload=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = str(args.run_id or uuid.uuid4().hex[:16])
    cfg = _load_config(args.config)
    voice_cfg = cfg.get("voice", {})

    if args.wav:
        samples, sample_rate = read_wav(Path(args.wav))
        tts_result = {
            "ok": True,
            "samples": samples,
            "sample_rate": sample_rate,
            "backend": "wav",
            "elapsed_ms": 0.0,
            "chunks": 1,
            "error": "",
        }
    else:
        tts_result = generate_tts_audio(args.text, voice_cfg.get("tts", {}))
        samples = tts_result["samples"]
        sample_rate = int(tts_result["sample_rate"])

    if args.output_wav and len(samples) > 0:
        write_wav(Path(args.output_wav), samples, sample_rate)

    asr_result = transcribe_generated_audio(
        samples,
        sample_rate,
        voice_cfg,
        use_cloud=bool(args.cloud),
    ) if len(samples) > 0 else {
        "ok": False,
        "text": "",
        "source": "",
        "is_noise": False,
        "elapsed_ms": 0.0,
        "error": "no generated audio",
    }

    transcript_route = route_skill_text(
        asr_result.get("text", ""),
        expected_skill=args.expected_skill,
    ) if asr_result.get("text") else {
        "ok": False,
        "text": "",
        "intent": "",
        "skill": None,
        "expected_skill": args.expected_skill,
        "trigger_count": 0,
    }
    skill_probe = route_skill_text(args.skill_text, expected_skill=args.expected_skill)
    memory_result = memory_scale_probe(args.memory_items, args.memory_dim, args.memory_top_k)

    audio_report = {
        "ok": bool(tts_result["ok"]) and bool(asr_result["ok"]),
        "text": args.text,
        "tts": {
            key: value for key, value in tts_result.items() if key != "samples"
        },
        "metrics": _audio_metrics(samples, sample_rate),
        "asr": asr_result,
        "output_wav": args.output_wav or "",
    }
    status_ok = (
        audio_report["ok"]
        and skill_probe["ok"]
        and transcript_route["ok"]
        and memory_result["ok"]
    )
    cases = [
        _build_case_report(
            run_id=run_id,
            name="generated_tts_asr",
            ok=audio_report["ok"],
            input_audio={
                "text": args.text,
                "wav": args.output_wav or "",
                "metrics": audio_report["metrics"],
            },
            asr_text=asr_result.get("text", ""),
            expected_intent="transcript",
            tts_playback=audio_report["tts"],
            failure_reason=_case_failure(
                tts_result.get("error", ""),
                asr_result.get("error", ""),
            ),
        ),
        _build_case_report(
            run_id=run_id,
            name="voice_transcript_skill_route",
            ok=transcript_route["ok"],
            asr_text=asr_result.get("text", ""),
            expected_intent=args.expected_skill,
            skill_called=transcript_route.get("skill"),
            failure_reason=(
                ""
                if transcript_route["ok"]
                else f"expected {args.expected_skill}, got {transcript_route.get('skill')}"
            ),
        ),
        _build_case_report(
            run_id=run_id,
            name="voice_skill_probe",
            ok=skill_probe["ok"],
            asr_text=args.skill_text,
            expected_intent=args.expected_skill,
            skill_called=skill_probe.get("skill"),
            failure_reason=(
                ""
                if skill_probe["ok"]
                else f"expected {args.expected_skill}, got {skill_probe.get('skill')}"
            ),
        ),
        _build_case_report(
            run_id=run_id,
            name="memory_scale_probe",
            ok=memory_result["ok"],
            memory_assertion={
                "items": memory_result["items"],
                "top_k": memory_result["top_k"],
                "first_id": memory_result["first_id"],
                "target_id": memory_result["target_id"],
                "schema_ok": memory_result["schema_ok"],
            },
            failure_reason="" if memory_result["ok"] else "memory search assertion failed",
        ),
    ]
    return {
        "run_id": run_id,
        "status": "ok" if status_ok else "fail",
        "cases": cases,
        "generated_voice_loopback": audio_report,
        "transcript_skill_route": transcript_route,
        "skill_probe": skill_probe,
        "memory_scale": memory_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to config.yaml override")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--wav", help="Use an existing PCM16 wav instead of TTS generation")
    parser.add_argument(
        "--output-wav",
        default=str(ROOT / ".tmp" / "generated_voice_loopback.wav"),
        help="Where to write generated/captured audio",
    )
    parser.add_argument("--cloud", action="store_true", help="Use configured Cloud ASR too")
    parser.add_argument("--skill-text", default=DEFAULT_TEXT, help="Recognized text skill probe")
    parser.add_argument("--expected-skill", default=DEFAULT_EXPECTED_SKILL)
    parser.add_argument("--memory-items", type=int, default=5000)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument("--run-id", default="", help="Attach a stable run id to the report")
    parser.add_argument("--json-out", default="", help="Also write the JSON report to this path")
    parser.add_argument("--json", action="store_true", help="Print compact JSON only")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run(args)
    payload = json.dumps(report, ensure_ascii=False, indent=None if args.json else 2)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
