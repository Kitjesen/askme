from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path("scripts/bench/generated_voice_capability_check.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "generated_voice_capability_check",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_voice_report_includes_case_artifacts(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()

    monkeypatch.setattr(module, "_load_config", lambda _config: {"voice": {"tts": {}}})
    monkeypatch.setattr(
        module,
        "generate_tts_audio",
        lambda _text, _tts_cfg: {
            "ok": True,
            "samples": np.ones(1600, dtype=np.float32) * 0.1,
            "sample_rate": 16000,
            "backend": "fake",
            "elapsed_ms": 1.0,
            "chunks": 1,
            "error": "",
        },
    )
    monkeypatch.setattr(
        module,
        "transcribe_generated_audio",
        lambda *_args, **_kwargs: {
            "ok": True,
            "text": "\u51e0\u70b9\u4e86",
            "source": "fake",
            "is_noise": False,
            "elapsed_ms": 2.0,
            "error": "",
        },
    )
    monkeypatch.setattr(
        module,
        "route_skill_text",
        lambda text, *, expected_skill=None: {
            "ok": True,
            "text": text,
            "intent": "voice_trigger",
            "skill": expected_skill,
            "expected_skill": expected_skill,
            "trigger_count": 1,
        },
    )
    monkeypatch.setattr(
        module,
        "memory_scale_probe",
        lambda items, dim, top_k: {
            "ok": True,
            "items": items,
            "dim": dim,
            "top_k": top_k,
            "first_id": 3,
            "target_id": 3,
            "elapsed_ms": 1.5,
            "embedding_bytes": 1024,
            "schema_ok": True,
        },
    )

    args = argparse.Namespace(
        config=None,
        text="\u51e0\u70b9\u4e86",
        wav="",
        output_wav=str(tmp_path / "generated.wav"),
        cloud=False,
        skill_text="\u51e0\u70b9\u4e86",
        expected_skill="get_time",
        memory_items=8,
        memory_dim=4,
        memory_top_k=1,
        run_id="run-test",
    )

    report = module.run(args)

    assert report["status"] == "ok"
    assert report["run_id"] == "run-test"
    assert [case["case_name"] for case in report["cases"]] == [
        "generated_tts_asr",
        "voice_transcript_skill_route",
        "voice_skill_probe",
        "memory_scale_probe",
    ]
    assert report["cases"][0]["input_audio"]["wav"] == str(tmp_path / "generated.wav")
    assert report["cases"][0]["asr_text"] == "\u51e0\u70b9\u4e86"
    assert report["cases"][1]["skill_called"] == "get_time"
    assert report["cases"][3]["memory_assertion"]["schema_ok"] is True


def test_generated_voice_main_writes_json_out(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    out_path = tmp_path / "report.json"

    monkeypatch.setattr(
        module,
        "run",
        lambda _args: {
            "run_id": "run-test",
            "status": "ok",
            "cases": [],
        },
    )

    assert module.main(["--json", "--json-out", str(out_path)]) == 0
    assert '"run-test"' in out_path.read_text(encoding="utf-8")
