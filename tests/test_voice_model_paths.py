"""Regression tests for cwd-independent voice model loading."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from askme import config as config_module
from askme.config import get_config, project_root
from askme.voice.input import vad as vad_module


def _configured_local_paths(node, *, path_map: bool = False):
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if isinstance(value, dict):
            yield from _configured_local_paths(value, path_map=key == "paths")
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        if (
            path_map
            or key in {"path", "file", "dir"}
            or key.endswith(("_path", "_file", "_dir"))
        ):
            yield value


def test_local_config_paths_are_resolved_from_project_root(
    monkeypatch,
) -> None:
    monkeypatch.chdir(project_root().parent)

    config = get_config(reload=True)
    root = project_root().resolve()

    for configured_path in _configured_local_paths(config):
        path = Path(configured_path)
        assert path.is_absolute()
        assert path.is_relative_to(root)


def test_absolute_config_path_is_preserved(tmp_path) -> None:
    absolute_model = tmp_path / "custom-silero-vad.onnx"
    config = {"voice": {"vad": {"model_path": str(absolute_model)}}}

    config_module._apply_project_relative_paths(config)

    assert config["voice"]["vad"]["model_path"] == str(absolute_model.resolve())


def test_vad_engine_honours_model_path_config(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeVadModelConfig:
        def __init__(self) -> None:
            self.silero_vad = SimpleNamespace(
                model="",
                threshold=0.0,
                min_silence_duration=0.0,
                min_speech_duration=0.0,
            )
            self.sample_rate = 0

    def fake_detector(config, *, buffer_size_in_seconds):
        captured["model"] = config.silero_vad.model
        captured["buffer_size_in_seconds"] = buffer_size_in_seconds
        return SimpleNamespace()

    monkeypatch.setattr(vad_module.sherpa_onnx, "VadModelConfig", FakeVadModelConfig)
    monkeypatch.setattr(
        vad_module.sherpa_onnx,
        "VoiceActivityDetector",
        fake_detector,
    )

    monkeypatch.chdir(project_root().parent)
    voice = get_config(reload=True)["voice"]
    vad_module.VADEngine(voice["vad"])

    expected_model = project_root() / "models" / "vad" / "silero_vad.onnx"
    assert Path(str(captured["model"])).resolve() == expected_model.resolve()
    assert captured["buffer_size_in_seconds"] == 30
