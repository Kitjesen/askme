"""Tests for ImageArchive — filesystem-backed capture store with JSON sidecars."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from askme.perception.image_archive import ImageArchive


def _make_archive(tmp_path: Path) -> ImageArchive:
    return ImageArchive(captures_dir=str(tmp_path / "captures"))


class TestSave:
    def test_returns_metadata_dict(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"\xff\xd8\xff", label="anomaly")
        assert isinstance(meta, dict)
        assert "id" in meta
        assert "timestamp" in meta

    def test_jpeg_file_written(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"\xff\xd8\xff", label="test")
        jpg = Path(meta["file_path"])
        assert jpg.exists()
        assert jpg.read_bytes() == b"\xff\xd8\xff"

    def test_json_sidecar_written(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"img", label="patrol")
        sidecar = Path(meta["file_path"]).with_suffix(".json")
        assert sidecar.exists()
        loaded = json.loads(sidecar.read_text())
        assert loaded["label"] == "patrol"

    def test_label_in_capture_id(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"img", label="door_check")
        assert "door_check" in meta["id"]

    def test_special_chars_in_label_sanitised(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"img", label="room 3/A")
        # slashes and spaces replaced with underscore
        assert "/" not in meta["id"]
        assert " " not in meta["id"]

    def test_description_stored(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"img", label="scan", description="机器人拍摄的图像")
        assert meta["description"] == "机器人拍摄的图像"

    def test_width_height_stored(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"img", label="scan", width=640, height=480)
        assert meta["width"] == 640
        assert meta["height"] == 480

    def test_file_size_bytes_correct(self, tmp_path):
        arch = _make_archive(tmp_path)
        data = b"x" * 100
        meta = arch.save(data, label="test")
        assert meta["file_size_bytes"] == 100

    def test_creates_day_subdirectory(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"img", label="test")
        # Parent is a YYYYMMDD directory
        day_dir = Path(meta["file_path"]).parent
        assert day_dir.name.isdigit()
        assert len(day_dir.name) == 8


class TestListCaptures:
    def test_empty_dir_returns_empty(self, tmp_path):
        arch = _make_archive(tmp_path)
        assert arch.list_captures() == []

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        arch = ImageArchive(captures_dir=str(tmp_path / "nonexistent"))
        assert arch.list_captures() == []

    def test_lists_saved_captures(self, tmp_path):
        arch = _make_archive(tmp_path)
        arch.save(b"img1", label="a")
        arch.save(b"img2", label="b")
        captures = arch.list_captures()
        assert len(captures) == 2

    def test_does_not_include_image_base64(self, tmp_path):
        arch = _make_archive(tmp_path)
        arch.save(b"img", label="test")
        captures = arch.list_captures()
        assert "image_base64" not in captures[0]

    def test_limit_respected(self, tmp_path):
        arch = _make_archive(tmp_path)
        for i in range(5):
            arch.save(b"img", label=f"cap{i}")
        captures = arch.list_captures(limit=3)
        assert len(captures) == 3

    def test_label_filter(self, tmp_path):
        arch = _make_archive(tmp_path)
        arch.save(b"img", label="anomaly")
        arch.save(b"img", label="patrol")
        captures = arch.list_captures(label_filter="anomaly")
        assert len(captures) == 1
        assert captures[0]["label"] == "anomaly"

    def test_bad_sidecar_skipped(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"img", label="test")
        # Corrupt the sidecar
        Path(meta["file_path"]).with_suffix(".json").write_text("bad json")
        captures = arch.list_captures()
        assert captures == []


class TestGetCapture:
    def test_returns_metadata_with_image_base64(self, tmp_path):
        arch = _make_archive(tmp_path)
        data = b"\xff\xd8\xff\xe0"
        meta = arch.save(data, label="test")
        result = arch.get_capture(meta["id"])
        assert result is not None
        assert "image_base64" in result
        assert base64.b64decode(result["image_base64"]) == data

    def test_returns_none_for_unknown_id(self, tmp_path):
        arch = _make_archive(tmp_path)
        assert arch.get_capture("20260101_120000_nonexistent") is None

    def test_returns_none_when_jpeg_missing(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"img", label="test")
        Path(meta["file_path"]).unlink()  # remove JPEG
        result = arch.get_capture(meta["id"])
        assert result is None

    def test_capture_id_roundtrip(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"data", label="roundtrip", description="test")
        result = arch.get_capture(meta["id"])
        assert result["label"] == "roundtrip"
        assert result["description"] == "test"


class TestDeleteCapture:
    def test_deletes_both_files(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"img", label="test")
        jpg = Path(meta["file_path"])
        sidecar = jpg.with_suffix(".json")
        assert arch.delete_capture(meta["id"]) is True
        assert not jpg.exists()
        assert not sidecar.exists()

    def test_returns_false_for_unknown_id(self, tmp_path):
        arch = _make_archive(tmp_path)
        assert arch.delete_capture("20260101_120000_ghost") is False

    def test_deleted_capture_not_in_list(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"img", label="to_delete")
        arch.delete_capture(meta["id"])
        captures = arch.list_captures()
        assert all(c["id"] != meta["id"] for c in captures)

    def test_get_capture_returns_none_after_delete(self, tmp_path):
        arch = _make_archive(tmp_path)
        meta = arch.save(b"img", label="test")
        arch.delete_capture(meta["id"])
        assert arch.get_capture(meta["id"]) is None
