from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

from scripts.dev import install_kws_model


def _write_valid_model(directory: Path) -> None:
    directory.mkdir(parents=True)
    for relative_path in install_kws_model.REQUIRED_MODEL_FILES:
        path = directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model-data")


def test_check_mode_verifies_existing_model_without_network(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    target = tmp_path / install_kws_model.MODEL_DIRECTORY_NAME
    _write_valid_model(target)

    def _unexpected_download(*args, **kwargs):
        raise AssertionError("check mode must not access the network")

    monkeypatch.setattr(install_kws_model, "_download_archive", _unexpected_download)

    result = install_kws_model.main(["--check", "--target", str(target)])

    assert result == 0
    output = capsys.readouterr().out
    assert "KWS model check passed" in output
    assert str(target) in output


def test_install_from_verified_archive_publishes_complete_model_atomically(
    tmp_path: Path,
) -> None:
    source = tmp_path / "archive-source" / install_kws_model.MODEL_DIRECTORY_NAME
    _write_valid_model(source)
    archive = tmp_path / "model.tar.bz2"
    with tarfile.open(archive, mode="w:bz2") as model_archive:
        model_archive.add(source, arcname=install_kws_model.MODEL_DIRECTORY_NAME)
    expected_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    target = tmp_path / "deployed" / install_kws_model.MODEL_DIRECTORY_NAME

    result = install_kws_model.install_model(
        target,
        archive_path=archive,
        expected_sha256=expected_sha256,
    )

    assert result == target.resolve()
    assert install_kws_model.model_problems(target) == []
    assert archive.is_file()
    assert not any(
        path.name.startswith(
            f".{install_kws_model.MODEL_DIRECTORY_NAME}.install-"
        )
        for path in target.parent.iterdir()
    )
