"""Install or verify the official sherpa-onnx Chinese KWS model.

Installation requires outbound HTTPS access to GitHub Releases. ``--check``
is completely offline and is the preferred deployment-health command.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

MODEL_DIRECTORY_NAME = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
OFFICIAL_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    f"{MODEL_DIRECTORY_NAME}.tar.bz2"
)
MODEL_ARCHIVE_SHA256 = (
    "b2f7c89690dc8ce4c6ed6afeab7cd800c36ad1421fb6b6302b4a4b194cf7f35f"
)
DEFAULT_TARGET = Path("models") / "kws" / MODEL_DIRECTORY_NAME
REQUIRED_MODEL_FILES: tuple[str, ...] = (
    "encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
    "decoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
    "joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
    "tokens.txt",
)


class ModelInstallError(RuntimeError):
    """Raised when a model cannot be installed without risking deployment state."""


def model_problems(target: Path) -> list[str]:
    """Return observable validation failures for a deployed model directory."""

    target = Path(target)
    if not target.is_dir():
        return [f"model directory is missing: {target}"]
    problems: list[str] = []
    for relative_path in REQUIRED_MODEL_FILES:
        path = target / relative_path
        if not path.is_file():
            problems.append(f"required file is missing: {relative_path}")
        elif path.stat().st_size <= 0:
            problems.append(f"required file is empty: {relative_path}")
    return problems


def verify_model(target: Path) -> None:
    """Raise ``ModelInstallError`` unless the model is structurally complete."""

    problems = model_problems(target)
    if problems:
        raise ModelInstallError("; ".join(problems))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(destination: Path) -> None:
    request = urllib.request.Request(
        OFFICIAL_MODEL_URL,
        headers={"User-Agent": "askme-kws-installer/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def _safe_archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        raise ModelInstallError("model archive is empty")
    for member in members:
        relative = PurePosixPath(member.name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != MODEL_DIRECTORY_NAME
        ):
            raise ModelInstallError(f"unsafe archive path: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ModelInstallError(f"unsupported archive entry: {member.name}")
    return members


def install_model(
    target: Path,
    *,
    archive_path: Path | None = None,
    expected_sha256: str = MODEL_ARCHIVE_SHA256,
) -> Path:
    """Install the verified model atomically, or reuse an existing valid model."""

    target = Path(target).expanduser().resolve()
    if not model_problems(target):
        return target
    if target.exists():
        raise ModelInstallError(
            f"refusing to replace incomplete existing path: {target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=target.parent,
        prefix=f".{MODEL_DIRECTORY_NAME}.install-",
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        if archive_path is None:
            archive = temporary / f"{MODEL_DIRECTORY_NAME}.tar.bz2"
            _download_archive(archive)
        else:
            archive = Path(archive_path).expanduser().resolve()
            if not archive.is_file():
                raise ModelInstallError(f"archive does not exist: {archive}")

        actual_sha256 = _sha256(archive)
        if actual_sha256.lower() != expected_sha256.lower():
            raise ModelInstallError(
                "model archive checksum mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        extraction_root = temporary / "extracted"
        extraction_root.mkdir()
        try:
            with tarfile.open(archive, mode="r:bz2") as model_archive:
                members = _safe_archive_members(model_archive)
                model_archive.extractall(extraction_root, members=members)
        except (OSError, tarfile.TarError) as exc:
            raise ModelInstallError(f"invalid model archive: {exc}") from exc

        extracted_model = extraction_root / MODEL_DIRECTORY_NAME
        verify_model(extracted_model)
        try:
            os.replace(extracted_model, target)
        except OSError as exc:
            raise ModelInstallError(f"atomic model install failed: {exc}") from exc

    verify_model(target)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install or verify AskMe's official sherpa-onnx KWS model.",
        epilog=(
            "Installation downloads from GitHub Releases and requires outbound HTTPS; "
            "--check never uses the network."
        ),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"model directory (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the deployed model without downloading or changing files",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="install from an existing official archive instead of downloading",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    target = args.target.expanduser().resolve()
    try:
        if args.check:
            verify_model(target)
            print(f"KWS model check passed: {target}")
            return 0
        installed = install_model(target, archive_path=args.archive)
        print(f"KWS model ready: {installed}")
        return 0
    except ModelInstallError as exc:
        print(f"KWS model error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
