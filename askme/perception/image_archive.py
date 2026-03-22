"""Image archive: persist camera snapshots with JSON sidecar metadata.

Each capture is stored as::

    {captures_dir}/{YYYYMMDD}/{capture_id}.jpg
    {captures_dir}/{YYYYMMDD}/{capture_id}.json

The JSON sidecar contains all searchable metadata; the JPEG contains the
raw image bytes.  No in-memory index is maintained — the filesystem is the
source of truth.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CAPTURES_DIR = "data/captures"


class ImageArchive:
    """Filesystem-backed image archive with JSON sidecar per capture.

    Thread-safe for concurrent reads; writes are serialised via asyncio
    (caller is responsible for not issuing concurrent saves for the same
    capture_id, which is timestamp-derived and therefore unique in practice).
    """

    def __init__(self, captures_dir: str = _DEFAULT_CAPTURES_DIR) -> None:
        self._captures_dir = Path(captures_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        image_bytes: bytes,
        label: str,
        description: str = "",
        width: int = 0,
        height: int = 0,
    ) -> dict[str, Any]:
        """Save *image_bytes* to disk and write a JSON sidecar.

        Returns the metadata dict (same shape as ``get_capture`` but without
        ``image_base64``).

        Raises ``OSError`` on disk failure (caller should handle).
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y%m%d")
        ts_str = now.strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        capture_id = f"{ts_str}_{safe_label}"

        day_dir = self._captures_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        jpg_path = day_dir / f"{capture_id}.jpg"
        json_path = day_dir / f"{capture_id}.json"

        # Write JPEG
        jpg_path.write_bytes(image_bytes)
        file_size = len(image_bytes)

        # Build metadata
        metadata: dict[str, Any] = {
            "id": capture_id,
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "label": label,
            "description": description,
            "width": width,
            "height": height,
            "file_path": str(jpg_path).replace("\\", "/"),
            "file_size_bytes": file_size,
        }

        # Write JSON sidecar (UTF-8, pretty-printed for human readability)
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("[ImageArchive] Saved capture: %s (%d bytes)", capture_id, file_size)
        return metadata

    def list_captures(
        self,
        limit: int = 50,
        label_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return metadata for recent captures, newest first.

        Does not include ``image_base64`` — call ``get_capture()`` for that.
        """
        captures: list[dict[str, Any]] = []

        if not self._captures_dir.exists():
            return captures

        # Collect all JSON sidecar files, sorted by name descending (newest first)
        json_files: list[Path] = []
        try:
            for day_dir in sorted(self._captures_dir.iterdir(), reverse=True):
                if not day_dir.is_dir():
                    continue
                for json_path in sorted(day_dir.glob("*.json"), reverse=True):
                    json_files.append(json_path)
        except OSError as exc:
            logger.warning("[ImageArchive] list_captures scan error: %s", exc)
            return captures

        for json_path in json_files:
            if len(captures) >= limit:
                break
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.debug("[ImageArchive] Skipping bad sidecar %s: %s", json_path, exc)
                continue

            if label_filter is not None and meta.get("label") != label_filter:
                continue

            captures.append(meta)

        return captures

    def get_capture(self, capture_id: str) -> dict[str, Any] | None:
        """Return metadata + ``image_base64`` for *capture_id*, or ``None``."""
        json_path = self._find_sidecar(capture_id)
        if json_path is None:
            return None

        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[ImageArchive] Failed to read sidecar %s: %s", json_path, exc)
            return None

        jpg_path = json_path.with_suffix(".jpg")
        if not jpg_path.exists():
            logger.warning("[ImageArchive] JPEG missing for capture %s", capture_id)
            return None

        try:
            image_b64 = base64.b64encode(jpg_path.read_bytes()).decode("ascii")
        except OSError as exc:
            logger.warning("[ImageArchive] Failed to read JPEG %s: %s", jpg_path, exc)
            return None

        result = dict(meta)
        result["image_base64"] = image_b64
        return result

    def delete_capture(self, capture_id: str) -> bool:
        """Delete the JPEG and JSON sidecar for *capture_id*.

        Returns ``True`` if at least the JSON was deleted, ``False`` if not found.
        """
        json_path = self._find_sidecar(capture_id)
        if json_path is None:
            return False

        deleted = False
        jpg_path = json_path.with_suffix(".jpg")

        for p in (jpg_path, json_path):
            try:
                if p.exists():
                    p.unlink()
                    deleted = True
            except OSError as exc:
                logger.warning("[ImageArchive] Failed to delete %s: %s", p, exc)

        if deleted:
            logger.info("[ImageArchive] Deleted capture: %s", capture_id)
        return deleted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_sidecar(self, capture_id: str) -> Path | None:
        """Locate the JSON sidecar for *capture_id* by scanning day directories."""
        if not self._captures_dir.exists():
            return None

        # capture_id is "{YYYYMMDD}_{HHMMSS}_{label}" — extract date prefix for fast lookup
        date_prefix = capture_id[:8]  # "YYYYMMDD"
        candidate_dirs: list[Path] = []

        # Try exact date directory first, then fall back to full scan
        exact_dir = self._captures_dir / date_prefix
        if exact_dir.is_dir():
            candidate_dirs.append(exact_dir)
        else:
            try:
                candidate_dirs = [
                    d for d in self._captures_dir.iterdir() if d.is_dir()
                ]
            except OSError:
                return None

        for day_dir in candidate_dirs:
            json_path = day_dir / f"{capture_id}.json"
            if json_path.exists():
                return json_path

        return None
