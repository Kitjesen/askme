"""Vision capture, analysis, and archive API response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VisionSnapshotResponse(BaseModel):
    """Camera snapshot response with optional archive capture id."""

    model_config = ConfigDict(extra="allow")

    image_base64: str = ""
    width: int = 0
    height: int = 0
    timestamp: str = ""
    capture_id: str = ""
    error: str = ""


class VisionAnalyzeResponse(BaseModel):
    """Visual-language image analysis response."""

    model_config = ConfigDict(extra="allow")

    description: str = ""
    error: str = ""


class VisionCaptureListResponse(BaseModel):
    """Archived capture metadata list without required image bytes."""

    model_config = ConfigDict(extra="allow")

    captures: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class VisionCaptureDetailResponse(BaseModel):
    """Single archived capture detail, usually including image_base64."""

    model_config = ConfigDict(extra="allow")

    id: str = ""
    capture_id: str = ""
    image_base64: str = ""
    label: str = ""
    description: str = ""
    width: int = 0
    height: int = 0
    timestamp: str = ""
    error: str = ""


class VisionCaptureDeleteResponse(BaseModel):
    """Delete result for one archived capture."""

    model_config = ConfigDict(extra="allow")

    deleted: bool = False
    capture_id: str = ""
    error: str = ""
