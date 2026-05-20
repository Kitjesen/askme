"""Health, trace, status, and dashboard-page response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthSnapshotResponse(BaseModel):
    """Runtime health snapshot returned by /health and /healthz."""

    model_config = ConfigDict(extra="allow")

    status: str = ""
    service: str = ""
    version: str = ""
    uptime_seconds: float | int = 0
    degraded_reasons: list[str] = Field(default_factory=list)
    voice_pipeline_status: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class TraceSnapshotResponse(BaseModel):
    """Recent pipeline timing trace summary."""

    model_config = ConfigDict(extra="allow")

    summary: dict[str, Any] = Field(default_factory=dict)
    recent: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""


class SystemStatusResponse(BaseModel):
    """Dashboard status payload for perception, camera, and memory probes."""

    model_config = ConfigDict(extra="allow")

    timestamp: float | int = 0
    perception: dict[str, Any] = Field(default_factory=dict)
    orbbec_camera: bool = False
    memory: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class DashboardPageRegistryResponse(BaseModel):
    """Product dashboard page registry and page ownership policy."""

    model_config = ConfigDict(extra="allow")

    ok: bool = True
    pages: list[dict[str, Any]] = Field(default_factory=list)
    sections: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
