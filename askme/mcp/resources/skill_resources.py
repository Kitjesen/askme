"""MCP resources for skills catalog and application configuration."""

from __future__ import annotations

import json
import logging

from askme.mcp.registration import mcp
from askme.mcp.resource_surface import get_resource_surface

logger = logging.getLogger(__name__)


@mcp.resource("askme://skills")
def skills_catalog() -> str:
    """Catalog of all available skills with generated contract metadata."""
    return json.dumps(get_resource_surface().skills_catalog_payload(), ensure_ascii=False)


@mcp.resource("askme://skills/openapi")
def skills_openapi() -> str:
    """OpenAPI document generated from the loaded skill contracts."""
    return json.dumps(get_resource_surface().skills_openapi_payload(), ensure_ascii=False)


@mcp.resource("askme://config")
def askme_config() -> str:
    """Current askme configuration (sanitised — API keys removed)."""
    return json.dumps(
        get_resource_surface().sanitized_config_payload(),
        ensure_ascii=False,
        default=str,
    )
