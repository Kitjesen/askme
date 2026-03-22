"""MCP resources for skills catalog and application configuration."""

from __future__ import annotations

import json
import logging

from askme.mcp_server import mcp

logger = logging.getLogger(__name__)


@mcp.resource("askme://skills")
def skills_catalog() -> str:
    """Catalog of all available skills with generated contract metadata."""
    from askme.skills.skill_manager import SkillManager

    mgr = SkillManager()
    mgr.load()
    skills = mgr.get_contract_catalog()
    return json.dumps({"skills": skills, "count": len(skills)}, ensure_ascii=False)


@mcp.resource("askme://skills/openapi")
def skills_openapi() -> str:
    """OpenAPI document generated from the loaded skill contracts."""
    from askme.skills.skill_manager import SkillManager

    mgr = SkillManager()
    mgr.load()
    return json.dumps(mgr.openapi_document(), ensure_ascii=False)


@mcp.resource("askme://config")
def askme_config() -> str:
    """Current askme configuration (sanitised — API keys removed)."""
    from askme.config import get_config

    cfg = get_config()
    sanitised: dict = {}

    for section_name, section_val in cfg.items():
        if isinstance(section_val, dict):
            sanitised[section_name] = {
                k: v
                for k, v in section_val.items()
                if "key" not in k.lower() and "secret" not in k.lower()
            }
        else:
            sanitised[section_name] = section_val

    return json.dumps(sanitised, ensure_ascii=False, default=str)
