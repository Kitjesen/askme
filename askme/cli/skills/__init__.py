"""Skills CLI commands — inspect loaded skills and generated contracts."""

from __future__ import annotations

from typing import Any

from askme.cli.utils import _emit_payload, _load_skill_manager


def _handle_skills_command(args: Any) -> None:
    """Handle the 'skills' command group: list, show, openapi."""
    manager = _load_skill_manager()

    if args.skills_command == "list":
        payload = {
            "skills": manager.get_contract_catalog(),
            "count": len(manager.get_all()),
        }
        _emit_payload(payload, json_output=args.json)
        return

    if args.skills_command == "show":
        skill = manager.get(args.skill_name)
        contract = manager.get_contract(args.skill_name)
        if skill is None or contract is None:
            raise SystemExit(f"Unknown skill: {args.skill_name}")
        payload = {
            "name": skill.name,
            "enabled": skill.enabled,
            "trigger": skill.trigger,
            "voice_trigger": skill.voice_trigger,
            "source": skill.source,
            "contract": contract.summary(),
            "parameters": [
                {
                    "name": parameter.name,
                    "type": parameter.type,
                    "description": parameter.description,
                    "required": parameter.required,
                    "default": parameter.default,
                    "enum": list(parameter.enum),
                }
                for parameter in contract.parameters
            ],
        }
        _emit_payload(payload, json_output=args.json)
        return

    if args.skills_command == "openapi":
        _emit_payload(manager.openapi_document(), json_output=True)
        return

    raise SystemExit(f"Unknown skills command: {args.skills_command}")
