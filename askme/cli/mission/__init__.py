"""Mission CLI commands — draft, run, and report industrial inspection missions."""

from __future__ import annotations

from typing import Any

from askme.cli.utils import (
    _cli_root_override,
    _emit_payload,
    _get_json,
    _load_local_mission_service,
    _load_mission_source,
    _normalise_server_url,
    _post_json,
)


def _handle_mission_command(args: Any) -> None:
    """Handle the 'mission' command group: draft, run, report."""
    if args.mission_command == "draft":
        draft_mission = _cli_root_override("_draft_mission_sync", _draft_mission_sync)
        payload = draft_mission(
            " ".join(args.text),
            operator_id=args.operator_id,
            robot_id=args.robot_id,
            site_id=args.site_id,
            server=args.server,
        )
        _emit_payload(payload, json_output=args.json)
        return

    if args.mission_command == "run":
        run_mission = _cli_root_override("_run_mission_sync", _run_mission_sync)
        payload = run_mission(
            " ".join(args.source),
            dry_run=(not args.submit) or args.dry_run,
            confirmed=args.confirm,
            operator_id=args.operator_id,
            robot_id=args.robot_id,
            site_id=args.site_id,
            server=args.server,
        )
        _emit_payload(payload, json_output=args.json)
        return

    if args.mission_command == "report":
        mission_report = _cli_root_override("_mission_report_sync", _mission_report_sync)
        payload = mission_report(args.mission_id, server=args.server)
        _emit_payload(payload, json_output=args.json)
        return

    raise SystemExit(f"Unknown mission command: {args.mission_command}")


def _draft_mission_sync(
    text: str,
    *,
    operator_id: str = "",
    robot_id: str = "",
    site_id: str = "",
    server: str = "",
) -> dict[str, Any]:
    """Draft a high-level mission plan from operator text."""
    payload = _mission_context_payload(
        {"text": text, "channel": "cli"},
        operator_id=operator_id,
        robot_id=robot_id,
        site_id=site_id,
    )
    if server:
        post_json = _cli_root_override("_post_json", _post_json)
        return post_json(f"{_normalise_server_url(server)}/api/missions/draft", payload)

    load_service = _cli_root_override(
        "_load_local_mission_service", _load_local_mission_service
    )
    service = load_service()
    return service.draft_from_payload(payload)


def _run_mission_sync(
    source: str,
    *,
    dry_run: bool,
    confirmed: bool,
    operator_id: str = "",
    robot_id: str = "",
    site_id: str = "",
    server: str = "",
) -> dict[str, Any]:
    """Dry-run or submit a mission plan through the runtime arbiter."""
    payload = _load_mission_source(source)
    payload = _mission_context_payload(
        payload,
        operator_id=operator_id,
        robot_id=robot_id,
        site_id=site_id,
    )
    payload.setdefault("channel", "cli")
    payload["dry_run"] = dry_run
    payload["confirmed"] = confirmed

    if server:
        post_json = _cli_root_override("_post_json", _post_json)
        return post_json(f"{_normalise_server_url(server)}/api/missions", payload)

    load_service = _cli_root_override(
        "_load_local_mission_service", _load_local_mission_service
    )
    service = load_service()
    return service.submit_from_payload(payload, trusted_confirmation=True)


def _mission_report_sync(mission_id: str, *, server: str = "") -> dict[str, Any]:
    """Build an inspection report shell for a mission."""
    if server:
        get_json = _cli_root_override("_get_json", _get_json)
        return get_json(
            f"{_normalise_server_url(server)}/api/missions/{mission_id}/report"
        )

    load_service = _cli_root_override(
        "_load_local_mission_service", _load_local_mission_service
    )
    service = load_service()
    return service.report_payload(mission_id)


def _mission_context_payload(
    payload: dict[str, Any],
    *,
    operator_id: str = "",
    robot_id: str = "",
    site_id: str = "",
) -> dict[str, Any]:
    """Augment a payload with optional operator/robot/site context."""
    result = dict(payload)
    if operator_id:
        result["operator_id"] = operator_id
    if robot_id:
        result["robot_id"] = robot_id
    if site_id:
        result["site_id"] = site_id
    return result
