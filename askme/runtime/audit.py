"""Persistent JSONL audit helper for runtime handoff records."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from askme.runtime.handoff import RuntimeEvent, TaskRun

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeAuditConfig:
    """Opt-in config for persistent runtime audit logging."""

    enabled: bool = False
    path: str | Path | None = None
    swallow_errors: bool = True

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> RuntimeAuditConfig:
        if not isinstance(config, dict):
            return cls()
        return cls(
            enabled=bool(config.get("enabled", False)),
            path=config.get("path") or config.get("jsonl_path"),
            swallow_errors=bool(config.get("swallow_errors", True)),
        )


class RuntimeAuditLog:
    """Append runtime handoff audit records to JSONL when explicitly enabled."""

    def __init__(self, config: RuntimeAuditConfig | dict[str, Any] | None = None) -> None:
        if isinstance(config, RuntimeAuditConfig):
            self.config = config
        else:
            self.config = RuntimeAuditConfig.from_mapping(config)
        self.path = Path(self.config.path).expanduser() if self.config.path else None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.path is not None)

    def append_event(self, event: RuntimeEvent) -> None:
        self.append_record(
            {
                "kind": "runtime_event",
                "created_at": time.time(),
                "event": event.to_dict(),
            }
        )

    def append_terminal_snapshot(self, run: TaskRun) -> None:
        if not run.terminal:
            return
        self.append_record(
            {
                "kind": "task_run_terminal_snapshot",
                "created_at": time.time(),
                "run": run.to_dict(),
                "report": run.report,
            }
        )

    def append_operator_action(self, run: TaskRun, action: dict[str, Any]) -> None:
        self.append_record(
            {
                "kind": "operator_action",
                "created_at": time.time(),
                "run_id": run.run_id,
                "handoff_id": run.handoff.handoff_id,
                "plan_id": run.handoff.plan_id,
                "profile": run.profile,
                "state": run.current_state,
                "action": dict(action),
            }
        )

    def append_record(self, record: dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.write("\n")
        except OSError:
            if not self.config.swallow_errors:
                raise
            logger.exception("Runtime audit append failed for %s", self.path)

