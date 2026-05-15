"""Run the AskMe dashboard/API without opening microphone or speaker devices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from askme.health_server import build_health_snapshot, create_health_app


def _health_snapshot() -> dict:
    return build_health_snapshot(
        app_name="askme",
        app_version="4.1.0-dev",
        model_name="MiniMax-M2.7-highspeed",
        metrics_snapshot={
            "uptime_seconds": 0.0,
            "conversation_count": 0,
            "llm": {},
            "voice_pipeline": {},
        },
        active_skills=[],
        voice_status={
            "mode": "text",
            "enabled": False,
            "pipeline_ok": True,
            "interaction": {
                "state": "dashboard_only",
                "can_talk": False,
                "hint": "microphone_not_started",
            },
        },
        ota_status={"enabled": False},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args()

    app = create_health_app(_health_snapshot)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
