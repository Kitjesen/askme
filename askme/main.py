"""Legacy askme runtime entrypoints."""

from __future__ import annotations

import asyncio

from askme.app import AskmeApp
from askme.health_server import AskmeHealthServer


async def run_legacy_app(*, voice_mode: bool, robot_mode: bool) -> None:
    """Run the legacy askme runtime with the health HTTP server attached."""
    app = AskmeApp(
        voice_mode=voice_mode,
        robot_mode=robot_mode,
    )
    health_server = AskmeHealthServer(
        app.cfg.get("health_server", {}),
        health_provider=app.health_snapshot,
        metrics_provider=app.metrics_snapshot,
    )
    health_task: asyncio.Task[None] | None = None

    try:
        if health_server.enabled:
            health_task = asyncio.create_task(
                health_server.serve(),
                name="askme-health-http",
            )
            await health_server.wait_started(health_task)

        await app.run()
    finally:
        if health_task is not None:
            await health_server.stop()
            await asyncio.gather(health_task, return_exceptions=True)
