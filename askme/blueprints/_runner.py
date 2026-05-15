"""Shared runtime runner for blueprint modules.

Each blueprint can expose a direct module entrypoint without repeating the
build/start/signal/stop boilerplate.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askme.runtime.module import Runtime


def run_blueprint(blueprint: Runtime, label: str = "Runtime") -> None:
    """Build, start, wait for SIGINT/SIGTERM, and stop a blueprint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    async def _main() -> None:
        from askme.config import get_config

        cfg = get_config()
        app = await blueprint.build(cfg)

        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_running_loop().add_signal_handler(sig, stop.set)
            except NotImplementedError:
                pass

        await app.start()
        print(f"{label} running - {len(app.modules)} modules", flush=True)
        try:
            await stop.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await app.stop()
            print(f"{label} stopped.", flush=True)

    asyncio.run(_main())
