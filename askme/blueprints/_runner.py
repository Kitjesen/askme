"""Shared blueprint runner — eliminates __main__ boilerplate.

Usage in a blueprint module::

    if __name__ == "__main__":
        from askme.blueprints._runner import run_blueprint
        run_blueprint(voice, "Voice")
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
    """Build, start, wait-for-signal, and stop a blueprint.

    Handles SIGINT/SIGTERM on Unix and falls back to KeyboardInterrupt
    on Windows where ``add_signal_handler`` is not implemented.
    """
    # Basic logging so module build/start messages are visible
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
                pass  # Windows — falls back to KeyboardInterrupt

        await app.start()
        print(f"{label} running — {len(app.modules)} modules", flush=True)
        try:
            await stop.wait()
        except asyncio.CancelledError:
            pass
        await app.stop()
        print(f"{label} stopped.", flush=True)

    asyncio.run(_main())
