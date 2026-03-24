"""Shared blueprint runner — eliminates __main__ boilerplate.

Usage in a blueprint module::

    if __name__ == "__main__":
        from askme.blueprints._runner import run_blueprint
        run_blueprint(voice, "Voice")
"""

from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askme.runtime.module import Runtime


def run_blueprint(blueprint: Runtime, label: str = "Runtime") -> None:
    """Build, start, wait-for-signal, and stop a blueprint.

    Handles SIGINT/SIGTERM on Unix and falls back to KeyboardInterrupt
    on Windows where ``add_signal_handler`` is not implemented.
    """

    async def _main() -> None:
        from askme.config import load_config

        cfg = load_config()
        app = await blueprint.build(cfg)

        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_running_loop().add_signal_handler(sig, stop.set)
            except NotImplementedError:
                pass  # Windows — falls back to KeyboardInterrupt

        await app.start()
        print(f"{label} running — {len(app.modules)} modules")
        await stop.wait()
        await app.stop()

    asyncio.run(_main())
