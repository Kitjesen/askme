"""Run the voice runtime on a selectable local Dashboard port."""

from __future__ import annotations

import argparse
import asyncio

import askme.interfaces.register_defaults  # noqa: F401
from askme.blueprints import load_runtime_blueprint_for_modes
from askme.config import get_config


async def _run(port: int) -> None:
    cfg = get_config(reload=True)
    cfg.setdefault("health_server", {})["port"] = int(port)
    blueprint = load_runtime_blueprint_for_modes(voice_mode=True, robot_mode=False)
    runtime = await blueprint.build(cfg)
    await runtime.start()
    try:
        await asyncio.Event().wait()
    finally:
        await runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    asyncio.run(_run(args.port))


if __name__ == "__main__":
    main()
