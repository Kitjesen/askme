"""Shared runtime runner for blueprint modules."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askme.runtime.core.module import Runtime

logger = logging.getLogger(__name__)


def run_blueprint(blueprint: Runtime, label: str = "Runtime") -> None:
    """Build, start, wait for SIGINT/SIGTERM, and stop a blueprint."""
    args = set(sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if any(arg in {"-h", "--help"} for arg in args):
        logger.info(
            f"{label}\n\n"
            "用途：启动一个 askme 产品蓝图，用于演示、实验或试点验证。\n"
            "Purpose: start one askme product runtime blueprint for demo, lab, "
            "or pilot validation.\n\n"
            "启动命令：\n"
            "Start:\n"
            "  python -m askme.blueprints.presets.<name>\n\n"
            "Preflight without opening microphone, camera, or robot services:\n"
            "  python -m askme.blueprints.presets.<name> --preflight --json\n\n"
            "Arguments:\n"
            "  -h, --help        Show this help and exit.\n"
            "  --preflight       Validate module composition without starting runtime IO.\n"
            "  --json            Print preflight output as JSON.\n\n"
            "交付检查：\n"
            "Catalog and delivery gates:\n"
            "  python -m askme runtime blueprints --help",
        )
        return

    if "--preflight" in args:
        payload = _preflight_payload(blueprint, label)
        if "--json" in args:
            logger.info(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            logger.info(
                f"{payload['label']} preflight ok: "
                f"{payload['module_count']} modules "
                f"({', '.join(payload['modules'])})",
            )
        return

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
        logger.info(f"{label} started with {len(app.modules)} modules.")
        try:
            await stop.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await app.stop()
            logger.info(f"{label} stopped.")

    asyncio.run(_main())


def _preflight_payload(blueprint: Runtime, label: str) -> dict[str, object]:
    """Return a no-IO blueprint summary for customer delivery checks."""
    module_names = list(blueprint.module_names())
    duplicates = sorted({name for name in module_names if module_names.count(name) > 1})
    return {
        "ok": not duplicates,
        "label": label,
        "module_count": len(module_names),
        "modules": module_names,
        "duplicates": duplicates,
        "opens_runtime_io": False,
    }
