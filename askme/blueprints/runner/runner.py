"""Shared runtime runner for blueprint modules."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from askme.runtime.core.module import Module, Runtime

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def print_blueprint_help(label: str = "Runtime") -> None:
    """Print CLI help without importing or composing the runtime modules."""

    _configure_logging()
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


RuntimeFactory = Callable[[], "Runtime"]


class _RuntimeLike(Protocol):
    def module_names(self) -> Sequence[str]: ...


def compose_runtime(module_classes: Sequence[type[Module]]) -> Runtime:
    """Compose a Runtime from module classes only when full runtime import is intended."""

    from askme.runtime.core.module import Runtime

    runtime = Runtime()
    for module_class in module_classes:
        runtime = runtime + Runtime.use(module_class)
    return runtime


def run_blueprint(
    blueprint: Runtime | RuntimeFactory,
    label: str = "Runtime",
    *,
    module_names: Sequence[str] | None = None,
) -> None:
    """Build, start, wait for SIGINT/SIGTERM, and stop a blueprint."""
    args = set(sys.argv[1:])
    _configure_logging()
    if any(arg in {"-h", "--help"} for arg in args):
        print_blueprint_help(label)
        return

    if "--preflight" in args:
        payload = _preflight_payload(blueprint, label, module_names=module_names)
        if "--json" in args:
            logger.info(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            modules = [str(name) for name in cast(Sequence[object], payload["modules"])]
            logger.info(
                f"{payload['label']} preflight ok: "
                f"{payload['module_count']} modules "
                f"({', '.join(modules)})",
            )
        return

    async def _main() -> None:
        from askme.config import get_config

        cfg = get_config()
        runtime = blueprint() if callable(blueprint) else blueprint
        app = await runtime.build(cfg)

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


def _preflight_payload(
    blueprint: Runtime | RuntimeFactory | Any,
    label: str,
    *,
    module_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return a no-IO blueprint summary for customer delivery checks."""
    if module_names is None:
        if callable(blueprint):
            raise ValueError(
                "module_names must be provided when preflighting a runtime factory"
            )
        names = list(cast(_RuntimeLike, blueprint).module_names())
    else:
        names = list(module_names)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    return {
        "ok": not duplicates,
        "label": label,
        "module_count": len(names),
        "modules": names,
        "duplicates": duplicates,
        "opens_runtime_io": False,
    }
