"""Small runnable examples for the runtime module system.

Run:

    python -m askme.runtime.examples
"""

from __future__ import annotations

import asyncio
from typing import Any

from askme.runtime.core import In, Module, ModuleRegistry, Out, Runtime


class ClockService:
    """Tiny service used by the runtime example."""

    def now(self) -> str:
        return "09:30"


class ClockModule(Module):
    """Provider module: creates and exposes a ClockService."""

    name = "clock"
    provides = ("clock",)

    clock: Out[ClockService]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        _ = cfg, registry
        self.service = ClockService()

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "time": self.service.now()}


class GreeterModule(Module):
    """Consumer module: receives ClockModule through In[ClockService]."""

    name = "greeter"
    depends_on = ("clock",)

    clock: In[ClockService]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        _ = cfg, registry
        self.message = f"greeter ready at {self.clock.service.now()}"

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "message": self.message,
            "started": getattr(self, "started", False),
        }


async def build_example_runtime() -> None:
    """Build, start, inspect, and stop a minimal runtime."""
    runtime = Runtime.use(ClockModule) + Runtime.use(GreeterModule)
    app = await runtime.build({})
    await app.start()
    try:
        print(app.greeter.message)
        print(app.health())
        print(app.flow_stats())
    finally:
        await app.stop()


def describe_blueprint_modules(runtime: Runtime) -> list[str]:
    """Return module names from a blueprint-style Runtime object.

    Blueprint presets expose assembled ``Runtime`` objects. Delivery and
    preflight code can inspect their module list before building the app, which
    lets a newcomer verify composition without opening microphones, cameras, or
    robot services.
    """
    return list(runtime.module_names())


def assemble_blueprint_example() -> Runtime:
    """Assemble a tiny blueprint from runtime modules."""
    return Runtime.use(ClockModule) + Runtime.use(GreeterModule)


def main() -> None:
    blueprint = assemble_blueprint_example()
    print(f"blueprint modules: {describe_blueprint_modules(blueprint)}")
    asyncio.run(build_example_runtime())


if __name__ == "__main__":
    main()
