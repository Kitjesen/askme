"""Legacy askme runtime entrypoints."""

from __future__ import annotations

from askme.app import AskmeApp


async def run_legacy_app(*, voice_mode: bool, robot_mode: bool) -> None:
    """Run the interactive askme runtime."""
    app = AskmeApp(
        voice_mode=voice_mode,
        robot_mode=robot_mode,
    )
    await app.run()
