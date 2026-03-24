"""Legacy askme runtime entrypoints."""

from __future__ import annotations

import logging

import askme.interfaces.register_defaults  # noqa: F401 — register all backends

from askme import __version__ as ASKME_VERSION
from askme.config import get_config
from askme.runtime.profiles import legacy_profile_for

logger = logging.getLogger(__name__)


def _select_blueprint(*, voice_mode: bool, robot_mode: bool):
    """Select the appropriate blueprint based on mode flags."""
    if voice_mode and robot_mode:
        from askme.blueprints.edge_robot import edge_robot
        return edge_robot
    if voice_mode:
        from askme.blueprints.voice import voice
        return voice
    from askme.blueprints.text import text
    return text


def _setup_logging(cfg: dict) -> None:
    """Configure logging from config."""
    level_str = cfg.get("app", {}).get("log_level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)
    fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

    log_file = cfg.get("app", {}).get("log_file")
    if log_file:
        handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logging.getLogger().addHandler(handler)


async def run_app(*, voice_mode: bool, robot_mode: bool) -> None:
    """Run the interactive askme runtime."""
    cfg = get_config()
    _setup_logging(cfg)

    blueprint = _select_blueprint(voice_mode=voice_mode, robot_mode=robot_mode)
    app = await blueprint.build(cfg)
    await app.start()
    try:
        if voice_mode:
            voice_mod = app.modules.get("voice")
            vl = getattr(voice_mod, "voice_loop", None) if voice_mod else None
            if vl is None:
                raise RuntimeError("voice profile is missing the voice loop")
            await vl.run()
        else:
            text_mod = app.modules.get("text")
            tl = getattr(text_mod, "text_loop", None) if text_mod else None
            if tl is None:
                raise RuntimeError("text profile is missing the text loop")
            await tl.run()
    finally:
        await app.stop()


# Backward compatibility alias
run_legacy_app = run_app
