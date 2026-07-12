#!/usr/bin/env python3
"""ZeroClaw <-> Askme MCP bridge.

Orchestrates the ZeroClaw gateway alongside the Askme MCP server,
monitors health on both sides, and handles graceful shutdown.

Usage::

    # Start bridge (ZeroClaw gateway + Askme MCP together)
    python scripts/zeroclaw_bridge.py

    # Start with a specific ZeroClaw config
    python scripts/zeroclaw_bridge.py --config .zeroclaw/config.toml

    # Health check mode (query both processes)
    python scripts/zeroclaw_bridge.py --check
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("zeroclaw-bridge")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / ".zeroclaw" / "config.toml"


# ---------------------------------------------------------------------------
# Health state
# ---------------------------------------------------------------------------

@dataclass
class BridgeState:
    """Track liveness of both processes."""

    zeroclaw_alive: bool = False
    mcp_alive: bool = False
    zeroclaw_pid: int | None = None
    mcp_pid: int | None = None
    started_at: float = field(default_factory=time.time)
    last_zeroclaw_heartbeat: float = 0.0
    last_mcp_heartbeat: float = 0.0


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

async def _start_process(
    name: str,
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> asyncio.subprocess.Process:
    """Launch a subprocess and return the handle."""
    logger.info("Starting %s: %s", name, " ".join(cmd))
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd or str(ROOT),
        env=merged_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    logger.info("%s started (pid=%d)", name, proc.pid)
    return proc


async def _read_stream(
    stream: asyncio.StreamReader,
    name: str,
    callback: callable,
) -> None:
    """Read lines from *stream* and pipe to logging callback."""
    while True:
        line = await stream.readline()
        if not line:
            break
        callback(f"[{name}] {line.decode(errors='replace').rstrip()}")


# ---------------------------------------------------------------------------
# Health monitoring
# ---------------------------------------------------------------------------

async def _health_check(state: BridgeState, interval: float = 5.0) -> None:
    """Periodically log bridge health."""
    while True:
        await asyncio.sleep(interval)
        elapsed = time.time() - state.started_at
        logger.info(
            "Health — zeroclaw=%s mcp=%s elapsed=%.0fs",
            "alive" if state.zeroclaw_alive else "dead",
            "alive" if state.mcp_alive else "dead",
            elapsed,
        )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def _run_bridge(
    zeroclaw_config: Path,
    *,
    zeroclaw_bin: str = "zeroclaw",
) -> int:
    """Launch ZeroClaw gateway and Askme MCP server, then monitor both."""
    state = BridgeState()
    tasks: list[asyncio.Task] = []
    zeroclaw_proc: asyncio.subprocess.Process | None = None
    mcp_proc: asyncio.subprocess.Process | None = None

    # Resolve config path
    if not zeroclaw_config.exists():
        logger.error("ZeroClaw config not found: %s", zeroclaw_config)
        return 1

    def _log_info(text: str) -> None:
        logger.info(text)

    def _log_warn(text: str) -> None:
        logger.warning(text)

    def _log_error(text: str) -> None:
        logger.error(text)

    try:
        # 1. Start Askme MCP server (stdio mode)
        logger.info("Starting Askme MCP server...")
        mcp_proc = await _start_process(
            "askme-mcp",
            [sys.executable, "-m", "askme.mcp.server"],
        )
        state.mcp_pid = mcp_proc.pid

        tasks.append(
            asyncio.create_task(
                _read_stream(mcp_proc.stdout, "askme-mcp", _log_info),
                name="mcp-stdout",
            )
        )
        tasks.append(
            asyncio.create_task(
                _read_stream(mcp_proc.stderr, "askme-mcp", _log_warn),
                name="mcp-stderr",
            )
        )
        # Give MCP server a moment to initialise
        await asyncio.sleep(1.5)
        state.mcp_alive = mcp_proc.returncode is None
        state.last_mcp_heartbeat = time.time()

        if not state.mcp_alive:
            logger.error("Askme MCP server failed to start")
            return 1

        # 2. Start ZeroClaw gateway
        logger.info("Starting ZeroClaw gateway...")
        zeroclaw_proc = await _start_process(
            "zeroclaw",
            [zeroclaw_bin, "gateway", "--config", str(zeroclaw_config)],
            cwd=str(ROOT),
        )
        state.zeroclaw_pid = zeroclaw_proc.pid

        tasks.append(
            asyncio.create_task(
                _read_stream(zeroclaw_proc.stdout, "zeroclaw", _log_info),
                name="zeroclaw-stdout",
            )
        )
        tasks.append(
            asyncio.create_task(
                _read_stream(zeroclaw_proc.stderr, "zeroclaw", _log_warn),
                name="zeroclaw-stderr",
            )
        )
        # Give ZeroClaw time to connect to MCP
        await asyncio.sleep(2.0)
        state.zeroclaw_alive = zeroclaw_proc.returncode is None
        state.last_zeroclaw_heartbeat = time.time()

        if not state.zeroclaw_alive:
            logger.error("ZeroClaw gateway failed to start")
            return 1

        # 3. Health monitor
        health_task = asyncio.create_task(
            _health_check(state),
            name="health-check",
        )
        tasks.append(health_task)

        logger.info(
            "Bridge ready — zeroclaw(gateway pid=%d) askme-mcp(pid=%d)",
            state.zeroclaw_pid,
            state.mcp_pid,
        )

        # 4. Wait for either process to exit
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(
                    _wait_proc(zeroclaw_proc, "zeroclaw", state, is_zeroclaw=True),
                    name="wait-zeroclaw",
                ),
                asyncio.create_task(
                    _wait_proc(mcp_proc, "askme-mcp", state, is_zeroclaw=False),
                    name="wait-mcp",
                ),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel remaining tasks
        for task in done:
            task.result()

    except asyncio.CancelledError:
        logger.info("Bridge cancelled")
    except Exception:
        logger.exception("Bridge error")
        return 1
    finally:
        await _shutdown(zeroclaw_proc, mcp_proc, tasks)

    return 0


async def _wait_proc(
    proc: asyncio.subprocess.Process,
    name: str,
    state: BridgeState,
    *,
    is_zeroclaw: bool,
) -> None:
    """Wait for a subprocess to finish and update health state."""
    code = await proc.wait()
    if is_zeroclaw:
        state.zeroclaw_alive = False
    else:
        state.mcp_alive = False
    logger.warning("%s exited (code=%d)", name, code)


async def _shutdown(
    zeroclaw_proc: asyncio.subprocess.Process | None,
    mcp_proc: asyncio.subprocess.Process | None,
    tasks: list[asyncio.Task],
) -> None:
    """Gracefully tear down both processes and background tasks."""
    logger.info("Shutting down bridge...")

    # Cancel all background readers
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # Graceful: SIGTERM first, wait, then SIGKILL
    for proc, name in [(zeroclaw_proc, "zeroclaw"), (mcp_proc, "askme-mcp")]:
        if proc is None or proc.returncode is not None:
            continue
        logger.info("Sending SIGTERM to %s (pid=%d)...", name, proc.pid)
        try:
            proc.send_signal(signal.SIGTERM)
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            logger.info("%s stopped gracefully", name)
        except TimeoutError:
            logger.warning("%s did not respond to SIGTERM; sending SIGKILL", name)
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass

    logger.info("Bridge shutdown complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ZeroClaw <-> Askme MCP bridge",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="ZeroClaw config path (default: .zeroclaw/config.toml)",
    )
    parser.add_argument(
        "--zeroclaw-bin",
        default="zeroclaw",
        help="ZeroClaw binary name or path (default: zeroclaw)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print bridge health summary and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def _check_health() -> int:
    """Quick health summary without running the bridge."""
    config_path = Path(DEFAULT_CONFIG)
    config_ok = config_path.exists()
    print("ZeroClaw <-> Askme Bridge Health Check")
    print(f"  Config file:        {config_path}  {'OK' if config_ok else 'MISSING'}")
    print(f"  Working directory:  {ROOT}")
    print("  To start bridge:    python scripts/zeroclaw_bridge.py")
    return 0 if config_ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.check:
        return _check_health()

    return asyncio.run(
        _run_bridge(
            Path(args.config),
            zeroclaw_bin=args.zeroclaw_bin,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
