#!/usr/bin/env python3
"""ZeroClaw AskMe MCP policy audit.

ZeroClaw v0.1.7 does not expose a verified native MCP connector for AskMe. This
module therefore audits the LiteLLM-only launch policy and refuses all runtime
launch attempts before spawning any process.

Usage::

    # Audit the selected ZeroClaw config and LiteLLM policy
    python scripts/zeroclaw_bridge.py --check

    # Non-check launch is intentionally blocked until native MCP is verified
    python scripts/zeroclaw_bridge.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("zeroclaw-policy-audit")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / ".zeroclaw" / "config.toml"
ZEROCLAW_MODEL_ALIAS = "robot-action"


class PolicyError(RuntimeError):
    """Raised when ZeroClaw could bypass the AskMe LiteLLM control plane."""


@dataclass(frozen=True)
class LaunchPolicy:
    """Audited process-launch values derived from AskMe configuration."""

    base_url: str
    provider: str
    model: str
    api_key: str

    def environment(self) -> dict[str, str]:
        return {
            "ZEROCLAW_API_KEY": self.api_key,
            "ZEROCLAW_PROVIDER": self.provider,
            "ZEROCLAW_MODEL": self.model,
        }


def _normalise_url(value: object) -> str:
    return str(value or "").strip().rstrip("/")


def _load_askme_policy(environ: Mapping[str, str]) -> LaunchPolicy:
    from askme.config import get_config

    config = get_config(reload=True)
    brain = config.get("brain", {}) if isinstance(config, dict) else {}
    provider = str(brain.get("provider") or "").strip().lower()
    if provider != "litellm":
        raise PolicyError("AskMe brain.provider must be exactly 'litellm'")

    expected_base_url = _normalise_url(environ.get("LITELLM_BASE_URL"))
    configured_base_url = _normalise_url(brain.get("base_url"))
    if not expected_base_url:
        raise PolicyError("LITELLM_BASE_URL is required")
    if configured_base_url != expected_base_url:
        raise PolicyError(
            "AskMe brain.base_url does not match LITELLM_BASE_URL "
            f"({configured_base_url!r} != {expected_base_url!r})"
        )

    api_key = str(environ.get("ZEROCLAW_LITELLM_VIRTUAL_KEY") or "").strip()
    if len(api_key) < 10:
        raise PolicyError(
            "ZEROCLAW_LITELLM_VIRTUAL_KEY is required and must be a scoped virtual key"
        )
    askme_key = str(environ.get("LITELLM_VIRTUAL_KEY") or "").strip()
    if askme_key and api_key == askme_key:
        raise PolicyError("ZEROCLAW_LITELLM_VIRTUAL_KEY must be dedicated to ZeroClaw")

    return LaunchPolicy(
        base_url=expected_base_url,
        provider=f"custom:{expected_base_url}",
        model=ZEROCLAW_MODEL_ALIAS,
        api_key=api_key,
    )


def _validate_config(config_path: Path, policy: LaunchPolicy) -> None:
    try:
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise PolicyError(f"ZeroClaw config not found: {config_path}") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PolicyError(f"ZeroClaw config cannot be audited: {exc}") from exc

    _validate_config_values(config, policy)


def _validate_config_values(config: Mapping[str, object], policy: LaunchPolicy) -> None:
    """Validate parsed ZeroClaw values against the only allowed launch policy."""

    violations: list[str] = []
    expected_top_level: dict[str, object] = {
        "default_provider": policy.provider,
        "default_model": policy.model,
        "model_routes": [],
    }
    for key, expected in expected_top_level.items():
        if config.get(key) != expected:
            violations.append(f"{key} must be {expected!r}")
    if config.get("api_key") not in (None, ""):
        violations.append("api_key must not be persisted")

    reliability = config.get("reliability")
    if not isinstance(reliability, dict):
        violations.append("[reliability] must be present")
    else:
        if (
            type(reliability.get("provider_retries")) is not int
            or reliability.get("provider_retries") != 0
        ):
            violations.append("reliability.provider_retries must be 0")
        for key, expected_value in (
            ("fallback_providers", []),
            ("api_keys", []),
            ("model_fallbacks", {}),
        ):
            if reliability.get(key) != expected_value:
                violations.append(f"reliability.{key} must be {expected_value!r}")

    if violations:
        raise PolicyError("unsafe ZeroClaw config: " + "; ".join(violations))


def _audit_launch_policy(
    config_path: Path, environ: Mapping[str, str] | None = None
) -> LaunchPolicy:
    policy = _load_askme_policy(environ if environ is not None else os.environ)
    _validate_config(config_path, policy)
    return policy


def _require_verified_native_mcp_connector(_config_path: Path) -> None:
    """Fail closed until ZeroClaw exposes a verified native MCP connector."""

    raise PolicyError(
        "ZeroClaw v0.1.7 has no verified native MCP connector; MCP integration is BLOCKED"
    )


async def _run_bridge(
    zeroclaw_config: Path,
    *,
    zeroclaw_bin: str = "zeroclaw",
) -> int:
    """Audit policy, then refuse runtime launch before spawning any process."""
    _ = zeroclaw_bin
    try:
        _audit_launch_policy(zeroclaw_config)
        _require_verified_native_mcp_connector(zeroclaw_config)
    except PolicyError as exc:
        logger.error("MCP integration blocked: %s", exc)
        return 1
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ZeroClaw AskMe MCP policy audit",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="ZeroClaw config path (default: .zeroclaw/config.toml)",
    )
    parser.add_argument(
        "--zeroclaw-bin",
        default="zeroclaw",
        help="Reserved for future native MCP launch; currently blocked",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Audit policy and exit without launching processes",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def _check_health(config_path: Path) -> int:
    """Audit the selected config without starting either process."""
    try:
        _audit_launch_policy(config_path)
    except PolicyError as exc:
        config_status = "INVALID"
        detail = str(exc)
        result = 1
    else:
        config_status = "OK"
        detail = "LiteLLM-only launch policy verified"
        result = 0

    print("ZeroClaw AskMe MCP Policy Audit")
    print(f"  Config file:        {config_path}  {config_status}")
    print(f"  Policy:             {detail}")
    print("  MCP integration:    BLOCKED until verified native MCP connector exists")
    print(f"  Working directory:  {ROOT}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.check:
        return _check_health(Path(args.config))

    return asyncio.run(
        _run_bridge(
            Path(args.config),
            zeroclaw_bin=args.zeroclaw_bin,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
