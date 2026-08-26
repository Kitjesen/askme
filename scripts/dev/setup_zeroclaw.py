"""Configure ZeroClaw to use the AskMe LiteLLM control plane.

The application key is a dedicated LiteLLM virtual key authorized only for the
robot-action alias. Provider credentials and the LiteLLM master key never enter
ZeroClaw. Unverified native MCP tables are removed; v0.1.7 integration remains
blocked until a supported connector is validated.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZEROCLAW_HOME = Path.home() / ".zeroclaw"
ZEROCLAW_WORKSPACE = ZEROCLAW_HOME / "workspace"
ZEROCLAW_RUNTIME_VERSION = "0.1.7"
ZEROCLAW_RUNTIME_PROFILE = "standard"
ZEROCLAW_EDGE_PROFILE = "nano"
ZEROCLAW_MODEL_ALIAS = "robot-action"

sys.path.insert(0, str(PROJECT_ROOT))
from askme.config import get_config


def _brain_config() -> dict[str, Any]:
    config = get_config()
    brain = config["brain"] if isinstance(config, dict) else config.brain
    if isinstance(brain, dict):
        return brain
    return dict(vars(brain))


def _litellm_credentials() -> tuple[str, str]:
    brain = _brain_config()
    provider = str(brain.get("provider") or "").strip().lower()
    if provider != "litellm":
        raise RuntimeError("AskMe brain.provider must be exactly litellm")

    base_url = str(brain.get("base_url") or "").strip().rstrip("/")
    expected_base_url = os.environ.get("LITELLM_BASE_URL", "").strip().rstrip("/")
    api_key = os.environ.get("ZEROCLAW_LITELLM_VIRTUAL_KEY", "").strip()
    if not expected_base_url:
        raise RuntimeError("LITELLM_BASE_URL is required")
    if base_url != expected_base_url:
        raise RuntimeError("AskMe brain.base_url must match LITELLM_BASE_URL")
    if len(api_key) < 10:
        raise RuntimeError(
            "ZEROCLAW_LITELLM_VIRTUAL_KEY is required and must be a scoped virtual key"
        )
    askme_key = os.environ.get("LITELLM_VIRTUAL_KEY", "").strip()
    if askme_key and api_key == askme_key:
        raise RuntimeError("ZEROCLAW_LITELLM_VIRTUAL_KEY must be dedicated to ZeroClaw")
    return base_url, api_key


def _set_toml_value(text: str, key: str, value: str, section: str | None) -> str:
    """Replace or add one TOML assignment without disturbing other sections."""

    if section is None:
        start = 0
        next_header = re.search(r"(?m)^[ \t]*\[", text)
        end = next_header.start() if next_header else len(text)
    else:
        header = re.search(rf"(?m)^[ \t]*\[{re.escape(section)}\][ \t]*$", text)
        if header is None:
            text = text.rstrip() + f"\n\n[{section}]\n"
            header = re.search(rf"(?m)^[ \t]*\[{re.escape(section)}\][ \t]*$", text)
            assert header is not None
        start = header.end()
        next_header = re.search(r"(?m)^[ \t]*\[", text[start:])
        end = start + next_header.start() if next_header else len(text)

    segment = text[start:end]
    pattern = rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=.*$"
    replacement = f"{key} = {value}"
    segment, count = re.subn(pattern, replacement, segment, count=1)
    if count == 0:
        segment = segment.rstrip() + f"\n{replacement}\n\n"
    return text[:start] + segment + text[end:]


def _toml_table_name(line: str) -> str | None:
    """Return the simple TOML table path used by legacy ZeroClaw config."""

    header = line.strip().split("#", 1)[0].rstrip()
    if header.startswith("[[") and header.endswith("]]"):
        return header[2:-2].strip()
    if header.startswith("[") and header.endswith("]"):
        return header[1:-1].strip()
    return None


def _remove_unverified_mcp_tables(text: str) -> str:
    """Remove legacy MCP table families unsupported by the pinned v0.1.7 runtime."""

    kept: list[str] = []
    dropping = False
    for line in text.splitlines(keepends=True):
        table_name = _toml_table_name(line)
        if table_name is not None:
            dropping = table_name == "mcp" or table_name.startswith("mcp.")
        if not dropping:
            kept.append(line)
    return "".join(kept)


def _sanitise_zeroclaw_config(text: str) -> str:
    """Remove credentials, routing escapes, and unverified MCP declarations."""

    text = _remove_unverified_mcp_tables(text)
    for key, value in (("api_key", '""'), ("model_routes", "[]")):
        text = _set_toml_value(text, key, value, None)
    for key, value in (
        ("provider_retries", "0"),
        ("fallback_providers", "[]"),
        ("api_keys", "[]"),
        ("model_fallbacks", "{}"),
    ):
        text = _set_toml_value(text, key, value, "reliability")

    parsed = tomllib.loads(text)
    reliability = parsed.get("reliability", {})
    expected = {
        "api_key": "",
        "model_routes": [],
        "provider_retries": 0,
        "fallback_providers": [],
        "api_keys": [],
        "model_fallbacks": {},
        "mcp_configured": False,
    }
    actual = {
        "api_key": parsed.get("api_key"),
        "model_routes": parsed.get("model_routes"),
        "provider_retries": reliability.get("provider_retries"),
        "fallback_providers": reliability.get("fallback_providers"),
        "api_keys": reliability.get("api_keys"),
        "model_fallbacks": reliability.get("model_fallbacks"),
        "mcp_configured": "mcp" in parsed,
    }
    if actual != expected:
        raise RuntimeError("ZeroClaw config sanitization could not be verified")
    return text


def _disable_zeroclaw_model_routing(config_path: Path) -> None:
    """Keep LiteLLM as the only retry/fallback/model-routing owner."""

    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(_sanitise_zeroclaw_config(text), encoding="utf-8")


def _audit_zeroclaw_config(config_path: Path, base_url: str) -> None:
    """Verify the persisted onboarding result cannot bypass LiteLLM."""

    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"ZeroClaw config cannot be audited: {exc}") from exc

    reliability = config.get("reliability", {})
    expected = {
        "api_key": "",
        "default_provider": f"custom:{base_url}",
        "default_model": ZEROCLAW_MODEL_ALIAS,
        "model_routes": [],
        "provider_retries": 0,
        "fallback_providers": [],
        "api_keys": [],
        "model_fallbacks": {},
        "mcp_configured": False,
    }
    actual = {
        "api_key": config.get("api_key"),
        "default_provider": config.get("default_provider"),
        "default_model": config.get("default_model"),
        "model_routes": config.get("model_routes"),
        "provider_retries": reliability.get("provider_retries"),
        "fallback_providers": reliability.get("fallback_providers"),
        "api_keys": reliability.get("api_keys"),
        "model_fallbacks": reliability.get("model_fallbacks"),
        "mcp_configured": "mcp" in config,
    }
    violations = [
        f"{key} must be {expected[key]!r}"
        for key, value in actual.items()
        if value != expected[key]
    ]
    if violations:
        raise RuntimeError("unsafe ZeroClaw config: " + "; ".join(violations))


def _configure_zeroclaw_litellm_policy(base_url: str) -> Path:
    """Write a keyless LiteLLM-only ZeroClaw policy for manual secure key injection."""

    config_path = ZEROCLAW_HOME / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    text = _sanitise_zeroclaw_config(text)
    text = _set_toml_value(text, "default_provider", f'"custom:{base_url}"', None)
    text = _set_toml_value(text, "default_model", f'"{ZEROCLAW_MODEL_ALIAS}"', None)
    config_path.write_text(text, encoding="utf-8")
    _audit_zeroclaw_config(config_path, base_url)
    return config_path


def _run_zeroclaw_onboard(base_url: str, api_key: str) -> None:
    """Refuse automatic onboarding because ZeroClaw CLI would expose keys in argv."""

    _ = (base_url, api_key)
    raise RuntimeError(
        "automatic zeroclaw onboard is disabled: passing scoped keys via "
        "--api-key exposes them in OS argv/process listings. Configure the "
        "key through a verified non-argv secret path, then rotate any key that "
        "was previously used with the CLI."
    )


def _install_persona() -> None:
    persona_files = [
        "IDENTITY.md",
        "SOUL.md",
        "AGENTS.md",
        "TOOLS.md",
        "MEMORY.md",
        "HEARTBEAT.md",
    ]
    agent_dir = PROJECT_ROOT / "agent"
    ZEROCLAW_WORKSPACE.mkdir(parents=True, exist_ok=True)
    for filename in persona_files:
        source = agent_dir / filename
        if source.exists():
            shutil.copy2(source, ZEROCLAW_WORKSPACE / filename)


_LEGACY_BRIDGE_MARKERS = (
    "# Askme Bridge Skill",
    'kind = "shell"',
    "curl -s -X POST http://localhost:8765/api/v1/chat",
    "curl -s http://localhost:8765/api/v1/robot/state",
    "curl -s 'http://localhost:8765/api/v1/memory/search?q={{query}}'",
)


def _remove_legacy_bridge_skill() -> str:
    """Remove only the legacy curl bridge generated by older setup versions."""

    skill_dir = ZEROCLAW_WORKSPACE / "skills" / "askme-bridge"
    skill_file = skill_dir / "SKILL.toml"
    if not skill_file.exists():
        return "missing"
    text = skill_file.read_text(encoding="utf-8")
    if not all(marker in text for marker in _LEGACY_BRIDGE_MARKERS):
        return "preserved"
    skill_file.unlink()
    try:
        skill_dir.rmdir()
    except OSError:
        pass
    return "removed"


def _verify_runtime() -> str:
    result = subprocess.run(
        ["zeroclaw", "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"zeroclaw --version failed: {result.stderr.strip()}")
    return result.stdout.strip()


def main() -> int:
    try:
        base_url, _api_key = _litellm_credentials()
        print("[1/4] Writing keyless ZeroClaw LiteLLM policy...")
        config_path = _configure_zeroclaw_litellm_policy(base_url)
        print(
            f"[OK]   {config_path}; model=robot-action; routing_owner=litellm; native_mcp=BLOCKED"
        )

        print("[2/4] Copying agent persona files...")
        _install_persona()
        print("[OK]   Persona files installed")

        print("[3/4] Removing unsafe legacy AskMe API bridge skill...")
        bridge_status = _remove_legacy_bridge_skill()
        print(f"[OK]   Legacy bridge skill {bridge_status}; MCP integration remains BLOCKED")

        print("[4/4] Verifying ZeroClaw installation...")
        version_output = _verify_runtime()
        if ZEROCLAW_RUNTIME_VERSION not in version_output:
            print(
                f"[WARN] ZeroClaw version mismatch: expected "
                f"{ZEROCLAW_RUNTIME_VERSION}, got {version_output}"
            )
        else:
            print(f"[OK]   ZeroClaw {version_output} ready")
        print(
            f"[OK]   Runtime profile: {ZEROCLAW_RUNTIME_PROFILE}; "
            f"edge profile reserved as {ZEROCLAW_EDGE_PROFILE}"
        )
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
