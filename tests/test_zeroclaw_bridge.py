from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_bridge():
    return importlib.import_module("scripts.zeroclaw_bridge")


def test_check_rejects_selected_direct_provider_config(monkeypatch, capsys) -> None:
    bridge = _load_bridge()
    config_path = Path("historical.toml")
    audited_paths = []

    def reject_selected(path):
        audited_paths.append(path)
        raise bridge.PolicyError("unsafe ZeroClaw config: direct provider")

    monkeypatch.setattr(bridge, "_audit_launch_policy", reject_selected)

    result = bridge.main(["--check", "--config", str(config_path)])

    assert result == 1
    assert audited_paths == [config_path]
    output = capsys.readouterr().out
    assert str(config_path) in output
    assert "INVALID" in output
    assert "MCP integration:    BLOCKED" in output
    assert "To start bridge" not in output


def test_check_accepts_audited_repo_config(monkeypatch, capsys) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://127.0.0.1:4000/v1")
    monkeypatch.setenv("LITELLM_VIRTUAL_KEY", "sk-askme-different")
    monkeypatch.setenv("ZEROCLAW_LITELLM_VIRTUAL_KEY", "sk-zeroclaw-dedicated")

    bridge = _load_bridge()
    result = bridge.main(["--check", "--config", str(ROOT / ".zeroclaw" / "config.toml")])

    assert result == 0
    output = capsys.readouterr().out
    assert "LiteLLM-only launch policy verified" in output
    assert "MCP integration:    BLOCKED" in output
    assert "To start bridge" not in output


def test_setup_clears_persisted_credentials_and_all_local_routes() -> None:
    import tomllib

    setup = importlib.import_module("scripts.dev.setup_zeroclaw")
    historical_config = """
api_key = "historical-secret"
default_provider = "custom:http://127.0.0.1:4000/v1"
default_model = "robot-action"
model_routes = [{ model = "robot-action", provider = "direct" }]

[reliability]
provider_retries = 3
fallback_providers = ["direct"]
api_keys = ["historical-secret"]
model_fallbacks = { robot-action = ["direct-model"] }

[mcp]
enabled = true

[[mcp.servers]]
name = "askme"
command = "python"

[telemetry]
enabled = false
""".strip()

    config = tomllib.loads(setup._sanitise_zeroclaw_config(historical_config))
    assert config["api_key"] == ""
    assert config["model_routes"] == []
    assert config["reliability"]["provider_retries"] == 0
    assert config["reliability"]["fallback_providers"] == []
    assert config["reliability"]["api_keys"] == []
    assert config["reliability"]["model_fallbacks"] == {}
    assert "mcp" not in config
    assert config["telemetry"]["enabled"] is False


def test_setup_rejects_noncanonical_askme_provider(monkeypatch) -> None:
    import pytest

    setup = importlib.import_module("scripts.dev.setup_zeroclaw")
    monkeypatch.setattr(
        setup,
        "_brain_config",
        lambda: {
            "provider": "litellm-proxy",
            "base_url": "http://127.0.0.1:4000/v1",
        },
    )

    with pytest.raises(RuntimeError, match="exactly litellm"):
        setup._litellm_credentials()


def test_policy_rejects_each_local_routing_escape_hatch() -> None:
    import copy

    import pytest

    bridge = _load_bridge()
    policy = bridge.LaunchPolicy(
        base_url="http://127.0.0.1:4000/v1",
        provider="custom:http://127.0.0.1:4000/v1",
        model="robot-action",
        api_key="sk-zeroclaw-dedicated",
    )
    audited = {
        "api_key": "",
        "default_provider": policy.provider,
        "default_model": policy.model,
        "model_routes": [],
        "reliability": {
            "provider_retries": 0,
            "fallback_providers": [],
            "api_keys": [],
            "model_fallbacks": {},
        },
    }
    unsafe_changes = (
        (("api_key",), "persisted-secret"),
        (("default_provider",), "minimax-cn"),
        (("default_model",), "direct-model"),
        (("model_routes",), [{"model": "direct-model"}]),
        (("reliability", "provider_retries"), 1),
        (("reliability", "fallback_providers"), ["direct"]),
        (("reliability", "api_keys"), ["persisted-secret"]),
        (("reliability", "model_fallbacks"), {"robot-action": ["direct"]}),
    )

    for path, value in unsafe_changes:
        candidate = copy.deepcopy(audited)
        target = candidate
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = value
        with pytest.raises(bridge.PolicyError):
            bridge._validate_config_values(candidate, policy)


def test_bridge_audits_before_runtime_launch(monkeypatch) -> None:
    import asyncio

    bridge = _load_bridge()
    audited_paths = []

    def reject(config_path):
        audited_paths.append(config_path)
        raise bridge.PolicyError("unsafe")

    monkeypatch.setattr(bridge, "_audit_launch_policy", reject)

    assert asyncio.run(bridge._run_bridge(Path("unsafe.toml"))) == 1
    assert audited_paths == [Path("unsafe.toml")]
    assert not hasattr(bridge, "_start_process")


def test_bridge_refuses_to_report_ready_without_verified_native_mcp(monkeypatch) -> None:
    import asyncio

    bridge = _load_bridge()
    policy = bridge.LaunchPolicy(
        base_url="http://127.0.0.1:4000/v1",
        provider="custom:http://127.0.0.1:4000/v1",
        model="robot-action",
        api_key="sk-zeroclaw-dedicated",
    )

    monkeypatch.setattr(bridge, "_audit_launch_policy", lambda _path: policy)

    assert asyncio.run(bridge._run_bridge(Path("audited.toml"))) == 1
    source = Path(bridge.__file__).read_text(encoding="utf-8")
    assert "Bridge ready" not in source
    assert "create_subprocess_exec" not in source


def test_setup_adds_missing_fail_closed_routing_fields() -> None:
    import tomllib

    setup = importlib.import_module("scripts.dev.setup_zeroclaw")
    config = tomllib.loads(
        setup._sanitise_zeroclaw_config('api_key = "historical-secret"\n\n[mcp]\nenabled = true\n')
    )

    assert config["api_key"] == ""
    assert config["model_routes"] == []
    assert config["reliability"] == {
        "provider_retries": 0,
        "fallback_providers": [],
        "api_keys": [],
        "model_fallbacks": {},
    }
    assert "mcp" not in config


def test_setup_refuses_onboard_without_putting_secret_in_argv(monkeypatch) -> None:
    import pytest

    setup = importlib.import_module("scripts.dev.setup_zeroclaw")

    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("zeroclaw onboard must not be invoked with argv secrets")

    monkeypatch.setattr(setup.subprocess, "run", forbidden_run)

    with pytest.raises(RuntimeError, match="argv|--api-key"):
        setup._run_zeroclaw_onboard("http://127.0.0.1:4000/v1", "sk-dedicated")


def test_setup_writes_keyless_litellm_policy(tmp_path, monkeypatch) -> None:
    import tomllib

    setup = importlib.import_module("scripts.dev.setup_zeroclaw")
    monkeypatch.setattr(setup, "ZEROCLAW_HOME", tmp_path)

    config_path = setup._configure_zeroclaw_litellm_policy("http://127.0.0.1:4000/v1")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["api_key"] == ""
    assert config["default_provider"] == "custom:http://127.0.0.1:4000/v1"
    assert config["default_model"] == "robot-action"
    assert config["model_routes"] == []
    assert config["reliability"]["provider_retries"] == 0
    assert "mcp" not in config


def test_repo_manual_config_does_not_advertise_unverified_native_mcp() -> None:
    import tomllib

    config = tomllib.loads((ROOT / ".zeroclaw" / "config.toml").read_text(encoding="utf-8"))

    assert "mcp" not in config


def test_setup_audit_rejects_any_unverified_mcp_table(tmp_path) -> None:
    import pytest

    setup = importlib.import_module("scripts.dev.setup_zeroclaw")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
api_key = ""
default_provider = "custom:http://127.0.0.1:4000/v1"
default_model = "robot-action"
model_routes = []

[reliability]
provider_retries = 0
fallback_providers = []
api_keys = []
model_fallbacks = {}

[mcp]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="mcp"):
        setup._audit_zeroclaw_config(config_path, "http://127.0.0.1:4000/v1")


def test_setup_removes_only_known_legacy_rest_bridge(tmp_path, monkeypatch) -> None:
    setup = importlib.import_module("scripts.dev.setup_zeroclaw")
    monkeypatch.setattr(setup, "ZEROCLAW_WORKSPACE", tmp_path)
    skill_dir = tmp_path / "skills" / "askme-bridge"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.toml"
    skill_file.write_text(
        """# Askme Bridge Skill
description = "Connect ZeroClaw to Askme voice/memory/robot API"
kind = "shell"
command = "curl -s -X POST http://localhost:8765/api/v1/chat"
command = "curl -s http://localhost:8765/api/v1/robot/state"
command = "curl -s 'http://localhost:8765/api/v1/memory/search?q={{query}}'"
""",
        encoding="utf-8",
    )

    assert setup._remove_legacy_bridge_skill() == "removed"
    assert not skill_file.exists()


def test_setup_preserves_user_bridge_content(tmp_path, monkeypatch) -> None:
    setup = importlib.import_module("scripts.dev.setup_zeroclaw")
    monkeypatch.setattr(setup, "ZEROCLAW_WORKSPACE", tmp_path)
    skill_dir = tmp_path / "skills" / "askme-bridge"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.toml"
    skill_file.write_text("# user maintained native mcp config\n", encoding="utf-8")

    assert setup._remove_legacy_bridge_skill() == "preserved"
    assert skill_file.read_text(encoding="utf-8") == "# user maintained native mcp config\n"


def test_askme_policy_requires_canonical_endpoint_and_dedicated_key(monkeypatch) -> None:
    import pytest

    from askme import config as askme_config

    bridge = _load_bridge()
    environment = {
        "LITELLM_BASE_URL": "http://127.0.0.1:4000/v1",
        "LITELLM_VIRTUAL_KEY": "sk-askme-shared",
        "ZEROCLAW_LITELLM_VIRTUAL_KEY": "sk-zeroclaw-dedicated",
    }

    unsafe_brains = (
        {"provider": "minimax", "base_url": environment["LITELLM_BASE_URL"]},
        {"provider": "litellm", "base_url": "https://direct.invalid/v1"},
    )
    for brain in unsafe_brains:
        monkeypatch.setattr(
            askme_config, "get_config", lambda *, reload, brain=brain: {"brain": brain}
        )
        with pytest.raises(bridge.PolicyError):
            bridge._load_askme_policy(environment)

    safe_brain = {"provider": "litellm", "base_url": environment["LITELLM_BASE_URL"]}
    monkeypatch.setattr(askme_config, "get_config", lambda *, reload: {"brain": safe_brain})
    environment["ZEROCLAW_LITELLM_VIRTUAL_KEY"] = environment["LITELLM_VIRTUAL_KEY"]
    with pytest.raises(bridge.PolicyError, match="dedicated"):
        bridge._load_askme_policy(environment)
