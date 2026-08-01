from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from askme.runtime.module import ModuleRegistry

from askme.runtime.modules.health_module import HealthModule
from askme.runtime.modules.llm_module import LLMModule
from askme.runtime.modules.voice_module import VoiceModule

ROOT = Path(__file__).resolve().parents[1]


def _chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_board_config_exposes_the_litellm_hot_switch_preset() -> None:
    config = yaml.safe_load((ROOT / "config.board.yaml").read_text(encoding="utf-8"))

    preset = config["brain"]["provider_presets"]["litellm"]

    assert preset["base_url"] == "${LITELLM_BASE_URL}"
    assert preset["api_key"] == "${LITELLM_VIRTUAL_KEY}"
    assert preset["fallback_models"] == []
    assert preset["max_retries"] == 0
    assert preset["minimax_api_key"] == ""


def test_voice_control_plane_never_synthesizes_an_implicit_direct_provider() -> None:
    module = VoiceModule()
    module._base_cfg = {
        "brain": {
            "provider": "litellm",
            "api_key": "sk-scoped",
            "base_url": "http://127.0.0.1:4000/v1",
            "model": "voice-fast",
            "fallback_models": [],
            "minimax_api_key": "legacy-direct-key",
            "provider_presets": {},
        }
    }

    assert "minimax" not in module._llm_presets()

    module._base_cfg["brain"]["provider_presets"]["minimax"] = {
        "api_key": "explicit-key",
        "base_url": "https://api.minimaxi.com/v1",
        "model": "MiniMax-M2.7-highspeed",
        "fallback_models": [],
    }

    assert module._llm_presets()["minimax"]["api_key"] == "explicit-key"


def test_litellm_startup_fails_closed_without_proxy_credentials() -> None:
    module = LLMModule()

    with pytest.raises(ValueError, match="api_key.*base_url"):
        module.build(
            {
                "brain": {
                    "provider": "litellm",
                    "api_key": "",
                    "base_url": "",
                    "model": "voice-fast",
                    "max_retries": 0,
                    "fallback_models": [],
                    "minimax_api_key": "",
                }
            },
            ModuleRegistry(),
        )


def test_prepare_client_rejects_missing_health_alias_before_transport() -> None:
    module = LLMModule()
    module.ota_metrics = None

    with pytest.raises(ValueError, match="health_model.*health-probe"):
        module.prepare_client(
            {
                "provider": "litellm",
                "api_key": "sk-scoped-virtual-key",
                "base_url": "http://127.0.0.1:4000/v1",
                "model": "voice-fast",
                "fallback_models": [],
            }
        )


def test_litellm_startup_rejects_competing_local_routing_policy() -> None:
    module = LLMModule()

    with pytest.raises(
        ValueError,
        match="max_retries.*fallback_models.*minimax_api_key",
    ):
        module.build(
            {
                "brain": {
                    "provider": "litellm",
                    "api_key": "sk-scoped-virtual-key",
                    "base_url": "http://127.0.0.1:4000/v1",
                    "model": "voice-fast",
                    "max_retries": 1,
                    "fallback_models": ["voice-quality"],
                    "minimax_api_key": "direct-provider-key",
                }
            },
            ModuleRegistry(),
        )


def test_llm_health_exposes_complete_active_routing_policy() -> None:
    module = LLMModule()
    module.client = SimpleNamespace(
        provider_name="litellm",
        model="voice-fast",
        provider_status=lambda: {
            "routing_owner": "litellm",
            "fallback_models": [],
        },
    )
    module._llm_config = SimpleNamespace(
        provider="litellm",
        model="voice-fast",
        health_model="health-probe",
        fallback_models=[],
    )
    module._warmup_model = "health-probe"

    health = module.health()

    assert health == {
        "status": "ok",
        "probe_status": "not_run",
        "provider": "litellm",
        "model": "voice-fast",
        "health_model": "health-probe",
        "fallback_models": [],
        "routing_owner": "litellm",
    }


def test_component_health_preserves_provider_and_routing_owner() -> None:
    checks = {}

    class _HealthService:
        def register(self, name, check):
            checks[name] = check

    llm_module = SimpleNamespace(
        health=lambda: {
            "status": "ok",
            "provider": "litellm",
            "model": "voice-fast",
            "health_model": "health-probe",
            "fallback_models": [],
            "routing_owner": "litellm",
        }
    )
    module = HealthModule()
    module.health_service = _HealthService()
    module._register_component_health_checks({}, {"llm": llm_module})

    assert checks["llm"]() == {
        "status": "healthy",
        "provider": "litellm",
        "model": "voice-fast",
        "health_model": "health-probe",
        "fallback_models": [],
        "routing_owner": "litellm",
    }


@pytest.mark.asyncio
async def test_llm_validation_waits_for_semantic_payload_and_rejects_empty_stream() -> None:
    seen = []
    contexts = []

    class _SemanticClient:
        async def chat_stream(self, *args, **kwargs):
            contexts.append(kwargs)
            seen.append("empty")
            yield _chunk()
            seen.append("semantic")
            yield _chunk("好")

    class _EmptyClient:
        async def chat_stream(self, *args, **kwargs):
            yield _chunk()

    module = LLMModule()
    await module.validate_client(
        _SemanticClient(),
        timeout_s=1.0,
        model="health-probe",
        purpose="health_probe",
    )

    assert seen == ["empty", "semantic"]
    assert contexts[0]["model"] == "health-probe"
    assert contexts[0]["context"].purpose == "health_probe"
    assert contexts[0]["context"].request_class == "health_probe"
    assert contexts[0]["context"].call_id
    with pytest.raises(RuntimeError, match="semantic payload"):
        await module.validate_client(_EmptyClient(), timeout_s=1.0)


@pytest.mark.asyncio
async def test_failed_llm_validation_closes_uncommitted_candidate() -> None:
    old_client = SimpleNamespace(provider_name="deepseek", model="direct")
    closed = 0

    async def _close_candidate() -> None:
        nonlocal closed
        closed += 1

    candidate = SimpleNamespace(
        provider_name="litellm",
        model="proxy",
        aclose=_close_candidate,
    )

    class _FakeLLMModule:
        def __init__(self):
            self.client = old_client
            self.llm_client = object()

        def prepare_client(self, brain_cfg):
            return candidate

        async def validate_client(
            self,
            client,
            *,
            timeout_s,
            model=None,
            purpose="assistant_response",
        ):
            raise RuntimeError("validation failed")

        def commit_client(self, client, *, warmup_model=None):
            raise AssertionError("failed candidate must not be committed")

    llm_module = _FakeLLMModule()
    module = VoiceModule()
    module._registry = {"llm": llm_module}
    module._control_state = {}
    module._resolve_llm_config = lambda payload: {
        "provider": "litellm",
        "model": "proxy",
        "voice_model": "proxy",
        "health_model": "health-probe",
        "timeout": 1.0,
    }

    with pytest.raises(RuntimeError, match="validation failed"):
        await module._switch_llm({"provider": "litellm", "validate": True})

    assert llm_module.client is old_client
    assert closed == 1


@pytest.mark.asyncio
async def test_successful_litellm_switch_commits_health_alias_and_retires_previous_client() -> None:
    old_client = SimpleNamespace(provider_name="deepseek", model="direct")
    candidate = SimpleNamespace(
        provider_name="litellm",
        model="voice-fast",
        provider_status=lambda: {"routing_owner": "litellm"},
    )
    validated = []
    commits = []
    retired = []
    publications = []

    class _FakeLLMModule:
        def __init__(self):
            self.client = old_client
            self.llm_client = object()

        def prepare_client(self, brain_cfg):
            assert brain_cfg["health_model"] == "health-probe"
            return candidate

        async def validate_client(
            self,
            client,
            *,
            timeout_s,
            model=None,
            purpose="assistant_response",
        ):
            validated.append((client, timeout_s, model, purpose))

        def commit_client(self, client, *, warmup_model=None):
            self.client = client
            commits.append((client, warmup_model))

        def retire_client(self, client):
            retired.append(client)

    def _resolve(payload):
        if payload.get("provider") == "litellm":
            return {
                "provider": "litellm",
                "model": "voice-fast",
                "voice_model": "voice-fast",
                "health_model": "health-probe",
                "fallback_models": [],
                "timeout": 3.0,
            }
        return {
            "provider": "deepseek",
            "model": "direct",
            "voice_model": "direct",
            "health_model": "direct-health",
            "fallback_models": [],
        }

    llm_module = _FakeLLMModule()
    module = VoiceModule()
    module._registry = {"llm": llm_module}
    module._control_state = {}
    module._resolve_llm_config = _resolve
    module._publish_llm = lambda client, brain_cfg: publications.append(
        (client, brain_cfg["health_model"])
    )

    result = await module._switch_llm({"provider": "litellm", "validate": True})

    assert result["runtime"] == {"routing_owner": "litellm"}
    assert validated == [
        (candidate, 3.0, None, "assistant_response"),
        (candidate, 3.0, "health-probe", "health_probe"),
    ]
    assert commits == [(candidate, "health-probe")]
    assert publications == [(llm_module.llm_client, "health-probe")]
    assert retired == [old_client]
    assert module._control_state["llm"] == {
        "provider": "litellm",
        "model": "voice-fast",
        "voice_model": "voice-fast",
        "health_model": "health-probe",
        "fallback_models": [],
    }


async def test_failed_llm_publication_rolls_back_the_module_and_consumers() -> None:
    old_client = SimpleNamespace(provider_name="deepseek", model="direct")
    candidate = SimpleNamespace(
        provider_name="litellm",
        model="proxy",
        provider_status=lambda: {"routing_owner": "litellm"},
    )

    class _FakeLLMModule:
        def __init__(self):
            self.client = old_client
            self.llm_client = object()
            self.commits = []

        def prepare_client(self, brain_cfg):
            return candidate

        def commit_client(self, client, *, warmup_model=None):
            self.client = client
            self.commits.append((client, warmup_model))

    llm_module = _FakeLLMModule()
    module = VoiceModule()
    module._registry = {"llm": llm_module}
    module._base_cfg = {
        "brain": {
            "provider": "deepseek",
            "api_key": "direct",
            "base_url": "https://direct.invalid/v1",
            "model": "direct",
            "health_model": "direct-health",
            "fallback_models": [],
            "provider_presets": {
                "litellm": {
                    "api_key": "virtual",
                    "base_url": "http://127.0.0.1:4000/v1",
                    "model": "proxy",
                    "health_model": "health-probe",
                    "fallback_models": [],
                }
            },
        }
    }
    module._control_state = {}
    publications = []

    def _publish(client, brain_cfg):
        publications.append((client, brain_cfg["provider"], brain_cfg["health_model"]))
        if brain_cfg["provider"] == "litellm":
            raise RuntimeError("consumer publish failed")

    module._publish_llm = _publish

    with pytest.raises(RuntimeError, match="consumer publish failed"):
        await module._switch_llm({"provider": "litellm", "validate": False})

    assert llm_module.client is old_client
    assert llm_module.commits == [
        (candidate, "health-probe"),
        (old_client, "direct-health"),
    ]
    assert publications == [
        (llm_module.llm_client, "litellm", "health-probe"),
        (llm_module.llm_client, "deepseek", "direct-health"),
    ]


@pytest.mark.asyncio
async def test_concurrent_llm_switch_failure_preserves_latest_committed_client() -> None:
    old_client = SimpleNamespace(provider_name="deepseek", model="direct")
    candidate_a = SimpleNamespace(
        provider_name="litellm",
        model="proxy-a",
        provider_status=lambda: {"routing_owner": "litellm"},
    )
    candidate_b = SimpleNamespace(
        provider_name="litellm",
        model="proxy-b",
        provider_status=lambda: {"routing_owner": "litellm"},
    )
    b_validation_started = asyncio.Event()
    release_b_validation = asyncio.Event()

    class _FakeLLMModule:
        def __init__(self):
            self.client = old_client
            self.llm_client = object()
            self.commits = []

        def prepare_client(self, brain_cfg):
            if brain_cfg["model"] == "proxy-b":
                return candidate_b
            return candidate_a

        async def validate_client(
            self,
            client,
            *,
            timeout_s,
            model=None,
            purpose="assistant_response",
        ):
            if client is candidate_b:
                b_validation_started.set()
                await release_b_validation.wait()

        def commit_client(self, client, *, warmup_model=None):
            self.client = client
            self.commits.append((client, warmup_model))

    class _StateStore:
        def __init__(self):
            self.saved = []

        def save(self, state):
            saved = deepcopy(state)
            self.saved.append(saved)
            return saved

    llm_module = _FakeLLMModule()
    module = VoiceModule()
    module._registry = {"llm": llm_module}
    module._base_cfg = {
        "brain": {
            "provider": "deepseek",
            "api_key": "direct",
            "base_url": "https://direct.invalid/v1",
            "model": "direct",
            "health_model": "direct-health",
            "fallback_models": [],
            "provider_presets": {
                "litellm": {
                    "api_key": "virtual",
                    "base_url": "http://127.0.0.1:4000/v1",
                    "model": "proxy",
                    "health_model": "health-probe",
                    "fallback_models": [],
                }
            },
        }
    }
    module._control_state = {}
    module._state_store = _StateStore()
    consumer = {"client": old_client}
    publications = []

    def _publish(client, brain_cfg):
        consumer["client"] = client
        publications.append((client, brain_cfg["model"]))
        if brain_cfg["model"] == "proxy-b":
            raise RuntimeError("consumer publish failed")

    module._publish_llm = _publish

    task_b = asyncio.create_task(
        module.switch_system_component_payload(
            {
                "component": "llm",
                "provider": "litellm",
                "model": "proxy-b",
            }
        )
    )
    await asyncio.wait_for(b_validation_started.wait(), timeout=1.0)

    task_a = asyncio.create_task(
        module.switch_system_component_payload(
            {
                "component": "llm",
                "provider": "litellm",
                "model": "proxy-a",
            }
        )
    )
    await asyncio.sleep(0)
    assert not task_a.done()

    release_b_validation.set()
    with pytest.raises(RuntimeError, match="consumer publish failed"):
        await task_b
    result_a = await asyncio.wait_for(task_a, timeout=1.0)

    assert result_a["runtime"] == {"routing_owner": "litellm"}
    assert llm_module.client is candidate_a
    assert consumer["client"] is llm_module.llm_client
    assert module._control_state["llm"]["model"] == "proxy-a"
    assert module._state_store.saved[-1]["llm"]["model"] == "proxy-a"
    assert llm_module.commits == [
        (candidate_b, "health-probe"),
        (old_client, "direct-health"),
        (candidate_a, "health-probe"),
    ]
    assert publications == [
        (llm_module.llm_client, "proxy-b"),
        (llm_module.llm_client, "direct"),
        (llm_module.llm_client, "proxy-a"),
    ]
