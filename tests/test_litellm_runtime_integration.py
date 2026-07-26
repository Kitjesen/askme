from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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


def test_llm_health_exposes_the_active_routing_owner() -> None:
    module = LLMModule()
    module.client = SimpleNamespace(
        provider_name="litellm",
        model="deepseek-v4-flash",
        provider_status=lambda: {"routing_owner": "litellm"},
    )
    module._llm_config = SimpleNamespace(provider="litellm", model="deepseek-v4-flash")

    health = module.health()

    assert health["routing_owner"] == "litellm"


def test_component_health_preserves_provider_and_routing_owner() -> None:
    checks = {}

    class _HealthService:
        def register(self, name, check):
            checks[name] = check

    llm_module = SimpleNamespace(
        health=lambda: {
            "status": "ok",
            "provider": "litellm",
            "model": "deepseek-v4-flash",
            "routing_owner": "litellm",
        }
    )
    module = HealthModule()
    module.health_service = _HealthService()
    module._register_component_health_checks({}, {"llm": llm_module})

    assert checks["llm"]() == {
        "status": "healthy",
        "provider": "litellm",
        "model": "deepseek-v4-flash",
        "routing_owner": "litellm",
    }


@pytest.mark.asyncio
async def test_llm_validation_waits_for_semantic_payload_and_rejects_empty_stream() -> None:
    seen = []

    class _SemanticClient:
        async def chat_stream(self, *args, **kwargs):
            seen.append("empty")
            yield _chunk()
            seen.append("semantic")
            yield _chunk("好")

    class _EmptyClient:
        async def chat_stream(self, *args, **kwargs):
            yield _chunk()

    module = LLMModule()
    await module.validate_client(_SemanticClient(), timeout_s=1.0)

    assert seen == ["empty", "semantic"]
    with pytest.raises(RuntimeError, match="semantic payload"):
        await module.validate_client(_EmptyClient(), timeout_s=1.0)


@pytest.mark.asyncio
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
            self.commits = []

        def prepare_client(self, brain_cfg):
            return candidate

        def commit_client(self, client):
            self.client = client
            self.commits.append(client)

    llm_module = _FakeLLMModule()
    module = VoiceModule()
    module._registry = {"llm": llm_module}
    module._base_cfg = {
        "brain": {
            "provider": "deepseek",
            "api_key": "direct",
            "base_url": "https://direct.invalid/v1",
            "model": "direct",
            "fallback_models": [],
            "provider_presets": {
                "litellm": {
                    "api_key": "virtual",
                    "base_url": "http://127.0.0.1:4000/v1",
                    "model": "proxy",
                    "fallback_models": [],
                }
            },
        }
    }
    module._control_state = {}
    publications = []

    def _publish(client, brain_cfg):
        publications.append(client)
        if client is candidate:
            raise RuntimeError("consumer publish failed")

    module._publish_llm = _publish

    with pytest.raises(RuntimeError, match="consumer publish failed"):
        await module._switch_llm({"provider": "litellm", "validate": False})

    assert llm_module.client is old_client
    assert llm_module.commits == [candidate, old_client]
    assert publications == [candidate, old_client]


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
            self.commits = []

        def prepare_client(self, brain_cfg):
            if brain_cfg["model"] == "proxy-b":
                return candidate_b
            return candidate_a

        async def validate_client(self, client, *, timeout_s):
            if client is candidate_b:
                b_validation_started.set()
                await release_b_validation.wait()

        def commit_client(self, client):
            self.client = client
            self.commits.append(client)

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
            "fallback_models": [],
            "provider_presets": {
                "litellm": {
                    "api_key": "virtual",
                    "base_url": "http://127.0.0.1:4000/v1",
                    "model": "proxy",
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
        publications.append(client)
        if client is candidate_b:
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
    assert consumer["client"] is candidate_a
    assert module._control_state["llm"]["model"] == "proxy-a"
    assert module._state_store.saved[-1]["llm"]["model"] == "proxy-a"
    assert llm_module.commits == [candidate_b, old_client, candidate_a]
    assert publications == [candidate_b, old_client, candidate_a]
