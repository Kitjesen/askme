"""Runtime module for the product LLM gateway."""

from __future__ import annotations

import asyncio
import inspect
import logging
import secrets
import threading
from dataclasses import dataclass
from typing import Any

from askme.llm.core.client import LLMClient
from askme.llm.core.config import LLMConfig
from askme.llm.core.contracts import LLMCallContext
from askme.llm.core.live_facade import LiveLLMClientFacade
from askme.runtime.core.module import Module, ModuleRegistry, Out
from askme.telemetry.ota_bridge import OTABridgeMetrics

logger = logging.getLogger(__name__)
_LLM_SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 2.0
_LLM_SHUTDOWN_DRAIN_POLL_SECONDS = 0.025


@dataclass
class _TrackedLLMGenerationLease:
    owner: LLMModule
    key: int
    client: Any
    model: str
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.owner._release_generation_lease(self.key)


def _has_semantic_payload(chunk: Any) -> bool:
    """Return whether a provider probe produced user-visible text or a tool call."""

    try:
        choices = chunk.choices
    except (AttributeError, TypeError):
        return False
    if not choices:
        return False
    for choice in choices:
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if content is not None and str(content).strip():
            return True
        if getattr(delta, "tool_calls", None):
            return True
    return False


class LLMModule(Module):
    """Provide a configured LLM gateway and a live warm-target seam.

    This is the runtime boundary that reads the ``brain`` section from
    config.yaml.  Downstream modules receive a configured LLM object and should
    not construct provider SDK clients directly.
    """

    name = "llm"
    provides = ("llm",)

    llm_client: Out[LLMClient]

    def __init__(self) -> None:
        super().__init__()
        self._switch_lock = threading.RLock()
        self._retired_clients: dict[int, LLMClient] = {}
        self._retirement_tasks: set[asyncio.Task[None]] = set()
        self._generation_leases: dict[int, int] = {}
        self._live_client = LiveLLMClientFacade(
            self.acquire_generation_lease,
            self.acquire_warm_generation_lease,
        )

    llm_config_out: Out[LLMConfig]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        brain_cfg = cfg.get("brain", {})
        self.ota_metrics = OTABridgeMetrics()
        self._llm_config = LLMConfig.from_cfg(brain_cfg)
        validation_errors = self._llm_config.validate()
        if validation_errors:
            raise ValueError("Invalid LLM configuration: " + "; ".join(validation_errors))
        self.client = LLMClient(llm_config=self._llm_config, metrics=self.ota_metrics)
        self._warmup_model = str(
            brain_cfg.get("health_model") or brain_cfg.get("voice_model") or self.client.model
        ).strip()
        self._switch_lock = threading.RLock()
        self._retired_clients = {}
        self._retirement_tasks = set()
        self._generation_leases = {}
        self._live_client = LiveLLMClientFacade(
            self.acquire_generation_lease,
            self.acquire_warm_generation_lease,
        )
        logger.info("LLMModule: built (model=%s)", self.client.model)

    async def stop(self) -> None:
        """Release the active provider transport during runtime shutdown."""

        retirement_tasks = tuple(self._retirement_tasks)
        for task in retirement_tasks:
            task.cancel()
        if retirement_tasks:
            await asyncio.gather(*retirement_tasks, return_exceptions=True)

        with self._switch_lock:
            clients = {id(self.client): self.client, **self._retired_clients}
            self._retired_clients.clear()
            self._retirement_tasks.clear()
        await self._drain_shutdown_clients(tuple(clients.values()))
        for client in clients.values():
            await self._close_client(client, label="runtime shutdown")

    async def _drain_shutdown_clients(self, clients: tuple[Any, ...]) -> None:
        for client in clients:
            self._cancel_client_warm_probes(client)

        timeout_s = max(
            0.0,
            float(
                getattr(
                    self,
                    "_shutdown_drain_timeout_seconds",
                    _LLM_SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
                )
            ),
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            active_clients = [client for client in clients if self._client_is_active(client)]
            if not active_clients:
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning(
                    "LLMModule: shutdown closing %d provider(s) with active LLM generation "
                    "or provider request activity after %.2fs grace period",
                    len(active_clients),
                    timeout_s,
                )
                return
            await asyncio.sleep(min(_LLM_SHUTDOWN_DRAIN_POLL_SECONDS, remaining))

    @property  # type: ignore[no-redef]
    def llm_client(self) -> LiveLLMClientFacade:  # type: ignore[no-redef, override]
        return self._ensure_live_client()

    @property  # type: ignore[no-redef]
    def llm_config_out(self) -> LLMConfig:  # type: ignore[no-redef, override]
        """Expose the resolved LLMConfig so downstream modules can read it."""

        return self._llm_config

    @property
    def metrics(self) -> Any:
        return self.ota_metrics

    def _ensure_live_client(self) -> LiveLLMClientFacade:
        live_client = getattr(self, "_live_client", None)
        if live_client is None:
            live_client = LiveLLMClientFacade(
                self.acquire_generation_lease,
                self.acquire_warm_generation_lease,
            )
            self._live_client = live_client
        return live_client

    def acquire_generation_lease(self) -> _TrackedLLMGenerationLease:
        """Lease the current raw generation for one business LLM call."""

        with self._switch_lock:
            client = self.client
            key = id(client)
            self._generation_leases[key] = self._generation_leases.get(key, 0) + 1
            return _TrackedLLMGenerationLease(
                owner=self,
                key=key,
                client=client,
                model=str(getattr(client, "model", "") or ""),
            )

    def acquire_warm_generation_lease(self) -> _TrackedLLMGenerationLease:
        """Lease the current raw generation and matching warm-probe model."""

        with self._switch_lock:
            client = self.client
            key = id(client)
            self._generation_leases[key] = self._generation_leases.get(key, 0) + 1
            return _TrackedLLMGenerationLease(
                owner=self,
                key=key,
                client=client,
                model=str(self._warmup_model or getattr(client, "model", "") or "").strip(),
            )

    def _release_generation_lease(self, key: int) -> None:
        with self._switch_lock:
            current = self._generation_leases.get(key, 0)
            if current <= 1:
                self._generation_leases.pop(key, None)
            else:
                self._generation_leases[key] = current - 1

    def _lease_count(self, client: Any) -> int:
        with self._switch_lock:
            return int(self._generation_leases.get(id(client), 0) or 0)

    def replace_config(self, brain_cfg: dict[str, Any]) -> LLMClient:
        """Atomically route subsequent requests to a newly configured gateway."""

        previous_client = self.client
        next_client = self.prepare_client(brain_cfg)
        warmup_model = str(
            brain_cfg.get("health_model") or brain_cfg.get("voice_model") or next_client.model
        ).strip()
        self.commit_client(next_client, warmup_model=warmup_model)
        self.retire_client(previous_client)
        return next_client

    def prepare_client(self, brain_cfg: dict[str, Any]) -> LLMClient:
        """Construct and validate a candidate without changing live routing."""

        next_config = LLMConfig.from_cfg(brain_cfg)
        errors = next_config.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return LLMClient(llm_config=next_config, metrics=self.ota_metrics)

    def commit_client(
        self,
        next_client: LLMClient,
        *,
        warmup_model: str | None = None,
    ) -> None:
        """Publish a prepared client for subsequent requests."""

        with self._switch_lock:
            previous_client = self.client
            if previous_client is not next_client:
                self._cancel_client_warm_probes(previous_client)
            self._llm_config = next_client.config
            self._warmup_model = str(warmup_model or next_client.model).strip()
            self.client = next_client
        logger.info(
            "LLMModule: hot switched provider=%s model=%s",
            next_client.provider_name,
            next_client.model,
        )

    @staticmethod
    def _cancel_client_warm_probes(client: Any) -> None:
        cancel_warm_probes = getattr(client, "cancel_warm_probes", None)
        if not callable(cancel_warm_probes):
            return
        try:
            cancel_warm_probes()
        except Exception:
            logger.warning("LLMModule: failed to cancel retiring warm probes", exc_info=True)

    def resolve_warm_target(self) -> tuple[LLMClient, str]:
        """Return one consistent raw client/model pair for legacy warm probes."""

        with self._switch_lock:
            return self.client, self._warmup_model

    def resolve_live_warm_target(self) -> tuple[LiveLLMClientFacade, str]:
        """Return the stable facade for manager-owned warm probes."""

        with self._switch_lock:
            return self._ensure_live_client(), self._warmup_model

    def retire_client(self, client: LLMClient) -> None:
        """Close a superseded gateway only after its active requests drain."""

        key = id(client)
        with self._switch_lock:
            if client is self.client or key in self._retired_clients:
                return
            self._retired_clients[key] = client
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if not self._client_is_active(client):
                self._retire_idle_client_without_loop(key, client)
            return
        task = loop.create_task(
            self._retire_when_idle(key, client),
            name=f"llm-retire-{key}",
        )
        self._retirement_tasks.add(task)
        task.add_done_callback(self._retirement_tasks.discard)

    async def _retire_when_idle(self, key: int, client: LLMClient) -> None:
        closed = False
        try:
            while self._client_is_active(client):
                await asyncio.sleep(0.05)
            closed = await self._close_client(client, label="superseded provider")
        finally:
            if closed:
                self._forget_retired_client(key, client)

    def _retire_idle_client_without_loop(self, key: int, client: LLMClient) -> None:
        """Use only an explicit bounded synchronous-close capability."""

        close_sync = getattr(client, "close_sync", None)
        if not callable(close_sync):
            return
        try:
            closed = close_sync() is True
        except Exception:
            logger.warning(
                "LLMModule: failed synchronous close of idle superseded provider; "
                "retaining it for shutdown retry",
                exc_info=True,
            )
            return
        if closed:
            self._forget_retired_client(key, client)

    def _forget_retired_client(self, key: int, client: LLMClient) -> None:
        with self._switch_lock:
            if self._retired_clients.get(key) is client:
                self._retired_clients.pop(key, None)

    def _client_is_active(self, client: Any) -> bool:
        return self._lease_count(client) > 0 or self._raw_client_is_active(client)

    @staticmethod
    def _raw_client_is_active(client: Any) -> bool:
        activity = getattr(client, "request_activity", None)
        if not callable(activity):
            return False
        try:
            snapshot = activity()
        except Exception:
            logger.warning("LLMModule: failed to inspect retiring client", exc_info=True)
            return True
        if not isinstance(snapshot, dict):
            return True
        return bool(
            int(snapshot.get("active_business_requests", 0) or 0)
            or int(snapshot.get("active_warm_probes", 0) or 0)
        )

    @staticmethod
    async def _close_client(client: Any, *, label: str) -> bool:
        close = getattr(client, "aclose", None)
        if not callable(close):
            return True
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.warning("LLMModule: failed to close %s", label, exc_info=True)
            return False
        return True

    async def validate_client(
        self,
        client: LLMClient,
        *,
        timeout_s: float = 10.0,
        model: str | None = None,
        purpose: str = "assistant_response",
    ) -> None:
        """Require a real payload before committing a requested provider switch."""

        request_class = "health_probe" if purpose == "health_probe" else "text"
        call_context = LLMCallContext(
            call_id=secrets.token_hex(16),
            purpose=purpose,
            channel="system",
            request_class=request_class,
            privacy_class="operational",
            allow_cache=False,
        )

        async def _probe() -> None:
            async for chunk in client.chat_stream(
                [{"role": "user", "content": "只回复好"}],
                model=model,
                max_tokens=2,
                temperature=0.0,
                context=call_context,
            ):
                if _has_semantic_payload(chunk):
                    return
            raise RuntimeError("LLM validation produced no semantic payload")

        await asyncio.wait_for(_probe(), timeout=max(1.0, float(timeout_s)))

    def health(self) -> dict[str, Any]:
        with self._switch_lock:
            client = self.client
            llm_config = self._llm_config
            health_model = self._warmup_model

        provider_status = getattr(client, "provider_status", None)
        runtime = provider_status() if callable(provider_status) else {}
        if not isinstance(runtime, dict):
            runtime = {}
        routing_owner = str(runtime.get("routing_owner") or "askme")
        configured_fallbacks = getattr(llm_config, "fallback_models", [])
        fallback_models = runtime.get("fallback_models", configured_fallbacks)

        probe_status = "not_run"
        recent_diagnostics = getattr(client, "recent_call_diagnostics", None)
        if callable(recent_diagnostics):
            try:
                diagnostics = recent_diagnostics(limit=20)
            except Exception:
                logger.debug("LLMModule: failed to read provider probe diagnostics", exc_info=True)
            else:
                if isinstance(diagnostics, (list, tuple)):
                    for diagnostic in reversed(diagnostics):
                        if not isinstance(diagnostic, dict):
                            continue
                        if str(diagnostic.get("purpose") or "").strip() != "health_probe":
                            continue
                        outcome = str(diagnostic.get("outcome") or "failed").strip()
                        if outcome in {"cancelled", "abandoned", "deferred"}:
                            continue
                        probe_status = outcome
                        break

        probe_failed = probe_status not in {"not_run", "success"}
        return {
            "status": "degraded" if probe_failed else "ok",
            "probe_status": probe_status,
            "provider": getattr(
                client,
                "provider_name",
                getattr(llm_config, "provider", "unknown"),
            ),
            "model": getattr(
                client,
                "model",
                getattr(llm_config, "model", "unknown"),
            ),
            "health_model": health_model,
            "fallback_models": list(fallback_models or []),
            "routing_owner": routing_owner,
        }
