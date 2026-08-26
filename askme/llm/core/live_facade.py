"""Stable live facade for hot-switched LLM gateways."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Protocol

_RAW_TRANSPORT_ATTRIBUTES = frozenset(
    {
        "raw_client",
        "client",
        "_client",
        "minimax_client",
        "_minimax_client",
    }
)
_ALLOWED_DYNAMIC_ATTRIBUTES = frozenset(
    {
        "model",
        "provider_name",
        "config",
        "provider_status",
        "request_activity",
        "recent_call_diagnostics",
        "cancel_warm_probes",
        "capabilities",
    }
)


class LLMGenerationLease(Protocol):
    """A leased gateway generation that must be released after use."""

    client: Any
    model: str

    def release(self) -> None:
        """Release the leased generation."""


class LiveLLMClientFacade:
    """Stable business-facing facade over the current LLM generation.

    Control-plane code can keep inspecting the raw active client on
    ``LLMModule.client``. Business code receives this object once and it keeps
    resolving the current generation at call time while holding a lease until
    each request, stream, or early stream close has fully finished.
    """

    def __init__(
        self,
        acquire_lease: Callable[[], LLMGenerationLease],
        acquire_warm_lease: Callable[[], LLMGenerationLease] | None = None,
    ) -> None:
        self._acquire_lease = acquire_lease
        self._acquire_warm_lease = acquire_warm_lease or acquire_lease

    async def chat_completion(
        self,
        messages: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        lease = self._acquire_lease()
        try:
            return await lease.client.chat_completion(messages, **kwargs)
        finally:
            lease.release()

    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        lease = self._acquire_lease()
        try:
            return await lease.client.chat(messages, **kwargs)
        finally:
            lease.release()

    async def chat_stream(
        self,
        messages: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        lease = self._acquire_lease()
        iterator: Any | None = None
        try:
            stream = lease.client.chat_stream(messages, **kwargs)
            iterator = stream.__aiter__()
            while True:
                try:
                    yield await anext(iterator)
                except StopAsyncIteration:
                    return
        finally:
            try:
                if iterator is not None:
                    close = getattr(iterator, "aclose", None)
                    if callable(close):
                        await close()
            finally:
                lease.release()

    def acquire_warm_target(self) -> LLMGenerationLease:
        """Lease the current generation with its matching warm-probe model."""

        return self._acquire_warm_lease()

    def __getattr__(self, name: str) -> Any:
        if name in _RAW_TRANSPORT_ATTRIBUTES or name.startswith("_"):
            raise AttributeError(name)
        if name not in _ALLOWED_DYNAMIC_ATTRIBUTES:
            raise AttributeError(name)

        lease = self._acquire_lease()
        try:
            return getattr(lease.client, name)
        finally:
            lease.release()
