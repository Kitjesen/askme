"""Provider-neutral contracts for external runtime task execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RuntimeExecutorUpdate:
    """One cursor-addressable update from an external executor."""

    event_id: str
    status: str
    message: str = ""
    cursor: str = ""
    observed_at: float | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))


@dataclass(frozen=True)
class RuntimeExecutorSubmitRequest:
    """Immutable task submission envelope."""

    handoff: Mapping[str, Any]
    idempotency_key: str
    correlation_id: str
    thread_id: str = ""
    turn_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "handoff", _freeze_mapping(self.handoff))


@dataclass(frozen=True)
class RuntimeExecutorSubmitResult:
    """Accepted remote task identity and its first observed state."""

    remote_task_id: str
    status: str
    correlation_id: str
    idempotency_key: str
    cursor: str = ""
    result_summary: str = ""
    updates: tuple[RuntimeExecutorUpdate, ...] = ()
    observed_at: float | None = None


@dataclass(frozen=True)
class RuntimeExecutorStatusRequest:
    """Request updates for one remote task after an optional cursor."""

    remote_task_id: str
    correlation_id: str
    cursor: str = ""


@dataclass(frozen=True)
class RuntimeExecutorStatusUpdate:
    """Current remote state plus any newly observed updates."""

    remote_task_id: str
    status: str
    correlation_id: str
    cursor: str = ""
    result_summary: str = ""
    updates: tuple[RuntimeExecutorUpdate, ...] = ()
    observed_at: float | None = None


@dataclass(frozen=True)
class RuntimeExecutorCancelRequest:
    """Idempotent request to cancel a remote task."""

    remote_task_id: str
    idempotency_key: str
    correlation_id: str
    reason: str = ""


@dataclass(frozen=True)
class RuntimeExecutorCancelResult:
    """Remote acknowledgement of a cancellation request."""

    remote_task_id: str
    status: str
    correlation_id: str
    idempotency_key: str
    cursor: str = ""
    result_summary: str = ""
    updates: tuple[RuntimeExecutorUpdate, ...] = ()
    observed_at: float | None = None


class RuntimeExecutorTransportError(RuntimeError):
    """Typed, secret-free transport failure."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        ambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.retryable = retryable
        self.ambiguous = ambiguous


class AmbiguousRuntimeSubmissionError(RuntimeExecutorTransportError):
    """Submission may have been accepted, so callers must reconcile by key."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(
            "ambiguous_submission",
            message,
            status_code=status_code,
            retryable=True,
            ambiguous=True,
        )


@runtime_checkable
class RuntimeExecutorTransport(Protocol):
    """Provider-neutral task executor transport."""

    def submit(self, request: RuntimeExecutorSubmitRequest) -> RuntimeExecutorSubmitResult: ...

    def get_status(self, request: RuntimeExecutorStatusRequest) -> RuntimeExecutorStatusUpdate: ...

    def cancel(self, request: RuntimeExecutorCancelRequest) -> RuntimeExecutorCancelResult: ...


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


__all__ = [
    "AmbiguousRuntimeSubmissionError",
    "RuntimeExecutorCancelRequest",
    "RuntimeExecutorCancelResult",
    "RuntimeExecutorStatusRequest",
    "RuntimeExecutorStatusUpdate",
    "RuntimeExecutorSubmitRequest",
    "RuntimeExecutorSubmitResult",
    "RuntimeExecutorTransport",
    "RuntimeExecutorTransportError",
    "RuntimeExecutorUpdate",
]
