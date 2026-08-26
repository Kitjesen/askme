"""Application composition seams shared by product blueprints."""

from __future__ import annotations

from askme.providers.runtime_executor import build_runtime_executor_transport
from askme.runtime.modules.runtime_handoff_module import (
    RuntimeHandoffModule as BaseRuntimeHandoffModule,
)


class RuntimeHandoffModule(BaseRuntimeHandoffModule):
    """Runtime handoff composed with the external executor provider adapter."""

    def __init__(self) -> None:
        super().__init__(executor_transport_factory=build_runtime_executor_transport)


__all__ = ["RuntimeHandoffModule"]
