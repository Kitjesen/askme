"""Backend registry — Strategy + Registry pattern for pluggable algorithms.

Every subsystem (LLM, ASR, TTS, Detector, etc.) defines an ABC interface.
Concrete implementations register themselves with ``@register``.
Modules resolve implementations by name from config at build time.

Usage::

    # 1. Define interface
    class LLMBackend(ABC):
        @abstractmethod
        async def chat(self, messages) -> str: ...

    # 2. Register implementations
    @llm_registry.register("minimax")
    class MiniMaxLLM(LLMBackend): ...

    @llm_registry.register("openai")
    class OpenAILLM(LLMBackend): ...

    # 3. Resolve from config
    backend = llm_registry.create(cfg)  # reads cfg["backend"]

    # 4. Add new backend = one class + one decorator. Zero changes elsewhere.
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BackendRegistry:
    """Registry for pluggable backend implementations.

    Each registry is typed to a specific ABC interface (LLM, ASR, TTS, etc.).
    Implementations register with ``@registry.register("name")``.
    """

    def __init__(self, name: str, interface: type, default: str = "") -> None:
        self.name = name
        self.interface = interface
        self.default = default
        self._backends: dict[str, type] = {}

    def register(self, name: str):
        """Decorator: register a backend implementation.

        ::

            @llm_registry.register("minimax")
            class MiniMaxLLM(LLMBackend):
                ...
        """
        def decorator(cls):
            if not issubclass(cls, self.interface):
                raise TypeError(
                    f"{cls.__name__} must implement {self.interface.__name__}"
                )
            self._backends[name] = cls
            logger.debug("Registered %s backend: %s → %s", self.name, name, cls.__name__)
            return cls
        return decorator

    def create(self, cfg: dict[str, Any] | None = None) -> Any:
        """Create a backend instance from config.

        Reads ``cfg["backend"]`` to select implementation.
        Falls back to ``self.default`` if not specified.
        """
        cfg = cfg or {}
        name = cfg.get("backend", self.default)
        if not name:
            raise ValueError(
                f"No backend specified for {self.name}. "
                f"Set 'backend' in config. Available: {self.available()}"
            )
        cls = self._backends.get(name)
        if cls is None:
            raise KeyError(
                f"Unknown {self.name} backend: '{name}'. "
                f"Available: {self.available()}"
            )
        return cls(cfg)

    def available(self) -> list[str]:
        """List registered backend names."""
        return sorted(self._backends.keys())

    def get_class(self, name: str) -> type | None:
        """Get backend class by name without instantiating."""
        return self._backends.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._backends

    def __len__(self) -> int:
        return len(self._backends)

    def __repr__(self) -> str:
        return f"BackendRegistry({self.name!r}, backends={self.available()})"
