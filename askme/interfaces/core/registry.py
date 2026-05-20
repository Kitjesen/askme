"""Backend registry for pluggable interface implementations."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Registry for named backend implementations of one interface."""

    def __init__(self, name: str, interface: type, default: str = "") -> None:
        self.name = name
        self.interface = interface
        self.default = default
        self._backends: dict[str, type] = {}

    def register(self, name: str):
        """Decorator: register a backend implementation under ``name``."""

        def decorator(cls):
            # Existing implementations predate the ABCs, so registration stays
            # soft until every concrete backend inherits its interface.
            if not issubclass(cls, self.interface):
                logger.debug(
                    "%s backend %r (%s) does not inherit %s; registered anyway",
                    self.name,
                    name,
                    cls.__name__,
                    self.interface.__name__,
                )
            self._backends[name] = cls
            logger.debug("Registered %s backend: %s -> %s", self.name, name, cls.__name__)
            return cls

        return decorator

    def create(self, cfg: dict[str, Any] | None = None) -> Any:
        """Create a backend instance from config."""

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
                f"Unknown {self.name} backend: {name!r}. "
                f"Available: {self.available()}"
            )
        try:
            return cls(**cfg)
        except TypeError:
            return cls(cfg)

    def available(self) -> list[str]:
        """List registered backend names."""

        return sorted(self._backends.keys())

    def get_class(self, name: str) -> type | None:
        """Get a backend class without instantiating it."""

        return self._backends.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._backends

    def __len__(self) -> int:
        return len(self._backends)

    def __repr__(self) -> str:
        return f"BackendRegistry({self.name!r}, backends={self.available()})"


__all__ = ["BackendRegistry"]
