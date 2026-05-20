"""Import compatibility helpers for package reorganizations."""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from collections.abc import Mapping
from importlib import import_module
from types import ModuleType
from typing import Any


class LegacyAliasLoader(importlib.abc.Loader):
    """Loader that redirects a historical module name to its canonical module."""

    def __init__(self, fullname: str, target: str) -> None:
        self.fullname = fullname
        self.target = target
        self._canonical_loader: Any = None
        self._canonical_package: str | None = None
        self._canonical_spec: importlib.machinery.ModuleSpec | None = None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = import_module(self.target)
        self._canonical_loader = getattr(module, "__loader__", None)
        self._canonical_package = getattr(module, "__package__", None)
        self._canonical_spec = getattr(module, "__spec__", None)
        sys.modules[self.fullname] = module
        return module

    def exec_module(self, module: ModuleType) -> None:
        if self._canonical_spec is not None:
            module.__spec__ = self._canonical_spec
        if self._canonical_loader is not None:
            module.__loader__ = self._canonical_loader
        if self._canonical_package is not None:
            module.__package__ = self._canonical_package
        sys.modules[self.fullname] = module

    def get_code(self, fullname: str) -> Any:
        _ = fullname
        spec = importlib.util.find_spec(self.target)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "get_code"):
            raise ImportError(f"Cannot load code for legacy alias {self.fullname}")
        return spec.loader.get_code(self.target)  # type: ignore[attr-defined]


class LegacyAliasFinder(importlib.abc.MetaPathFinder):
    """Meta path finder for one package's historical module aliases."""

    def __init__(self, namespace: str, aliases: Mapping[str, str]) -> None:
        self.namespace = namespace
        self.aliases = dict(aliases)

    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        _ = path, target
        alias_target = self.aliases.get(fullname)
        if not alias_target:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            LegacyAliasLoader(fullname, alias_target),
            origin=f"legacy-alias:{alias_target}",
        )


def install_legacy_aliases(namespace: str, aliases: Mapping[str, str]) -> None:
    """Install a package-scoped legacy alias finder once per interpreter."""
    if not aliases:
        return
    for finder in sys.meta_path:
        if isinstance(finder, LegacyAliasFinder) and finder.namespace == namespace:
            return
    sys.meta_path.insert(0, LegacyAliasFinder(namespace, aliases))
