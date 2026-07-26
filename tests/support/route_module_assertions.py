"""Shared assertions for route split and registration tests."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def parse_python_module(path: str | Path) -> ast.AST:
    module_path = Path(path)
    return ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))


def function_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def imports_by_module(tree: ast.AST) -> dict[str, set[str]]:
    imports: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.setdefault(node.module, set()).update(alias.name for alias in node.names)
    return imports


def route_paths(app: Any, prefix: str) -> list[str]:
    paths: list[str] = []
    for route in _iter_effective_routes(app):
        path = getattr(route, "path", "")
        if path.startswith(prefix):
            paths.append(path)
    return paths


def route_method_counts(
    app: Any,
    prefix: str,
    *,
    ignored_methods: Iterable[str] = ("HEAD",),
) -> dict[tuple[str, str], int]:
    ignored = set(ignored_methods)
    route_methods: dict[tuple[str, str], int] = {}
    for route in _iter_effective_routes(app):
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set()) or set()
        if not path.startswith(prefix):
            continue
        for method in methods:
            if method in ignored:
                continue
            key = (path, method)
            route_methods[key] = route_methods.get(key, 0) + 1
    return route_methods


def duplicate_route_methods(
    app: Any,
    prefix: str,
    *,
    ignored_methods: Iterable[str] = ("HEAD",),
) -> dict[tuple[str, str], int]:
    counts = route_method_counts(app, prefix, ignored_methods=ignored_methods)
    return {key: count for key, count in counts.items() if count != 1}


def _iter_effective_routes(app: Any) -> Iterator[Any]:
    for route in app.router.routes:
        contexts = getattr(route, "effective_route_contexts", None)
        if callable(contexts):
            yield from contexts()
        else:
            yield route
