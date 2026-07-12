"""API surface composition for the legacy health app factory."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.admin.routes import register_admin_routes
from askme.api.internal.routes import register_internal_routes
from askme.api.platform.routes import register_platform_routes
from askme.api.product.routes import register_product_routes
from askme.api.routes.governance import (
    AuthorizationPayload,
    CurrentOperatorPayload,
)
from askme.api.routes.runtime import OperatorActionKwargs
from askme.api.routes.vision import (
    ArchiveDeleteHandler,
    ArchiveGetHandler,
    ArchiveListHandler,
    ArchiveSnapshotHandler,
    VisionAnalyzeHandler,
    VisionSnapshotHandler,
)
from askme.api.services.conversation_service import ConversationService
from askme.api.services.monitor_service import MonitorService


@dataclass(frozen=True, slots=True)
class ApiSurfaceSpec:
    """Product contract for one HTTP API surface.

    This is intentionally small and code-owned. It lets Dashboard, docs, tests,
    and future app factories reason about audience boundaries without parsing
    filenames or copying comments.
    """

    name: str
    package: str
    registrar: str
    audience: str
    owns: tuple[str, ...]
    route_modules: tuple[str, ...] = ()
    must_not_expose: tuple[str, ...] = ()
    customer_visible: bool = False
    hardware_authority_allowed: bool = False
    production_claim_allowed: bool = False
    customer_boundary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "package": self.package,
            "registrar": self.registrar,
            "audience": self.audience,
            "owns": list(self.owns),
            "route_modules": list(self.route_modules),
            "must_not_expose": list(self.must_not_expose),
            "customer_visible": self.customer_visible,
            "hardware_authority_allowed": self.hardware_authority_allowed,
            "production_claim_allowed": self.production_claim_allowed,
            "customer_boundary": self.customer_boundary,
        }


API_SURFACES: tuple[ApiSurfaceSpec, ...] = (
    ApiSurfaceSpec(
        name="platform",
        package="askme.api.platform",
        registrar="register_platform_routes",
        audience="operations and deployment monitoring",
        owns=("health", "metrics", "system status", "monitor snapshots"),
        route_modules=(
            "askme.api.routes.health",
            "askme.api.routes.monitor",
            "askme.api.routes.system",
        ),
        customer_boundary="只展示系统健康、指标和接口分层状态，不承载客户业务操作。",
    ),
    ApiSurfaceSpec(
        name="product",
        package="askme.api.product",
        registrar="register_product_routes",
        audience="customer dashboard and operator workflows",
        owns=(
            "conversation",
            "customer knowledge",
            "field events",
            "space guidance",
            "capabilities",
            "missions",
            "voice profiles",
            "dashboard pages",
        ),
        route_modules=(
            "askme.api.routes.capabilities",
            "askme.api.routes.conversation",
            "askme.api.routes.dashboard",
            "askme.api.routes.field_customer_project_acceptance",
            "askme.api.routes.field_customer_project_artifacts",
            "askme.api.routes.field_customer_project_execution",
            "askme.api.routes.field_customer_project_profiles",
            "askme.api.routes.field_customer_project_templates",
            "askme.api.routes.field_events",
            "askme.api.routes.field_product_catalog",
            "askme.api.routes.memory",
            "askme.api.routes.mission",
            "askme.api.routes.space",
            "askme.api.routes.voice",
        ),
        must_not_expose=(
            "arbiter",
            "raw handoff internals",
            "runtime advance controls",
            "direct hardware authority",
        ),
        customer_visible=True,
        customer_boundary=(
            "客户和操作员可见的业务入口，只能展示对话、知识、能力、任务和现场事件；"
            "不能直接暴露硬件控制或内部运行仲裁。"
        ),
    ),
    ApiSurfaceSpec(
        name="admin",
        package="askme.api.admin",
        registrar="register_admin_routes",
        audience="supervisors, delivery engineers, and product operators",
        owns=("identity readiness", "authorization", "audit", "skill governance", "agent profiles"),
        route_modules=(
            "askme.api.routes.agent_profiles",
            "askme.api.routes.audit",
            "askme.api.routes.field_admin",
            "askme.api.routes.field_delivery_resources",
            "askme.api.routes.governance",
            "askme.api.routes.skills",
        ),
        customer_boundary="面向主管、交付和产品管理员；高风险配置、审批和审计不应出现在游客或普通客户入口。",
    ),
    ApiSurfaceSpec(
        name="internal",
        package="askme.api.internal",
        registrar="register_internal_routes",
        audience="robot runtime, devices, and low-level integrations",
        owns=(
            "runtime callbacks",
            "cognition",
            "vision",
            "device ingest",
            "device onboarding evidence",
            "robot bridges",
        ),
        route_modules=(
            "askme.api.routes.cognition",
            "askme.api.routes.field_internal",
            "askme.api.routes.runtime",
            "askme.api.routes.vision",
        ),
        must_not_expose=("customer copy", "sales-facing release claims"),
        hardware_authority_allowed=True,
        customer_boundary=(
            "只面向机器人运行时、设备和底层集成；可以承接硬件相关回调，"
            "但不能作为客户 UI 或销售口径。"
        ),
    ),
)


def _surface_registrars() -> dict[str, Callable[[FastAPI, Any], None]]:
    """Return live API surface registrars.

    Kept as a function so tests and app factories can monkeypatch the registrar
    symbols without relying on string lookups through module globals.
    """

    return {
        "platform": register_platform_routes,
        "product": register_product_routes,
        "admin": register_admin_routes,
        "internal": register_internal_routes,
    }


@dataclass(slots=True)
class ApiRouteDependencies:
    """Dependencies required to register every HTTP API surface."""

    health_provider: Callable[[], Any]
    metrics_provider: Callable[[], Any]
    render_prometheus_metrics: Callable[[Mapping[str, Any]], str]
    json_snapshot_response: Callable[[Mapping[str, Any]], JSONResponse]
    snapshot_payload: Callable[[], Mapping[str, Any]]
    prometheus_content_type: str
    governance_payload: Callable[[], dict[str, Any]]
    identity_readiness_payload: Callable[[], dict[str, Any]]
    current_operator_payload: CurrentOperatorPayload
    authorization_payload: AuthorizationPayload
    mission_json: Callable[..., JSONResponse]
    cors_options_response: Callable[[str], Response]
    dispatch_memory: Callable[[str, dict[str, Any]], Any]
    logger: logging.Logger
    authorize: Callable[..., JSONResponse | None]
    dispatch_cognition: Callable[..., Any]
    json_error: Callable[..., JSONResponse]
    cors_headers: Mapping[str, str]
    dispatch_runtime: Callable[..., Any]
    optional_json_body: Callable[[Request], Any]
    operator_action_kwargs: OperatorActionKwargs
    dispatch_voice: Callable[..., Any]
    dispatch_space: Callable[[str, dict[str, Any]], Any]
    dispatch_field_operations: Callable[..., Any]
    field_manual_trigger_body: Callable[[Request, dict[str, Any]], dict[str, Any]]
    looks_like_device_ingest_without_scenario: Callable[[Mapping[str, Any]], bool]
    dispatch_field_voice_directive: Callable[[dict[str, Any]], Any]
    dispatch_field_runtime_policy: Callable[..., Any]
    runtime_callback_trust: Callable[..., dict[str, Any]]
    runtime_callback_delivery_body: Callable[..., dict[str, Any]]
    runtime_callback_secret: str | None
    runtime_callback_max_age_s: float
    field_path_roots: Mapping[str, Any]
    config_provider: Callable[[], Mapping[str, Any]]
    dashboard_html: str
    dashboard_asset_dir: Any
    dashboard_pages: Mapping[str, Any]
    capabilities_provider: Callable[[], Mapping[str, Any]]
    blueprints_provider: Callable[[], Mapping[str, Any]]
    operator_id_from_request: Callable[[Request, dict[str, Any]], str]
    conversation_service: ConversationService
    runtime_available: bool
    runtime_voice_turn_timeout_s: float
    monitor_service: MonitorService
    dispatch_mission: Callable[..., Any]
    request_has_control_auth: Callable[[Request], bool]
    skill_growth_candidate_prompt: Callable[[dict[str, Any]], str]
    vision_snapshot_handler: VisionSnapshotHandler | None
    vision_analyze_handler: VisionAnalyzeHandler | None
    archive_snapshot_handler: ArchiveSnapshotHandler | None
    archive_list_handler: ArchiveListHandler | None
    archive_get_handler: ArchiveGetHandler | None
    archive_delete_handler: ArchiveDeleteHandler | None


def register_api_routes(app: FastAPI, deps: ApiRouteDependencies) -> None:
    """Register all API surfaces that already live outside health_server."""

    registrars = _surface_registrars()
    for spec in API_SURFACES:
        registrar = registrars[spec.name]
        registrar(app, deps)


def api_surface_manifest() -> list[dict[str, Any]]:
    """Return the customer/product boundary map for HTTP API surfaces."""

    return [spec.to_dict() for spec in API_SURFACES]


def api_surface_module_map() -> dict[str, str]:
    """Return the module-to-surface ownership contract for HTTP routes."""

    module_map: dict[str, str] = {}
    for spec in API_SURFACES:
        for module in spec.route_modules:
            existing = module_map.get(module)
            if existing and existing != spec.name:
                raise ValueError(f"API route module {module!r} is assigned to {existing!r} and {spec.name!r}")
            module_map[module] = spec.name
    return module_map


def api_surface_for_route_module(module: str) -> str:
    """Classify an endpoint module into an API surface, or ``unclassified``."""

    module_map = api_surface_module_map()
    if module in module_map:
        return module_map[module]
    for owner_module, surface in module_map.items():
        if module.startswith(owner_module + "."):
            return surface
    return "unclassified"


_INVENTORY_PATH_PREFIXES = (
    "/api/",
    "/dashboard",
)
_INVENTORY_PATHS = {
    "/health",
    "/healthz",
    "/metrics",
    "/metrics/prometheus",
    "/trace",
}
_FRAMEWORK_DOC_ROUTE_NAMES = {
    "openapi",
    "swagger_ui_html",
    "swagger_ui_redirect",
    "redoc_html",
}


def _route_methods(route: Any) -> list[str]:
    methods = set(getattr(route, "methods", set()) or set())
    methods.discard("HEAD")
    return sorted(methods)


def _joined_route_path(prefix: str, path: str) -> str:
    """Join an included-router prefix with a child route path."""

    clean_prefix = str(prefix or "").rstrip("/")
    clean_path = str(path or "")
    if not clean_prefix:
        return clean_path
    if not clean_path:
        return clean_prefix or "/"
    return clean_prefix + (clean_path if clean_path.startswith("/") else f"/{clean_path}")


def _iter_inventory_route_candidates(
    routes: Any,
    *,
    prefix: str = "",
) -> list[tuple[Any, str]]:
    """Flatten FastAPI routes, including new lazy ``include_router`` wrappers."""

    candidates: list[tuple[Any, str]] = []
    for route in routes:
        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            include_context = getattr(route, "include_context", None)
            include_prefix = str(getattr(include_context, "prefix", "") or "")
            candidates.extend(
                _iter_inventory_route_candidates(
                    getattr(original_router, "routes", ()) or (),
                    prefix=_joined_route_path(prefix, include_prefix),
                )
            )
            continue
        path = _joined_route_path(prefix, str(getattr(route, "path", "")))
        candidates.append((route, path))
    return candidates


def _route_is_inventory_candidate(route: Any, *, path: str | None = None) -> bool:
    resolved_path = str(path if path is not None else getattr(route, "path", ""))
    name = str(getattr(route, "name", ""))
    if name in _FRAMEWORK_DOC_ROUTE_NAMES:
        return False
    if resolved_path in _INVENTORY_PATHS:
        return True
    return any(resolved_path.startswith(prefix) for prefix in _INVENTORY_PATH_PREFIXES)


def api_route_inventory(app: FastAPI) -> dict[str, Any]:
    """Return a machine-checkable inventory of HTTP routes by product surface."""

    routes: list[dict[str, Any]] = []
    for route, path in _iter_inventory_route_candidates(app.routes):
        if not _route_is_inventory_candidate(route, path=path):
            continue
        endpoint = getattr(route, "endpoint", None)
        module = str(getattr(endpoint, "__module__", ""))
        surface = api_surface_for_route_module(module)
        routes.append(
            {
                "path": path,
                "methods": _route_methods(route),
                "name": str(getattr(route, "name", "")),
                "module": module,
                "surface": surface,
            }
        )

    routes.sort(key=lambda item: (item["surface"], item["path"], ",".join(item["methods"]), item["name"]))
    route_counts = Counter(route["surface"] for route in routes)
    module_counts: dict[str, int] = {}
    for surface in route_counts:
        module_counts[surface] = len({route["module"] for route in routes if route["surface"] == surface})

    surface_summaries = {
        spec.name: {
            "route_count": route_counts.get(spec.name, 0),
            "module_count": module_counts.get(spec.name, 0),
            "declared_modules": list(spec.route_modules),
            "customer_visible": spec.customer_visible,
            "hardware_authority_allowed": spec.hardware_authority_allowed,
            "production_claim_allowed": spec.production_claim_allowed,
            "customer_boundary": spec.customer_boundary,
        }
        for spec in API_SURFACES
    }
    unclassified_routes = [route for route in routes if route["surface"] == "unclassified"]
    return {
        "summary": {
            "total_route_count": len(routes),
            "unclassified_count": len(unclassified_routes),
            "surface_route_counts": dict(sorted(route_counts.items())),
            "surface_module_counts": dict(sorted(module_counts.items())),
        },
        "surfaces": surface_summaries,
        "routes": routes,
        "unclassified_routes": unclassified_routes,
        "policy": {
            "route_modules_are_contract_source": True,
            "new_customer_routes_must_be_product_classified": True,
            "new_robot_runtime_routes_must_be_internal_classified": True,
            "unclassified_routes_should_block_product_readiness": True,
        },
    }


def api_surface_readiness(route_inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Return a customer-readable gate for API surface separation."""

    summary = route_inventory.get("summary") if isinstance(route_inventory.get("summary"), Mapping) else {}
    surfaces = route_inventory.get("surfaces") if isinstance(route_inventory.get("surfaces"), Mapping) else {}
    unclassified_count = int(summary.get("unclassified_count") or 0)
    missing_surfaces = [
        spec.name
        for spec in API_SURFACES
        if not int(
            (
                surfaces.get(spec.name, {})
                if isinstance(surfaces.get(spec.name), Mapping)
                else {}
            ).get("route_count")
            or 0
        )
    ]
    product_count = int(
        (surfaces.get("product", {}) if isinstance(surfaces.get("product"), Mapping) else {})
        .get("route_count")
        or 0
    )
    internal_count = int(
        (surfaces.get("internal", {}) if isinstance(surfaces.get("internal"), Mapping) else {})
        .get("route_count")
        or 0
    )
    status = "ready" if unclassified_count == 0 and not missing_surfaces else "blocked"
    product_surface = (
        surfaces.get("product", {}) if isinstance(surfaces.get("product"), Mapping) else {}
    )
    internal_surface = (
        surfaces.get("internal", {}) if isinstance(surfaces.get("internal"), Mapping) else {}
    )
    product_hardware_allowed = bool(product_surface.get("hardware_authority_allowed"))
    product_production_claim_allowed = bool(product_surface.get("production_claim_allowed"))
    internal_customer_visible = bool(internal_surface.get("customer_visible"))
    if product_hardware_allowed or product_production_claim_allowed or internal_customer_visible:
        status = "blocked"
    policy = {
        "product_surface_is_customer_visible": product_count > 0,
        "internal_surface_must_not_drive_customer_ui": internal_count > 0 and unclassified_count == 0,
        "product_surface_must_not_allow_hardware_authority": not product_hardware_allowed,
        "api_surface_must_not_be_production_claim_source": not product_production_claim_allowed,
        "internal_surface_must_not_be_customer_visible": not internal_customer_visible,
        "admin_surface_requires_operator_governance": (
            int(
                (surfaces.get("admin", {}) if isinstance(surfaces.get("admin"), Mapping) else {})
                .get("route_count")
                or 0
            )
            > 0
        ),
        "legacy_paths_may_remain_during_migration": True,
        "route_inventory_has_no_unclassified_routes": unclassified_count == 0,
        "all_declared_surfaces_have_routes": not missing_surfaces,
        "surface_manifest_is_customer_boundary_contract": True,
    }
    blockers: list[str] = []
    if unclassified_count:
        blockers.append("Classify every HTTP route into platform, product, admin, or internal.")
    if missing_surfaces:
        blockers.append("Register routes for missing API surfaces: " + ", ".join(missing_surfaces))
    if product_hardware_allowed:
        blockers.append("Product API surface must not allow direct hardware authority.")
    if product_production_claim_allowed:
        blockers.append("Product API surface must not be used as a production launch claim source.")
    if internal_customer_visible:
        blockers.append("Internal robot/runtime API surface must not be customer visible.")
    return {
        "readiness_type": "askme.api_surface_readiness.v1",
        "overall_status": status,
        "customer_status": (
            "客户可见接口、治理接口和机器人内部接口边界清晰。"
            if status == "ready"
            else "API 边界仍有未分类或缺失的接口，不能作为产品交付口径。"
        ),
        "release_claim": (
            "可以向客户说明 Dashboard 按页面和权限使用客户、治理和平台接口，"
            "内部机器人控制接口不会驱动客户 UI 或客户交付口径。"
            if status == "ready"
            else "只能说明 API 分层仍在整改，不能承诺接口边界已完成。"
        ),
        "policy": policy,
        "blockers": blockers,
        "summary": {
            "surface_count": len(API_SURFACES),
            "total_route_count": int(summary.get("total_route_count") or 0),
            "unclassified_count": unclassified_count,
            "missing_surface_count": len(missing_surfaces),
            "missing_surfaces": missing_surfaces,
        },
    }


__all__ = [
    "API_SURFACES",
    "ApiRouteDependencies",
    "ApiSurfaceSpec",
    "api_route_inventory",
    "api_surface_readiness",
    "api_surface_for_route_module",
    "api_surface_manifest",
    "api_surface_module_map",
    "register_api_routes",
]
