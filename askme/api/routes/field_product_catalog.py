"""Customer-project product catalog routes for field operations."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.customer_projects import CustomerProjectAcceptanceRegistryResponse
from askme.api.schemas.customer_projects import CustomerProjectCatalogResponse
from askme.api.schemas.customer_projects import CustomerProjectResourceCatalogResponse
from askme.api.schemas.customer_projects import CustomerProjectWorkbenchResponse
from askme.api.schemas.customer_projects import ManagedObjectDirectoryResponse
from askme.api.schemas.customer_projects import SiteProfileCatalogResponse
from askme.api.schemas.delivery_readiness import ProductLaunchReadinessResponse
from askme.api.schemas.delivery_readiness import SolutionDeliveryReadinessResponse
from askme.api.routes.field_template_scope import (
    scope_template_catalog as _scope_template_catalog,
)
from askme.api.services.field_customer_project_workbench import (
    build_customer_project_workbench_payload,
)
from askme.api.services.field_managed_object_directory import (
    filter_managed_object_directory_rows,
    managed_object_directory_rows,
    managed_object_directory_summary,
)
from askme.api.services.field_project_catalog_scope import (
    scope_project_catalog,
    scope_site_catalog,
)
from askme.api.services.field_resource_catalog_scope import (
    scope_acceptance_registry,
    scope_resource_catalog,
)
from askme.pipeline.field.customer_project_templates import (
    list_customer_project_templates,
)
from askme.pipeline.field.customer_projects import (
    build_customer_project_acceptance_registry,
    build_customer_project_catalog,
    build_customer_project_resource_catalog,
    build_site_profile_catalog,
    build_solution_delivery_readiness,
)
from askme.pipeline.field.delivery_resources import (
    list_delivery_resource_governance_requests,
)
from askme.pipeline.field.product_launch_readiness import build_product_launch_readiness

Dispatch = Callable[..., Awaitable[dict[str, Any]]]
MissionJson = Callable[..., JSONResponse]
CorsOptions = Callable[[str], Response]
ProjectReadAuth = Callable[[Request], tuple[JSONResponse | None, dict[str, Any]]]
OperatorProjectScope = Callable[[dict[str, Any]], dict[str, list[str]]]
ScopeAllows = Callable[[dict[str, list[str]], dict[str, Any]], bool]
ScopeItem = Callable[[dict[str, Any]], dict[str, Any]]
HasExplicitResourceScope = Callable[[dict[str, Any]], bool]
PathProvider = Callable[[], Path]
IdentityReadinessPayload = Callable[[], dict[str, Any]]
DashboardPagesPayload = Callable[[], dict[str, Any]]
CatalogBuilder = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class FieldProductCatalogPipeline:
    """Pipeline callables used by the field product catalog routes."""

    build_site_profile_catalog: CatalogBuilder = build_site_profile_catalog
    build_customer_project_catalog: CatalogBuilder = build_customer_project_catalog
    build_customer_project_acceptance_registry: CatalogBuilder = (
        build_customer_project_acceptance_registry
    )
    build_customer_project_resource_catalog: CatalogBuilder = (
        build_customer_project_resource_catalog
    )
    build_solution_delivery_readiness: CatalogBuilder = build_solution_delivery_readiness
    list_customer_project_templates: CatalogBuilder = list_customer_project_templates
    list_delivery_resource_governance_requests: CatalogBuilder = (
        list_delivery_resource_governance_requests
    )
    build_product_launch_readiness: CatalogBuilder = build_product_launch_readiness


def register_field_product_catalog_routes(
    app: FastAPI,
    *,
    dispatch_field_operations: Dispatch,
    mission_json: MissionJson,
    project_read_auth: ProjectReadAuth,
    operator_project_scope: OperatorProjectScope,
    scope_allows: ScopeAllows,
    scope_item_from_site: ScopeItem,
    scope_item_from_resource: ScopeItem,
    resource_has_explicit_scope: HasExplicitResourceScope,
    site_profile_root: PathProvider,
    template_root: PathProvider,
    delivery_resource_root: PathProvider,
    identity_readiness_payload: IdentityReadinessPayload,
    dashboard_pages_payload: DashboardPagesPayload | None = None,
    pipeline: FieldProductCatalogPipeline | None = None,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
) -> None:
    """Register customer-readable project catalog, resource, and readiness routes."""

    app.include_router(
        create_field_product_catalog_router(
            dispatch_field_operations=dispatch_field_operations,
            mission_json=mission_json,
            project_read_auth=project_read_auth,
            operator_project_scope=operator_project_scope,
            scope_allows=scope_allows,
            scope_item_from_site=scope_item_from_site,
            scope_item_from_resource=scope_item_from_resource,
            resource_has_explicit_scope=resource_has_explicit_scope,
            site_profile_root=site_profile_root,
            template_root=template_root,
            delivery_resource_root=delivery_resource_root,
            identity_readiness_payload=identity_readiness_payload,
            dashboard_pages_payload=dashboard_pages_payload,
            pipeline=pipeline,
            cors_options_response=cors_options_response,
            logger=logger,
        )
    )


def create_field_product_catalog_router(
    *,
    dispatch_field_operations: Dispatch,
    mission_json: MissionJson,
    project_read_auth: ProjectReadAuth,
    operator_project_scope: OperatorProjectScope,
    scope_allows: ScopeAllows,
    scope_item_from_site: ScopeItem,
    scope_item_from_resource: ScopeItem,
    resource_has_explicit_scope: HasExplicitResourceScope,
    site_profile_root: PathProvider,
    template_root: PathProvider,
    delivery_resource_root: PathProvider,
    identity_readiness_payload: IdentityReadinessPayload,
    dashboard_pages_payload: DashboardPagesPayload | None = None,
    pipeline: FieldProductCatalogPipeline | None = None,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
) -> APIRouter:
    """Create the field product catalog router without binding it to an app factory."""

    pipeline = pipeline or FieldProductCatalogPipeline()
    router = APIRouter(tags=["Field Operations"])

    @router.get(
        "/api/field/site-profiles",
        response_model=SiteProfileCatalogResponse,
        response_model_exclude_none=True,
    )
    async def field_site_profiles(request: Request, check_env: bool = False) -> JSONResponse:
        """Return the multi-site field deployment catalog."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            root = site_profile_root()
            result = pipeline.build_site_profile_catalog(root, check_env=check_env)
            result = scope_site_catalog(
                result,
                scope,
                scope_allows=scope_allows,
                scope_item_from_site=scope_item_from_site,
            )
            response = SiteProfileCatalogResponse.model_validate(result)
            return mission_json(response.model_dump(mode="python"))
        except Exception as exc:
            logger.error("Field site profile catalog endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/customer-projects",
        response_model=CustomerProjectCatalogResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_projects(
        request: Request,
        check_env: bool = False,
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        industry: str = "",
        gate_status: str = "",
        deployment_stage: str = "",
    ) -> JSONResponse:
        """Return customer, project, site, and managed-object rollout coverage."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            root = site_profile_root()
            result = pipeline.build_customer_project_catalog(
                root,
                check_env=check_env,
                tenant_id=tenant_id,
                delivery_namespace=delivery_namespace,
                customer_id=customer_id,
                project_id=project_id,
                site_id=site_id,
                industry=industry,
                gate_status=gate_status,
                deployment_stage=deployment_stage,
            )
            result = scope_project_catalog(
                result,
                scope,
                scope_allows=scope_allows,
            )
            response = CustomerProjectCatalogResponse.model_validate(result)
            return mission_json(response.model_dump(mode="python"))
        except Exception as exc:
            logger.error("Field customer project catalog endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/customer-projects/managed-object-directory",
        response_model=ManagedObjectDirectoryResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_managed_object_directory(
        request: Request,
        check_env: bool = False,
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        industry: str = "",
        gate_status: str = "",
        deployment_stage: str = "",
        delivery_status: str = "",
        category: str = "",
        customer_visible: str = "",
    ) -> JSONResponse:
        """Return scoped managed-object bindings for delivery and acceptance review."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            root = site_profile_root()
            catalog = pipeline.build_customer_project_catalog(
                root,
                check_env=check_env,
                tenant_id=tenant_id,
                delivery_namespace=delivery_namespace,
                customer_id=customer_id,
                project_id=project_id,
                site_id=site_id,
                industry=industry,
                gate_status=gate_status,
                deployment_stage=deployment_stage,
            )
            catalog = scope_project_catalog(
                catalog,
                scope,
                scope_allows=scope_allows,
            )
            projects = [
                project
                for project in catalog.get("projects", [])
                if isinstance(project, dict)
            ]
            rows = managed_object_directory_rows(projects)
            rows, object_filters = filter_managed_object_directory_rows(
                rows,
                delivery_status=delivery_status,
                category=category,
                customer_visible=customer_visible,
            )
            base_summary = catalog.get("summary") if isinstance(catalog.get("summary"), dict) else {}
            summary = managed_object_directory_summary(
                rows,
                projects=projects,
                base_summary=base_summary,
                filtered=bool(object_filters) or bool(base_summary.get("filtered")),
            )
            filters = dict(catalog.get("filters") or {})
            filters.update(object_filters)
            response = ManagedObjectDirectoryResponse.model_validate(
                {
                    "directory_type": "askme.customer_project_managed_object_directory",
                    "root": catalog.get("root") or str(root),
                    "check_env": check_env,
                    "filters": filters,
                    "summary": summary,
                    "objects": rows,
                    "customer_status": (
                        "对象目录已按当前操作人范围过滤，可用于交付复核。"
                        if rows
                        else "当前操作人范围和筛选条件下没有可见的现场对象。"
                    ),
                    "next_step": (
                        "导出客户交付包前，先处理阻断或待复核的对象。"
                        if rows
                        else "调整客户项目筛选条件，或检查操作人的项目权限范围。"
                    ),
                }
            )
            return mission_json(response.model_dump(mode="python"))
        except Exception as exc:
            logger.error("Field managed object directory endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/customer-project-acceptance-registry",
        response_model=CustomerProjectAcceptanceRegistryResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_acceptance_registry(request: Request) -> JSONResponse:
        """Return managed-object acceptance references across projects and templates."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            root = site_profile_root()
            result = pipeline.build_customer_project_acceptance_registry(
                root,
                template_root=template_root(),
            )
            result = scope_acceptance_registry(
                result,
                scope,
                scope_allows=scope_allows,
            )
            response = CustomerProjectAcceptanceRegistryResponse.model_validate(result)
            return mission_json(response.model_dump(mode="python"))
        except Exception as exc:
            logger.error("Field customer project acceptance registry endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/customer-project-resource-catalog",
        response_model=CustomerProjectResourceCatalogResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_resource_catalog(request: Request) -> JSONResponse:
        """Return model, protocol, skill, and acceptance bindings used by projects."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            result = pipeline.build_customer_project_resource_catalog(
                site_profile_root(),
                template_root=template_root(),
            )
            result = scope_resource_catalog(
                result,
                scope,
                scope_allows=scope_allows,
                scope_item_from_resource=scope_item_from_resource,
                resource_has_explicit_scope=resource_has_explicit_scope,
            )
            response = CustomerProjectResourceCatalogResponse.model_validate(result)
            return mission_json(response.model_dump(mode="python"))
        except Exception as exc:
            logger.error("Field customer project resource catalog endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/solution-delivery-readiness",
        response_model=SolutionDeliveryReadinessResponse,
        response_model_exclude_none=True,
    )
    async def field_solution_delivery_readiness(
        request: Request,
        check_env: bool = False,
    ) -> JSONResponse:
        """Return one product-facing readiness gate for solution-provider delivery."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            project_root = site_profile_root()
            template_catalog_root = template_root()
            resource_root = delivery_resource_root()
            project_catalog = scope_project_catalog(
                pipeline.build_customer_project_catalog(project_root, check_env=check_env),
                scope,
                scope_allows=scope_allows,
            )
            template_catalog = _scope_template_catalog(
                pipeline.list_customer_project_templates(template_catalog_root),
                scope,
            )
            resource_catalog = scope_resource_catalog(
                pipeline.build_customer_project_resource_catalog(
                    project_root,
                    template_root=template_catalog_root,
                    delivery_resource_root=resource_root,
                ),
                scope,
                scope_allows=scope_allows,
                scope_item_from_resource=scope_item_from_resource,
                resource_has_explicit_scope=resource_has_explicit_scope,
            )
            if any(scope.values()):
                governance_requests = {
                    "skipped": True,
                    "reason": "resource_governance_requests_require_unrestricted_operator",
                }
            else:
                governance_requests = pipeline.list_delivery_resource_governance_requests(
                    resource_root,
                    limit=20,
                )
            result = pipeline.build_solution_delivery_readiness(
                project_catalog=project_catalog,
                template_catalog=template_catalog,
                resource_catalog=resource_catalog,
                governance_requests=governance_requests,
            )
            response = SolutionDeliveryReadinessResponse.model_validate(result)
            return mission_json(response.model_dump(mode="python"))
        except Exception as exc:
            logger.error("Field solution delivery readiness endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    def _customer_project_workbench_payload(
        *,
        scope: dict[str, list[str]],
        check_env: bool = False,
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        industry: str = "",
        gate_status: str = "",
        deployment_stage: str = "",
    ) -> dict[str, Any]:
        project_root = site_profile_root()
        template_catalog_root = template_root()
        resource_root = delivery_resource_root()
        project_catalog = scope_project_catalog(
            pipeline.build_customer_project_catalog(
                project_root,
                check_env=check_env,
                tenant_id=tenant_id,
                delivery_namespace=delivery_namespace,
                customer_id=customer_id,
                project_id=project_id,
                site_id=site_id,
                industry=industry,
                gate_status=gate_status,
                deployment_stage=deployment_stage,
            ),
            scope,
            scope_allows=scope_allows,
        )
        template_catalog = _scope_template_catalog(
            pipeline.list_customer_project_templates(template_catalog_root),
            scope,
        )
        resource_catalog = scope_resource_catalog(
            pipeline.build_customer_project_resource_catalog(
                project_root,
                template_root=template_catalog_root,
                delivery_resource_root=resource_root,
            ),
            scope,
            scope_allows=scope_allows,
            scope_item_from_resource=scope_item_from_resource,
            resource_has_explicit_scope=resource_has_explicit_scope,
        )
        projects = [
            project
            for project in project_catalog.get("projects", [])
            if isinstance(project, dict)
        ]
        object_rows = managed_object_directory_rows(projects)
        object_summary = managed_object_directory_summary(
            object_rows,
            projects=projects,
            base_summary=(
                project_catalog.get("summary")
                if isinstance(project_catalog.get("summary"), dict)
                else {}
            ),
            filtered=bool(project_catalog.get("filters")),
        )
        governance_requests = (
            {
                "skipped": True,
                "reason": "resource_governance_requests_require_unrestricted_operator",
            }
            if any(scope.values())
            else pipeline.list_delivery_resource_governance_requests(resource_root, limit=20)
        )
        readiness = pipeline.build_solution_delivery_readiness(
            project_catalog=project_catalog,
            template_catalog=template_catalog,
            resource_catalog=resource_catalog,
            governance_requests=governance_requests,
        )
        return build_customer_project_workbench_payload(
            project_catalog=project_catalog,
            template_catalog=template_catalog,
            resource_catalog=resource_catalog,
            object_summary=object_summary,
            object_rows=object_rows,
            projects=projects,
            readiness=readiness,
            scope_filtered=any(scope.values()),
        )

    @router.get(
        "/api/field/customer-project-workbench",
        response_model=CustomerProjectWorkbenchResponse,
        response_model_exclude_none=True,
    )
    async def field_customer_project_workbench(
        request: Request,
        check_env: bool = False,
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        industry: str = "",
        gate_status: str = "",
        deployment_stage: str = "",
    ) -> JSONResponse:
        """Return one solution-provider workbench payload for customer delivery."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            response = CustomerProjectWorkbenchResponse.model_validate(
                _customer_project_workbench_payload(
                    scope=scope,
                    check_env=check_env,
                    tenant_id=tenant_id,
                    delivery_namespace=delivery_namespace,
                    customer_id=customer_id,
                    project_id=project_id,
                    site_id=site_id,
                    industry=industry,
                    gate_status=gate_status,
                    deployment_stage=deployment_stage,
                )
            )
            return mission_json(response.model_dump(mode="python"))
        except Exception as exc:
            logger.error("Field customer project workbench endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.get(
        "/api/field/product-launch-readiness",
        response_model=ProductLaunchReadinessResponse,
        response_model_exclude_none=True,
    )
    async def field_product_launch_readiness(
        request: Request,
        check_env: bool = False,
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        industry: str = "",
        gate_status: str = "",
        deployment_stage: str = "",
    ) -> JSONResponse:
        """Return one customer-facing launch decision across product gates."""
        try:
            failure, auth_body = project_read_auth(request)
            if failure is not None:
                return failure
            scope = operator_project_scope(auth_body)
            field_readiness = await dispatch_field_operations("readiness_payload")
            workbench = _customer_project_workbench_payload(
                scope=scope,
                check_env=check_env,
                tenant_id=tenant_id,
                delivery_namespace=delivery_namespace,
                customer_id=customer_id,
                project_id=project_id,
                site_id=site_id,
                industry=industry,
                gate_status=gate_status,
                deployment_stage=deployment_stage,
            )
            result = pipeline.build_product_launch_readiness(
                identity_readiness=identity_readiness_payload(),
                field_readiness=field_readiness,
                solution_delivery_readiness=workbench.get("solution_delivery_readiness", {}),
                customer_project_workbench=workbench,
                dashboard_pages=dashboard_pages_payload() if dashboard_pages_payload else None,
            )
            response = ProductLaunchReadinessResponse.model_validate(result)
            return mission_json(response.model_dump(mode="python"))
        except Exception as exc:
            logger.error("Field product launch readiness endpoint failed: %s", exc)
            return mission_json({"error": str(exc)}, status_code=500)

    @router.options("/api/field/site-profiles", include_in_schema=False)
    async def field_site_profiles_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/customer-projects", include_in_schema=False)
    async def field_customer_projects_cors() -> Response:
        return cors_options_response("GET, POST, OPTIONS")

    @router.options("/api/field/customer-project-workbench", include_in_schema=False)
    async def field_customer_project_workbench_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/product-launch-readiness", include_in_schema=False)
    async def field_product_launch_readiness_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/customer-projects/managed-object-directory", include_in_schema=False)
    async def field_customer_project_managed_object_directory_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/customer-project-acceptance-registry", include_in_schema=False)
    async def field_customer_project_acceptance_registry_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/customer-project-resource-catalog", include_in_schema=False)
    async def field_customer_project_resource_catalog_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @router.options("/api/field/solution-delivery-readiness", include_in_schema=False)
    async def field_solution_delivery_readiness_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    return router

__all__ = [
    "FieldProductCatalogPipeline",
    "create_field_product_catalog_router",
    "register_field_product_catalog_routes",
]
