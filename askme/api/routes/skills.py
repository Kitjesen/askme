"""Skill growth, generated-skill, and skill-package FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from askme.api.routes._request_validation import (
    RequestFieldError,
    field_error_response,
    optional_int_field,
    route_error_response,
)
from askme.api.schemas.skills import (
    GeneratedSkillPreviewResponse,
    GeneratedSkillReviewResponse,
    GeneratedSkillValidationResponse,
    GeneratedSkillsResponse,
    SkillGrowthBacklogResponse,
    SkillGrowthDraftResponse,
    SkillGrowthMutationResponse,
    SkillPackageCatalogResponse,
    SkillPackageHistoryResponse,
    SkillPackageMutationResponse,
)

OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]
OperatorIdFromRequest = Callable[[Request, dict[str, Any]], str]
SkillGrowthPrompt = Callable[[dict[str, Any]], str]
SkillManagerFactory = Callable[[], Any]
SkillGrowthBacklogFactory = Callable[[], Any]
ValidateGeneratedSkill = Callable[..., dict[str, Any]]

NO_STORE_HEADERS = {"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"}
CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}


def register_skill_routes(
    app: FastAPI,
    *,
    optional_json_body: OptionalJsonBody,
    authorize: Authorize,
    operator_id_from_request: OperatorIdFromRequest,
    skill_growth_candidate_prompt: SkillGrowthPrompt,
    logger: logging.Logger,
    skill_manager_factory: SkillManagerFactory | None = None,
    skill_growth_backlog_factory: SkillGrowthBacklogFactory | None = None,
    validate_generated_skill_func: ValidateGeneratedSkill | None = None,
) -> None:
    """Register skill governance and customer skill-package routes."""

    def _skill_manager() -> Any:
        if skill_manager_factory is not None:
            return skill_manager_factory()
        from askme.skills.core.skill_manager import SkillManager

        return SkillManager()

    def _skill_growth_backlog() -> Any:
        if skill_growth_backlog_factory is not None:
            return skill_growth_backlog_factory()
        from askme.skills.governance.growth_backlog import SkillGrowthBacklog

        return SkillGrowthBacklog()

    def _validate_generated_skill(skill: Any, *, all_skills: Any) -> dict[str, Any]:
        if validate_generated_skill_func is not None:
            return validate_generated_skill_func(skill, all_skills=all_skills)
        from askme.skills.core.validation import validate_generated_skill

        return validate_generated_skill(skill, all_skills=all_skills)

    @app.get(
        "/api/skill-growth/backlog",
        tags=["System"],
        response_model=SkillGrowthBacklogResponse,
        response_model_exclude_none=True,
    )
    async def skill_growth_backlog(min_occurrences: int = 2, limit: int = 20) -> JSONResponse:
        """Return reviewable online-growth candidates from skill audit evidence."""
        try:
            backlog = _skill_growth_backlog()
            return _skill_json(
                backlog.payload(min_occurrences=min_occurrences, limit=limit),
                SkillGrowthBacklogResponse,
                headers=NO_STORE_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.post(
        "/api/skill-growth/backlog/{candidate_id}",
        tags=["System"],
        response_model=SkillGrowthMutationResponse,
        response_model_exclude_none=True,
    )
    async def mark_skill_growth_candidate(candidate_id: str, request: Request) -> JSONResponse:
        """Mark a skill-growth candidate as promoted, dismissed, or reopened."""
        try:
            body = await optional_json_body(request)
            denied = authorize(request, body, "skill:review")
            if denied is not None:
                return denied
            backlog = _skill_growth_backlog()
            result = backlog.mark(
                candidate_id,
                action=str(body.get("action") or "observe"),
                operator_id=operator_id_from_request(request, body),
                note=str(body.get("note") or ""),
            )
            return _skill_json(
                result,
                SkillGrowthMutationResponse,
                status_code=200 if result.get("ok") else 400,
                headers=NO_STORE_HEADERS,
            )
        except RequestFieldError as exc:
            return field_error_response(exc, headers=CORS_HEADERS)
        except ValueError as exc:
            return _skill_json(
                {"error": str(exc)},
                SkillGrowthMutationResponse,
                status_code=400,
                headers=CORS_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.post(
        "/api/skill-growth/backlog/{candidate_id}/draft",
        tags=["System"],
        response_model=SkillGrowthDraftResponse,
        response_model_exclude_none=True,
    )
    async def draft_skill_from_growth_candidate(
        candidate_id: str,
        request: Request,
    ) -> JSONResponse:
        """Create a generated SKILL.md draft from a reviewed growth candidate."""
        try:
            body = await optional_json_body(request)
            denied = authorize(request, body, "skill:review")
            if denied is not None:
                return denied
            operator_id = operator_id_from_request(request, body)
            backlog = _skill_growth_backlog()
            min_occurrences = optional_int_field(
                body,
                "min_occurrences",
                default=1,
                min_value=1,
            )
            candidate = backlog.get_candidate(
                candidate_id,
                min_occurrences=min_occurrences,
            )
            if candidate is None:
                return _skill_json(
                    {
                        "ok": False,
                        "error": "skill growth candidate not found",
                        "candidate_id": candidate_id,
                    },
                    SkillGrowthDraftResponse,
                    status_code=404,
                    headers=CORS_HEADERS,
                )
            candidate_payload = candidate.to_dict()
            prompt = str(body.get("prompt") or skill_growth_candidate_prompt(candidate_payload))
            manager = _skill_manager()
            draft = manager.create_generated_skill_draft(
                name=str(body.get("skill_name") or candidate.suggested_skill_name),
                description=str(
                    body.get("description")
                    or f"Handle repeated site request: {candidate.summary}"
                ),
                prompt=prompt,
                voice_trigger=str(body.get("voice_trigger") or candidate.suggested_voice_trigger),
                tools_section=str(body.get("tools_section") or ""),
                tags=body.get("tags") or ["generated", "growth", "candidate"],
                safety_level=str(body.get("safety_level") or candidate.risk_level),
                confirm_before_execute=bool(
                    body.get("confirm_before_execute", candidate.risk_level != "normal")
                ),
                operator_id=operator_id,
                source="skill_growth_backlog",
                overwrite=bool(body.get("overwrite", False)),
            )
            if not draft.get("ok"):
                return _skill_json(
                    {
                        "ok": False,
                        "candidate": candidate_payload,
                        "draft": draft,
                        "error": draft.get("error"),
                    },
                    SkillGrowthDraftResponse,
                    status_code=400,
                    headers=CORS_HEADERS,
                )
            marked = backlog.mark(
                candidate_id,
                action="promote",
                operator_id=operator_id,
                note=f"drafted generated skill {draft.get('skill_name')}",
            )
            return _skill_json(
                {
                    "ok": True,
                    "candidate": candidate_payload,
                    "draft": draft,
                    "backlog": marked.get("backlog", {}),
                },
                SkillGrowthDraftResponse,
                headers=NO_STORE_HEADERS,
            )
        except RequestFieldError as exc:
            return field_error_response(exc, headers=CORS_HEADERS)
        except ValueError as exc:
            return _skill_json(
                {"error": str(exc)},
                SkillGrowthDraftResponse,
                status_code=400,
                headers=CORS_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.get(
        "/api/skills/generated",
        tags=["System"],
        response_model=GeneratedSkillsResponse,
        response_model_exclude_none=True,
    )
    async def generated_skills() -> JSONResponse:
        """Return generated-skill review queue."""
        try:
            manager = _skill_manager()
            manager.load()
            return _skill_json(
                manager.get_generated_skill_governance(),
                GeneratedSkillsResponse,
                headers=NO_STORE_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.get(
        "/api/skill-packages",
        tags=["System"],
        response_model=SkillPackageCatalogResponse,
        response_model_exclude_none=True,
    )
    async def skill_packages() -> JSONResponse:
        """Return customer/site ability packages for approved generated skills."""
        try:
            manager = _skill_manager()
            manager.load()
            return _skill_json(
                manager.get_skill_packages(),
                SkillPackageCatalogResponse,
                headers=NO_STORE_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.post(
        "/api/skill-packages",
        tags=["System"],
        response_model=SkillPackageMutationResponse,
        response_model_exclude_none=True,
    )
    async def upsert_skill_package(request: Request) -> JSONResponse:
        """Create or update a customer/site ability package."""
        try:
            body = await optional_json_body(request)
            denied = authorize(request, body, "skill:review")
            if denied is not None:
                return denied
            rollout_percent = optional_int_field(
                body,
                "rollout_percent",
                min_value=0,
                max_value=100,
            )
            manager = _skill_manager()
            result = manager.upsert_skill_package(
                package_id=str(body.get("package_id") or "default-demo"),
                display_name=str(body.get("display_name") or ""),
                site_id=str(body.get("site_id") or "demo"),
                customer_name=str(body.get("customer_name") or ""),
                description=str(body.get("description") or ""),
                enabled=bool(body.get("enabled", True)),
                release_channel=str(body.get("release_channel") or ""),
                rollout_percent=rollout_percent,
                operator_id=operator_id_from_request(request, body),
            )
            return _skill_json(
                result,
                SkillPackageMutationResponse,
                status_code=200 if result.get("ok") else 400,
                headers=NO_STORE_HEADERS,
            )
        except RequestFieldError as exc:
            return field_error_response(exc, headers=CORS_HEADERS)
        except ValueError as exc:
            return _skill_json(
                {"error": str(exc)},
                SkillPackageMutationResponse,
                status_code=400,
                headers=CORS_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.post(
        "/api/skill-packages/{package_id}/skills/{skill_name}",
        tags=["System"],
        response_model=SkillPackageMutationResponse,
        response_model_exclude_none=True,
    )
    async def update_skill_package(
        package_id: str,
        skill_name: str,
        request: Request,
    ) -> JSONResponse:
        """Assign or remove a generated skill from a customer/site ability package."""
        try:
            body = await optional_json_body(request)
            denied = authorize(request, body, "skill:review")
            if denied is not None:
                return denied
            manager = _skill_manager()
            result = manager.update_skill_package(
                skill_name=skill_name,
                package_id=package_id,
                action=str(body.get("action") or "assign"),
                operator_id=operator_id_from_request(request, body),
            )
            return _skill_json(
                result,
                SkillPackageMutationResponse,
                status_code=200 if result.get("ok") else 400,
                headers=NO_STORE_HEADERS,
            )
        except RequestFieldError as exc:
            return field_error_response(exc, headers=CORS_HEADERS)
        except ValueError as exc:
            return _skill_json(
                {"error": str(exc)},
                SkillPackageMutationResponse,
                status_code=400,
                headers=CORS_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.get(
        "/api/skill-packages/{package_id}/history",
        tags=["System"],
        response_model=SkillPackageHistoryResponse,
        response_model_exclude_none=True,
    )
    async def skill_package_history(package_id: str, limit: int = 20) -> JSONResponse:
        """Return version snapshots for one customer/site ability package."""
        try:
            manager = _skill_manager()
            return _skill_json(
                manager.get_skill_package_history(
                    package_id=package_id,
                    limit=max(1, min(int(limit), 100)),
                ),
                SkillPackageHistoryResponse,
                headers=NO_STORE_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.post(
        "/api/skill-packages/{package_id}/release",
        tags=["System"],
        response_model=SkillPackageMutationResponse,
        response_model_exclude_none=True,
    )
    async def release_skill_package(package_id: str, request: Request) -> JSONResponse:
        """Publish or gray-release a customer/site ability package."""
        try:
            body = await optional_json_body(request)
            denied = authorize(request, body, "skill:review")
            if denied is not None:
                return denied
            rollout_percent = optional_int_field(
                body,
                "rollout_percent",
                default=100,
                min_value=0,
                max_value=100,
            )
            manager = _skill_manager()
            result = manager.release_skill_package(
                package_id=package_id,
                release_channel=str(body.get("release_channel") or "pilot"),
                rollout_percent=rollout_percent,
                operator_id=operator_id_from_request(request, body),
                note=str(body.get("note") or ""),
            )
            return _skill_json(
                result,
                SkillPackageMutationResponse,
                status_code=200 if result.get("ok") else 400,
                headers=NO_STORE_HEADERS,
            )
        except RequestFieldError as exc:
            return field_error_response(exc, headers=CORS_HEADERS)
        except ValueError as exc:
            return _skill_json(
                {"error": str(exc)},
                SkillPackageMutationResponse,
                status_code=400,
                headers=CORS_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.post(
        "/api/skill-packages/{package_id}/rollback",
        tags=["System"],
        response_model=SkillPackageMutationResponse,
        response_model_exclude_none=True,
    )
    async def rollback_skill_package(package_id: str, request: Request) -> JSONResponse:
        """Rollback a customer/site ability package to a previous snapshot."""
        try:
            body = await optional_json_body(request)
            denied = authorize(request, body, "skill:review")
            if denied is not None:
                return denied
            target_version = optional_int_field(body, "target_version")
            manager = _skill_manager()
            result = manager.rollback_skill_package(
                package_id=package_id,
                target_version=target_version,
                operator_id=operator_id_from_request(request, body),
                note=str(body.get("note") or ""),
            )
            return _skill_json(
                result,
                SkillPackageMutationResponse,
                status_code=200 if result.get("ok") else 400,
                headers=NO_STORE_HEADERS,
            )
        except RequestFieldError as exc:
            return field_error_response(exc, headers=CORS_HEADERS)
        except ValueError as exc:
            return _skill_json(
                {"error": str(exc)},
                SkillPackageMutationResponse,
                status_code=400,
                headers=CORS_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.get(
        "/api/skills/generated/{skill_name}/validation",
        tags=["System"],
        response_model=GeneratedSkillValidationResponse,
        response_model_exclude_none=True,
    )
    async def generated_skill_validation(skill_name: str) -> JSONResponse:
        """Return preflight validation for one generated skill."""
        try:
            manager = _skill_manager()
            manager.load()
            skill = manager.get(skill_name)
            if skill is None or skill.source != "generated":
                return _skill_json(
                    {"ok": False, "error": "generated skill not found", "skill_name": skill_name},
                    GeneratedSkillValidationResponse,
                    status_code=404,
                    headers=CORS_HEADERS,
                )
            result = _validate_generated_skill(skill, all_skills=manager.get_all())
            result["skill_name"] = skill_name
            return _skill_json(
                result,
                GeneratedSkillValidationResponse,
                headers=NO_STORE_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.get(
        "/api/skills/generated/{skill_name}/preview",
        tags=["System"],
        response_model=GeneratedSkillPreviewResponse,
        response_model_exclude_none=True,
    )
    async def generated_skill_preview(skill_name: str) -> JSONResponse:
        """Return the reviewable SKILL.md body and parsed policy for one generated skill."""
        try:
            manager = _skill_manager()
            manager.load()
            skill = manager.get(skill_name)
            if skill is None or skill.source != "generated":
                return _skill_json(
                    {"ok": False, "error": "generated skill not found", "skill_name": skill_name},
                    GeneratedSkillPreviewResponse,
                    status_code=404,
                    headers=CORS_HEADERS,
                )
            raw_body_available = True
            raw_body_error = ""
            try:
                raw_body = Path(skill.path).read_text(encoding="utf-8")
            except OSError as exc:
                raw_body = ""
                raw_body_available = False
                raw_body_error = f"{type(exc).__name__}: {exc}"
            return _skill_json(
                {
                    "ok": True,
                    "skill_name": skill.name,
                    "description": skill.description,
                    "voice_trigger": skill.voice_trigger or "",
                    "safety_level": skill.safety_level,
                    "execution": skill.execution,
                    "enabled": skill.enabled,
                    "tags": list(skill.tags),
                    "path": skill.path,
                    "prompt": skill.prompt_template,
                    "tools": skill.tools_section,
                    "raw_body": raw_body,
                    "raw_body_available": raw_body_available,
                    "raw_body_error": raw_body_error,
                    "validation": _validate_generated_skill(skill, all_skills=manager.get_all()),
                },
                GeneratedSkillPreviewResponse,
                headers=NO_STORE_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.post(
        "/api/skills/generated/{skill_name}/review",
        tags=["System"],
        response_model=GeneratedSkillReviewResponse,
        response_model_exclude_none=True,
    )
    async def review_generated_skill(skill_name: str, request: Request) -> JSONResponse:
        """Approve, reject, disable, or return a generated skill to review."""
        try:
            body = await optional_json_body(request)
            denied = authorize(request, body, "skill:review")
            if denied is not None:
                return denied
            operator_id = operator_id_from_request(request, body)
            manager = _skill_manager()
            result = manager.review_generated_skill(
                skill_name,
                action=str(body.get("action") or "request_review"),
                operator_id=str(operator_id),
                note=str(body.get("note") or ""),
            )
            return _skill_json(
                result,
                GeneratedSkillReviewResponse,
                status_code=200 if result.get("ok") else 400,
                headers=NO_STORE_HEADERS,
            )
        except RequestFieldError as exc:
            return field_error_response(exc, headers=CORS_HEADERS)
        except ValueError as exc:
            return _skill_json(
                {"error": str(exc)},
                GeneratedSkillReviewResponse,
                status_code=400,
                headers=CORS_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="skill route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )


__all__ = ["register_skill_routes"]


def _skill_json(
    payload: dict[str, Any],
    schema: type[BaseModel],
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Return a response that is validated against the public API contract."""

    return JSONResponse(
        schema.model_validate(payload).model_dump(mode="python", exclude_unset=True),
        status_code=status_code,
        headers=headers,
    )
