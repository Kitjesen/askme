from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from askme.api.composition import API_SURFACES, api_surface_for_route_module, api_surface_module_map
from askme.api.routes.agent_profiles import register_agent_profile_routes
from askme.api.routes.cognition import create_cognition_router, register_cognition_routes
from askme.api.routes.conversation import create_conversation_router
from askme.api.routes.field_admin import create_field_admin_router
from askme.api.routes.field_events import create_field_event_router
from askme.api.routes.field_internal import create_field_internal_router
from askme.api.routes.field_product_catalog import create_field_product_catalog_router
from askme.api.routes.governance import create_governance_router, register_governance_routes
from askme.api.routes.memory import create_memory_router, register_memory_routes
from askme.api.routes.runtime import create_runtime_router
from askme.api.routes.skills import register_skill_routes
from askme.api.routes.space import create_space_router


async def _optional_json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def _operator_id_from_request(request: Request, body: dict[str, Any]) -> str:
    return str(request.headers.get("x-operator-id") or body.get("operator_id") or "test.operator")


def test_api_surface_route_manifest_has_unique_module_ownership() -> None:
    module_map = api_surface_module_map()
    declared_modules = [module for spec in API_SURFACES for module in spec.route_modules]

    assert len(module_map) == len(declared_modules)
    assert api_surface_for_route_module("askme.api.routes.conversation") == "product"
    assert api_surface_for_route_module("askme.api.routes.runtime") == "internal"
    assert api_surface_for_route_module("askme.api.routes.agent_profiles") == "admin"
    assert api_surface_for_route_module("askme.api.routes.system") == "platform"


class FakeSkillGrowthBacklog:
    def __init__(self) -> None:
        self.payload_calls: list[dict[str, Any]] = []
        self.mark_calls: list[dict[str, Any]] = []

    def payload(self, *, min_occurrences: int, limit: int) -> dict[str, Any]:
        self.payload_calls.append({"min_occurrences": min_occurrences, "limit": limit})
        return {"ok": True, "candidates": [], "min_occurrences": min_occurrences, "limit": limit}

    def mark(
        self,
        candidate_id: str,
        *,
        action: str,
        operator_id: str,
        note: str,
    ) -> dict[str, Any]:
        self.mark_calls.append(
            {
                "candidate_id": candidate_id,
                "action": action,
                "operator_id": operator_id,
                "note": note,
            }
        )
        return {"ok": action != "bad", "candidate_id": candidate_id, "operator_id": operator_id}


class ExplodingSkillGrowthBacklog:
    def payload(self, *, min_occurrences: int, limit: int) -> dict[str, Any]:
        _ = (min_occurrences, limit)
        raise RuntimeError("internal skill failure detail")


@dataclass
class FakeGeneratedSkill:
    name: str = "route_helper"
    source: str = "generated"
    path: str = ""
    description: str = "Route helper"
    voice_trigger: str = "带路"
    safety_level: str = "normal"
    execution: str = "assistive"
    enabled: bool = True
    tags: tuple[str, ...] = ("generated",)
    prompt_template: str = "Only answer park route questions."
    tools_section: str = ""


class FakeSkillManager:
    def __init__(self) -> None:
        self.loaded = False
        self.skill = FakeGeneratedSkill()

    def load(self) -> None:
        self.loaded = True

    def get(self, name: str) -> FakeGeneratedSkill | None:
        return self.skill if name == self.skill.name else None

    def get_all(self) -> list[FakeGeneratedSkill]:
        return [self.skill]


class FakeAgentProfileRegistry:
    def __init__(self) -> None:
        self.write_calls: list[dict[str, Any]] = []

    def catalog(self) -> dict[str, Any]:
        return {"profiles": [{"name": "field_operator"}], "profile_count": 1}

    def write_project_profile(self, **kwargs: Any) -> dict[str, Any]:
        self.write_calls.append(kwargs)
        tools = set(kwargs.get("tools") or [])
        known_tools = set(kwargs.get("known_tools") or [])
        unknown_tools = sorted(tools - known_tools)
        if unknown_tools:
            return {
                "ok": False,
                "error": "unknown tools requested",
                "unknown_tools": unknown_tools,
            }
        return {
            "ok": True,
            "name": kwargs.get("name"),
            "operator_id": kwargs.get("operator_id"),
            "known_tools": sorted(known_tools),
        }

    def preview(self, profile_name: str) -> dict[str, Any]:
        if profile_name == "known":
            return {"ok": True, "name": profile_name, "raw_body": "profile"}
        return {"ok": False, "error": "agent profile not found", "profile_name": profile_name}


class ExplodingAgentProfileRegistry:
    def catalog(self) -> dict[str, Any]:
        raise RuntimeError("internal agent profile failure detail")


def _memory_answer_contract() -> dict[str, Any]:
    return {
        "contract_type": "askme.customer_knowledge_answer_contract.v1",
        "evidence_required": True,
        "approved_knowledge_only": True,
        "current_knowledge_only": True,
        "conflict_free_knowledge_only": True,
        "show_evidence_in_answer": True,
        "refuse_when_no_evidence": True,
        "refuse_when_expired": True,
        "refuse_when_conflicting": True,
        "robot_behavior_memory_enters_customer_prompt": False,
    }


def _identity_readiness_payload() -> dict[str, Any]:
    return {
        "gate_type": "identity_gateway",
        "status": "demo_only",
        "production_ready": False,
        "identity_mode": "demo",
        "identity_provider": "local_config",
        "production_binding_required": True,
        "production_target": "enterprise_sso_or_iam",
        "customer_status": "当前是演示账号目录",
        "release_claim": "生产环境必须绑定企业身份系统",
        "next_step": "接入客户企业身份网关",
    }


def _skill_route_client(
    *,
    backlog: Any | None = None,
    manager: FakeSkillManager | None = None,
    auth_calls: list[tuple[str, dict[str, Any]]] | None = None,
    deny_permission: bool = False,
    validator_calls: list[dict[str, Any]] | None = None,
) -> TestClient:
    app = FastAPI()
    backlog = backlog or FakeSkillGrowthBacklog()
    manager = manager or FakeSkillManager()

    def authorize(_request: Request, body: dict[str, Any], permission: str) -> JSONResponse | None:
        if auth_calls is not None:
            auth_calls.append((permission, dict(body)))
        if deny_permission:
            return JSONResponse(
                {"error": "forbidden", "permission": permission},
                status_code=403,
            )
        return None

    def validate_generated_skill(skill: Any, *, all_skills: list[Any]) -> dict[str, Any]:
        if validator_calls is not None:
            validator_calls.append({"skill": skill, "all_skills": all_skills})
        return {"ok": True, "checks": [{"name": "fake", "status": "pass"}]}

    register_skill_routes(
        app,
        optional_json_body=_optional_json_body,
        authorize=authorize,
        operator_id_from_request=_operator_id_from_request,
        skill_growth_candidate_prompt=lambda candidate: f"Draft {candidate.get('summary', '')}",
        logger=logging.getLogger("tests.skill_routes"),
        skill_manager_factory=lambda: manager,
        skill_growth_backlog_factory=lambda: backlog,
        validate_generated_skill_func=validate_generated_skill,
    )
    return TestClient(app)


def _agent_profile_route_client(
    *,
    registry: Any | None = None,
    auth_calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> TestClient:
    app = FastAPI()
    registry = registry or FakeAgentProfileRegistry()

    def authorize(_request: Request, body: dict[str, Any], permission: str) -> JSONResponse | None:
        if auth_calls is not None:
            auth_calls.append((permission, dict(body)))
        return None

    register_agent_profile_routes(
        app,
        optional_json_body=_optional_json_body,
        authorize=authorize,
        operator_id_from_request=_operator_id_from_request,
        logger=logging.getLogger("tests.agent_profile_routes"),
        known_tools_provider=lambda: {"server_tool"},
        agent_profile_registry_factory=lambda: registry,
    )
    return TestClient(app)


def _memory_route_client(
    *,
    dispatch_calls: list[tuple[str, dict[str, Any]]] | None = None,
    auth_calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> TestClient:
    app = FastAPI()

    async def dispatch_memory(method_name: str, body: dict[str, Any]) -> dict[str, Any]:
        if dispatch_calls is not None:
            dispatch_calls.append((method_name, dict(body)))
        if method_name == "health_payload":
            return {
                "status": "ready",
                "ready": True,
                "customer_status": "客户知识库可用于回答",
                "customer_next_step": "继续维护过期和冲突知识",
                "answer_contract": _memory_answer_contract(),
            }
        if method_name == "search_payload":
            return {
                "query": body.get("query") or body.get("text") or "",
                "results": [{"title": "梵木咖啡", "score": 0.9}],
                "answer_contract": _memory_answer_contract(),
            }
        if method_name == "preview_payload":
            return {"source": body.get("filename") or "manual", "parsed": 1, "records": []}
        return {
            "updated": True,
            "record_id": body.get("record_id", ""),
            "action": body.get("action", ""),
        }

    def mission_json(payload: dict[str, Any], status_code: int = 200, **_: Any) -> JSONResponse:
        return JSONResponse(payload, status_code=status_code)

    def cors_options_response(methods: str) -> JSONResponse:
        return JSONResponse({"methods": methods})

    def authorize(_request: Request, body: dict[str, Any], permission: str) -> JSONResponse | None:
        if auth_calls is not None:
            auth_calls.append((permission, dict(body)))
        return None

    register_memory_routes(
        app,
        dispatch_memory=dispatch_memory,
        mission_json=mission_json,
        cors_options_response=cors_options_response,
        logger=logging.getLogger("tests.memory_routes"),
        authorize=authorize,
    )
    return TestClient(app)


def _governance_route_client(
    *,
    call_log: list[tuple[str, Any]] | None = None,
    allow: bool = True,
) -> TestClient:
    app = FastAPI()

    def operator() -> dict[str, Any]:
        return {
            "operator_id": "dashboard.operator",
            "display_name": "演示操作员",
            "roles": ["operator"],
            "source": "demo_config",
            "authenticated": True,
            "known": True,
        }

    def governance_payload() -> dict[str, Any]:
        if call_log is not None:
            call_log.append(("directory", None))
        return {
            "mode": "demo",
            "identity_provider": "local_config",
            "operators": [operator()],
            "permissions": {"operator": ["knowledge:read"]},
            "readiness": {"status": "demo_only", "production_ready": False},
            "identity_gateway_readiness": _identity_readiness_payload(),
        }

    def identity_readiness_payload() -> dict[str, Any]:
        if call_log is not None:
            call_log.append(("identity_readiness", None))
        return _identity_readiness_payload()

    def current_operator_payload(operator_id: str | None, headers: Any) -> dict[str, Any]:
        if call_log is not None:
            call_log.append(("current_operator", operator_id or headers.get("x-operator-id")))
        return {
            "operator": operator(),
            "permissions": ["knowledge:read"],
            "known": True,
            "authenticated": True,
            "directory_mode": "demo",
            "identity_provider": "local_config",
            "readiness": {"status": "demo_only", "production_ready": False},
            "identity_gateway_readiness": _identity_readiness_payload(),
        }

    def authorization_payload(
        operator_id: str | None,
        permission: str,
        _headers: Any,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        if call_log is not None:
            call_log.append(
                ("authorize", {"operator_id": operator_id, "permission": permission, "body": body})
            )
        return {
            "allowed": allow,
            "permission": permission,
            "operator": operator(),
            "reason": "allowed" if allow else "denied",
            "audit": {"operator_id": operator_id},
        }

    def mission_json(payload: dict[str, Any], status_code: int = 200, **_: Any) -> JSONResponse:
        return JSONResponse(payload, status_code=status_code)

    def cors_options_response(methods: str) -> JSONResponse:
        return JSONResponse({"methods": methods})

    register_governance_routes(
        app,
        governance_payload=governance_payload,
        identity_readiness_payload=identity_readiness_payload,
        current_operator_payload=current_operator_payload,
        authorization_payload=authorization_payload,
        mission_json=mission_json,
        cors_options_response=cors_options_response,
    )
    return TestClient(app)


def _cognition_route_client(
    *,
    dispatch_calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> TestClient:
    app = FastAPI()

    async def dispatch_cognition(method_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if dispatch_calls is not None:
            payload = dict(args[0]) if args and isinstance(args[0], dict) else dict(kwargs)
            dispatch_calls.append((method_name, payload))
        if method_name == "context_payload":
            return {
                "world_state": {"site": "fanmu"},
                "working_memory": {"last_intent": "guide"},
                "perception": {},
                "runtime": {},
            }
        if method_name == "plan_from_payload":
            body = dict(args[0]) if args and isinstance(args[0], dict) else dict(kwargs)
            return {
                "planned": True,
                "status": "awaiting_confirmation",
                "plan": {"intent": body.get("intent") or body.get("text")},
                "planning_session_id": "plan-test",
                "confirmation_status": "awaiting_confirmation",
            }
        return {}

    def json_error(message: str, status_code: int = 500, **_: Any) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status_code)

    def cors_options_response(methods: str) -> JSONResponse:
        return JSONResponse({"methods": methods})

    register_cognition_routes(
        app,
        dispatch_cognition=dispatch_cognition,
        json_error=json_error,
        cors_options_response=cors_options_response,
        cors_headers={"Access-Control-Allow-Origin": "*"},
    )
    return TestClient(app)


def test_cognition_route_exposes_router_factory_for_app_composition() -> None:
    router = create_cognition_router(
        dispatch_cognition=lambda *_args, **_kwargs: None,  # type: ignore[arg-type,return-value]
        json_error=lambda message, **kwargs: JSONResponse(
            {"error": message}, status_code=kwargs.get("status_code", 500)
        ),
        cors_options_response=lambda methods: JSONResponse({"methods": methods}),
        cors_headers={"Access-Control-Allow-Origin": "*"},
    )

    paths = {route.path for route in router.routes}

    assert "/api/cognition/context" in paths
    assert "/api/cognition/plan" in paths


def test_cognition_context_route_uses_injected_dispatcher() -> None:
    dispatch_calls: list[tuple[str, dict[str, Any]]] = []
    client = _cognition_route_client(dispatch_calls=dispatch_calls)

    response = client.get("/api/cognition/context?refresh_perception=true")

    assert response.status_code == 200
    assert response.json()["world_state"]["site"] == "fanmu"
    assert dispatch_calls == [("context_payload", {"refresh_perception": True})]


def test_cognition_plan_route_rejects_non_object_json_before_dispatch() -> None:
    dispatch_calls: list[tuple[str, dict[str, Any]]] = []
    client = _cognition_route_client(dispatch_calls=dispatch_calls)

    response = client.post("/api/cognition/plan", json=["bad"])

    assert response.status_code == 400
    assert dispatch_calls == []


def test_conversation_route_exposes_router_factory_for_app_composition() -> None:
    router = create_conversation_router(
        conversation_service=object(),  # type: ignore[arg-type]
        runtime_available=False,
        dispatch_runtime=lambda *_args, **_kwargs: None,  # type: ignore[arg-type,return-value]
        cors_options_response=lambda methods: JSONResponse({"methods": methods}),
        logger=logging.getLogger("tests.conversation_router_factory"),
        authorize=lambda *_args: None,
        runtime_voice_turn_timeout_s=0.1,
    )

    paths = {route.path for route in router.routes}

    assert "/api/chat" in paths
    assert "/api/conversation/diagnostics" in paths
    assert "/api/runtime/voice-turn" in paths


def test_chat_runtime_mutation_denial_stops_dispatch() -> None:
    class ConversationServiceStub:
        def __init__(self) -> None:
            self.chat_calls: list[dict[str, Any]] = []

        async def chat_payload_from_body(
            self,
            body: dict[str, Any],
            *,
            trace_id: str | None = None,
        ) -> dict[str, Any]:
            self.chat_calls.append({"body": dict(body), "trace_id": trace_id})
            return {"reply": "should not run"}

        def diagnostics_snapshot(self) -> dict[str, Any]:
            return {}

    service = ConversationServiceStub()
    authorization_calls: list[tuple[dict[str, Any], str]] = []

    def authorize(
        _request: Request,
        body: dict[str, Any],
        permission: str,
    ) -> JSONResponse | None:
        authorization_calls.append((dict(body), permission))
        return JSONResponse({"error": "operator not authorized"}, status_code=403)

    app = FastAPI()
    app.include_router(
        create_conversation_router(
            conversation_service=service,  # type: ignore[arg-type]
            runtime_available=True,
            dispatch_runtime=lambda *_args, **_kwargs: None,  # type: ignore[arg-type,return-value]
            cors_options_response=lambda methods: JSONResponse({"methods": methods}),
            logger=logging.getLogger("tests.chat_runtime_authorization"),
            authorize=authorize,
        )
    )

    client = TestClient(app)
    responses = [
        client.post(
            "/api/chat",
            json={"text": text, "conversation_session_id": "thread-7"},
        )
        for text in ("pause current task", "resume current task", "cancel task")
    ]

    assert [response.status_code for response in responses] == [403, 403, 403]
    assert authorization_calls == [
        (
            {"text": "pause current task", "conversation_session_id": "thread-7"},
            "runtime:pause",
        ),
        (
            {"text": "resume current task", "conversation_session_id": "thread-7"},
            "runtime:resume",
        ),
        (
            {"text": "cancel task", "conversation_session_id": "thread-7"},
            "runtime:cancel",
        ),
    ]
    assert service.chat_calls == []


def test_chat_runtime_words_in_ordinary_prose_skip_control_authorization() -> None:
    class ConversationServiceStub:
        def __init__(self) -> None:
            self.chat_calls: list[dict[str, Any]] = []

        async def chat_payload_from_body(
            self,
            body: dict[str, Any],
            *,
            trace_id: str | None = None,
        ) -> dict[str, Any]:
            self.chat_calls.append({"body": dict(body), "trace_id": trace_id})
            return {"reply": "ordinary chat"}

        def diagnostics_snapshot(self) -> dict[str, Any]:
            return {}

    service = ConversationServiceStub()
    authorization_calls: list[str] = []

    def authorize(
        _request: Request,
        _body: dict[str, Any],
        permission: str,
    ) -> JSONResponse | None:
        authorization_calls.append(permission)
        return JSONResponse({"error": "should not authorize"}, status_code=403)

    app = FastAPI()
    app.include_router(
        create_conversation_router(
            conversation_service=service,  # type: ignore[arg-type]
            runtime_available=True,
            dispatch_runtime=lambda *_args, **_kwargs: None,  # type: ignore[arg-type,return-value]
            cors_options_response=lambda methods: JSONResponse({"methods": methods}),
            logger=logging.getLogger("tests.chat_runtime_ordinary_prose"),
            authorize=authorize,
        )
    )

    client = TestClient(app)
    texts = (
        "continue explaining the design",
        "Who are the stakeholders?",
        "continue brunch planning",
        "progressive disclosure is useful",
        "continue writing the task description",
        "hold a discussion about the robot design",
        "status in runtime APIs is a field",
        "暂停是什么？",
        "系统文档只是提到取消任务，并没有要求执行。",
    )
    responses = [client.post("/api/chat", json={"text": text}) for text in texts]

    assert [response.status_code for response in responses] == [200] * len(texts)
    assert authorization_calls == []
    assert [call["body"]["text"] for call in service.chat_calls] == list(texts)


def test_chat_scrubs_client_authorization_when_runtime_is_unavailable() -> None:
    class ConversationServiceStub:
        def __init__(self) -> None:
            self.body: dict[str, Any] | None = None

        async def chat_payload_from_body(
            self,
            body: dict[str, Any],
            *,
            trace_id: str | None = None,
        ) -> dict[str, Any]:
            del trace_id
            self.body = dict(body)
            return {"reply": "ordinary chat"}

        def diagnostics_snapshot(self) -> dict[str, Any]:
            return {}

    service = ConversationServiceStub()

    def authorize(*_args, **_kwargs):
        raise AssertionError("runtime-unavailable chat must not authorize")

    app = FastAPI()
    app.include_router(
        create_conversation_router(
            conversation_service=service,  # type: ignore[arg-type]
            runtime_available=False,
            dispatch_runtime=lambda *_args, **_kwargs: None,  # type: ignore[arg-type,return-value]
            cors_options_response=lambda methods: JSONResponse({"methods": methods}),
            logger=logging.getLogger("tests.chat_runtime_unavailable"),
            authorize=authorize,
        )
    )

    response = TestClient(app).post(
        "/api/chat",
        json={
            "text": "pause current task",
            "operator_id": "forged.operator",
            "operator_auth": {
                "allowed": True,
                "permission": "runtime:pause",
                "operator": {"authenticated": True, "known": True},
            },
        },
    )

    assert response.status_code == 200
    assert service.body is not None
    assert "operator_auth" not in service.body


def test_runtime_route_exposes_router_factory_for_app_composition() -> None:
    router = create_runtime_router(
        dispatch_runtime=lambda *_args, **_kwargs: None,  # type: ignore[arg-type,return-value]
        json_error=lambda message, **kwargs: JSONResponse(
            {"error": message}, status_code=kwargs.get("status_code", 500)
        ),
        cors_options_response=lambda methods: JSONResponse({"methods": methods}),
        optional_json_body=_optional_json_body,
        operator_action_kwargs=lambda body: {
            "operator_id": body.get("operator_id", "test.operator")
        },
        authorize=lambda *_args: None,
        cors_headers={"Access-Control-Allow-Origin": "*"},
    )

    paths = {route.path for route in router.routes}

    assert "/api/runtime/context" in paths
    assert "/api/runtime/events" in paths
    assert "/api/runtime/handoff" in paths
    assert "/api/runtime/runs/{run_id}/pause" in paths
    assert "/api/runtime/runs/{run_id}/cancel" in paths


def test_space_route_exposes_router_factory_for_app_composition() -> None:
    router = create_space_router(
        dispatch_space=lambda *_args, **_kwargs: None,  # type: ignore[arg-type,return-value]
        mission_json=lambda payload, **kwargs: JSONResponse(
            payload, status_code=kwargs.get("status_code", 200)
        ),
        optional_json_body=_optional_json_body,
        cors_options_response=lambda methods: JSONResponse({"methods": methods}),
        logger=logging.getLogger("tests.space_router_factory"),
        authorize=None,
    )

    paths = {route.path for route in router.routes}

    assert "/api/space/health" in paths
    assert "/api/space/resolve-destination" in paths
    assert "/api/space/service-point-trigger" in paths
    assert "/api/space/guide" in paths
    assert "/api/space/manage" in paths


def test_field_admin_route_exposes_router_factory_for_app_composition() -> None:
    router = create_field_admin_router(
        dispatch_field_operations=lambda *_args, **_kwargs: None,  # type: ignore[arg-type,return-value]
        mission_json=lambda payload, **kwargs: JSONResponse(
            payload, status_code=kwargs.get("status_code", 200)
        ),
        optional_json_body=_optional_json_body,
        cors_options_response=lambda methods: JSONResponse({"methods": methods}),
        logger=logging.getLogger("tests.field_admin_router_factory"),
        authorize=lambda *_args: None,
        site_profile_root=lambda: Path("."),
        template_root=lambda: Path("."),
        operator_project_scope=lambda _body: {"tenant_ids": ["default"]},
        scope_allows=lambda _scope, _item: True,
        scope_item_from_create_body=lambda body: body,
        project_scope_forbidden=lambda: JSONResponse({"error": "forbidden"}, status_code=403),
    )

    paths = {route.path for route in router.routes}

    assert "/api/field/notification-test" in paths
    assert "/api/field/notification-preflight" in paths
    assert "/api/field/readiness" in paths
    assert "/api/field/audit/integrity" in paths
    assert "/api/field/customer-projects/from-template" in paths


def test_field_internal_route_exposes_router_factory_for_app_composition() -> None:
    router = create_field_internal_router(
        dispatch_field_operations=lambda *_args, **_kwargs: None,  # type: ignore[arg-type,return-value]
        mission_json=lambda payload, **kwargs: JSONResponse(
            payload, status_code=kwargs.get("status_code", 200)
        ),
        optional_json_body=_optional_json_body,
        cors_options_response=lambda methods: JSONResponse({"methods": methods}),
        logger=logging.getLogger("tests.field_internal_router_factory"),
        dispatch_field_voice_directive=lambda result: result,  # type: ignore[arg-type,return-value]
        dispatch_field_runtime_policy=lambda result, **_kwargs: result,  # type: ignore[arg-type,return-value]
        runtime_callback_trust=lambda *_args, **_kwargs: {"trusted": True},
        runtime_callback_delivery_body=lambda body, **_kwargs: body,
        runtime_callback_secret=None,
        runtime_callback_max_age_s=30.0,
    )

    paths = {route.path for route in router.routes}

    assert "/api/field/events/{event_id}/runtime-delivery" in paths
    assert "/api/field/devices" in paths
    assert "/api/field/device-onboarding" in paths
    assert "/api/field/ingest" in paths


def test_field_event_route_exposes_router_factory_for_app_composition() -> None:
    router = create_field_event_router(
        dispatch_field_operations=lambda *_args, **_kwargs: None,  # type: ignore[arg-type,return-value]
        mission_json=lambda payload, **kwargs: JSONResponse(
            payload, status_code=kwargs.get("status_code", 200)
        ),
        optional_json_body=_optional_json_body,
        authorize=lambda *_args: None,
        project_read_auth=lambda _request: (None, {}),
        operator_project_scope=lambda _body: {
            "tenant_ids": ["default"],
            "delivery_namespaces": ["default"],
            "customer_ids": [],
            "project_ids": [],
            "site_ids": [],
        },
        scoped_query_value=lambda value, _scope, _key: (True, value),
        scope_allows=lambda _scope, _item: True,
        scope_item_from_event_detail=lambda item: item,
        scope_item_from_event_payload=lambda item: item,
        has_explicit_project_scope=lambda _body: False,
        apply_single_scope_defaults=lambda _body, _scope: None,
        project_scope_forbidden=lambda: JSONResponse({"error": "forbidden"}, status_code=403),
        field_event_scope_failure=lambda *_args: None,  # type: ignore[arg-type,return-value]
        field_manual_trigger_body=lambda _request, body: body,
        looks_like_device_ingest_without_scenario=lambda _body: False,
        dispatch_field_voice_directive=lambda result: result,  # type: ignore[arg-type,return-value]
        dispatch_field_runtime_policy=lambda result, **_kwargs: result,  # type: ignore[arg-type,return-value]
        cors_headers={"Access-Control-Allow-Origin": "*"},
        cors_options_response=lambda methods: JSONResponse({"methods": methods}),
        logger=logging.getLogger("tests.field_event_router_factory"),
    )

    paths = {route.path for route in router.routes}

    assert "/api/field/scenarios" in paths
    assert "/api/field/scenario-acceptance" in paths
    assert "/api/field/events" in paths
    assert "/api/field/events/{event_id}" in paths
    assert "/api/field/evidence" in paths
    assert "/api/field/events/{event_id}/acknowledge" in paths
    assert "/api/field/events/{event_id}/report" in paths


def test_field_product_catalog_route_exposes_router_factory_for_app_composition() -> None:
    router = create_field_product_catalog_router(
        dispatch_field_operations=lambda *_args, **_kwargs: None,  # type: ignore[arg-type,return-value]
        mission_json=lambda payload, **kwargs: JSONResponse(
            payload, status_code=kwargs.get("status_code", 200)
        ),
        project_read_auth=lambda _request: (None, {}),
        operator_project_scope=lambda _body: {
            "tenant_ids": ["default"],
            "delivery_namespaces": ["default"],
        },
        scope_allows=lambda _scope, _item: True,
        scope_item_from_site=lambda item: item,
        scope_item_from_resource=lambda item: item,
        resource_has_explicit_scope=lambda _item: False,
        site_profile_root=lambda: Path("deploy/site-profiles"),
        template_root=lambda: Path("deploy/customer-project-templates"),
        delivery_resource_root=lambda: Path("deploy/delivery-resources"),
        identity_readiness_payload=lambda: {"status": "demo_only"},
        dashboard_pages_payload=lambda: {"pages": []},
        cors_options_response=lambda methods: JSONResponse({"methods": methods}),
        logger=logging.getLogger("tests.field_product_catalog_router_factory"),
    )

    paths = {route.path for route in router.routes}

    assert "/api/field/site-profiles" in paths
    assert "/api/field/customer-projects" in paths
    assert "/api/field/customer-projects/managed-object-directory" in paths
    assert "/api/field/customer-project-acceptance-registry" in paths
    assert "/api/field/customer-project-resource-catalog" in paths
    assert "/api/field/customer-project-workbench" in paths
    assert "/api/field/solution-delivery-readiness" in paths
    assert "/api/field/product-launch-readiness" in paths


def test_governance_route_exposes_router_factory_for_app_composition() -> None:
    router = create_governance_router(
        governance_payload=lambda: {"identity_gateway_readiness": _identity_readiness_payload()},
        identity_readiness_payload=_identity_readiness_payload,
        current_operator_payload=lambda *_args: {},
        authorization_payload=lambda *_args: {},
        mission_json=lambda payload, **kwargs: JSONResponse(
            payload, status_code=kwargs.get("status_code", 200)
        ),
        cors_options_response=lambda methods: JSONResponse({"methods": methods}),
    )

    paths = {route.path for route in router.routes}

    assert "/api/governance/operator-directory" in paths
    assert "/api/governance/identity-readiness" in paths
    assert "/api/governance/current-operator" in paths
    assert "/api/governance/authorize" in paths


def test_governance_current_operator_uses_injected_identity_resolver() -> None:
    call_log: list[tuple[str, Any]] = []
    client = _governance_route_client(call_log=call_log)

    response = client.get(
        "/api/governance/current-operator?operator_id=dashboard.operator",
        headers={"x-operator-id": "header.operator"},
    )

    assert response.status_code == 200
    assert response.json()["operator"]["operator_id"] == "dashboard.operator"
    assert call_log == [("current_operator", "dashboard.operator")]


def test_governance_authorize_maps_denial_to_403() -> None:
    call_log: list[tuple[str, Any]] = []
    client = _governance_route_client(call_log=call_log, allow=False)

    response = client.post(
        "/api/governance/authorize",
        headers={"x-operator-id": "ops.viewer"},
        json={"permission": "field:event:close"},
    )

    assert response.status_code == 403
    assert response.json()["allowed"] is False
    assert call_log == [
        (
            "authorize",
            {
                "operator_id": "ops.viewer",
                "permission": "field:event:close",
                "body": {"permission": "field:event:close"},
            },
        )
    ]


def test_memory_route_exposes_router_factory_for_app_composition() -> None:
    router = create_memory_router(
        dispatch_memory=lambda _method, _body: None,  # type: ignore[arg-type,return-value]
        mission_json=lambda payload, **kwargs: JSONResponse(
            payload, status_code=kwargs.get("status_code", 200)
        ),
        cors_options_response=lambda methods: JSONResponse({"methods": methods}),
        logger=logging.getLogger("tests.memory_router_factory"),
        authorize=None,
    )

    paths = {route.path for route in router.routes}

    assert "/api/memory/search" in paths
    assert "/api/memory/health" in paths
    assert "/api/knowledge/preview" in paths
    assert "/api/knowledge/import" in paths
    assert "/api/knowledge/list" in paths
    assert "/api/knowledge/update" in paths


def test_memory_search_route_uses_injected_dispatcher_and_authorize() -> None:
    dispatch_calls: list[tuple[str, dict[str, Any]]] = []
    auth_calls: list[tuple[str, dict[str, Any]]] = []
    client = _memory_route_client(dispatch_calls=dispatch_calls, auth_calls=auth_calls)

    response = client.post("/api/memory/search", json={"query": "咖啡店在哪", "top_k": 3})

    assert response.status_code == 200
    assert response.json()["query"] == "咖啡店在哪"
    assert dispatch_calls == [("search_payload", {"query": "咖啡店在哪", "top_k": 3})]
    assert auth_calls == [("knowledge:read", {"query": "咖啡店在哪", "top_k": 3})]


def test_memory_health_route_uses_product_answer_contract() -> None:
    dispatch_calls: list[tuple[str, dict[str, Any]]] = []
    auth_calls: list[tuple[str, dict[str, Any]]] = []
    client = _memory_route_client(dispatch_calls=dispatch_calls, auth_calls=auth_calls)

    response = client.get("/api/memory/health")

    assert response.status_code == 200
    assert response.json()["customer_status"] == "客户知识库可用于回答"
    assert response.json()["answer_contract"]["evidence_required"] is True
    assert dispatch_calls == [("health_payload", {})]
    assert auth_calls == [("knowledge:read", {})]


def test_memory_preview_rejects_non_object_json_before_dispatch() -> None:
    dispatch_calls: list[tuple[str, dict[str, Any]]] = []
    client = _memory_route_client(dispatch_calls=dispatch_calls)

    response = client.post("/api/knowledge/preview", json=["bad"])

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"
    assert dispatch_calls == []


def test_skill_growth_backlog_route_uses_injected_backlog_and_no_store_headers() -> None:
    backlog = FakeSkillGrowthBacklog()
    client = _skill_route_client(backlog=backlog)

    response = client.get("/api/skill-growth/backlog?min_occurrences=4&limit=7")

    assert response.status_code == 200
    assert response.json()["min_occurrences"] == 4
    assert response.json()["limit"] == 7
    assert backlog.payload_calls == [{"min_occurrences": 4, "limit": 7}]
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["access-control-allow-origin"] == "*"


def test_skill_route_backend_exception_returns_stable_500() -> None:
    client = _skill_route_client(backlog=ExplodingSkillGrowthBacklog())

    response = client.get("/api/skill-growth/backlog")

    assert response.status_code == 500
    assert response.json() == {"ok": False, "error": "skill route failed"}
    assert "internal skill failure detail" not in response.text


def test_skill_growth_backlog_mark_route_uses_authorize_and_operator_provider() -> None:
    backlog = FakeSkillGrowthBacklog()
    auth_calls: list[tuple[str, dict[str, Any]]] = []
    client = _skill_route_client(backlog=backlog, auth_calls=auth_calls)

    response = client.post(
        "/api/skill-growth/backlog/candidate-1",
        headers={"x-operator-id": "ops.lead"},
        json={"action": "promote", "note": "approved"},
    )

    assert response.status_code == 200
    assert auth_calls == [("skill:review", {"action": "promote", "note": "approved"})]
    assert backlog.mark_calls == [
        {
            "candidate_id": "candidate-1",
            "action": "promote",
            "operator_id": "ops.lead",
            "note": "approved",
        }
    ]


def test_skill_write_route_denial_stops_before_backend_call() -> None:
    backlog = FakeSkillGrowthBacklog()
    client = _skill_route_client(backlog=backlog, deny_permission=True)

    response = client.post("/api/skill-growth/backlog/candidate-1", json={"action": "promote"})

    assert response.status_code == 403
    assert response.json() == {"error": "forbidden", "permission": "skill:review"}
    assert backlog.mark_calls == []


def test_generated_skill_validation_route_uses_injected_manager_and_validator() -> None:
    manager = FakeSkillManager()
    validator_calls: list[dict[str, Any]] = []
    client = _skill_route_client(manager=manager, validator_calls=validator_calls)

    response = client.get("/api/skills/generated/route_helper/validation")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["skill_name"] == "route_helper"
    assert manager.loaded is True
    assert validator_calls == [{"skill": manager.skill, "all_skills": [manager.skill]}]


def test_generated_skill_preview_exposes_unavailable_raw_body() -> None:
    manager = FakeSkillManager()
    client = _skill_route_client(manager=manager)

    response = client.get("/api/skills/generated/route_helper/preview")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["raw_body"] == ""
    assert response.json()["raw_body_available"] is False
    assert response.json()["raw_body_error"]


def test_skill_package_invalid_rollout_percent_returns_400() -> None:
    client = _skill_route_client()

    response = client.post(
        "/api/skill-packages",
        json={"package_id": "demo", "rollout_percent": "not-a-number"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request_field"
    assert response.json()["field"] == "rollout_percent"


def test_agent_profile_route_uses_server_tool_allowlist_and_operator_provider() -> None:
    registry = FakeAgentProfileRegistry()
    auth_calls: list[tuple[str, dict[str, Any]]] = []
    client = _agent_profile_route_client(registry=registry, auth_calls=auth_calls)

    response = client.post(
        "/api/agent-profiles",
        headers={"x-operator-id": "product.owner"},
        json={
            "name": "park_service",
            "display_name": "Park service",
            "description": "Customer-visible park assistant",
            "instructions": "Answer only park service questions.",
            "tools": ["server_tool"],
            "known_tools": ["client_supplied_tool"],
        },
    )

    assert response.status_code == 200
    assert response.json()["operator_id"] == "product.owner"
    assert response.json()["known_tools"] == ["server_tool"]
    assert auth_calls[0][0] == "skill:review"
    assert registry.write_calls[0]["known_tools"] == {"server_tool"}


def test_agent_profile_route_backend_exception_returns_stable_500() -> None:
    client = _agent_profile_route_client(registry=ExplodingAgentProfileRegistry())

    response = client.get("/api/agent-profiles")

    assert response.status_code == 500
    assert response.json() == {"ok": False, "error": "agent profile route failed"}
    assert "internal agent profile failure detail" not in response.text


def test_agent_profile_route_does_not_trust_client_supplied_tool_allowlist() -> None:
    registry = FakeAgentProfileRegistry()
    client = _agent_profile_route_client(registry=registry)

    response = client.post(
        "/api/agent-profiles",
        json={
            "name": "unsafe_profile",
            "display_name": "Unsafe profile",
            "tools": ["client_supplied_tool"],
            "known_tools": ["client_supplied_tool"],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == "unknown tools requested"
    assert response.json()["unknown_tools"] == ["client_supplied_tool"]
    assert registry.write_calls[0]["known_tools"] == {"server_tool"}


def test_agent_profile_invalid_timeout_returns_400() -> None:
    registry = FakeAgentProfileRegistry()
    client = _agent_profile_route_client(registry=registry)

    response = client.post(
        "/api/agent-profiles",
        json={
            "name": "park_service",
            "display_name": "Park service",
            "tools": ["server_tool"],
            "timeoutSeconds": "slow",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request_field"
    assert response.json()["field"] == "timeout_seconds"
    assert registry.write_calls == []


def test_agent_profile_preview_route_maps_missing_profile_to_404() -> None:
    client = _agent_profile_route_client()

    response = client.get("/api/agent-profiles/missing/preview")

    assert response.status_code == 404
    assert response.json()["profile_name"] == "missing"
