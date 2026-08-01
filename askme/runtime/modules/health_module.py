"""HealthModule -wraps AskmeHealthServer as a declarative module.

Canonical wiring::

    health_server = AskmeHealthServer(
        cfg.get("health_server", {}),
        snapshot_provider=runtime.health_snapshot,
        metrics_provider=runtime.metrics_snapshot,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from askme.api.routes.health import register_health_routes
from askme.api.services.conversation_service import current_chat_runtime_context
from askme.api.services.health_service import HealthService
from askme.config import project_root
from askme.pipeline.channels.external_turns import (
    begin_external_turn,
    cancel_external_turn,
    complete_external_turn,
    record_external_turn,
)
from askme.runtime.control_intent import (
    runtime_control_candidate_intent,
    runtime_control_intent,
)
from askme.runtime.core.module import Module, ModuleRegistry
from askme.voice.diagnostics.status_privacy import sanitize_voice_status

logger = logging.getLogger(__name__)


class HealthModule(Module):
    """Provides the AskmeHealthServer to the runtime."""

    name = "health"
    depends_on = (
        "memory",
        "pipeline",
        "skill",
        "text",
        "mission",
        "cognition",
        "runtime_handoff",
        "warm_sessions",
    )
    provides = ("health_http", "http_chat", "capabilities", "missions", "runtime_handoff")

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        from askme.health_server import AskmeHealthServer, build_health_snapshot

        health_cfg = cfg.get("health_server", {})
        include_voice_transcripts = (
            health_cfg.get("debug_include_voice_transcripts") is True
            if isinstance(health_cfg, dict)
            else False
        )

        # Collect health from all registered modules in the runtime.
        def _runtime_health_provider() -> dict[str, Any]:
            components: dict[str, dict[str, Any]] = {}
            degraded_components: list[str] = []
            for mod_name, mod in registry.items():
                try:
                    module_health = mod.health()
                except Exception:
                    module_health = {"status": "error"}
                components[mod_name] = module_health
                if module_health.get("status") in {"degraded", "error"}:
                    degraded_components.append(mod_name)

            snapshot = build_health_snapshot(
                app_name=cfg.get("app", {}).get("name", "askme"),
                app_version=str(cfg.get("app", {}).get("version", "unknown")),
                model_name=self._model_name(cfg, registry),
                metrics_snapshot=self._metrics_snapshot(registry),
                active_skills=self._active_skill_names(registry),
                voice_status=self._voice_status(registry),
                ota_status=self._ota_status(registry),
                voice_bridge=self._voice_bridge_status(registry),
            )
            snapshot["model_routing"] = self._model_routing(cfg, registry, snapshot)
            snapshot["skill_callability"] = self._skill_callability(registry)
            snapshot["components"] = components
            snapshot.update(components)
            snapshot["rag_trust"] = self._rag_trust_report(cfg)
            snapshot["voice_e2e"] = self._voice_e2e_report(cfg)
            snapshot["field_operations"] = self._field_operations_report(cfg)
            if degraded_components:
                snapshot["status"] = "degraded"
                reasons = list(snapshot.get("degraded_reasons", []))
                reasons.extend(f"component:{name}" for name in degraded_components)
                snapshot["degraded_reasons"] = reasons
            return sanitize_voice_status(
                snapshot,
                include_transcripts=include_voice_transcripts,
            )

        self.server = AskmeHealthServer(
            health_cfg,
            snapshot_provider=_runtime_health_provider,
        )
        self._wire_runtime_handlers(cfg, registry)

        # Set up HealthService with component-level checks for K8s probes.
        self.health_service = HealthService()
        self._register_component_health_checks(cfg, registry)
        register_health_routes(
            self.server._app,
            self.health_service,
            routes=("ready",),  # /healthz and /health already handled by system.py
        )

        logger.info(
            "HealthModule: built (enabled=%s, port=%d)",
            self.server.enabled,
            self.server.port,
        )

    def _register_component_health_checks(
        self,
        cfg: dict[str, Any],
        registry: ModuleRegistry,
    ) -> None:
        """Register component-level health checks on ``self.health_service``.

        Checks are wired to the runtime's Module ``health()`` methods.  Each
        check function is a closure that reads live state from the registry at
        call time rather than capturing a snapshot during build.
        """
        health_cfg = cfg.get("health_server", {})
        include_voice_transcripts = (
            health_cfg.get("debug_include_voice_transcripts") is True
            if isinstance(health_cfg, dict)
            else False
        )

        # ── LLM ──────────────────────────────────────────────────────────
        def _check_llm() -> dict[str, Any]:
            llm_mod = registry.get("llm")
            if llm_mod is None:
                return {"status": "unhealthy", "error": "llm module not registered"}
            health = llm_mod.health()
            model = health.get("model", "unknown")
            provider = health.get("provider", "unknown")
            routing_owner = health.get("routing_owner", "askme")
            fallback_models = health.get("fallback_models", [])
            return {
                "status": "healthy" if health.get("status") == "ok" else "degraded",
                "provider": str(provider),
                "model": str(model),
                "health_model": str(health.get("health_model") or ""),
                "fallback_models": list(fallback_models or []),
                "routing_owner": str(routing_owner),
            }

        self.health_service.register("llm", _check_llm)

        # ── Memory ────────────────────────────────────────────────────────
        def _check_memory() -> dict[str, Any]:
            mem_mod = registry.get("memory")
            if mem_mod is None:
                return {"status": "unhealthy", "error": "memory module not registered"}
            health = mem_mod.health()
            bridge = health.get("rag", {}) if isinstance(health.get("rag"), dict) else {}
            backend = str(bridge.get("backend") or health.get("backend") or "unknown")
            return {
                "status": "healthy" if health.get("status") == "ok" else "degraded",
                "backend": backend,
                "conversation_len": health.get("conversation_len", 0),
                "episodic_buffer_len": health.get("episodic_buffer_len", 0),
            }

        self.health_service.register("memory", _check_memory)

        # ── TTS ──────────────────────────────────────────────────────────
        def _check_tts() -> dict[str, Any]:
            voice_mod = registry.get("voice")
            if voice_mod is None:
                return {"status": "unhealthy", "error": "voice module not registered"}
            tts = getattr(voice_mod, "tts_provider", None)
            if tts is None:
                tts = self._tts_from_text_module(registry)
            if tts is None:
                return {"status": "degraded", "message": "TTS provider not available"}
            voice_health = voice_mod.health()
            audio = voice_health.get("audio") if isinstance(voice_health, dict) else None
            tts_status = audio.get("tts") if isinstance(audio, dict) else None
            output_ready = audio.get("output_ready") if isinstance(audio, dict) else None
            return {
                "status": "healthy" if output_ready is True else "degraded",
                "provider": str(
                    (tts_status or {}).get("backend")
                    if isinstance(tts_status, dict)
                    else type(tts).__name__
                ),
                "available": output_ready is True,
                "details": tts_status if isinstance(tts_status, dict) else {},
            }

        self.health_service.register("tts", _check_tts)

        # ── ASR ──────────────────────────────────────────────────────────
        def _check_asr() -> dict[str, Any]:
            voice_mod = registry.get("voice")
            if voice_mod is None:
                return {"status": "unhealthy", "error": "voice module not registered"}
            asr = getattr(voice_mod, "asr_provider", None)
            if asr is None:
                return {"status": "degraded", "message": "ASR provider not available"}
            voice_health = voice_mod.health()
            audio = voice_health.get("audio") if isinstance(voice_health, dict) else None
            asr_status = audio.get("asr") if isinstance(audio, dict) else None
            local = asr_status.get("local") if isinstance(asr_status, dict) else None
            available = bool(
                local.get("available")
                if isinstance(local, dict)
                else audio.get("input_ready") is True
                if isinstance(audio, dict)
                else False
            )
            return {
                "status": "healthy" if available else "degraded",
                "provider": str(
                    asr_status.get("provider")
                    if isinstance(asr_status, dict)
                    else type(asr).__name__
                ),
                "available": available,
                "details": sanitize_voice_status(
                    asr_status,
                    include_transcripts=include_voice_transcripts,
                )
                if isinstance(asr_status, dict)
                else {},
            }

        self.health_service.register("asr", _check_asr)

        # ── Conversation Core ────────────────────────────────────────────
        def _check_conversation_core() -> dict[str, Any]:
            pipeline_mod = registry.get("pipeline")
            if pipeline_mod is None:
                return {
                    "status": "healthy",
                    "enabled": False,
                    "message": "Conversation Core not wired in this runtime",
                }
            health = pipeline_mod.health()
            raw_conversation_core = health.get("conversation_core")
            conversation_core: dict[str, Any] = (
                dict(raw_conversation_core) if isinstance(raw_conversation_core, dict) else {}
            )
            enabled = bool(conversation_core.get("enabled", False))
            raw_status = str(conversation_core.get("status") or health.get("status") or "ok")
            if not enabled:
                status = "healthy"
            elif raw_status in {"ok", "healthy"}:
                status = "healthy"
            elif raw_status in {"error", "unhealthy"}:
                status = "unhealthy"
            else:
                status = "degraded"
            return {
                **conversation_core,
                "enabled": enabled,
                "status": status,
            }

        self.health_service.register("conversation_core", _check_conversation_core)

        # ── Warm Provider Sessions ───────────────────────────────────────
        def _check_warm_sessions() -> dict[str, Any]:
            warm_mod = registry.get("warm_sessions")
            if warm_mod is None:
                return {
                    "status": "healthy",
                    "enabled": False,
                    "latency_warm": False,
                    "message": "Warm session manager not wired in this runtime",
                }
            health = warm_mod.health()
            enabled = bool(health.get("enabled", False))
            if not enabled:
                status = "healthy"
            elif health.get("status") == "ok":
                status = "healthy"
            else:
                status = "degraded"
            return {
                "status": status,
                "enabled": enabled,
                "running": bool(health.get("running", False)),
                "latency_warm": bool(health.get("latency_warm", False)),
                "manager_status": str(health.get("manager_status", "")),
                "degraded_targets": health.get("degraded_targets", []),
                "targets": health.get("targets", {}),
            }

        self.health_service.register("warm_sessions", _check_warm_sessions)

    @staticmethod
    def _tts_from_text_module(registry: ModuleRegistry) -> Any | None:
        """Fallback: look for a TTS provider on the text module."""
        text_mod = registry.get("text")
        if text_mod is None:
            return None
        text_loop = getattr(text_mod, "text_loop", None)
        if text_loop is None:
            return None
        audio = getattr(text_loop, "_audio", None)
        if audio is None:
            return None
        return getattr(audio, "tts", None)

    def _wire_runtime_handlers(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        """Connect HTTP surfaces to built runtime modules when they exist."""
        chat_handler = self._chat_handler(registry)
        if chat_handler is not None and hasattr(self.server, "set_chat_handler"):
            self.server.set_chat_handler(chat_handler)

        if hasattr(self.server, "set_capabilities_provider"):
            self.server.set_capabilities_provider(
                lambda: self._capabilities_snapshot(cfg, registry)
            )

        if hasattr(self.server, "set_conversation_provider"):
            self.server.set_conversation_provider(lambda: self._conversation_snapshot(registry))

        mission_handler = self._mission_handler(registry)
        if mission_handler is not None and hasattr(self.server, "set_mission_handler"):
            self.server.set_mission_handler(mission_handler)

        cognition_handler = self._cognition_handler(registry)
        if cognition_handler is not None and hasattr(self.server, "set_cognition_handler"):
            self.server.set_cognition_handler(cognition_handler)

        runtime_handler = self._runtime_handler(registry)
        if runtime_handler is not None and hasattr(self.server, "set_runtime_handler"):
            self.server.set_runtime_handler(runtime_handler)

        memory_handler = self._memory_handler(registry)
        if memory_handler is not None and hasattr(self.server, "set_memory_handler"):
            self.server.set_memory_handler(memory_handler)

        voice_handler = self._voice_handler(registry)
        if voice_handler is not None and hasattr(self.server, "set_voice_handler"):
            self.server.set_voice_handler(voice_handler)

        field_operations_handler = self._field_operations_handler(cfg, registry)
        if hasattr(self.server, "set_field_operations_handler"):
            self.server.set_field_operations_handler(field_operations_handler)

    def _chat_handler(self, registry: ModuleRegistry) -> Any | None:
        text_mod = registry.get("text")
        text_loop = getattr(text_mod, "text_loop", None) if text_mod else None
        runtime_handler = self._runtime_handler(registry)
        process_turn = getattr(text_loop, "process_turn", None)
        if callable(process_turn):
            accepts_speak = self._call_accepts_keyword(process_turn, "speak")

            async def _handle_text_chat(
                text: str,
                *,
                speak: bool = False,
                conversation_session_id: str | None = None,
                planning_session_id: str | None = None,
                runtime_policy: str = "disabled",
            ) -> dict[str, Any] | str:
                runtime_control = await self._maybe_handle_runtime_control(
                    runtime_handler,
                    text,
                    text_loop=text_loop,
                    conversation_session_id=conversation_session_id,
                )
                if runtime_control is not None:
                    return runtime_control

                if accepts_speak:
                    kwargs: dict[str, Any] = {"speak": speak}
                    if self._call_accepts_keyword(
                        process_turn,
                        "conversation_session_id",
                    ):
                        kwargs["conversation_session_id"] = conversation_session_id
                    if self._call_accepts_keyword(
                        process_turn,
                        "planning_session_id",
                    ):
                        kwargs["planning_session_id"] = planning_session_id
                    if self._call_accepts_keyword(
                        process_turn,
                        "runtime_policy",
                    ):
                        kwargs["runtime_policy"] = runtime_policy
                    reply = await process_turn(text, **kwargs)
                    rag_payload = self._rag_evidence_payload(
                        registry,
                        turn_rag=self._current_turn_rag(text_loop),
                    )
                    self._attach_rag_to_last_assistant(
                        registry,
                        rag_payload,
                        conversation_session_id=conversation_session_id,
                    )
                    cognition_result = self._last_cognition_result(text_loop)
                    if cognition_result is not None:
                        runtime_result = await self._maybe_submit_runtime_handoff(
                            runtime_handler,
                            cognition_result,
                        )
                        if runtime_result is not None:
                            cognition_result["runtime"] = runtime_result
                        payload = {
                            "reply": reply,
                            "cognition": cognition_result,
                            "evidence": rag_payload["evidence"],
                            "rag": rag_payload["rag"],
                        }
                        if runtime_result is not None:
                            payload["runtime"] = runtime_result
                        if speak:
                            payload["spoken"] = bool(isinstance(reply, str) and reply.strip())
                        return payload
                    if not speak:
                        return {
                            "reply": reply,
                            "evidence": rag_payload["evidence"],
                            "rag": rag_payload["rag"],
                        }
                    return {
                        "reply": reply,
                        "spoken": bool(isinstance(reply, str) and reply.strip()),
                        "evidence": rag_payload["evidence"],
                        "rag": rag_payload["rag"],
                    }

                reply = await process_turn(text)
                rag_payload = self._rag_evidence_payload(
                    registry,
                    turn_rag=self._current_turn_rag(text_loop),
                )
                self._attach_rag_to_last_assistant(registry, rag_payload)
                if not speak:
                    return {
                        "reply": reply,
                        "evidence": rag_payload["evidence"],
                        "rag": rag_payload["rag"],
                    }
                spoken_payload: dict[str, Any] = {"reply": reply, "spoken": False}
                try:
                    spoken_payload["spoken"] = await self._speak_text_loop_reply(
                        text_loop,
                        reply,
                    )
                except Exception as exc:
                    logger.warning("HealthModule: HTTP chat speak failed: %s", exc)
                    spoken_payload["speak_error"] = str(exc)
                spoken_payload["evidence"] = rag_payload["evidence"]
                spoken_payload["rag"] = rag_payload["rag"]
                return spoken_payload

            return _handle_text_chat

        pipeline_mod = registry.get("pipeline")
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None
        process = getattr(pipeline, "process", None)
        if callable(process):
            if runtime_handler is None:
                return process

            async def _handle_pipeline_chat(
                text: str,
                *,
                speak: bool = False,
                conversation_session_id: str | None = None,
                planning_session_id: str | None = None,
                runtime_policy: str = "disabled",
            ) -> dict[str, Any] | str:
                del speak, planning_session_id, runtime_policy
                runtime_control = await self._maybe_handle_runtime_control(
                    runtime_handler,
                    text,
                    text_loop=pipeline,
                    conversation_session_id=conversation_session_id,
                )
                if runtime_control is not None:
                    return runtime_control
                process_kwargs: dict[str, Any] = {}
                if self._call_accepts_keyword(process, "conversation_session_id"):
                    process_kwargs["conversation_session_id"] = conversation_session_id
                if self._call_accepts_keyword(process, "source"):
                    process_kwargs["source"] = "text"
                result = process(text, **process_kwargs)
                return await result if asyncio.iscoroutine(result) else result

            return _handle_pipeline_chat
        return None

    def _voice_handler(self, registry: ModuleRegistry) -> Any | None:
        voice_mod = registry.get("voice")
        if voice_mod is not None and hasattr(voice_mod, "system_control_payload"):
            return voice_mod
        audio = getattr(voice_mod, "audio", None) if voice_mod else None
        return getattr(audio, "tts", None) if audio is not None else None

    def _field_operations_handler(self, cfg: dict[str, Any], registry: ModuleRegistry) -> Any:
        from askme.pipeline.field.field_operations import FieldOperationsService

        field_cfg = dict(cfg.get("field_operations", {}) or {})
        llm_mod = registry.get("llm")
        llm_client = getattr(llm_mod, "llm_client", None) if llm_mod is not None else None
        if field_cfg.get("llm_narrative_enabled") and llm_client is None:
            logger.warning(
                "HealthModule: field LLM narrative disabled; llm module is not available"
            )
        return FieldOperationsService(config=field_cfg, llm_client=llm_client)

    async def _maybe_handle_runtime_control(
        self,
        runtime_handler: Any | None,
        text: str,
        *,
        text_loop: Any,
        conversation_session_id: str | None = None,
    ) -> dict[str, Any] | None:
        if runtime_handler is None:
            return None
        candidate_intent = runtime_control_candidate_intent(text)
        intent = runtime_control_intent(text)
        if intent in {"pause", "resume", "cancel"}:
            return await self._handle_authorized_runtime_mutation(
                runtime_handler,
                intent,
                text=text,
                text_loop=text_loop,
                conversation_session_id=conversation_session_id,
            )
        if candidate_intent is not None and intent is None:
            return None
        handle = getattr(runtime_handler, "handle_chat_control", None)
        if not callable(handle):
            return None
        try:
            result = handle(text)
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as exc:
            logger.warning("HealthModule: runtime control failed: %s", exc)
            return None
        if not isinstance(result, dict) or not result.get("handled"):
            return None
        payload = {
            "reply": str(result.get("reply", "")),
            "runtime": result.get("runtime", result),
        }
        self._record_runtime_control_turn(
            text_loop,
            text,
            payload,
            conversation_session_id=conversation_session_id,
        )
        return payload

    async def _handle_authorized_runtime_mutation(
        self,
        runtime_handler: Any,
        intent: str,
        *,
        text: str,
        text_loop: Any,
        conversation_session_id: str | None,
    ) -> dict[str, Any] | None:
        context = current_chat_runtime_context()
        session_id = str(conversation_session_id or "").strip()
        expected_permission = f"runtime:{intent}"

        def denied(reason: str) -> dict[str, Any]:
            payload = self._runtime_control_denied(intent, reason=reason)
            self._record_runtime_control_turn(
                text_loop,
                text,
                payload,
                conversation_session_id=conversation_session_id,
            )
            return payload

        if context is None:
            return denied("runtime_control_authorization_required")
        authorization = context.authorization
        operator = authorization.get("operator")
        authorized_roles = (
            tuple(str(role).strip() for role in operator.get("roles", ()) if str(role).strip())
            if isinstance(operator, dict)
            else ()
        )
        if (
            not session_id
            or session_id != context.conversation_session_id
            or context.permission != expected_permission
            or authorization.get("allowed") is not True
            or str(authorization.get("permission") or "") != expected_permission
            or not isinstance(operator, dict)
            or str(operator.get("operator_id") or "").strip() != context.operator_id
            or authorized_roles != context.operator_roles
            or bool(operator.get("authenticated")) != context.operator_authenticated
            or str(operator.get("source") or "").strip() != context.operator_source
        ):
            return denied("runtime_control_provenance_mismatch")

        context_payload = getattr(runtime_handler, "context_payload", None)
        if not callable(context_payload):
            return denied("runtime_control_target_unavailable")
        try:
            runtime_context = context_payload()
            if asyncio.iscoroutine(runtime_context):
                runtime_context = await runtime_context
        except Exception as exc:
            logger.warning("HealthModule: runtime context lookup failed: %s", exc)
            return denied("runtime_control_target_unavailable")
        active_run = (
            runtime_context.get("active_run") if isinstance(runtime_context, dict) else None
        )
        run_id = str(active_run.get("run_id") or "").strip() if isinstance(active_run, dict) else ""
        if not run_id:
            return denied("runtime_control_no_active_run")

        action = getattr(runtime_handler, f"{intent}_payload", None)
        if not callable(action) or not self._call_accepts_keyword(action, "operator_id"):
            return denied("runtime_control_operator_provenance_unsupported")
        candidate_kwargs: dict[str, Any] = {
            "operator_id": context.operator_id,
            "operator_roles": list(context.operator_roles),
            "operator_authenticated": context.operator_authenticated,
            "operator_source": context.operator_source,
            "operator_auth": dict(context.authorization),
            "conversation_session_id": session_id,
        }
        kwargs = {
            name: value
            for name, value in candidate_kwargs.items()
            if self._call_accepts_keyword(action, name)
        }
        pipeline = self._runtime_control_pipeline(text_loop)
        if pipeline is None:
            return denied("conversation_turn_owner_unavailable")
        metadata = self._runtime_control_turn_metadata(text)
        turn_handle = begin_external_turn(
            pipeline,
            text,
            source="runtime_control",
            channel="text",
            conversation_session_id=conversation_session_id,
            metadata=metadata,
        )
        if turn_handle is None:
            return denied("conversation_turn_admission_unavailable")
        try:
            result = action(run_id, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
        except asyncio.CancelledError:
            cancel_external_turn(
                pipeline,
                turn_handle,
                user_text=text,
                source="runtime_control",
                reason="runtime_control_interrupted",
                conversation_session_id=conversation_session_id,
                metadata=metadata,
            )
            raise
        except Exception as exc:
            logger.warning("HealthModule: authorized runtime control failed: %s", exc)
            payload = self._runtime_control_denied(
                intent,
                reason="runtime_control_execution_failed",
            )
        else:
            payload = (
                {
                    "reply": str(result.get("reply", "")),
                    "runtime": result,
                }
                if isinstance(result, dict)
                else self._runtime_control_denied(
                    intent,
                    reason="runtime_control_invalid_response",
                )
            )
        complete_external_turn(
            pipeline,
            turn_handle,
            user_text=text,
            assistant_text=str(payload.get("reply") or ""),
            source="runtime_control",
            conversation_session_id=conversation_session_id,
            metadata=metadata,
        )
        return payload

    @staticmethod
    def _runtime_control_denied(intent: str, *, reason: str) -> dict[str, Any]:
        if reason == "runtime_control_no_active_run":
            reply = "Runtime control was not applied because no TaskRun is active."
            error = "no active run"
        elif reason in {
            "runtime_control_authorization_required",
            "runtime_control_provenance_mismatch",
        }:
            reply = (
                "Runtime control was not applied because operator authorization "
                "could not be verified."
            )
            error = "operator not authorized"
        else:
            reply = "Runtime control was not applied because its safe control path is unavailable."
            error = "runtime control unavailable"
        return {
            "reply": reply,
            "runtime": {
                "handled": False,
                "error": error,
                "reason": reason,
                "runtime_control_intent": intent,
            },
        }

    @staticmethod
    def _runtime_control_turn_metadata(text: str) -> dict[str, Any]:
        intent = runtime_control_intent(text)
        context = current_chat_runtime_context()
        metadata: dict[str, Any] = {
            "handled_by": "runtime_handoff",
            "runtime_control_intent": intent or "",
        }
        if context is not None:
            metadata.update(
                {
                    "operator_id": context.operator_id,
                    "operator_roles": list(context.operator_roles),
                    "operator_authenticated": context.operator_authenticated,
                    "operator_source": context.operator_source,
                    "runtime_permission": context.permission,
                }
            )
        return metadata

    @staticmethod
    def _runtime_control_pipeline(turn_owner: Any) -> Any | None:
        process = getattr(turn_owner, "process", None)
        process_turn = getattr(turn_owner, "process_turn", None)
        if callable(process) and not callable(process_turn):
            return turn_owner
        return getattr(turn_owner, "_pipeline", None)

    @staticmethod
    def _record_runtime_control_turn(
        text_loop: Any,
        text: str,
        payload: dict[str, Any],
        *,
        conversation_session_id: str | None,
    ) -> None:
        reply = str(payload.get("reply") or "").strip()
        pipeline = HealthModule._runtime_control_pipeline(text_loop)
        if not reply or pipeline is None:
            return
        metadata = HealthModule._runtime_control_turn_metadata(text)
        record_external_turn(
            pipeline,
            text,
            reply,
            source="runtime_control",
            channel="text",
            conversation_session_id=conversation_session_id,
            metadata=metadata,
        )

    async def _maybe_submit_runtime_handoff(
        self,
        runtime_handler: Any | None,
        cognition_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if runtime_handler is None:
            return None
        plan = cognition_result.get("plan")
        if not isinstance(plan, dict) or not bool(plan.get("handoff_ready")):
            return None
        submit = getattr(runtime_handler, "submit_plan_payload", None)
        if not callable(submit):
            return None
        try:
            result = submit(plan)
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as exc:
            logger.warning("HealthModule: runtime handoff failed: %s", exc)
            return {"accepted": False, "error": str(exc)}
        return dict(result) if isinstance(result, dict) else None

    @staticmethod
    def _last_cognition_result(text_loop: Any) -> dict[str, Any] | None:
        result = getattr(text_loop, "last_cognition_result", None)
        return dict(result) if isinstance(result, dict) else None

    @staticmethod
    def _call_accepts_keyword(func: Any, name: str) -> bool:
        try:
            parameters = signature(func).parameters
        except (TypeError, ValueError):
            return True
        if name in parameters:
            return True
        return any(param.kind == Parameter.VAR_KEYWORD for param in parameters.values())

    async def _speak_text_loop_reply(self, text_loop: Any, reply: str) -> bool:
        if not isinstance(reply, str) or not reply.strip():
            return False
        audio = getattr(text_loop, "_audio", None) if text_loop is not None else None
        if audio is None:
            raise RuntimeError("runtime text loop has no audio output")

        if not bool(getattr(audio, "is_busy", False)):
            audio.speak(reply.strip())
            audio.start_playback()
        try:
            done = await asyncio.to_thread(audio.wait_speaking_done)
            if done is False:
                raise TimeoutError("TTS playback did not finish within timeout")
        finally:
            audio.stop_playback()
        return True

    def _conversation_snapshot(self, registry: ModuleRegistry) -> list[dict[str, Any]]:
        conversation = self._conversation(registry)
        history = getattr(conversation, "history", None)
        if not isinstance(history, list):
            return []
        return [dict(msg) for msg in history if isinstance(msg, dict)]

    @staticmethod
    def _current_turn_rag(text_loop: Any) -> dict[str, Any] | None:
        payload = getattr(text_loop, "current_turn_rag", None)
        if callable(payload):
            payload = payload()
        return dict(payload) if isinstance(payload, dict) else None

    def _rag_evidence_payload(
        self,
        registry: ModuleRegistry,
        *,
        turn_rag: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(turn_rag, dict):
            evidence = turn_rag.get("evidence")
            rag = turn_rag.get("rag")
            if isinstance(rag, dict) and rag.get("turn_scoped") is True:
                return {
                    "evidence": [dict(item) for item in evidence if isinstance(item, dict)]
                    if isinstance(evidence, list)
                    else [],
                    "rag": dict(rag),
                }

        mem_mod = registry.get("memory")
        bridge = getattr(mem_mod, "memory_bridge", None) if mem_mod else None
        health = {}
        health_fn = getattr(bridge, "health", None)
        if callable(health_fn):
            try:
                payload = health_fn()
                if isinstance(payload, dict):
                    health = payload
            except Exception as exc:
                logger.debug("HealthModule: memory RAG snapshot failed: %s", exc)
        evidence = health.get("last_evidence")
        dropped = health.get("last_dropped_evidence")
        answer_policy = health.get("last_answer_policy")
        return {
            "evidence": evidence if isinstance(evidence, list) else [],
            "rag": {
                "backend": health.get("last_backend") or health.get("backend") or "",
                "configured_backend": health.get("backend") or "",
                "retrieve_ms": health.get("last_retrieve_ms"),
                "fallback_reason": health.get("last_fallback_reason") or "",
                "dropped_evidence": dropped if isinstance(dropped, list) else [],
                "answer_policy": answer_policy if isinstance(answer_policy, dict) else {},
                "used_in_answer": bool(evidence),
            },
        }

    def _attach_rag_to_last_assistant(
        self,
        registry: ModuleRegistry,
        rag_payload: dict[str, Any],
        *,
        conversation_session_id: str | None = None,
    ) -> None:
        evidence = rag_payload.get("evidence")
        rag = rag_payload.get("rag")
        if not evidence and not rag:
            return
        conversation = self._conversation(registry)
        update = getattr(conversation, "update_last_assistant_metadata", None)
        if callable(update):
            update(
                {
                    "evidence": evidence if isinstance(evidence, list) else [],
                    "rag": rag if isinstance(rag, dict) else {},
                },
                conversation_session_id=conversation_session_id,
            )
            return
        if conversation_session_id:
            return
        history = getattr(conversation, "history", None)
        if not isinstance(history, list):
            return
        for msg in reversed(history):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                msg["evidence"] = evidence if isinstance(evidence, list) else []
                msg["rag"] = rag if isinstance(rag, dict) else {}
                save = getattr(conversation, "_save", None)
                if callable(save):
                    try:
                        save()
                    except Exception as exc:
                        logger.debug("HealthModule: conversation RAG save failed: %s", exc)
                return

    def _conversation(self, registry: ModuleRegistry) -> Any | None:
        mem_mod = registry.get("memory")
        conversation = getattr(mem_mod, "conversation", None) if mem_mod else None
        if conversation is not None:
            return conversation

        pipeline_mod = registry.get("pipeline")
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None
        return getattr(pipeline, "_conversation", None)

    def _memory_handler(self, registry: ModuleRegistry) -> Any | None:
        return registry.get("memory")

    def _audio(self, registry: ModuleRegistry) -> Any | None:
        voice_mod = registry.get("voice")
        audio = getattr(voice_mod, "audio", None) if voice_mod else None
        if audio is not None:
            return audio

        text_mod = registry.get("text")
        text_loop = getattr(text_mod, "text_loop", None) if text_mod else None
        return getattr(text_loop, "_audio", None)

    def _voice_status(self, registry: ModuleRegistry) -> dict[str, Any]:
        audio = self._audio(registry)
        status_snapshot = getattr(audio, "status_snapshot", None)
        if callable(status_snapshot):
            try:
                status = status_snapshot()
                if isinstance(status, dict):
                    return status
            except Exception as exc:
                logger.debug("HealthModule: audio status snapshot failed: %s", exc)

        has_voice = "voice" in registry
        return {
            "mode": "voice" if has_voice else "text",
            "enabled": has_voice,
            "input_ready": False,
            "output_ready": not has_voice,
            "pipeline_ok": not has_voice,
        }

    def _metrics_snapshot(self, registry: ModuleRegistry) -> dict[str, Any]:
        llm_mod = registry.get("llm")
        metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else None
        snapshot = getattr(metrics, "snapshot", None)
        if callable(snapshot):
            try:
                payload = snapshot()
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:
                logger.debug("HealthModule: OTA metrics snapshot failed: %s", exc)
        return {"uptime_seconds": 0.0, "conversation_count": 0}

    def _active_skill_names(self, registry: ModuleRegistry) -> list[str]:
        skill_mod = registry.get("skill")
        skill_manager = getattr(skill_mod, "skill_manager", None) if skill_mod else None
        get_enabled = getattr(skill_manager, "get_enabled", None)
        if not callable(get_enabled):
            return []
        try:
            return [getattr(skill, "name", str(skill)) for skill in get_enabled()]
        except Exception as exc:
            logger.debug("HealthModule: active skill snapshot failed: %s", exc)
            return []

    def _ota_status(self, registry: ModuleRegistry) -> dict[str, Any]:
        llm_mod = registry.get("llm")
        metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else None
        status_snapshot = getattr(metrics, "status_snapshot", None)
        if callable(status_snapshot):
            try:
                payload = status_snapshot()
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:
                logger.debug("HealthModule: OTA status snapshot failed: %s", exc)
        return {"enabled": False, "registered": False, "state": "disabled"}

    def _voice_bridge_status(self, registry: ModuleRegistry) -> dict[str, Any] | None:
        voice_mod = registry.get("voice")
        bridge = getattr(voice_mod, "voice_runtime_bridge", None) if voice_mod else None
        if bridge is None:
            text_mod = registry.get("text")
            text_loop = getattr(text_mod, "text_loop", None) if text_mod else None
            bridge = getattr(text_loop, "_voice_runtime_bridge", None)
        status_snapshot = getattr(bridge, "status_snapshot", None)
        if callable(status_snapshot):
            try:
                payload = status_snapshot()
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:
                logger.debug("HealthModule: voice bridge snapshot failed: %s", exc)
        return None

    def _rag_trust_report(self, cfg: dict[str, Any]) -> dict[str, Any]:
        report_path = self._rag_trust_report_path(cfg)
        if not report_path.exists():
            return {
                "status": "missing",
                "report_path": str(report_path),
                "suite": "askme-rag-trust",
                "scenario_count": 0,
                "passed": 0,
                "failed": 0,
                "scenarios": [],
            }
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "status": "error",
                "report_path": str(report_path),
                "error": str(exc),
                "suite": "askme-rag-trust",
                "scenario_count": 0,
                "passed": 0,
                "failed": 0,
                "scenarios": [],
            }
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, list):
            scenarios = []
        return {
            "status": str(payload.get("status") or "unknown"),
            "report_path": str(report_path),
            "suite": str(payload.get("suite") or "askme-rag-trust"),
            "scenario_count": int(payload.get("scenario_count") or len(scenarios)),
            "passed": int(payload.get("passed") or 0),
            "failed": int(payload.get("failed") or 0),
            "generated_at": payload.get("generated_at"),
            "external_services": bool(payload.get("external_services", False)),
            "scenarios": [
                {
                    "name": str(item.get("name") or ""),
                    "passed": bool(item.get("passed")),
                }
                for item in scenarios[:20]
                if isinstance(item, dict)
            ],
        }

    @staticmethod
    def _rag_trust_report_path(cfg: dict[str, Any]) -> Path:
        raw = (
            cfg.get("rag_trust", {}).get("report_path")
            or cfg.get("health_server", {}).get("rag_trust_report_path")
            or "artifacts/rag_trust/scenario-evaluation.json"
        )
        path = Path(str(raw))
        if not path.is_absolute():
            path = project_root() / path
        return path

    def _voice_e2e_report(self, cfg: dict[str, Any]) -> dict[str, Any]:
        report_path = self._voice_e2e_report_path(cfg)
        if not report_path.exists():
            return {
                "status": "missing",
                "report_path": str(report_path),
                "suite": "askme-voice-e2e",
                "scenario_count": 0,
                "passed": 0,
                "failed": 0,
                "metrics": {},
                "scenarios": [],
            }
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "status": "error",
                "report_path": str(report_path),
                "error": str(exc),
                "suite": "askme-voice-e2e",
                "scenario_count": 0,
                "passed": 0,
                "failed": 0,
                "metrics": {},
                "scenarios": [],
            }
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, list):
            scenarios = []
        metrics = payload.get("metrics")
        return {
            "status": str(payload.get("status") or "unknown"),
            "report_path": str(report_path),
            "suite": str(payload.get("suite") or "askme-voice-e2e"),
            "scenario_count": int(payload.get("scenario_count") or len(scenarios)),
            "passed": int(payload.get("passed") or 0),
            "failed": int(payload.get("failed") or 0),
            "generated_at": payload.get("generated_at"),
            "external_services": bool(payload.get("external_services", False)),
            "metrics": metrics if isinstance(metrics, dict) else {},
            "scenarios": [
                {
                    "name": str(item.get("name") or ""),
                    "passed": bool(item.get("passed")),
                    "gate_action": str((item.get("interaction_gate") or {}).get("action") or ""),
                }
                for item in scenarios[:20]
                if isinstance(item, dict)
            ],
        }

    @staticmethod
    def _voice_e2e_report_path(cfg: dict[str, Any]) -> Path:
        raw = (
            cfg.get("voice_e2e", {}).get("report_path")
            or cfg.get("health_server", {}).get("voice_e2e_report_path")
            or "artifacts/voice_e2e/scenario-evaluation.json"
        )
        path = Path(str(raw))
        if not path.is_absolute():
            path = project_root() / path
        return path

    def _field_operations_report(self, cfg: dict[str, Any]) -> dict[str, Any]:
        report_path = self._field_operations_report_path(cfg)
        if not report_path.exists():
            return {
                "status": "missing",
                "report_path": str(report_path),
                "suite": "askme-field-operations",
                "scenario_count": 0,
                "passed": 0,
                "failed": 0,
                "external_services": False,
                "hardware_dispatch": False,
                "scenarios": [],
            }
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "status": "error",
                "report_path": str(report_path),
                "error": str(exc),
                "suite": "askme-field-operations",
                "scenario_count": 0,
                "passed": 0,
                "failed": 0,
                "external_services": False,
                "hardware_dispatch": False,
                "scenarios": [],
            }
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, list):
            scenarios = []
        product_demo = payload.get("product_demo")
        if not isinstance(product_demo, dict):
            product_demo = {}
        return {
            "status": str(payload.get("status") or "unknown"),
            "report_path": str(report_path),
            "suite": str(payload.get("suite") or "askme-field-operations"),
            "scenario_count": int(payload.get("scenario_count") or len(scenarios)),
            "passed": int(payload.get("passed") or 0),
            "failed": int(payload.get("failed") or 0),
            "generated_at": payload.get("generated_at"),
            "external_services": bool(payload.get("external_services", False)),
            "hardware_dispatch": bool(payload.get("hardware_dispatch", False)),
            "product_demo": {
                "suite_name": str(product_demo.get("suite_name") or ""),
                "demo_ready": bool(product_demo.get("demo_ready", False)),
                "real_integration_ready": bool(product_demo.get("real_integration_ready", False)),
                "customer_scenario_count": int(product_demo.get("customer_scenario_count") or 0),
                "blocked_on_real_integrations": list(
                    product_demo.get("blocked_on_real_integrations") or []
                )[:10],
            },
            "scenarios": [
                {
                    "name": str(item.get("name") or ""),
                    "passed": bool(item.get("passed")),
                }
                for item in scenarios[:20]
                if isinstance(item, dict)
            ],
        }

    @staticmethod
    def _field_operations_report_path(cfg: dict[str, Any]) -> Path:
        raw = (
            cfg.get("field_operations", {}).get("scenario_report_path")
            or cfg.get("health_server", {}).get("field_operations_report_path")
            or "artifacts/field_operations/scenario-evaluation.json"
        )
        path = Path(str(raw))
        if not path.is_absolute():
            path = project_root() / path
        return path

    def _model_name(self, cfg: dict[str, Any], registry: ModuleRegistry) -> str:
        llm_mod = registry.get("llm")
        client = getattr(llm_mod, "client", None) if llm_mod else None
        model = getattr(client, "model", None)
        if model:
            return str(model)
        return str(cfg.get("brain", {}).get("model", "unknown"))

    @staticmethod
    def _agent_shell_deprecation(shell: Any | None) -> str:
        if shell is None:
            return ""
        replacement = getattr(shell, "deprecated_replacement", "")
        return str(replacement or "")

    def _model_routing(
        self,
        cfg: dict[str, Any],
        registry: ModuleRegistry,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Return product-readable model routing for voice, reasoning, and agent entrypoints."""

        brain_cfg = cfg.get("brain", {}) if isinstance(cfg.get("brain"), dict) else {}
        voice_cfg = cfg.get("voice", {}) if isinstance(cfg.get("voice"), dict) else {}
        tts_cfg = voice_cfg.get("tts", {}) if isinstance(voice_cfg.get("tts"), dict) else {}
        cloud_asr_cfg = (
            voice_cfg.get("cloud_asr", {}) if isinstance(voice_cfg.get("cloud_asr"), dict) else {}
        )

        voice_status = snapshot.get("voice_pipeline_status", {})
        asr_status = voice_status.get("asr", {}) if isinstance(voice_status, dict) else {}
        cloud_asr_status = asr_status.get("cloud", {}) if isinstance(asr_status, dict) else {}
        tts_status = voice_status.get("tts", {}) if isinstance(voice_status, dict) else {}
        tts_backend = str(voice_status.get("tts_backend") or tts_cfg.get("backend") or "unknown")
        provider_tts_status = (
            tts_status.get(tts_backend, {}) if isinstance(tts_status, dict) else {}
        )
        if not isinstance(provider_tts_status, dict):
            provider_tts_status = {}
        configured_tts_model = {
            "volcengine": tts_cfg.get("volcengine_tts_model"),
            "minimax": tts_cfg.get("minimax_tts_model"),
            "edge": tts_cfg.get("voice"),
            "local": Path(str(tts_cfg.get("model_dir") or "")).name,
        }.get(tts_backend)

        executor = registry.get("executor")
        shell = getattr(executor, "shell", None) if executor else None
        agent_profile = getattr(getattr(shell, "_profile", None), "name", "field_operator")
        agent_shell_replacement = self._agent_shell_deprecation(shell)
        agent_shell_deprecated = bool(agent_shell_replacement)

        return {
            "dialogue": {
                "llm_provider": str(brain_cfg.get("provider") or "unknown"),
                "llm_model": str(
                    brain_cfg.get("voice_model")
                    or brain_cfg.get("model")
                    or snapshot.get("model_name")
                    or "unknown"
                ),
                "asr_provider": str(asr_status.get("provider") or "unknown"),
                "asr_model": str(
                    cloud_asr_status.get("model") or cloud_asr_cfg.get("model") or "unknown"
                ),
                "tts_backend": tts_backend,
                "tts_model": str(
                    provider_tts_status.get("model") or configured_tts_model or "unknown"
                ),
                "voice_profile": str(
                    provider_tts_status.get("active_profile")
                    or provider_tts_status.get("speaker")
                    or tts_cfg.get("voice_profile")
                    or ""
                ),
            },
            "reasoning": {
                "provider": str(brain_cfg.get("provider") or "unknown"),
                "model": str(snapshot.get("model_name") or brain_cfg.get("model") or "unknown"),
            },
            "agent_shell": {
                "loaded": shell is not None,
                "enabled": bool(shell is not None and not agent_shell_deprecated),
                "status": "deprecated"
                if agent_shell_deprecated
                else "enabled"
                if shell is not None
                else "unavailable",
                "replacement": agent_shell_replacement,
                "model": str(
                    getattr(shell, "_model", "") or brain_cfg.get("agent_model") or "unknown"
                ),
                "profile": str(agent_profile or ""),
                "timeout_seconds": getattr(
                    shell, "_default_timeout", brain_cfg.get("agent_timeout")
                ),
                "max_iterations": getattr(shell, "_iteration_limit", None),
            },
        }

    def _skill_callability(self, registry: ModuleRegistry) -> dict[str, Any]:
        skill_mod = registry.get("skill")
        skill_manager = getattr(skill_mod, "skill_manager", None) if skill_mod else None
        enabled = self._active_skill_names(registry)
        get_agent_shell_skills = getattr(skill_manager, "get_agent_shell_skills", None)
        agent_shell_skills: list[str] = []
        if callable(get_agent_shell_skills):
            try:
                agent_shell_skills = sorted(str(name) for name in get_agent_shell_skills())
            except Exception as exc:
                logger.debug("HealthModule: agent shell skill snapshot failed: %s", exc)
        pipeline_ready = registry.get("pipeline") is not None
        executor = registry.get("executor")
        shell = getattr(executor, "shell", None) if executor else None
        agent_shell_replacement = self._agent_shell_deprecation(shell)
        executor_ready = executor is not None and not agent_shell_replacement
        return {
            "callable": bool(enabled and pipeline_ready),
            "active_skill_count": len(enabled),
            "agent_shell_callable": bool(agent_shell_skills and executor_ready),
            "agent_shell_status": "deprecated"
            if agent_shell_replacement
            else "enabled"
            if shell is not None
            else "unavailable",
            "agent_shell_replacement": agent_shell_replacement,
            "agent_shell_skill_count": len(agent_shell_skills),
            "agent_shell_skills": agent_shell_skills,
        }

    def _mission_handler(self, registry: ModuleRegistry) -> Any | None:
        mission_mod = registry.get("mission")
        if mission_mod is None:
            return None
        return getattr(mission_mod, "mission_service", None)

    def _cognition_handler(self, registry: ModuleRegistry) -> Any | None:
        return registry.get("cognition")

    def _runtime_handler(self, registry: ModuleRegistry) -> Any | None:
        return registry.get("runtime_handoff")

    def _capabilities_snapshot(
        self,
        cfg: dict[str, Any],
        registry: ModuleRegistry,
    ) -> dict[str, Any]:
        from askme import __version__ as ASKME_VERSION

        profile = self._runtime_profile(registry)
        skill_mod = registry.get("skill")
        skill_manager = getattr(skill_mod, "skill_manager", None) if skill_mod else None
        contracts = skill_manager.get_contracts() if skill_manager else []
        openapi_doc = (
            skill_manager.openapi_document()
            if skill_manager
            else {"info": {"title": "", "version": ""}, "paths": {}}
        )

        components: dict[str, dict[str, Any]] = {}
        for name, mod in registry.items():
            try:
                health = mod.health()
            except Exception:
                health = {"status": "error"}
            try:
                capabilities = mod.capabilities()
            except Exception:
                capabilities = {}
            components[name] = {
                "health": health,
                "capabilities": capabilities,
            }

        return {
            "app": {
                "name": cfg.get("app", {}).get("name", "askme"),
                "version": cfg.get("app", {}).get("version") or ASKME_VERSION,
                "voice_mode": profile.voice_io,
                "robot_mode": profile.robot_api,
            },
            "profile": profile.snapshot(),
            "components": components,
            "mission_adapter": components.get("mission", {}).get("capabilities", {}),
            "skills": {
                "count": len(skill_manager.get_all()) if skill_manager else 0,
                "enabled_count": len(skill_manager.get_enabled()) if skill_manager else 0,
                "contract_count": len(contracts),
                "code_contract_count": sum(
                    1 for contract in contracts if getattr(contract, "source", None) == "code"
                ),
                "legacy_contract_count": sum(
                    1 for contract in contracts if getattr(contract, "source", None) != "code"
                ),
                "catalog": (skill_manager.get_contract_catalog() if skill_manager else []),
                "capability_center": (
                    skill_manager.get_capability_center()
                    if skill_manager and hasattr(skill_manager, "get_capability_center")
                    else {}
                ),
                "generated_skill_governance": (
                    skill_manager.get_generated_skill_governance()
                    if skill_manager and hasattr(skill_manager, "get_generated_skill_governance")
                    else {}
                ),
                "skill_packages": (
                    skill_manager.get_skill_packages()
                    if skill_manager and hasattr(skill_manager, "get_skill_packages")
                    else {}
                ),
            },
            "openapi": {
                "title": openapi_doc.get("info", {}).get("title", ""),
                "version": openapi_doc.get("info", {}).get("version", ""),
                "path_count": len(openapi_doc.get("paths", {})),
            },
        }

    def _runtime_profile(self, registry: ModuleRegistry) -> Any:
        from askme.runtime.core.profiles import MCP_PROFILE, legacy_profile_for

        has_voice = "voice" in registry
        has_text = "text" in registry
        has_robot = any(
            name in registry for name in ("control", "executor", "led", "perception", "safety")
        )
        if has_voice and has_robot and not has_text:
            return MCP_PROFILE
        return legacy_profile_for(voice_mode=has_voice, robot_mode=has_robot)

    async def start(self) -> None:
        if self.server.enabled:
            await self.server.start()

    async def stop(self) -> None:
        if self.server.enabled:
            await self.server.stop()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "enabled": self.server.enabled,
            "port": self.server.port,
        }
