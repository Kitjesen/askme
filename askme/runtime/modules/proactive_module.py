"""ProactiveModule -wraps ProactiveAgent as a declarative module.

Canonical wiring::

    proactive = ProactiveAgent(
        vision=vision, audio=audio, episodic=episodic, llm=llm, config=cfg,
    )
    proactive.set_solve_callback(
        lambda anomaly_text: pipeline.execute_skill("solve_problem", anomaly_text)
    )
"""

from __future__ import annotations

import asyncio
import logging
from inspect import Parameter, signature
from typing import Any
from uuid import uuid4

from askme.llm.core.client import LLMClient
from askme.pipeline.core.brain_pipeline import BrainPipeline
from askme.pipeline.reactions.proactive_agent import ProactiveAgent
from askme.ports import AudioFrontendPort, VisionPort
from askme.runtime.core.module import In, Module, ModuleRegistry
from askme.schemas.messages import MemoryContext

logger = logging.getLogger(__name__)

_PROACTIVE_SOURCE = "proactive"


def _clean_optional(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _configured_system_scope(cfg: dict[str, Any]) -> tuple[str | None, str | None]:
    """Resolve deployment scope without treating transport IDs as Threads."""

    proactive_cfg = cfg.get("proactive", {})
    robot_cfg = cfg.get("robot", {})
    runtime_cfg = cfg.get("runtime", {})
    voice_bridge_cfg = runtime_cfg.get("voice_bridge", {})
    field_cfg = cfg.get("field_operations", {})
    ota_cfg = cfg.get("ota", {})
    ota_device_cfg = ota_cfg.get("device", {})
    scopes = (
        proactive_cfg,
        robot_cfg,
        voice_bridge_cfg,
        field_cfg,
        ota_device_cfg,
    )

    def first_configured(key: str) -> str | None:
        for scope in scopes:
            value = _clean_optional(scope.get(key))
            if value is not None:
                return value
        return None

    return first_configured("robot_id"), first_configured("site_id")


class ProactiveModule(Module):
    """Provides the ProactiveAgent to the runtime."""

    name = "proactive"
    depends_on = ("llm", "memory", "pipeline")
    provides = ("supervision",)

    llm_in: In[LLMClient]
    memory_in: In[MemoryContext]
    perception_in: In[VisionPort]
    voice_in: In[AudioFrontendPort]
    pipeline_in: In[BrainPipeline]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self._proactive_thread_id = uuid4().hex
        self._stop_event = asyncio.Event()

        # In[T] ports are pre-set to None by Module.__init__ and overwritten by
        # _auto_wire() when a provider exists.  Inner getattr guards against
        # providers that skip setting a specific attribute (e.g. fake test doubles).
        llm_mod = self.llm_in
        llm = getattr(llm_mod, "llm_client", None) if llm_mod else None

        mem_mod = self.memory_in
        episodic = getattr(mem_mod, "episodic", None) if mem_mod else None

        perception_mod = self.perception_in
        vision = getattr(perception_mod, "vision_bridge", None) if perception_mod else None

        voice_mod = self.voice_in
        audio = getattr(voice_mod, "audio", None) if voice_mod else None

        pipeline_mod = self.pipeline_in
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None

        self.agent = ProactiveAgent(
            vision=vision,
            audio=audio,
            episodic=episodic,
            llm=llm,
            config=cfg,
        )

        if pipeline is not None:
            robot_id, site_id = _configured_system_scope(cfg)
            turn_ledger = getattr(pipeline, "turn_ledger", None)
            resolve_thread = getattr(turn_ledger, "resolve_thread", None)
            if callable(resolve_thread):
                thread = resolve_thread(
                    **self._supported_kwargs(
                        resolve_thread,
                        {
                            "thread_id": self._proactive_thread_id,
                            "conversation_session_id": self._proactive_thread_id,
                            "channel": _PROACTIVE_SOURCE,
                            "robot_id": robot_id,
                            "site_id": site_id,
                            "metadata": {
                                "actor_type": "system",
                                "source": _PROACTIVE_SOURCE,
                            },
                        },
                    )
                )
                resolved_thread_id = _clean_optional(getattr(thread, "thread_id", None))
                if resolved_thread_id is not None:
                    self._proactive_thread_id = resolved_thread_id

            async def solve(anomaly_text: str) -> str:
                callback = pipeline.execute_skill
                return await callback(
                    "solve_problem",
                    anomaly_text,
                    **self._supported_kwargs(
                        callback,
                        {
                            "source": _PROACTIVE_SOURCE,
                            "conversation_session_id": self._proactive_thread_id,
                            "voice_turn_id": uuid4().hex,
                            "turn_cancel_token": self._stop_event,
                        },
                    ),
                )

            self.agent.set_solve_callback(solve)

        self.agent.set_interaction_context(
            session_id=self._proactive_thread_id,
            cancel_token=self._stop_event,
        )

        logger.info(
            "ProactiveModule: built (enabled=%s)",
            self.agent._enabled,
        )

    @staticmethod
    def _supported_kwargs(
        callback: Any,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Return context keywords supported by modern and legacy callbacks."""

        signature_target = getattr(callback, "side_effect", None)
        if not callable(signature_target):
            signature_target = callback
        try:
            parameters = signature(signature_target).parameters
        except (TypeError, ValueError):
            return {}
        if any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return {key: value for key, value in values.items() if value is not None}
        keyword_kinds = {
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        }
        return {
            key: value
            for key, value in values.items()
            if value is not None and key in parameters and parameters[key].kind in keyword_kinds
        }

    async def start(self) -> None:
        if self.agent._enabled:
            if self._stop_event.is_set():
                self._stop_event = asyncio.Event()
                self.agent.set_interaction_context(
                    session_id=self._proactive_thread_id,
                    cancel_token=self._stop_event,
                )
            self._task = asyncio.create_task(
                self.agent.run(self._stop_event),
                name="askme-proactive",
            )

    async def stop(self) -> None:
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        task = getattr(self, "_task", None)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "enabled": self.agent._enabled,
        }
