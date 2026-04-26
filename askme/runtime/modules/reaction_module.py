"""ReactionModule -- scene-aware reaction decision engine as a runtime module.

Sits between perception events and output channels (TTS/alerts).
Fully self-contained: subscribes to ChangeEvents itself, has its own
AlertDispatcher reference. Does NOT delegate to or from ProactiveAgent.
"""

from __future__ import annotations

import logging
from typing import Any

from askme.llm.client import LLMClient
from askme.runtime.module import In, Module, ModuleRegistry
from askme.schemas.messages import MemoryContext

try:
    from askme.voice.audio_agent import AudioAgent
except ModuleNotFoundError:
    AudioAgent = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class ReactionModule(Module):
    """Provides scene-aware reaction decisions to the runtime.

    Depends on perception (WorldState), memory (EpisodicMemory),
    pipeline (LLM), and optionally skill (SkillDispatcher).
    Fully independent of ProactiveModule.
    """

    name = "reaction"
    depends_on = ("llm", "memory")

    # In ports (auto-wired from provider modules)
    llm_in: In[LLMClient]
    memory_in: In[MemoryContext]
    voice_in: In[AudioAgent]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        from askme.pipeline.reaction_engine import (
            HybridReaction,
            LLMReaction,
            RuleBasedReaction,
        )

        llm_mod = self.llm_in
        llm = getattr(llm_mod, "client", None) if llm_mod else None

        mem_mod = self.memory_in
        episodic = getattr(mem_mod, "episodic", None) if mem_mod else None

        voice_mod = self.voice_in
        audio = getattr(voice_mod, "audio", None) if voice_mod else None

        # Create our own AlertDispatcher
        from askme.pipeline.alert_dispatcher import AlertDispatcher

        pro_cfg = cfg.get("proactive", {})
        alert_dispatcher = AlertDispatcher(
            voice=audio,
            config=pro_cfg.get("alerts", {}),
            robot_id=cfg.get("robot", {}).get("robot_id"),
            robot_name=cfg.get("robot", {}).get("robot_name", "Thunder"),
        )

        # Read reaction config
        reaction_cfg = pro_cfg.get("reaction", {})
        backend_name = reaction_cfg.get("backend", "hybrid")
        content_model = reaction_cfg.get(
            "llm_content_model",
            cfg.get("brain", {}).get("voice_model", ""),
        )
        content_timeout = float(reaction_cfg.get("llm_content_timeout", 5.0))

        # Business hours config
        bh = reaction_cfg.get("business_hours", [8, 18])
        self._business_hours_start = int(bh[0]) if len(bh) > 0 else 8
        self._business_hours_end = int(bh[1]) if len(bh) > 1 else 18

        # Construct backend
        if backend_name == "rules":
            self.engine = RuleBasedReaction(
                alert_dispatcher=alert_dispatcher,
                episodic=episodic,
            )
        elif backend_name == "llm":
            self.engine = LLMReaction(
                llm=llm,
                alert_dispatcher=alert_dispatcher,
                episodic=episodic,
                decision_model=content_model,
                decision_timeout=content_timeout,
            )
        else:
            # Default: hybrid
            self.engine = HybridReaction(
                llm=llm,
                alert_dispatcher=alert_dispatcher,
                episodic=episodic,
                content_model=content_model,
                content_timeout=content_timeout,
            )

        self._backend_name = backend_name

        logger.info(
            "ReactionModule: built (backend=%s, content_model=%s)",
            backend_name,
            content_model or "(default)",
        )

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": self._backend_name,
        }

