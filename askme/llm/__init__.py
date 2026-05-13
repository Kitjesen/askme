"""Product LLM package.

Root imports stay backward compatible.  Real implementation lives under
``core/``, provider adapters under ``providers/``, and runtime policy under
``policy/``.
"""

from askme.interaction.intent_router import IntentRouter
from askme.llm.core.client import LLMClient
from askme.llm.core.config import LLMConfig
from askme.llm.core.gateway import LLMGateway
from askme.memory.conversation import ConversationManager

__all__ = ["LLMClient", "LLMConfig", "LLMGateway", "ConversationManager", "IntentRouter"]
