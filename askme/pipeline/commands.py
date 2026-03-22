"""Built-in command handler: /help, /quit, /clear, /history, /skills."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askme.llm.conversation import ConversationManager
    from askme.skills.skill_manager import SkillManager

logger = logging.getLogger(__name__)


class CommandHandler:
    """Processes built-in slash commands.

    Returns ``True`` from :meth:`handle` when the caller should exit.
    """

    QUIT_COMMANDS = {"/quit", "/exit", "exit", "quit"}
    ALL_COMMANDS = QUIT_COMMANDS | {"/clear", "/history", "/help", "/skills"}

    def __init__(
        self,
        *,
        conversation: ConversationManager,
        skill_manager: SkillManager,
    ) -> None:
        self._conversation = conversation
        self._skill_manager = skill_manager

    def handle(self, cmd: str) -> bool:
        """Execute *cmd*. Return ``True`` if the app should quit."""
        if cmd in self.QUIT_COMMANDS:
            return True

        if cmd == "/clear":
            self._conversation.clear()
            logger.info("Conversation history cleared.")

        elif cmd == "/history":
            for msg in self._conversation.history:
                role = "You" if msg["role"] == "user" else "AI"
                logger.info("  [%s]: %s", role, msg.get("content", "")[:80])

        elif cmd == "/skills":
            skills = self._skill_manager.get_enabled()
            if skills:
                for s in skills:
                    vt = f" (voice: '{s.voice_trigger}')" if s.voice_trigger else ""
                    logger.info("  [%s] %s%s", s.name, s.description, vt)
            else:
                logger.info("  No skills loaded.")

        elif cmd == "/help":
            logger.info("  /clear    - Clear conversation history")
            logger.info("  /history  - Show conversation history")
            logger.info("  /skills   - List available skills")
            logger.info("  /quit     - Exit")

        return False
