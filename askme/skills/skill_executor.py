"""
Skill executor for askme.

Takes a SkillDefinition, builds a prompt with context, calls the LLM
with available tools, handles tool-call loops, and returns the final response.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .skill_model import SkillDefinition
from ..tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class SkillExecutor:
    """Execute a skill by calling the LLM with tools and handling tool loops."""

    def __init__(
        self,
        llm_client: Any,
        tool_registry: ToolRegistry,
        default_model: str = "deepseek-chat",
    ) -> None:
        """
        Args:
            llm_client: An AsyncOpenAI-compatible client.
            tool_registry: The tool registry for tool definitions and execution.
            default_model: Fallback model if skill doesn't specify one.
        """
        self._llm = llm_client
        self._tools = tool_registry
        self._default_model = default_model

    async def execute(
        self,
        skill: SkillDefinition,
        context: dict[str, str] | None = None,
    ) -> str:
        """Execute a skill end-to-end.

        1. Build the prompt from the skill template + context variables.
        2. Call the LLM with the prompt and tool definitions.
        3. If the LLM requests tool calls, execute them and feed results back.
        4. Repeat until the LLM returns a text response or timeout.

        Args:
            skill: The skill definition to execute.
            context: Template variables to substitute in the prompt.

        Returns:
            The final LLM text response.
        """
        prompt = skill.build_prompt(context)
        model = skill.model or self._default_model
        timeout = skill.timeout
        max_safety_level = skill.safety_level or "normal"

        # Determine which tools to expose
        tool_definitions = self._tools.get_definitions(
            max_safety_level=max_safety_level,
        )

        # If the skill specifies a tools section, filter to only those tools
        allowed_tools: list[str] | None = None
        if skill.tools_section:
            allowed_tools = [
                t.strip() for t in skill.tools_section.split("\n")
                if t.strip() and not t.strip().startswith("#")
            ]
            if allowed_tools:
                tool_definitions = [
                    td for td in tool_definitions
                    if td.get("function", {}).get("name") in allowed_tools
                ]
        allowed_tool_names = {
            td.get("function", {}).get("name")
            for td in tool_definitions
            if td.get("function", {}).get("name")
        }

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
        ]

        # If context has a user_input, add it as a user message
        if context and "user_input" in context:
            messages.append({"role": "user", "content": context["user_input"]})

        try:
            result = await asyncio.wait_for(
                self._run_tool_loop(
                    messages,
                    model,
                    tool_definitions,
                    allowed_tool_names=allowed_tool_names,
                    max_safety_level=max_safety_level,
                ),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Skill '%s' timed out after %ds", skill.name, timeout)
            return f"[Timeout] Skill '{skill.name}' execution timed out after {timeout}s."

    async def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tool_definitions: list[dict[str, Any]],
        *,
        allowed_tool_names: set[str] | None = None,
        max_safety_level: str = "critical",
        max_iterations: int = 5,
    ) -> str:
        """Run the LLM -> tool-call -> LLM loop until a text response is produced."""
        for iteration in range(max_iterations):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if tool_definitions:
                kwargs["tools"] = tool_definitions
                kwargs["tool_choice"] = "auto"

            response = await self._llm.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message

            # If the model returned tool calls, execute them
            if message.tool_calls:
                # Append the assistant message (with tool_calls) to conversation
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })

                # Execute each tool call and append results
                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    tool_args = tc.function.arguments
                    logger.info(
                        "Skill tool call [%d/%d]: %s(%s)",
                        iteration + 1, max_iterations, tool_name, tool_args,
                    )
                    result = await asyncio.to_thread(
                        self._tools.execute,
                        tool_name,
                        tool_args,
                        allowed_names=allowed_tool_names,
                        max_safety_level=max_safety_level,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })
                    if self._tools.has_pending_approval():
                        return str(result)
                continue

            # No tool calls -- return the text response
            return message.content or ""

        # Exhausted iterations
        logger.warning("Tool loop exhausted after %d iterations", max_iterations)
        return "[Error] Maximum tool-call iterations reached."
