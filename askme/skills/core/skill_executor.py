"""
Skill executor for askme.

Takes a SkillDefinition, builds a prompt with context, calls the LLM
with available tools, handles tool-call loops, and returns the final response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import replace
from inspect import Parameter, signature
from typing import Any

from askme.conversation import InteractionTurnContext
from askme.llm.core.contracts import LLMCallContext
from askme.telemetry.ota_bridge import OTABridgeMetrics
from askme.tools.core.tool_registry import ToolRegistry

from .skill_model import SkillDefinition

logger = logging.getLogger(__name__)

_NAV_PREFLIGHT_TIMEOUT_S = 0.35


class SkillExecutor:
    """Execute a skill by calling the LLM with tools and handling tool loops."""

    def __init__(
        self,
        llm_client: Any,
        tool_registry: ToolRegistry,
        default_model: str = "",
        metrics: OTABridgeMetrics | None = None,
    ) -> None:
        """
        Args:
            llm_client: An LLMClient or AsyncOpenAI-compatible client.
            tool_registry: The tool registry for tool definitions and execution.
            default_model: Fallback model if skill doesn't specify one.
        """
        self._llm = llm_client
        self._tools = tool_registry
        self._default_model = default_model
        self._metrics = metrics

    @staticmethod
    def _accepts_keyword(callback: Any, name: str) -> bool:
        """Return whether a callable accepts one optional compatibility keyword."""

        signature_target = callback
        side_effect = getattr(callback, "side_effect", None)
        if callable(side_effect):
            signature_target = side_effect
        try:
            parameters = signature(signature_target).parameters
        except (TypeError, ValueError):
            return False
        parameter = parameters.get(name)
        return bool(
            (parameter is not None and parameter.kind is not Parameter.POSITIONAL_ONLY)
            or any(candidate.kind is Parameter.VAR_KEYWORD for candidate in parameters.values())
        )

    @staticmethod
    def _interaction_context(
        llm_call_context: LLMCallContext,
    ) -> InteractionTurnContext | None:
        """Project complete LLM turn identity into canonical tool policy context."""

        thread_id = str(llm_call_context.session_id or "").strip()
        turn_id = str(llm_call_context.turn_id or "").strip()
        if not thread_id or not turn_id:
            return None
        channel = str(llm_call_context.channel or "").strip()
        operator_id = str(llm_call_context.operator_id or "").strip()
        context_kwargs: dict[str, Any] = {}
        cancel_token = getattr(llm_call_context, "cancel_token", None)
        if cancel_token is not None:
            context_kwargs["cancel_token"] = cancel_token
        return InteractionTurnContext(
            thread_id=thread_id,
            turn_id=turn_id,
            channel=channel,
            source=channel,
            user_text="",
            operator_id=operator_id or None,
            **context_kwargs,
        )

    def _has_pending_approval(
        self,
        interaction_context: InteractionTurnContext | None,
    ) -> bool:
        callback = self._tools.has_pending_approval
        if interaction_context is not None and self._accepts_keyword(
            callback,
            "interaction_context",
        ):
            return bool(callback(interaction_context=interaction_context))
        return bool(callback())

    async def preflight_skill(self, skill: SkillDefinition) -> tuple[bool, str]:
        """Check deterministic runtime prerequisites before an audible preface.

        Most skills have no cheap deterministic readiness probe and remain
        executable. ``nav_query`` is different: saying that location is being
        read before the navigation gateway or its pose is ready creates a
        customer-visible dead end. Query the registered ``nav_status`` adapter
        without invoking an LLM.
        """

        if str(getattr(skill, "name", "") or "") != "nav_query":
            return True, "ready"

        get_tool = getattr(self._tools, "get", None)
        nav_status = get_tool("nav_status") if callable(get_tool) else None
        if nav_status is None:
            return False, "nav_status_tool_missing"

        navigation_client = getattr(nav_status, "_navigation_client", None)
        if navigation_client is not None:
            is_configured = getattr(navigation_client, "is_configured", None)
            if callable(is_configured) and not bool(is_configured()):
                return False, "nav_gateway_unconfigured"
            status = getattr(navigation_client, "status", None)
            if not callable(status):
                return False, "nav_status_unavailable"
            try:
                payload = await asyncio.wait_for(
                    asyncio.to_thread(status),
                    timeout=_NAV_PREFLIGHT_TIMEOUT_S,
                )
            except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
                return False, "nav_gateway_unavailable"
            except Exception as exc:
                logger.warning("Navigation preflight failed: %s", exc)
                return False, "nav_gateway_unavailable"
        else:
            if not str(os.environ.get("NAV_GATEWAY_URL", "") or "").strip():
                return False, "nav_gateway_unconfigured"
            execute = getattr(nav_status, "execute", None)
            if not callable(execute):
                return False, "nav_status_unavailable"
            try:
                payload = await asyncio.wait_for(
                    asyncio.to_thread(execute),
                    timeout=_NAV_PREFLIGHT_TIMEOUT_S,
                )
            except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
                return False, "nav_gateway_unavailable"
            except Exception as exc:
                logger.warning("Navigation preflight failed: %s", exc)
                return False, "nav_gateway_unavailable"

        return _navigation_status_readiness(payload)

    async def execute(
        self,
        skill: SkillDefinition,
        context: dict[str, str] | None = None,
        *,
        prompt_seed: list[dict[str, Any]] | None = None,
        on_tool_call: Any | None = None,
        llm_call_context: LLMCallContext | None = None,
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
        if skill.execution == "read_only_tool":
            tool_names = [
                line.strip()
                for line in skill.tools_section.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if len(tool_names) != 1:
                return "[Error] read_only_tool skills must declare exactly one tool."
            tool_name = tool_names[0]
            tool = self._tools.get(tool_name)
            if (
                skill.safety_level != "normal"
                or tool is None
                or not getattr(tool, "read_only", False)
                or getattr(tool, "safety_level", "critical") != "normal"
            ):
                return f"[Error] Tool '{tool_name}' is not approved for read-only execution."
            return await asyncio.to_thread(
                self._tools.execute,
                tool_name,
                "{}",
                allowed_names={tool_name},
                max_safety_level="normal",
            )

        prompt = skill.build_prompt(context)
        model = skill.model or self._default_model
        timeout = skill.timeout
        max_safety_level = skill.safety_level or "normal"

        # Determine which tools to expose
        tool_definitions = self._tools.get_definitions(
            max_safety_level=max_safety_level,
        )

        # Exclude dispatch_skill from skill tool sets — prevents recursive dispatch
        # which would saturate the thread pool via asyncio.run_coroutine_threadsafe.
        tool_definitions = [
            td for td in tool_definitions if td.get("function", {}).get("name") != "dispatch_skill"
        ]

        # If the skill specifies a tools section, filter to only those tools
        allowed_tools: list[str] | None = None
        if skill.tools_section:
            allowed_tools = [
                t.strip()
                for t in skill.tools_section.split("\n")
                if t.strip() and not t.strip().startswith("#")
            ]
            if allowed_tools:
                tool_definitions = [
                    td
                    for td in tool_definitions
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

        # Inject fake-turn seed to establish robot identity.
        # The relay service overrides the system message with its own developer
        # prompt — fake turns are the only way to force Chinese-only output.
        if prompt_seed:
            messages.extend(prompt_seed)

        # If context has a user_input, add it as a user message
        if context and "user_input" in context:
            messages.append({"role": "user", "content": context["user_input"]})

        base_llm_context = llm_call_context or LLMCallContext(
            purpose="tool_followup",
            channel="text",
            request_class="robot_action",
            privacy_class="conversation",
            allow_cache=False,
        )

        import time as _time

        t_start = _time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._run_tool_loop(
                    messages,
                    model,
                    tool_definitions,
                    allowed_tool_names=allowed_tool_names,
                    max_safety_level=max_safety_level,
                    on_tool_call=on_tool_call,
                    llm_call_context=base_llm_context,
                ),
                timeout=timeout,
            )
            duration_s = _time.perf_counter() - t_start
            if self._metrics is not None:
                success = not result.startswith("[Error]") and not result.startswith("[Timeout]")
                self._metrics.record_skill_execution(
                    success=success,
                    skill_name=skill.name,
                    duration_s=duration_s,
                )
            return result
        except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
            duration_s = _time.perf_counter() - t_start
            logger.warning("Skill '%s' timed out after %ds", skill.name, timeout)
            if self._metrics is not None:
                self._metrics.record_skill_execution(
                    success=False,
                    skill_name=skill.name,
                    duration_s=duration_s,
                )
            return f"[Timeout] Skill '{skill.name}' execution timed out after {timeout}s."
        except Exception as exc:
            duration_s = _time.perf_counter() - t_start
            logger.warning("Skill '%s' failed: %s", skill.name, exc)
            if self._metrics is not None:
                self._metrics.record_skill_execution(
                    success=False,
                    skill_name=skill.name,
                    duration_s=duration_s,
                )
            return f"[Error] Skill '{skill.name}' execution failed: {exc}"

    async def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tool_definitions: list[dict[str, Any]],
        *,
        allowed_tool_names: set[str] | None = None,
        max_safety_level: str = "critical",
        max_iterations: int = 5,
        on_tool_call: Any | None = None,
        llm_call_context: LLMCallContext,
    ) -> str:
        """Run the LLM -> tool-call -> LLM loop until a text response is produced."""
        interaction_context = self._interaction_context(llm_call_context)
        for iteration in range(max_iterations):
            response = await self._create_completion(
                messages,
                model=model,
                tool_definitions=tool_definitions,
                llm_call_context=llm_call_context,
            )
            choice = response.choices[0]
            message = choice.message

            # If the model returned tool calls, execute them
            if message.tool_calls:
                # Append the assistant message (with tool_calls) to conversation
                messages.append(
                    {
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
                    }
                )

                # Execute each tool call and append results
                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    tool_args = tc.function.arguments
                    logger.info(
                        "Skill tool call [%d/%d]: %s(%s)",
                        iteration + 1,
                        max_iterations,
                        tool_name,
                        tool_args,
                    )
                    if on_tool_call:
                        on_tool_call(tool_name)
                    execute_kwargs: dict[str, Any] = {
                        "allowed_names": allowed_tool_names,
                        "max_safety_level": max_safety_level,
                    }
                    if interaction_context is not None and self._accepts_keyword(
                        self._tools.execute,
                        "interaction_context",
                    ):
                        execute_kwargs["interaction_context"] = interaction_context
                    result = await asyncio.to_thread(
                        self._tools.execute,
                        tool_name,
                        tool_args,
                        **execute_kwargs,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result),
                        }
                    )
                    if self._has_pending_approval(interaction_context):
                        return str(result)
                continue

            # No tool calls -- return the text response
            return message.content or ""

        # Exhausted iterations
        logger.warning("Tool loop exhausted after %d iterations", max_iterations)
        return "[Error] Maximum tool-call iterations reached."

    async def _create_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tool_definitions: list[dict[str, Any]],
        llm_call_context: LLMCallContext,
    ) -> Any:
        if hasattr(self._llm, "chat_completion"):
            call_context = replace(
                llm_call_context,
                call_id=uuid.uuid4().hex,
                purpose="tool_followup",
            )
            return await self._llm.chat_completion(
                messages,
                tools=tool_definitions or None,
                tool_choice="auto" if tool_definitions else None,
                model=model,
                context=call_context,
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tool_definitions:
            kwargs["tools"] = tool_definitions
            kwargs["tool_choice"] = "auto"
        return await self._llm.chat.completions.create(**kwargs)


def _navigation_status_readiness(payload: Any) -> tuple[bool, str]:
    """Interpret nav-gateway status conservatively for current-pose queries."""

    if isinstance(payload, str):
        text = payload.strip()
        try:
            payload = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            lowered = text.lower()
            if "not configured" in lowered or "未配置" in text or "nav_gateway_url" in lowered:
                return False, "nav_gateway_unconfigured"
            if any(token in lowered for token in ("stale", "odometry_missing", "pose_not_ready")):
                return False, "nav_pose_stale"
            if "查询失败" in text or "[error]" in lowered or "unreachable" in lowered:
                return False, "nav_gateway_unavailable"
            return False, "nav_status_unrecognized"

    if not isinstance(payload, dict):
        return False, "nav_status_unrecognized"

    for key, value in _walk_status_items(payload):
        normalized_key = str(key or "").strip().lower()
        if normalized_key == "error" and value:
            lowered = str(value).lower()
            if "not configured" in lowered or "nav_gateway_url" in lowered:
                return False, "nav_gateway_unconfigured"
            return False, "nav_gateway_unavailable"
        if normalized_key in {"ready", "pose_fresh", "has_odometry"} and value is False:
            return False, "nav_pose_stale"
        if normalized_key in {"stale", "pose_stale", "odometry_stale"} and value is True:
            return False, "nav_pose_stale"
        if normalized_key in {"reason", "code"}:
            lowered = str(value or "").lower()
            if any(
                token in lowered
                for token in (
                    "stale",
                    "odometry_missing",
                    "pose_not_ready",
                    "localization_not_ready",
                )
            ):
                return False, "nav_pose_stale"
        if normalized_key in {"status", "state", "readiness"}:
            lowered = str(value or "").strip().lower()
            if lowered in {"stale", "not_ready", "unready", "unavailable", "offline"}:
                return False, "nav_pose_stale"

    return True, "ready"


def _walk_status_items(payload: dict[str, Any]):
    for key, value in payload.items():
        yield key, value
        if isinstance(value, dict):
            yield from _walk_status_items(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield from _walk_status_items(item)
