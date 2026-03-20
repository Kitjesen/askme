"""ThunderAgentShell — Claude Agent SDK wrapper for agentic task execution.

Gives Thunder the same "write code, execute, debug, iterate" capability
that Claude Code has.  Unlike skills (fixed prompt + max 5 LLM turns),
ThunderAgentShell lets Claude autonomously decide the tool sequence and
iterate until the task is done.

Architecture:
    VoiceLoop/TextLoop
        → SkillDispatcher.handle_general("agent_task")
            → ThunderAgentShell.run_task(user_text)
                → Agentic loop: LLM ↔ tools until stop_reason="end_turn"
                    → BrainPipeline._audio.speak() for progress
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from askme.brain.llm_client import LLMClient
    from askme.tools.tool_registry import ToolRegistry
    from askme.voice.audio_agent import AudioAgent

logger = logging.getLogger(__name__)

# Tools exposed to the agentic loop — subset of full tool registry.
# Excludes voice-control tools (mute/unmute) and dispatch_skill
# (to avoid nested skill-in-skill recursion).
_AGENT_ALLOWED_TOOLS = {
    "bash",
    "write_file",
    "read_file",
    "list_directory",
    "http_request",
    "robot_api",
    "get_current_time",
    "speak_progress",
    "web_fetch",
    "web_search",
    "edit_file",     # surgical string replacement in any file
    "create_skill",  # agent can solidify solutions into reusable skills
    "spawn_agent",   # spawn child agent for focused sub-tasks
    "look_around",   # vision: describe current scene
    "find_target",   # vision: search for specific object by YOLO class
    "move_robot",    # motion: rotate, forward, go_to, stop
    "scan_around",   # fast 360° scan: rotate + capture + batch analyze
}

_DEFAULT_AGENT_MODEL = "MiniMax-M2.7-highspeed"  # fast + stable + reasoning, no relay dependency
_MAX_ITERATIONS = 10
_DEFAULT_TIMEOUT = 120.0
_MAX_DEPTH = 1  # max sub-agent nesting depth (0=root, 1=child, children cannot spawn)
# Keep the original task message + last N to cap prompt size (prevents TTFT blowup
# on later iterations and avoids memory pressure on Sunrise's constrained RAM).
_MAX_MESSAGES = 20

# Friendly Chinese names for TTS progress announcements
_TOOL_VOICE_LABELS: dict[str, str] = {
    "bash": "运行命令",
    "write_file": "写入文件",
    "edit_file": "编辑文件",
    "read_file": "读取文件",
    "list_directory": "查看目录",
    "web_search": "搜索网络",
    "web_fetch": "抓取网页",
    "http_request": "调用接口",
    "robot_api": "查询机器人",
    "spawn_agent": "启动子任务",
    "create_skill": "创建新技能",
    "get_current_time": "获取时间",
    "speak_progress": None,  # LLM-driven TTS — don't double-announce
    "look_around": "观察环境",
    "move_robot": "移动机器人",
    "scan_around": "全方位扫描",
    "find_target": "搜索物体",
}

# Inline schema for spawn_agent — not registered in ToolRegistry to avoid shared-state mutation
_SPAWN_AGENT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "spawn_agent",
        "description": (
            "启动子 Agent 自主完成一个专注的子任务（最多嵌套1层）。"
            "子 Agent 拥有独立执行上下文和工具权限，完成后返回结果字符串。"
            "适用场景：可独立完成的局部任务，如'写一个函数'、'查某API文档'、'分析某文件内容'。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "子 Agent 要完成的具体任务，要写清楚目标和约束",
                },
                "context": {
                    "type": "string",
                    "description": "给子 Agent 的额外上下文（可选），如相关文件路径、已知信息等",
                },
            },
            "required": ["task"],
        },
    },
}


def _build_agent_system_prompt(workspace: Path) -> str:
    return (
        "你是 Thunder 机器人上运行的自主 Agent，拥有真实的执行能力。\n"
        "你可以运行 shell 命令、读写文件、搜索网络、调用机器人 API、发送 HTTP 请求。\n\n"
        f"工作区：{workspace}（所有文件操作默认在此目录内）\n\n"
        "【工具使用指南】\n"
        "  bash         — shell 命令执行，支持 python/pip/curl 等；超时 30s\n"
        "  write_file   — 写文件到工作区；path 用相对路径（如 result.py）\n"
        "  edit_file    — 精确替换文件内容（old_string → new_string）；old_string 必须唯一\n"
        "  read_file    — 读取文件；path 为绝对路径\n"
        "  web_search   — 搜索网络获取摘要和链接；技术查询建议加版本号（如 'asyncio Python 3.10'）\n"
        "  web_fetch    — 抓取指定网页完整内容；web_search 找到 URL 后用此工具深读\n"
        "  http_request — 调用 REST API（已预授权 localhost:5050-5110）\n"
        "  robot_api    — Thunder runtime 快捷接口：\n"
        "                 service=telemetry  GET /api/v1/health → 电量/温度/IMU\n"
        "                 service=safety     GET /api/v1/safety/modes/estop → 急停状态\n"
        "                 service=control    POST /api/v1/control/commands {cmd,params} → 运动指令\n"
        "                 service=nav        GET /api/v1/missions → 导航任务列表\n"
        "                 service=arbiter    GET /api/v1/missions → 任务编排状态\n"
        "                 service=arm        GET /api/v1/arm/state → 机械臂状态\n"
        "                 service=ops        GET /api/v1/ops/logs → 运维日志\n"
        "  spawn_agent  — 启动子 Agent 执行独立子任务（最多嵌套1层）；适合可并行的局部工作\n"
        "  speak_progress — 主动向用户播报进度（不阻塞执行）\n"
        "  create_skill — 把当前解决方案固化为新语音技能（写 SKILL.md + 热加载）\n\n"
        "【执行原则】\n"
        "1. 行动优先：直接用工具做，不要先说'我将会...'再做\n"
        "2. 搜索后深读：web_search 拿到链接 → web_fetch 读全文 → 再综合回答\n"
        "3. 验证每步：bash/http_request 执行后检查输出，失败时换策略（最多3次重试）\n"
        "4. 并行子任务：多个独立子任务用 spawn_agent 并行处理，节省时间\n"
        "5. 进度播报：超过15秒的操作前用 speak_progress 告知用户在做什么\n"
        "6. 保存结果：有价值的输出写入工作区文件（write_file），避免丢失\n"
        "7. 固化方案：如果任务会重复执行，用 create_skill 固化为语音技能\n"
        "8. 口语回复：最终回复用简洁中文口语，说清楚'做了什么+结果是什么'，不用 markdown"
    )


class ThunderAgentShell:
    """Agentic execution shell for Thunder robot.

    Wraps the LLM client in an autonomous tool-use loop that continues
    until the model emits stop_reason='end_turn' (task complete) or
    the iteration/timeout limit is reached.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        audio: AudioAgent | None,
        *,
        model: str | None = None,
        workspace: Path | None = None,
        _depth: int = 0,
    ) -> None:
        self._llm = llm_client
        self._tools = tool_registry
        self._audio = audio
        self._model = model or os.environ.get("AGENT_MODEL", _DEFAULT_AGENT_MODEL)
        self._workspace = workspace or (
            Path(__file__).parent.parent.parent / "data" / "agent_workspace"
        )
        self._depth = _depth
        # Current action string — updated during tool execution so the heartbeat
        # can report what the agent is doing instead of a generic message.
        self._current_action = ""

    async def run_task(
        self,
        task: str,
        *,
        context: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> str:
        """Run an agentic task loop until completion or timeout.

        Returns the final assistant text response.
        """
        logger.info("[AgentShell] Starting task: %s", task[:80])
        self._workspace.mkdir(parents=True, exist_ok=True)

        self._did_stream_tts = False
        self._streamed_tts_text = ""

        # Announce start (only for root agent; child agents are silent)
        if self._audio is not None:
            self._audio.speak("好的，我来处理一下。")

        # Reset action tracker so heartbeat starts clean for each new task
        self._current_action = ""

        # Heartbeat: speak every 30s so the user knows we're still working.
        # Reads self._current_action set by _run_agent_loop so the message
        # reflects what the agent is actually doing ("第3步正在搜索网络，请稍候")
        # rather than cycling through generic placeholder strings.
        _hb_count = 0

        async def _heartbeat() -> None:
            nonlocal _hb_count
            await asyncio.sleep(30.0)
            while True:
                try:
                    action = self._current_action
                    if action:
                        msg = f"{action}，请稍候..."
                    else:
                        msg = "还在处理中，请稍候..." if _hb_count % 2 == 0 else "任务进行中，马上好..."
                    _hb_count += 1
                    self._audio.speak(msg)
                    logger.info("[AgentShell] Heartbeat: %s", msg)
                    await asyncio.sleep(30.0)
                except asyncio.CancelledError:
                    break

        heartbeat_task: asyncio.Task[None] | None = None
        if self._audio is not None:
            heartbeat_task = asyncio.create_task(_heartbeat())

        system_prompt = _build_agent_system_prompt(self._workspace)

        # Build context string
        ctx_parts = [f"任务：{task}"]
        if context:
            for k, v in context.items():
                if v:
                    ctx_parts.append(f"{k}: {v}")
        user_message = "\n".join(ctx_parts)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message},
        ]

        # Get tool definitions for allowed tools only
        tool_definitions = self._tools.get_definitions(
            allowed_names=_AGENT_ALLOWED_TOOLS,
            max_safety_level="dangerous",
        )
        # Inject spawn_agent inline schema (not in registry) when nesting allows it
        if self._depth < _MAX_DEPTH:
            tool_definitions = list(tool_definitions) + [_SPAWN_AGENT_SCHEMA]
        logger.info(
            "[AgentShell] Tools available (%d): %s",
            len(tool_definitions),
            [td.get("function", {}).get("name") for td in tool_definitions],
        )

        final_response = ""
        try:
            final_response = await asyncio.wait_for(
                self._run_agent_loop(messages, tool_definitions, system_prompt),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[AgentShell] Task timed out after %.0fs", timeout)
            final_response = f"任务执行超时（{int(timeout)}秒），已停止。"
        except Exception as exc:
            logger.error("[AgentShell] Task failed: %s", exc)
            final_response = f"任务执行出错：{exc}"
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

        return final_response or "任务已完成。"

    async def _run_agent_loop(
        self,
        messages: list[dict[str, Any]],
        tool_definitions: list[dict[str, Any]],
        system_prompt: str,
    ) -> str:
        """Inner agentic loop — separated for asyncio.wait_for compatibility (Python 3.10+)."""
        final_response = ""
        iterations = 0

        while iterations < _MAX_ITERATIONS:
            iterations += 1
            logger.info("[AgentShell] Iteration %d/%d", iterations, _MAX_ITERATIONS)

            response_text, tool_calls = await self._call_llm(
                messages, tool_definitions, system_prompt
            )

            if response_text:
                final_response = response_text

            if not tool_calls:
                self._did_stream_tts = bool(self._streamed_tts_text)
                logger.info(
                    "[AgentShell] Task complete after %d iterations (streamed_tts=%s)",
                    iterations, self._did_stream_tts,
                )
                break

            # Announce the first tool call so user knows what's happening.
            # Show step number from iteration 2 onward so user can gauge progress.
            # Also update _current_action so the heartbeat can report context
            # ("第3步正在搜索网络，请稍候") instead of a generic message.
            if tool_calls:
                first_name = tool_calls[0].get("name", "")
                label = _TOOL_VOICE_LABELS.get(first_name)
                if label:
                    step_prefix = f"第{iterations}步，" if iterations > 1 else ""
                    self._current_action = f"{step_prefix}正在{label}"
                    if self._audio is not None:
                        self._audio.speak(f"{step_prefix}正在{label}...")
                else:
                    # speak_progress or unknown tool — clear action text so
                    # the heartbeat falls back to a generic message.
                    self._current_action = ""

            # Execute tool calls in parallel (asyncio.gather preserves order).
            # return_exceptions=True: single tool failure stays isolated — the other
            # tools' results are still usable and LLM can decide how to recover.
            raw_results = await asyncio.gather(
                *[self._execute_tool(tc) for tc in tool_calls],
                return_exceptions=True,
            )
            tool_call_objs = []
            tool_results = []
            for tc, result in zip(tool_calls, raw_results):
                if isinstance(result, BaseException):
                    result = f"[Error] 工具 {tc.get('name', '?')} 执行异常: {result}"
                result = str(result)
                tool_call_objs.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                })
                tool_results.append({
                    "tool_call_id": tc["id"],
                    "content": result,
                })
                logger.info("[AgentShell] Tool %s → %s", tc["name"], result[:100])

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": tool_call_objs,
            }
            if response_text:
                assistant_msg["content"] = response_text
            messages.append(assistant_msg)

            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["content"],
                })

            # Sliding window: cap messages to avoid prompt blowup on later iterations.
            # Always keep messages[0] (the original user task) so the LLM never loses
            # the goal, then keep the most recent (_MAX_MESSAGES - 1) entries.
            if len(messages) > _MAX_MESSAGES:
                tail = messages[-(_MAX_MESSAGES - 1):]
                # Drop orphaned tool-result messages at the head of the tail.
                # They refer to tool_call_ids from an assistant message that was
                # trimmed away; sending them to the LLM causes a 400 Bad Request.
                while tail and tail[0].get("role") == "tool":
                    tail = tail[1:]
                messages = [messages[0]] + tail
                logger.debug(
                    "[AgentShell] Messages trimmed to %d (sliding window)", len(messages)
                )

        else:
            logger.warning("[AgentShell] Reached max iterations (%d)", _MAX_ITERATIONS)
            if not final_response:
                final_response = f"任务执行中，已完成 {_MAX_ITERATIONS} 步操作。"

        return final_response

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tool_definitions: list[dict[str, Any]],
        system_prompt: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Call LLM with tools, collect full response and tool calls.

        Retries up to 2 times with exponential backoff so a single relay
        hiccup at iteration 15 does not kill an entire 20-step task.
        CancelledError is never retried — it propagates immediately.

        Returns (response_text, tool_calls_list).
        """
        _MAX_RETRIES = 2
        _RETRY_BASE_DELAY = 0.5  # seconds; doubles on each attempt

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "[AgentShell] LLM retry %d/%d after %.1fs: %s",
                    attempt, _MAX_RETRIES, delay, last_exc,
                )
                await asyncio.sleep(delay)

            response_text = ""
            tool_calls_acc: dict[int, dict[str, str]] = {}
            self._streamed_tts_text = ""  # tracks what was already sent to TTS

            try:
                async for chunk in self._llm.chat_stream(
                    full_messages,
                    tools=tool_definitions,
                    tool_choice="auto",
                    model=self._model,
                ):
                    delta = chunk.choices[0].delta

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc.id:
                                tool_calls_acc[idx]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls_acc[idx]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls_acc[idx]["arguments"] += tc.function.arguments

                    if delta.content:
                        response_text += delta.content
                        # Stream text to TTS in real-time (only final turn,
                        # when no tool calls are being accumulated).
                        # Sentence boundaries trigger immediate TTS speak.
                        if (
                            self._audio is not None
                            and not tool_calls_acc
                            and len(response_text) > 0
                        ):
                            last_char = response_text[-1]
                            if last_char in "。！？\n；，、":
                                # Skip if we're inside a <think> block
                                in_think = response_text.count("<think>") > response_text.count("</think>")
                                if not in_think:
                                    from askme.pipeline.brain_pipeline import strip_think_blocks
                                    clean = strip_think_blocks(response_text).strip()
                                    if clean and clean != self._streamed_tts_text:
                                        new_part = clean[len(self._streamed_tts_text):]
                                        if new_part.strip():
                                            self._audio.speak(new_part.strip())
                                        self._streamed_tts_text = clean

                tool_calls = list(tool_calls_acc.values()) if tool_calls_acc else []

                return response_text, tool_calls

            except asyncio.CancelledError:
                raise  # never retry cancellation
            except Exception as exc:
                last_exc = exc
                logger.error(
                    "[AgentShell] LLM call failed (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES + 1, exc,
                )

        raise last_exc  # type: ignore[misc]

    async def _execute_tool(self, tc: dict[str, str]) -> str:
        """Execute a single tool call, return string result."""
        name = tc.get("name", "")
        args_json = tc.get("arguments", "{}")
        logger.info("[AgentShell] Executing tool: %s(%s)", name, args_json[:80])

        # spawn_agent is handled inline (not registered in ToolRegistry)
        if name == "spawn_agent":
            return await self._spawn_child_agent(args_json)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._tools.execute,
                    name,
                    args_json,
                    allowed_names=_AGENT_ALLOWED_TOOLS,
                    max_safety_level="dangerous",
                ),
                timeout=35.0,  # slightly above SandboxedBashTool's 30s limit
            )
            return str(result)
        except asyncio.TimeoutError:
            return f"[Error] 工具 {name} 执行超时（35s）"
        except Exception as exc:
            return f"[Error] {exc}"

    async def _spawn_child_agent(self, args_json: str) -> str:
        """Spawn a child ThunderAgentShell to handle a focused sub-task."""
        import json

        if self._depth >= _MAX_DEPTH:
            return f"[Error] 已达最大子 Agent 嵌套深度（{_MAX_DEPTH}层），无法再 spawn。"

        try:
            args = json.loads(args_json)
        except Exception:
            return "[Error] spawn_agent: 参数 JSON 解析失败"

        task = args.get("task", "").strip()
        if not task:
            return "[Error] spawn_agent: task 不能为空"

        context_str = args.get("context", "").strip()
        ctx = {"上下文": context_str} if context_str else None

        logger.info("[AgentShell] Spawning child agent (depth=%d): %s", self._depth + 1, task[:60])
        child = ThunderAgentShell(
            llm_client=self._llm,
            tool_registry=self._tools,
            audio=None,  # child agents are silent
            model=self._model,
            workspace=self._workspace,
            _depth=self._depth + 1,
        )
        try:
            result = await child.run_task(task, context=ctx, timeout=30.0)
            return f"[子任务完成]\n{result}"
        except Exception as exc:
            return f"[子任务失败] {exc}"
