"""Full-screen terminal UI for askme with non-blocking refresh."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import sys
import textwrap
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from colorama import just_fix_windows_console

import askme.interfaces.register_defaults  # noqa: F401 — register all backends
from askme.config import get_config
from askme.runtime.module import RuntimeApp
from askme.runtime.profiles import legacy_profile_for

logger = logging.getLogger(__name__)

# ANSI color codes
_C_RESET = "\x1b[0m"
_C_BOLD = "\x1b[1m"
_C_DIM = "\x1b[2m"
_C_CYAN = "\x1b[36m"
_C_GREEN = "\x1b[32m"
_C_YELLOW = "\x1b[33m"
_C_RED = "\x1b[31m"
_C_WHITE = "\x1b[97m"
_C_BG_RED = "\x1b[41m"
_C_BG_GREEN = "\x1b[42m"

_REFRESH_INTERVAL = 0.5  # seconds between screen refreshes


def _display_width(text: str) -> int:
    """Calculate terminal display width accounting for CJK double-width chars."""
    w = 0
    for ch in text:
        eaw = unicodedata.east_asian_width(ch)
        w += 2 if eaw in ("W", "F") else 1
    return w


def _pad_to_width(text: str, width: int) -> str:
    """Pad text with spaces to reach target display width."""
    current = _display_width(text)
    if current >= width:
        return text
    return text + " " * (width - current)


def _truncate_to_width(text: str, width: int) -> str:
    """Truncate text to fit within display width, adding '...' if needed."""
    if _display_width(text) <= width:
        return text
    if width <= 3:
        result = ""
        w = 0
        for ch in text:
            cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if w + cw > width:
                break
            result += ch
            w += cw
        return result
    result = ""
    w = 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > width - 3:
            break
        result += ch
        w += cw
    return result + "..."


@dataclass
class DisplayEntry:
    """Single rendered entry in the terminal transcript."""

    role: str
    content: str
    color: str = ""


@contextlib.contextmanager
def _alternate_screen() -> Any:
    """Switch to an alternate screen buffer for the terminal session."""
    just_fix_windows_console()
    sys.stdout.write("\x1b[?1049h\x1b[2J\x1b[H\x1b[?25l")
    sys.stdout.flush()
    try:
        yield
    finally:
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()


@contextlib.contextmanager
def _suppress_console_logs() -> Any:
    """Temporarily silence stdout/stderr log handlers while the TUI is active."""
    root = logging.getLogger()
    saved: list[tuple[logging.Handler, int]] = []
    for handler in root.handlers:
        stream = getattr(handler, "stream", None)
        if stream in (sys.stdout, sys.stderr):
            saved.append((handler, handler.level))
            handler.setLevel(100)
    try:
        yield
    finally:
        for handler, level in saved:
            handler.setLevel(level)


class SilentAudio:
    """Null-object audio replacement for TUI mode."""

    awaiting_confirmation = False
    is_muted = False

    def speak(self, *a: Any, **kw: Any) -> None: pass
    def start_playback(self, *a: Any, **kw: Any) -> None: pass
    def stop_playback(self, *a: Any, **kw: Any) -> None: pass
    def wait_speaking_done(self, *a: Any, **kw: Any) -> None: pass
    def drain_buffers(self, *a: Any, **kw: Any) -> None: pass
    def stop_immediately(self, *a: Any, **kw: Any) -> None: pass
    def acknowledge(self, *a: Any, **kw: Any) -> None: pass
    def play_thinking(self, *a: Any, **kw: Any) -> None: pass

    async def speak_and_wait(self, *a: Any, **kw: Any) -> None: pass


def _mod_attr(app: RuntimeApp, module_name: str, attr: str, default: Any = None) -> Any:
    """Safely get a module attribute from a RuntimeApp."""
    mod = app.modules.get(module_name)
    return getattr(mod, attr, default) if mod else default


class AskmeTerminalUI:
    """Full-screen chat UI with non-blocking 2Hz refresh.

    Background thread reads stdin -> asyncio.Queue.
    Main loop polls queue with 0.5s timeout -> re-renders on every tick.
    Events, status, and perception data update even while waiting for input.
    """

    def __init__(self, app: RuntimeApp, profile: Any) -> None:
        self.app = app
        self.profile = profile
        self._entries: list[DisplayEntry] = []
        self._known_history_len = 0
        self._pending_user_input = ""
        self._status_text = "就绪"
        self._started_at = time.monotonic()
        self._input_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._processing = False
        self._quit = False
        self._audio_override: Any = None

        # Cache module references
        self._conversation = _mod_attr(app, "memory", "conversation")
        self._text_loop = _mod_attr(app, "text", "text_loop")
        self._pipeline = _mod_attr(app, "pipeline", "brain_pipeline")
        self._skill_manager = _mod_attr(app, "skill", "skill_manager")
        self._dog_safety = _mod_attr(app, "safety", "client")
        self._dispatcher = _mod_attr(app, "skill", "skill_dispatcher")

        self._sync_history()
        self._append_system("欢迎使用 askme。输入 /help 查看命令。")

    @property
    def _audio(self) -> Any:
        """Get the current audio object (override or from voice/text module)."""
        if self._audio_override is not None:
            return self._audio_override
        audio = _mod_attr(self.app, "voice", "audio")
        if audio is not None:
            return audio
        return _mod_attr(self.app, "text", "_text_audio")

    async def run(self) -> None:
        """Run the terminal UI with non-blocking refresh."""
        # Replace audio with silent version
        silent = SilentAudio()
        self._audio_override = silent

        # Patch audio into voice module and pipeline
        voice_mod = self.app.modules.get("voice")
        if voice_mod is not None:
            voice_mod.audio = silent
        if self._pipeline is not None:
            self._pipeline._audio = silent

        # Start background stdin reader thread
        stdin_task = asyncio.create_task(self._stdin_reader())

        with _suppress_console_logs(), _alternate_screen():
            try:
                while not self._quit:
                    self._sync_history()
                    self._render()

                    try:
                        line = await asyncio.wait_for(
                            self._input_queue.get(), timeout=_REFRESH_INTERVAL,
                        )
                    except TimeoutError:
                        continue  # re-render with updated status/events

                    if line is None:
                        # EOF — stdin closed
                        break

                    line = line.strip()
                    if not line:
                        continue

                    should_exit = await self._handle_input(line)
                    if should_exit:
                        break
            finally:
                self._quit = True
                stdin_task.cancel()
                try:
                    await stdin_task
                except (asyncio.CancelledError, Exception):
                    pass

    async def _stdin_reader(self) -> None:
        """Background task: read lines from stdin in a thread, post to queue."""
        loop = asyncio.get_running_loop()

        def _blocking_read() -> str | None:
            try:
                return input()
            except EOFError:
                return None
            except KeyboardInterrupt:
                return "\x03"  # Ctrl+C sentinel

        while not self._quit:
            try:
                line = await loop.run_in_executor(None, _blocking_read)
                await self._input_queue.put(line)
                if line is None:
                    break
            except asyncio.CancelledError:
                break
            except Exception:
                break

    def _handle_estop(self) -> None:
        """Trigger emergency stop via Ctrl+C."""
        try:
            self._pipeline.handle_estop()
            self._append_system(f"{_C_RED}[ESTOP] 紧急停止已触发！{_C_RESET}")
        except Exception:
            self._append_system(f"{_C_RED}[ESTOP] 紧急停止（安全服务未连接）{_C_RESET}")
        self._status_text = "ESTOP"

    async def _handle_input(self, user_text: str) -> bool:
        """Handle slash commands or dispatch a normal turn."""
        # Ctrl+C sentinel
        if user_text == "\x03":
            self._handle_estop()
            return False

        if user_text in {"/quit", "/exit", "quit", "exit"}:
            return True

        if user_text == "/help":
            self._append_system(
                "命令: /help /skills /cap /status /estop /clear /quit\n"
                "快捷键: Ctrl+C = 紧急停止"
            )
            return False

        if user_text == "/estop":
            self._handle_estop()
            return False

        if user_text == "/clear":
            self._conversation.clear()
            self._entries = []
            self._known_history_len = 0
            self._append_system("会话历史已清空。")
            return False

        if user_text == "/skills":
            self._append_system(self._skills_summary())
            return False

        if user_text == "/cap":
            self._append_system(self._capability_summary())
            return False

        if user_text == "/status":
            self._append_system(self._status_summary())
            return False

        # Normal turn — process with periodic refresh
        before_history_len = len(self._conversation.history)
        self._pending_user_input = user_text
        self._status_text = "处理中..."
        self._processing = True
        self._render()

        try:
            # Run LLM turn as a task so we can refresh while it processes
            turn_task = asyncio.create_task(
                self._text_loop.process_turn(user_text)
            )

            # Refresh loop during processing
            while not turn_task.done():
                self._sync_history()
                self._render()

                # Also check for Ctrl+C during processing
                try:
                    line = await asyncio.wait_for(
                        self._input_queue.get(), timeout=_REFRESH_INTERVAL,
                    )
                    if line == "\x03":
                        self._handle_estop()
                    elif line is None:
                        self._quit = True
                        turn_task.cancel()
                        break
                except TimeoutError:
                    pass

            reply = await turn_task
        except asyncio.CancelledError:
            self._append_system(f"{_C_YELLOW}已取消{_C_RESET}")
            reply = ""
        except Exception as exc:
            logger.exception("TUI turn failed")
            self._append_system(f"{_C_RED}处理失败: {exc}{_C_RESET}")
            self._status_text = "错误"
            return False
        finally:
            self._pending_user_input = ""
            self._processing = False

        self._sync_history()
        if reply and len(self._conversation.history) == before_history_len:
            self._entries.append(DisplayEntry("user", user_text))
            self._entries.append(DisplayEntry("assistant", reply))

        self._status_text = "就绪"
        return False

    def render_text(self, *, width: int, height: int) -> str:
        """Return the current screen buffer without the input prompt line."""
        width = max(width, 80)
        height = max(height, 20)

        right_width = min(34, max(24, width // 3))
        left_width = max(40, width - right_width - 3)
        body_height = max(8, height - 6)

        chat_lines = self._build_chat_lines(left_width, body_height)
        status_lines = self._build_status_lines(right_width, body_height)

        header = self._build_header(width)
        context = self._build_context_bar(width)
        separator = f"{_C_DIM}{'─' * width}{_C_RESET}"
        footer = _truncate_to_width(
            " Ctrl+C:ESTOP  /help /skills /status /clear /quit ",
            width,
        )

        lines = [header, context, separator]
        for index in range(body_height):
            left = chat_lines[index] if index < len(chat_lines) else ""
            right = status_lines[index] if index < len(status_lines) else ""
            left_padded = _pad_to_width(left, left_width)
            right_padded = _pad_to_width(right, right_width)
            lines.append(f"{left_padded} {_C_DIM}│{_C_RESET} {right_padded}")
        lines.append(separator)
        lines.append(footer)
        return "\n".join(lines)

    def _build_header(self, width: int) -> str:
        """Build the top header bar with safety/connectivity status."""
        elapsed = int(time.monotonic() - self._started_at)
        now = time.strftime("%H:%M:%S")

        estop_active = False
        try:
            if self._dog_safety is not None:
                estop_active = self._dog_safety.is_estop_active()
        except Exception:
            pass

        if estop_active:
            estop_str = f"{_C_BG_RED}{_C_WHITE} ESTOP:激活 {_C_RESET}"
        else:
            estop_str = f"{_C_GREEN}ESTOP:正常{_C_RESET}"

        if self._status_text == "ESTOP":
            status_colored = f"{_C_BG_RED}{_C_WHITE} {self._status_text} {_C_RESET}"
        elif self._processing:
            status_colored = f"{_C_YELLOW}{self._status_text}{_C_RESET}"
        elif self._status_text == "错误":
            status_colored = f"{_C_RED}{self._status_text}{_C_RESET}"
        else:
            status_colored = f"{_C_GREEN}{self._status_text}{_C_RESET}"

        profile_name = getattr(self.profile, 'name', 'text')

        return (
            f" {_C_BOLD}THUNDER{_C_RESET}"
            f"  {status_colored}"
            f"  {estop_str}"
            f"  {_C_DIM}{profile_name}{_C_RESET}"
            f"  {_C_DIM}{now}  {elapsed}s{_C_RESET}"
        )

    def _build_context_bar(self, width: int) -> str:
        """Build row 2: scene summary + mission state."""
        parts: list[str] = []

        try:
            if self._dispatcher is not None:
                mission = getattr(self._dispatcher, 'current_mission', None)
                if mission:
                    state = getattr(mission, 'state', None)
                    steps = getattr(mission, 'steps', [])
                    state_str = state.value if state else "?"
                    parts.append(f"任务:{state_str} ({len(steps)}步)")
        except Exception:
            pass

        try:
            health = self._health_snapshot()
            lat = health.get('last_llm_latency_ms')
            if lat:
                parts.append(f"LLM:{int(lat)}ms")
        except Exception:
            pass

        if not parts:
            parts.append(f"{_C_DIM}无活动任务{_C_RESET}")

        return f" {_C_DIM}{'  │  '.join(parts)}{_C_RESET}"

    def _render(self) -> None:
        width, height = shutil.get_terminal_size((120, 36))
        # Reserve 1 line for input prompt
        screen = self.render_text(width=width, height=height - 1)
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(screen)
        sys.stdout.write("\n")
        # Show cursor position for input
        if not self._processing:
            sys.stdout.write("\x1b[?25h")  # show cursor
        else:
            sys.stdout.write("\x1b[?25l")  # hide cursor during processing
        sys.stdout.flush()

    def _sync_history(self) -> None:
        """Append new conversation history entries into the local transcript."""
        history = self._conversation.history
        if len(history) < self._known_history_len:
            self._entries = [self._from_history_entry(item) for item in history if item.get("content")]
            self._known_history_len = len(history)
            return

        for item in history[self._known_history_len:]:
            if item.get("content"):
                self._entries.append(self._from_history_entry(item))
        self._known_history_len = len(history)

    def _from_history_entry(self, item: dict[str, Any]) -> DisplayEntry:
        role = str(item.get("role", "assistant"))
        content = str(item.get("content", ""))
        return DisplayEntry(role=role, content=content)

    def _append_system(self, text: str) -> None:
        self._entries.append(DisplayEntry("system", text))

    def _build_chat_lines(self, width: int, height: int) -> list[str]:
        lines: list[str] = []
        for entry in self._entries:
            lines.extend(self._wrap_entry(entry, width))

        if self._pending_user_input:
            lines.extend(self._wrap_entry(DisplayEntry("user", self._pending_user_input), width))
            # Animated thinking indicator
            dots = "." * (1 + int(time.monotonic() * 2) % 3)
            lines.extend(self._wrap_entry(
                DisplayEntry("system", f"{_C_YELLOW}思考中{dots}{_C_RESET}"), width
            ))

        if len(lines) > height:
            lines = lines[-height:]
        if len(lines) < height:
            lines.extend([""] * (height - len(lines)))
        return lines

    def _health_snapshot(self) -> dict[str, Any]:
        """Build a minimal health snapshot from module data."""
        from askme import __version__ as ASKME_VERSION
        from askme.health_server import build_health_snapshot
        from askme.robot.runtime_health import merge_voice_pipeline_status

        cfg = get_config()
        app_name = cfg.get("app", {}).get("name", "askme")
        app_version = cfg.get("app", {}).get("version") or ASKME_VERSION

        llm_mod = self.app.modules.get("llm")
        llm = getattr(llm_mod, "client", None) if llm_mod else None
        ota_metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else None
        ota_snap = ota_metrics.snapshot() if ota_metrics else {}

        audio_obj = self._audio
        voice_status: dict[str, Any] = {}
        if audio_obj is not None and hasattr(audio_obj, "status_snapshot"):
            voice_status = audio_obj.status_snapshot()
        voice_status = merge_voice_pipeline_status(
            voice_status,
            ota_snap.get("voice_pipeline", {}),
        )

        vrb = _mod_attr(self.app, "voice", "voice_runtime_bridge")
        if vrb is None:
            vrb = _mod_attr(self.app, "text", "_voice_runtime_bridge")

        return build_health_snapshot(
            app_name=app_name,
            app_version=app_version,
            model_name=llm.model if llm else "unknown",
            metrics_snapshot=ota_snap,
            active_skills=[s.name for s in self._skill_manager.get_enabled()] if self._skill_manager else [],
            voice_status=voice_status,
            ota_status=None,
            voice_bridge=vrb.status_snapshot() if vrb else None,
        )

    def _capabilities_snapshot(self) -> dict[str, Any]:
        """Build a capabilities snapshot from module data."""
        from askme import __version__ as ASKME_VERSION

        cfg = get_config()
        app_name = cfg.get("app", {}).get("name", "askme")
        app_version = cfg.get("app", {}).get("version") or ASKME_VERSION

        sm = self._skill_manager
        contracts = sm.get_contracts() if sm else []
        openapi_doc = sm.openapi_document() if sm else {"info": {"title": "", "version": ""}, "paths": {}}

        components: dict[str, dict[str, Any]] = {}
        for name, mod in self.app.modules.items():
            components[name] = {
                "health": mod.health(),
                "capabilities": mod.capabilities(),
            }

        return {
            "app": {
                "name": app_name,
                "version": app_version,
                "voice_mode": False,
                "robot_mode": False,
            },
            "profile": self.profile.snapshot(),
            "components": components,
            "skills": {
                "count": len(sm.get_all()) if sm else 0,
                "enabled_count": len(sm.get_enabled()) if sm else 0,
                "contract_count": len(contracts),
                "code_contract_count": sum(
                    1 for c in contracts if c.source == "code"
                ),
                "legacy_contract_count": sum(
                    1 for c in contracts if c.source != "code"
                ),
                "catalog": sm.get_contract_catalog() if sm else [],
            },
            "openapi": {
                "title": openapi_doc["info"]["title"],
                "version": openapi_doc["info"]["version"],
                "path_count": len(openapi_doc["paths"]),
            },
        }

    def _build_status_lines(self, width: int, height: int) -> list[str]:
        capabilities = self._capabilities_snapshot()
        health = self._health_snapshot()
        elapsed = int(time.monotonic() - self._started_at)
        enabled_skills = capabilities["skills"]["enabled_count"]
        total_skills = capabilities["skills"]["count"]

        status_val = health.get('status', 'unknown')
        if status_val == 'ok':
            status_colored = f"{_C_GREEN}ok{_C_RESET}"
        elif status_val == 'degraded':
            status_colored = f"{_C_YELLOW}degraded{_C_RESET}"
        else:
            status_colored = f"{_C_RED}{status_val}{_C_RESET}"

        component_lines = []
        for name, component in capabilities["components"].items():
            s = component["health"].get("status", "unknown")
            if s == "ok":
                component_lines.append(f"  {_C_GREEN}●{_C_RESET} {name}")
            elif s == "degraded":
                component_lines.append(f"  {_C_YELLOW}●{_C_RESET} {name}")
            else:
                component_lines.append(f"  {_C_RED}●{_C_RESET} {name}")

        lat = health.get('last_llm_latency_ms')
        lat_str = f"{int(lat)}ms" if lat else "-"

        lines = [
            f"{_C_BOLD}系统状态{_C_RESET}",
            f"服务: {status_colored}",
            f"延迟: {lat_str}",
            f"技能: {enabled_skills}/{total_skills}",
            f"会话: {health.get('total_conversations', 0)}",
            f"运行: {elapsed}s",
            "",
            f"{_C_BOLD}组件{_C_RESET}",
            *component_lines[: max(0, height - 12)],
            "",
            f"{_C_BOLD}事件{_C_RESET}",
        ]

        event_lines = self._get_recent_events()
        if event_lines:
            lines.extend(event_lines[:max(0, height - len(lines) - 1)])
        else:
            lines.append(f"{_C_DIM}暂无事件{_C_RESET}")

        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(self._wrap_line(line, width))

        if len(wrapped) > height:
            wrapped = wrapped[:height]
        if len(wrapped) < height:
            wrapped.extend([""] * (height - len(wrapped)))
        return wrapped

    def _get_recent_events(self) -> list[str]:
        """Get recent perception events for the status panel."""
        return []

    def _wrap_entry(self, entry: DisplayEntry, width: int) -> list[str]:
        label_colors = {
            "user": (_C_CYAN, "[你]"),
            "assistant": (_C_GREEN, "[askme]"),
            "system": (_C_YELLOW, "[系统]"),
        }
        color, label = label_colors.get(entry.role, (_C_GREEN, "[askme]"))
        colored_label = f"{color}{label}{_C_RESET}"
        label_width = _display_width(label)

        content = entry.content.strip() or "-"
        wrap_width = max(12, width - label_width - 1)
        wrapped = textwrap.wrap(
            content,
            width=wrap_width,
            replace_whitespace=False,
            drop_whitespace=False,
        ) or ["-"]

        lines = [f"{colored_label} {wrapped[0]}"]
        indent = " " * (label_width + 1)
        lines.extend(f"{indent}{line}" for line in wrapped[1:])
        return [_truncate_to_width(line, width + 20) for line in lines]

    def _wrap_line(self, text: str, width: int) -> list[str]:
        if not text:
            return [""]
        if "\x1b[" in text:
            return [text]
        wrapped = textwrap.wrap(
            text,
            width=max(8, width),
            replace_whitespace=False,
            drop_whitespace=False,
        )
        return [_truncate_to_width(line, width) for line in (wrapped or [""])]

    def _skills_summary(self) -> str:
        skills = self._skill_manager.get_enabled() if self._skill_manager else []
        if not skills:
            return "当前没有启用技能。"
        summary = ", ".join(skill.name for skill in skills[:12])
        if len(skills) > 12:
            summary += f" 等 {len(skills)} 个"
        return f"已启用技能: {summary}"

    def _status_summary(self) -> str:
        health = self._health_snapshot()
        return (
            f"状态={health.get('status', 'unknown')} "
            f"会话={health.get('total_conversations', 0)} "
            f"技能={health.get('active_skill_count', 0)} "
            f"模型={health.get('model_name', 'unknown')}"
        )

    def _capability_summary(self) -> str:
        capabilities = self._capabilities_snapshot()
        parts = [
            f"profile={capabilities['profile']['name']}",
        ]
        for name, component in capabilities["components"].items():
            parts.append(f"{name}:{component['health'].get('status', 'unknown')}")
        return "组件能力: " + ", ".join(parts)


async def run_terminal_ui(*, robot_mode: bool = False) -> None:
    """Run askme inside the full-screen terminal UI."""
    from askme.main import _select_blueprint, _setup_logging

    cfg = get_config()
    _setup_logging(cfg)

    blueprint = _select_blueprint(voice_mode=False, robot_mode=robot_mode)
    app = await blueprint.build(cfg)
    profile = legacy_profile_for(voice_mode=False, robot_mode=robot_mode)
    await app.start()
    ui = AskmeTerminalUI(app, profile)
    try:
        await ui.run()
    finally:
        await app.stop()
