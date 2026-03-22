"""Full-screen terminal UI for askme."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any

from colorama import just_fix_windows_console

from askme.app import AskmeApp

logger = logging.getLogger(__name__)


@dataclass
class DisplayEntry:
    """Single rendered entry in the terminal transcript."""

    role: str
    content: str


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


def _install_silent_audio(audio: Any) -> None:
    """Disable audible playback for the terminal UI while keeping shutdown intact."""
    for name in (
        "speak",
        "start_playback",
        "stop_playback",
        "wait_speaking_done",
        "drain_buffers",
        "stop_immediately",
    ):
        setattr(audio, name, lambda *args, **kwargs: None)

    async def _no_speak_and_wait(*args: Any, **kwargs: Any) -> None:
        return None

    setattr(audio, "speak_and_wait", _no_speak_and_wait)


class AskmeTerminalUI:
    """Simple full-screen chat UI built on top of the text runtime."""

    def __init__(self, app: AskmeApp) -> None:
        self.app = app
        self._entries: list[DisplayEntry] = []
        self._known_history_len = 0
        self._pending_user_input = ""
        self._status_text = "就绪"
        self._started_at = time.monotonic()
        self._sync_history()
        self._append_system("欢迎使用 askme。直接输入需求，输入 /help 查看命令。")

    async def run(self) -> None:
        """Run the terminal UI until the user quits."""
        _install_silent_audio(self.app.audio)
        with _suppress_console_logs(), _alternate_screen():
            while True:
                self._sync_history()
                self._render()
                try:
                    user_text = await asyncio.to_thread(input, "askme> ")
                except (EOFError, KeyboardInterrupt):
                    break

                user_text = user_text.strip()
                if not user_text:
                    continue

                should_exit = await self._handle_input(user_text)
                if should_exit:
                    break

    async def _handle_input(self, user_text: str) -> bool:
        """Handle slash commands or dispatch a normal turn."""
        if user_text in {"/quit", "/exit", "quit", "exit"}:
            return True

        if user_text == "/help":
            self._append_system(
                "命令: /help /skills /cap /status /clear /quit"
            )
            return False

        if user_text == "/clear":
            self.app.conversation.clear()
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

        before_history_len = len(self.app.conversation.history)
        self._pending_user_input = user_text
        self._status_text = "处理中..."
        self._render()

        try:
            reply = await self.app._text_loop.process_turn(user_text)  # noqa: SLF001
        except Exception as exc:
            logger.exception("TUI turn failed")
            self._append_system(f"处理失败: {exc}")
            self._status_text = "错误"
            return False
        finally:
            self._pending_user_input = ""

        self._sync_history()
        if reply and len(self.app.conversation.history) == before_history_len:
            self._entries.append(DisplayEntry("user", user_text))
            self._entries.append(DisplayEntry("assistant", reply))

        self._status_text = "就绪"
        return False

    def render_text(self, *, width: int, height: int) -> str:
        """Return the current screen buffer without the input prompt line."""
        width = max(width, 80)
        height = max(height, 20)

        right_width = min(38, max(28, width // 3))
        left_width = max(40, width - right_width - 3)
        body_height = max(8, height - 5)

        chat_lines = self._build_chat_lines(left_width, body_height)
        status_lines = self._build_status_lines(right_width, body_height)

        header = self._truncate(
            f" askme | profile={self.app.profile.name} | {self._status_text} ",
            width,
        )
        separator = "=" * width
        footer = self._truncate(
            " 回车发送  /help /skills /cap /status /clear /quit ",
            width,
        )

        lines = [header, separator]
        for index in range(body_height):
            left = chat_lines[index] if index < len(chat_lines) else ""
            right = status_lines[index] if index < len(status_lines) else ""
            lines.append(f"{left.ljust(left_width)} | {right.ljust(right_width)}")
        lines.append(separator)
        lines.append(footer)
        return "\n".join(lines)

    def _render(self) -> None:
        width, height = shutil.get_terminal_size((120, 36))
        screen = self.render_text(width=width, height=height)
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(screen)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _sync_history(self) -> None:
        """Append new conversation history entries into the local transcript."""
        history = self.app.conversation.history
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
            lines.extend(self._wrap_entry(DisplayEntry("system", "askme 正在思考..."), width))

        if len(lines) > height:
            lines = lines[-height:]
        if len(lines) < height:
            lines.extend([""] * (height - len(lines)))
        return lines

    def _build_status_lines(self, width: int, height: int) -> list[str]:
        capabilities = self.app.capabilities_snapshot()
        health = self.app.health_snapshot()
        elapsed = int(time.monotonic() - self._started_at)
        enabled_skills = capabilities["skills"]["enabled_count"]
        total_skills = capabilities["skills"]["count"]

        component_lines = []
        for name, component in capabilities["components"].items():
            status = component["health"].get("status", "unknown")
            component_lines.append(f"{name}: {status}")

        recent_skills = [
            entry["name"]
            for entry in capabilities["skills"]["catalog"]
            if entry.get("enabled", False)
        ][:6]

        lines = [
            "状态",
            f"服务: {health.get('status', 'unknown')}",
            f"模式: {'voice' if capabilities['app']['voice_mode'] else 'text'}",
            f"机器人: {'on' if capabilities['app']['robot_mode'] else 'off'}",
            f"会话: {health.get('total_conversations', 0)}",
            f"延迟: {health.get('last_llm_latency_ms') or '-'} ms",
            f"技能: {enabled_skills}/{total_skills}",
            f"运行: {elapsed}s",
            "",
            "组件",
            *component_lines[: max(0, height - 18)],
            "",
            "已启用技能",
            ", ".join(recent_skills) if recent_skills else "-",
            "",
            "提示",
            "右侧是状态，左侧是对话。",
        ]

        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(self._wrap_line(line, width))

        if len(wrapped) > height:
            wrapped = wrapped[:height]
        if len(wrapped) < height:
            wrapped.extend([""] * (height - len(wrapped)))
        return wrapped

    def _wrap_entry(self, entry: DisplayEntry, width: int) -> list[str]:
        label = {
            "user": "[你]",
            "assistant": "[askme]",
            "system": "[系统]",
        }.get(entry.role, "[askme]")

        content = entry.content.strip() or "-"
        wrapped = textwrap.wrap(
            content,
            width=max(12, width - len(label) - 1),
            replace_whitespace=False,
            drop_whitespace=False,
        ) or ["-"]

        lines = [f"{label} {wrapped[0]}"]
        indent = " " * (len(label) + 1)
        lines.extend(f"{indent}{line}" for line in wrapped[1:])
        return [self._truncate(line, width) for line in lines]

    def _wrap_line(self, text: str, width: int) -> list[str]:
        if not text:
            return [""]
        wrapped = textwrap.wrap(
            text,
            width=max(8, width),
            replace_whitespace=False,
            drop_whitespace=False,
        )
        return [self._truncate(line, width) for line in (wrapped or [""])]

    def _truncate(self, text: str, width: int) -> str:
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return text[: width - 3] + "..."

    def _skills_summary(self) -> str:
        skills = self.app.skill_manager.get_enabled()
        if not skills:
            return "当前没有启用技能。"
        summary = ", ".join(skill.name for skill in skills[:12])
        if len(skills) > 12:
            summary += f" 等 {len(skills)} 个"
        return f"已启用技能: {summary}"

    def _status_summary(self) -> str:
        health = self.app.health_snapshot()
        return (
            f"状态={health.get('status', 'unknown')} "
            f"会话={health.get('total_conversations', 0)} "
            f"技能={health.get('active_skill_count', 0)} "
            f"模型={health.get('model_name', 'unknown')}"
        )

    def _capability_summary(self) -> str:
        capabilities = self.app.capabilities_snapshot()
        parts = [
            f"profile={capabilities['profile']['name']}",
        ]
        for name, component in capabilities["components"].items():
            parts.append(f"{name}:{component['health'].get('status', 'unknown')}")
        return "组件能力: " + ", ".join(parts)


async def run_terminal_ui(*, robot_mode: bool = False) -> None:
    """Run askme inside the full-screen terminal UI."""
    app = AskmeApp(voice_mode=False, robot_mode=robot_mode)
    ui = AskmeTerminalUI(app)
    try:
        await ui.run()
    finally:
        await app.shutdown()
