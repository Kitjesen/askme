from __future__ import annotations

from askme.agent_shell.agent_hooks import AgentHookRunner


def test_pre_tool_hook_blocks_matching_tool() -> None:
    runner = AgentHookRunner(
        {
            "PreToolUse": [
                {
                    "matcher": "robot_api",
                    "decision": "deny",
                    "reason": "robot writes require runtime handoff",
                }
            ]
        }
    )

    decision = runner.before_tool(tool_name="robot_api", arguments='{"path":"/control"}')

    assert decision.blocked is True
    assert decision.ok is False
    assert "runtime handoff" in decision.reason
    assert "robot_api" in decision.error_text()


def test_pre_tool_hook_respects_argument_contains_guard() -> None:
    runner = AgentHookRunner(
        {
            "PreToolUse": [
                {
                    "matcher": "bash",
                    "contains": "rm -rf",
                    "decision": "block",
                    "reason": "destructive shell command",
                }
            ]
        }
    )

    allowed = runner.before_tool(tool_name="bash", arguments='{"command":"ls"}')
    blocked = runner.before_tool(tool_name="bash", arguments='{"command":"rm -rf data"}')

    assert allowed.blocked is False
    assert blocked.blocked is True
    assert "destructive" in blocked.reason


def test_nested_claude_style_hook_blocks_without_running_shell() -> None:
    runner = AgentHookRunner(
        {
            "PreToolUse": [
                {
                    "matcher": "bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "./dangerous-hook.sh",
                        }
                    ],
                }
            ]
        }
    )

    decision = runner.before_tool(tool_name="bash", arguments='{"command":"echo ok"}')

    assert decision.blocked is True
    assert "unsupported executable hook type" in decision.reason
    assert "command" not in decision.matched_hooks[0]


def test_post_tool_hook_can_block_sensitive_result() -> None:
    runner = AgentHookRunner(
        {
            "PostToolUse": [
                {
                    "matcher": "read_file",
                    "contains": "secret",
                    "decision": "deny",
                    "reason": "secret content cannot be returned",
                }
            ]
        }
    )

    decision = runner.after_tool(
        tool_name="read_file",
        arguments='{"path":"secret.txt"}',
        result="secret=123",
    )

    assert decision.blocked is True
    assert decision.event == "PostToolUse"
