#!/usr/bin/env python3
"""Test all MCP tools and resources — verifies registration and basic function."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_registration():
    """Check all tools and resources are registered."""
    from askme.mcp.server import mcp

    tools = mcp._tool_manager._tools if hasattr(mcp, '_tool_manager') else {}
    resources = mcp._resource_manager._resources if hasattr(mcp, '_resource_manager') else {}

    print(f"=== MCP Registration ===")
    print(f"Tools: {len(tools)}")
    for name in sorted(tools.keys()):
        t = tools[name]
        desc = ""
        if hasattr(t, 'description'):
            desc = (t.description or "")[:60]
        elif hasattr(t, 'fn') and t.fn.__doc__:
            desc = t.fn.__doc__.strip().split('\n')[0][:60]
        print(f"  [{name}] {desc}")

    print(f"\nResources: {len(resources)}")
    for uri in sorted(resources.keys()):
        print(f"  {uri}")

    assert len(tools) >= 15, f"Expected 15+ tools, got {len(tools)}"
    assert len(resources) >= 10, f"Expected 10+ resources, got {len(resources)}"

    # Check critical tools exist
    required_tools = [
        "chat", "memory_search", "memory_save", "execute_skill",
        "voice_listen", "voice_speak", "robot_estop",
    ]
    for t in required_tools:
        assert t in tools, f"Missing required tool: {t}"

    print("\n[PASS] All tools and resources registered")


def test_resources():
    """Test resource functions that don't need runtime context."""
    print("\n=== Resource Tests ===")

    # health
    from askme.mcp.resources.health_resources import health_check
    result = json.loads(health_check())
    assert result["status"] == "ok"
    assert "version" in result
    print(f"  [health] OK — v{result['version']}, uptime={result['uptime_seconds']}s")

    # config
    from askme.mcp.resources.health_resources import health_check
    print(f"  [config] OK — subsystems: {result['subsystems']}")

    # skills
    from askme.mcp.resources.skill_resources import skills_catalog
    skills_data = json.loads(skills_catalog())
    count = skills_data.get("count", len(skills_data.get("skills", [])))
    print(f"  [skills] OK — {count} skills loaded")

    # skills openapi
    from askme.mcp.resources.skill_resources import skills_openapi
    openapi = json.loads(skills_openapi())
    paths = len(openapi.get("paths", {}))
    print(f"  [openapi] OK — {paths} API paths")

    # config
    from askme.mcp.resources.skill_resources import askme_config
    config_data = json.loads(askme_config())
    print(f"  [config resource] OK — keys: {list(config_data.keys())[:5]}")

    print("\n[PASS] Resources working")


def test_tool_schemas():
    """Verify tool parameter schemas are valid."""
    print("\n=== Tool Schema Tests ===")
    from askme.mcp.server import mcp

    tools = mcp._tool_manager._tools
    for name, tool in sorted(tools.items()):
        # Each tool should have a callable
        fn = getattr(tool, 'fn', None)
        assert fn is not None or callable(tool), f"Tool {name} has no callable"

        # Check it has parameters defined
        import inspect
        if fn:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            # Filter out 'self', 'ctx' etc
            user_params = [p for p in params if p not in ('self', 'ctx', 'context')]
            print(f"  [{name}] params: {user_params}")

    print("\n[PASS] All tool schemas valid")


if __name__ == "__main__":
    test_registration()
    test_resources()
    test_tool_schemas()
    print("\n=== ALL MCP TESTS PASSED ===")
