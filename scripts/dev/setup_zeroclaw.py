"""Complete ZeroClaw + Askme integration setup.

Configures:
1. MiniMax API key in ZeroClaw
2. Agent persona files (workspace)
3. Askme HTTP API bridge skill (MCP alternative for v0.1.7)

Usage: python scripts/dev/setup_zeroclaw.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZEROCLAW_HOME = Path.home() / ".zeroclaw"
ZEROCLAW_WORKSPACE = ZEROCLAW_HOME / "workspace"
ZEROCLAW_RUNTIME_VERSION = "0.1.7"
ZEROCLAW_RUNTIME_PROFILE = "standard"
ZEROCLAW_EDGE_PROFILE = "nano"

sys.path.insert(0, str(PROJECT_ROOT))
from askme.config import get_config

# ── Step 1: API Key ──────────────────────────────────────────────
cfg = get_config()
brain = cfg["brain"] if isinstance(cfg, dict) else cfg.brain
api_key = brain.get("minimax_api_key", "") if isinstance(brain, dict) else getattr(brain, "minimax_api_key", "")

if not api_key or len(api_key) < 10:
    print("[FAIL] Cannot read MiniMax API key from Askme config. Check .env")
    sys.exit(1)

print("[1/4] Configuring MiniMax API key in ZeroClaw...")
result = subprocess.run(
    ["zeroclaw", "onboard", "--provider", "minimax-cn",
     "--api-key", api_key, "--force"],
    capture_output=True, text=True, timeout=30,
)
if result.returncode != 0:
    print(f"[FAIL] zeroclaw onboard failed: {result.stderr}")
    sys.exit(1)
print("[OK]   MiniMax API key configured")

# ── Step 2: Agent Persona ────────────────────────────────────────
print("[2/4] Copying agent persona files to ZeroClaw workspace...")
PERSONA_FILES = ["IDENTITY.md", "SOUL.md", "AGENTS.md", "TOOLS.md", "MEMORY.md", "HEARTBEAT.md"]
agent_dir = PROJECT_ROOT / "agent"
ZEROCLAW_WORKSPACE.mkdir(parents=True, exist_ok=True)

for fname in PERSONA_FILES:
    src = agent_dir / fname
    if src.exists():
        shutil.copy2(src, ZEROCLAW_WORKSPACE / fname)

# Also copy to workspace root for ZeroClaw auto-loading
print("[OK]   Persona files installed")

# ── Step 3: Askme HTTP Bridge Skill ──────────────────────────────
print("[3/4] Creating Askme API bridge skill for ZeroClaw...")
skill_dir = ZEROCLAW_WORKSPACE / "skills" / "askme-bridge"
skill_dir.mkdir(parents=True, exist_ok=True)

bridge_skill = """# Askme Bridge Skill
description = "Connect ZeroClaw to Askme voice/memory/robot API"

[[tools]]
name = "askme_chat"
description = "Chat with Askme's LLM (voice/text)"
kind = "shell"
command = "curl -s -X POST http://localhost:8765/api/v1/chat -H 'Content-Type: application/json' -d '{\"message\": \"{{message}}\"}'"

[[tools]]
name = "askme_robot_state"
description = "Get robot arm current state"
kind = "shell"
command = "curl -s http://localhost:8765/api/v1/robot/state"

[[tools]]
name = "askme_memory_search"
description = "Search Askme's knowledge base"
kind = "shell"
command = "curl -s 'http://localhost:8765/api/v1/memory/search?q={{query}}'"
"""
(skill_dir / "SKILL.toml").write_text(bridge_skill, encoding="utf-8")
print("[OK]   Askme bridge skill created at ~/.zeroclaw/workspace/skills/askme-bridge/")

# ── Step 4: Verification ─────────────────────────────────────────
print("[4/4] Verifying ZeroClaw installation...")
result = subprocess.run(["zeroclaw", "--version"], capture_output=True, text=True, timeout=10)
version_output = result.stdout.strip()
if result.returncode != 0:
    print(f"[FAIL] zeroclaw --version failed: {result.stderr}")
    sys.exit(1)
if ZEROCLAW_RUNTIME_VERSION not in version_output:
    print(f"[WARN] ZeroClaw version mismatch: expected {ZEROCLAW_RUNTIME_VERSION}, got {version_output}")
else:
    print(f"[OK]   ZeroClaw {version_output} ready")
print(f"[OK]   Runtime profile: {ZEROCLAW_RUNTIME_PROFILE}; edge profile reserved as {ZEROCLAW_EDGE_PROFILE}")

print()
print("=" * 60)
print("  ZeroClaw + Askme 集成完成！")
print("=" * 60)
print()
print("  启动 ZeroClaw（独立对话）:")
print("    zeroclaw agent")
print()
print("  启动 Askme HTTP API（配合 bridge skill）:")
print("    python -m askme.blueprints.presets.edge_robot")
print()
print("  启动 Askme MCP（给 Claude Code 用）:")
print("    python -m askme.mcp.server")
print()
print("  一键验证:")
print("    python scripts/dev/run_full_verification.py")
