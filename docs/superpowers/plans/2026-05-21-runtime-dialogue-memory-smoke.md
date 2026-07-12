# Runtime Dialogue Memory Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify and harden AskMe's real communication flow from text input through memory retrieval, dialogue orchestration, and response output without using pytest as acceptance evidence.

**Architecture:** Add or improve a runtime-level smoke path that uses existing CLI/runtime components instead of test doubles. The smoke must report memory retrieval evidence, chat response evidence, health/diagnostics evidence, and clear failure reasons.

**Tech Stack:** Python CLI, AskMe runtime modules, `MemoryBridge`, `TextLoop`, `BrainPipeline`, health/conversation APIs.

---

### Task 1: Map The Live Flow

**Files:**
- Read: `askme/runtime/modules/text_module.py`
- Read: `askme/pipeline/channels/text_loop.py`
- Read: `askme/pipeline/core/brain_pipeline.py`
- Read: `askme/memory/retrieval/bridge.py`
- Read: `askme/api/services/conversation_service.py`
- Read: `askme/cli.py`

- [x] Identify the text chat runtime entrypoint.
- [x] Identify where memory retrieval is invoked.
- [x] Identify existing CLI smoke commands and whether they exercise memory retrieval.

### Task 2: Implement A Real Runtime Smoke

**Files:**
- Modify: `askme/cli.py`
- Create or modify: runtime diagnostics helper under `askme/voice/diagnostics` or another existing diagnostics package if a better home exists.

- [x] Add a CLI command that runs a real conversation turn through runtime components.
- [x] Include memory import or retrieval setup using an isolated artifact directory or explicit user-provided data.
- [x] Return JSON evidence for memory retrieval, chat response, latency, and health.
- [x] Avoid pytest dependency for acceptance.

### Task 3: Run Real Verification

**Commands:**
- `python -m py_compile askme\pipeline\core\prompt_builder.py askme\pipeline\core\turn_executor.py askme\pipeline\core\tool_executor.py askme\llm\providers\fake.py askme\runtime\diagnostics\dialogue_smoke.py askme\cli.py`
- `ruff check askme\pipeline\core\prompt_builder.py askme\pipeline\core\turn_executor.py askme\pipeline\core\tool_executor.py askme\llm\providers\fake.py askme\runtime\diagnostics\dialogue_smoke.py askme\cli.py --select F`
- `python -m askme.cli runtime dialogue-smoke --token ASKME-LIVE-REAL-003 --output-dir artifacts\runtime-dialogue-smoke\real-llm3 --json`
- `python -m askme.cli runtime dialogue-smoke --fake-llm --token ASKME-LIVE-LOCAL-004 --output-dir artifacts\runtime-dialogue-smoke\local-fake4 --json`
- `python -m askme.cli runtime capabilities --profile text --json`

- [x] Confirm commands return successful machine-readable output.
- [x] Confirm memory evidence appears in the dialogue path.
- [x] Confirm failure output is actionable if an external service or local memory backend is unavailable.

### Task 4: Report Evidence

- [x] Summarize changed files.
- [x] Include exact real commands run.
- [x] Include remaining risks, especially any external-service or local-device dependency.
