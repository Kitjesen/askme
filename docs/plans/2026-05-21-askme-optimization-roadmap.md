# Askme Optimization Roadmap Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Continue optimizing Askme with low-risk, measurable changes that preserve current public payloads and protect existing uncommitted work.

**Architecture:** Execute one bounded lane at a time: first remove a measured chat hot-path regression, then cache field event archive reads, then refactor TTS playback polling. Each lane starts with a failing regression/performance test, keeps response structures unchanged, and verifies with the lane tests listed in `docs/MODULE_OWNERSHIP.md`.

**Tech Stack:** Python 3.11, FastAPI/httpx, pytest, ruff, JSONL event archives, NumPy/audio playback, local benchmark scripts under `scripts/`.

---

## Current Constraints

- Worktree already contains uncommitted changes. Do not overwrite or revert:
  - `askme/voice/diagnostics/audio_devices.py`
  - `tests/test_audio_devices.py`
- Preserve the current in-progress optimization files unless intentionally continuing that lane:
  - `askme/memory/retrieval/vector_store.py`
  - `tests/test_vector_store.py`
  - `.python-version`
- Use Python 3.11-compatible verification. On this machine, run tests with:
  - `env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest ...`
- Keep public response schemas and payload keys stable.
- Do not add generated benchmark JSON files to git; `artifacts/` is generated output.

## Evidence From Scan

### Static hotspots

- Largest product files:
  - `askme/cli.py` — 5189 lines
  - `askme/pipeline/field/field_operations.py` — 4452 lines
  - `askme/runtime/task/handoff.py` — 3094 lines
  - `askme/pipeline/field/customer_project_acceptance.py` — 3092 lines
  - `askme/voice/output/tts.py` — 2230 lines
- Large functions that create maintainability risk:
  - `askme/health_server.py:create_health_app` — 594 lines
  - `askme/pipeline/channels/voice_loop.py:run` — 462 lines
  - `askme/voice/output/tts.py:_playback_loop` — 236 lines
- I/O/full-scan signals:
  - `askme/pipeline/field/field_operations.py` has repeated `_read_events()` callers and 24 full-scan/I/O matches.
  - `_read_events()` reparses the full JSONL archive on every list/detail/device-status call.
- Polling signals:
  - `askme/voice/output/tts.py` has 23 sleep/poll matches, including 20ms polling loops in playback paths.

### Runtime measurements

- `scripts/benchmark_core_paths.py --quick --iterations 40 --concurrency 8` exposed one failure:
  - `api_chat.p95_ms = 713.188`, threshold is `250.0`.
- Direct `/api/chat` probes showed normal chat requests spending most local time in `space_context_ms`:
  - example last turn: `space_context_ms = 43.674`, `total_ms = 47.017`.
- Microbench of scenario capability fallback:
  - `default_product_skill_names()` p50 approximately `39ms`, max approximately `111ms`.
  - `requested_or_runtime_skills({}, None)` p50 approximately `39ms`.
  - `classify_scenario_intent('probe', available_skills=requested_or_runtime_skills(...))` p50 approximately `41ms`.
- Memory retrieval benchmark is currently strong after cache/top-k work:
  - `memory_retrieve.p95_ms = 0.024` in the quick benchmark.

## Priority Order

1. P0 — Chat hot path: cache/default-skill fallback and skip unnecessary scenario preview work.
2. P1 — Field event archive cache/index: avoid repeated full JSONL parses for dashboard/API reads.
3. P2 — TTS playback loop: split helpers first, then replace idle polling with notification-based wakeups.
4. P3 — Maintainability cleanup: break oversized route/CLI functions only after P0-P2 have tests and metrics.

---

## P0: Chat Hot Path Optimization

**Objective:** Make ordinary `/api/chat` requests avoid rebuilding the default product capability catalog on every turn while preserving scenario preview behavior for relevant utterances.

**Write scope:**
- Modify: `askme/api/services/scenario_intent_payloads.py`
- Modify: `askme/api/services/conversation_service.py`
- Optional modify: `askme/api/services/space_preview.py`
- Test: `tests/test_conversation_service.py`
- Test: `tests/test_conversation_http.py`
- Optional benchmark script update: `scripts/benchmark_core_paths.py`

### Task 0.1: Add regression test for cached default product skill names

**Objective:** Prove repeated `requested_or_runtime_skills(..., provider=None)` does not rebuild the default capability center.

**Files:**
- Modify: `tests/test_conversation_service.py`

**Step 1: Write failing test**

Add a test near the other conversation-service unit tests:

```python
def test_requested_or_runtime_skills_caches_default_product_skills(monkeypatch):
    from askme.api.services import scenario_intent_payloads as payloads

    calls = 0

    def fake_center():
        nonlocal calls
        calls += 1
        return {
            "groups": [
                {
                    "group_id": "test",
                    "skills": [
                        {"skill_name": "lookup_place", "enabled": True, "installed": True},
                        {"skill_name": "detect_fire_smoke", "enabled": True, "installed": True},
                    ],
                }
            ]
        }

    if hasattr(payloads.default_product_skill_names, "cache_clear"):
        payloads.default_product_skill_names.cache_clear()
    monkeypatch.setattr(payloads, "default_product_capability_center", fake_center)

    assert "lookup_place" in payloads.requested_or_runtime_skills({}, None)
    assert "lookup_place" in payloads.requested_or_runtime_skills({}, None)
    assert calls == 1
```

**Step 2: Run test to verify failure**

Run:

```bash
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest tests/test_conversation_service.py::test_requested_or_runtime_skills_caches_default_product_skills -q --tb=short
```

Expected before implementation: FAIL because `calls == 2` or because `cache_clear` is missing.

### Task 0.2: Cache default product skill names

**Objective:** Add a small process-local cache for static default product skill names.

**Files:**
- Modify: `askme/api/services/scenario_intent_payloads.py`

**Step 1: Implement minimal cache**

Add the import and decorator:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def default_product_skill_names() -> frozenset[str]:
    """Return enabled skills from the local product catalog for dashboard-only mode."""

    center = default_product_capability_center()
    if not center:
        return frozenset()
    return frozenset(enabled_skill_names({"skills": {"capability_center": center}}))
```

Notes:
- Returning `frozenset[str]` avoids accidental mutation of cached state.
- `requested_or_runtime_skills()` can continue returning a set by wrapping if needed:

```python
return runtime_skills or set(default_product_skill_names())
```

**Step 2: Run focused test**

```bash
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest tests/test_conversation_service.py::test_requested_or_runtime_skills_caches_default_product_skills -q --tb=short
```

Expected: PASS.

### Task 0.3: Add a non-scenario chat fast-path test

**Objective:** Prove an ordinary chat like `hello` does not do scenario capability work when it cannot produce scenario/space preview evidence.

**Files:**
- Modify: `tests/test_conversation_service.py`

**Step 1: Write failing test**

Add a test that monkeypatches the expensive capability fallback to raise. This forces the implementation to short-circuit before the fallback for non-scenario text.

```python
async def test_plain_chat_skips_scenario_preview_capability_fallback(monkeypatch):
    import askme.api.services.conversation_service as service_mod

    async def chat_handler(text: str, *, speak: bool = False):
        return {"reply": f"reply:{text}", "spoken": speak}

    def fail_requested_skills(*args, **kwargs):
        raise AssertionError("plain chat should not load scenario skills")

    monkeypatch.setattr(service_mod, "requested_or_runtime_skills", fail_requested_skills)
    service = ConversationService(
        chat_handler=chat_handler,
        space_dispatch=lambda method, body: {"should_not": "run"},
    )

    payload = await service.chat_payload_from_body({"text": "hello"})

    assert payload["reply"] == "reply:hello"
    assert "scenario_preview" not in payload
    assert "space_resolution" not in payload
```

**Step 2: Run test to verify failure**

```bash
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest tests/test_conversation_service.py::test_plain_chat_skips_scenario_preview_capability_fallback -q --tb=short
```

Expected before implementation: FAIL because `attach_space_chat_context()` always calls `requested_or_runtime_skills()` when `space_dispatch` is configured.

### Task 0.4: Implement cheap prefilter before capability fallback

**Objective:** Avoid expensive default-skill derivation for text that cannot match any scenario or space preview rule.

**Files:**
- Modify: `askme/robot_interaction/scenario_intents.py`
- Modify: `askme/api/services/conversation_service.py`
- Optional modify: `askme/api/services/space_preview.py`

**Step 1: Add helper**

In `askme/robot_interaction/scenario_intents.py`, add a helper near `classify_scenario_intent()`:

```python
def could_match_scenario_intent(text: str) -> bool:
    """Cheap prefilter for whether text contains any scenario-intent term."""

    normalized = normalize_intent_text(text)
    if not normalized:
        return False
    for rule in SCENARIO_INTENT_RULES:
        for term in (*rule.any_terms, *rule.all_terms):
            if _has_term(normalized, term):
                return True
    return False
```

**Step 2: Short-circuit in chat context attachment**

In `ConversationService.attach_space_chat_context()`, import `could_match_scenario_intent` from `askme.robot_interaction.scenario_intents` and `should_resolve_space_preview` from `askme.api.services.space_preview`, then return before `requested_or_runtime_skills(...)` when:

- no caller supplied `available_skills`, and
- there is no capabilities provider, and
- text cannot match a scenario intent, and
- text does not contain a space-preview query term.

If using existing helpers, keep this function readable. One possible shape:

```python
if (
    self._capabilities_provider is None
    and not isinstance(body.get("available_skills"), list)
    and not could_match_scenario_intent(text)
    and not should_resolve_space_preview(text, None)
):
    return payload
```

**Step 3: Run focused tests**

```bash
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest tests/test_conversation_service.py tests/test_conversation_http.py -q --tb=short
```

Expected: PASS.

### Task 0.5: Re-run benchmark and adjust only if evidence supports it

**Objective:** Verify the P0 change moves `/api/chat` under the configured threshold.

Run:

```bash
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python scripts/benchmark_core_paths.py --quick --iterations 40 --concurrency 8 --output artifacts/perf/core_paths_latest.json
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python scripts/check_perf_thresholds.py --report artifacts/perf/core_paths_latest.json
```

Expected after P0: `api_chat.p95_ms <= 250.0` and threshold check PASS.

If still slow:
- inspect `/api/conversation/diagnostics` timings;
- do not blindly raise thresholds;
- add the next small failing test around the timing source.

---

## P1: Field Event Archive Cache and Index

**Objective:** Avoid reparsing the full field event JSONL archive for every dashboard/API read while preserving write consistency and response payloads.

**Write scope:**
- Modify: `askme/pipeline/field/field_operations.py`
- Test: `tests/test_field_operations.py`
- Optional benchmark script: `scripts/benchmark_core_paths.py` or a new `scripts/benchmark_field_events.py`

### Task 1.1: Add a failing test for repeated read cache

**Objective:** Prove repeated read-only calls reuse a cached event snapshot when the archive file did not change.

**Files:**
- Modify: `tests/test_field_operations.py`

**Step 1: Write test**

Use the existing `_service(tmp_path)` helper and monkeypatch `Path.open` or a small wrapper to count archive reads. The assertions should verify:

- first `service.list_payload()` reads the JSONL file;
- second `service.list_payload()` does not reopen the archive for reading;
- `service.detail_payload(event_id)` can use the same cached snapshot.

Pseudo-shape:

```python
@pytest.mark.asyncio
async def test_field_event_reads_reuse_archive_cache_until_file_changes(tmp_path, monkeypatch):
    service = _service(tmp_path)
    first = await service.trigger_payload({"scenario_id": "fire_or_smoke", "location": "A", "smoke_level": "high"})
    event_id = first["event"]["event_id"]

    read_opens = 0
    original_open = Path.open

    def counting_open(self, mode="r", *args, **kwargs):
        nonlocal read_opens
        if self == service._archive_path and "r" in mode:
            read_opens += 1
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)

    assert service.list_payload()["total"] == 1
    assert service.list_payload()["total"] == 1
    assert service.detail_payload(event_id)["found"] is True
    assert read_opens == 1
```

**Step 2: Run test to verify failure**

```bash
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest tests/test_field_operations.py::test_field_event_reads_reuse_archive_cache_until_file_changes -q --tb=short
```

Expected before implementation: FAIL because `_read_events()` opens the archive every call.

### Task 1.2: Implement fingerprinted event cache

**Objective:** Cache parsed events by archive `(exists, size, mtime_ns)` fingerprint.

**Files:**
- Modify: `askme/pipeline/field/field_operations.py`

**Implementation outline:**

Add instance fields in `FieldOperationsService.__init__`:

```python
self._events_cache_fingerprint: tuple[bool, int, int] | None = None
self._events_cache: list[dict[str, Any]] | None = None
```

Add helper:

```python
def _archive_fingerprint(self) -> tuple[bool, int, int]:
    try:
        stat = self._archive_path.stat()
    except FileNotFoundError:
        return (False, 0, 0)
    return (True, int(stat.st_size), int(stat.st_mtime_ns))
```

Change `_read_events()`:

```python
fingerprint = self._archive_fingerprint()
if self._events_cache_fingerprint == fingerprint and self._events_cache is not None:
    return [dict(event) for event in self._events_cache]
# existing parse code...
self._events_cache_fingerprint = fingerprint
self._events_cache = [dict(event) for event in events]
return events
```

Keep shallow copies at minimum so callers do not mutate the top-level cached dict by accident. If tests reveal nested mutation risks, use `copy.deepcopy()` only at the cache boundary and measure.

Invalidate cache after writes:

```python
def _invalidate_events_cache(self) -> None:
    self._events_cache_fingerprint = None
    self._events_cache = None
```

Call `_invalidate_events_cache()` after `_append_event()` and `_write_events()` complete.

**Step 2: Run focused field test**

```bash
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest tests/test_field_operations.py::test_field_event_reads_reuse_archive_cache_until_file_changes -q --tb=short
```

Expected: PASS.

### Task 1.3: Add cache invalidation test

**Objective:** Prove writes update subsequent reads.

**Files:**
- Modify: `tests/test_field_operations.py`

Add a test:

```python
@pytest.mark.asyncio
async def test_field_event_cache_invalidates_after_append(tmp_path):
    service = _service(tmp_path)
    await service.trigger_payload({"scenario_id": "fire_or_smoke", "location": "A", "smoke_level": "high"})
    assert service.list_payload()["total"] == 1

    await service.trigger_payload({"scenario_id": "illegal_parking", "location": "B", "plate_number": "沪A12345"})

    listed = service.list_payload()
    assert listed["total"] == 2
    assert listed["filtered_total"] == 2
```

Run:

```bash
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest tests/test_field_operations.py::test_field_event_cache_invalidates_after_append -q --tb=short
```

Expected: PASS.

### Task 1.4: Run Field Delivery Domain lane verification

Run:

```bash
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest tests/test_field_operations.py tests/test_field_ingest_adapters.py tests/test_field_contracts.py tests/test_dashboard_customer_project_contract.py -q --tb=short
ruff check askme/pipeline/field/field_operations.py tests/test_field_operations.py
```

Expected: all tests and ruff PASS.

---

## P2: TTS Playback Loop Split and Polling Reduction

**Objective:** Reduce maintenance risk and idle polling in `TTSEngine._playback_loop()` without changing public TTS behavior.

**Write scope:**
- Modify: `askme/voice/output/tts.py`
- Test: `tests/test_tts.py`
- Do not modify `askme/voice/diagnostics/audio_devices.py` or `tests/test_audio_devices.py` in this lane.

### Task 2.1: Split playback transport helpers with no behavior change

**Objective:** Make `_playback_loop()` smaller before changing synchronization behavior.

**Files:**
- Modify: `askme/voice/output/tts.py`
- Test: `tests/test_tts.py`

Extract helpers such as:

```python
def _pop_next_playback_chunk(self) -> np.ndarray | None: ...
def _apply_playback_volume(self, chunk: np.ndarray) -> np.ndarray: ...
def _play_chunk_via_aplay(...): ...
def _play_chunk_via_sounddevice(self, chunk: np.ndarray) -> None: ...
def _handle_playback_stop_requested(self) -> bool: ...
```

Run existing playback tests after each small extraction:

```bash
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest tests/test_tts.py::test_playback_loop_uses_configured_output_device tests/test_tts.py::test_playback_loop_uses_usb_direct_transport tests/test_tts.py::test_wait_done_waits_for_usb_direct_chunk_after_buffer_pop -q --tb=short
```

Expected: PASS after every extraction.

### Task 2.2: Add buffer wakeup primitive

**Objective:** Replace idle `time.sleep(0.02)` in playback loops with a wait/notify path for new buffered audio.

**Files:**
- Modify: `askme/voice/output/tts.py`
- Modify: `tests/test_tts.py`

Add a condition in `__init__`:

```python
self._buffer_condition = threading.Condition(self._buffer_lock)
```

Wrap buffer appends in a helper:

```python
def _queue_audio_chunk(self, samples: np.ndarray) -> None:
    with self._buffer_condition:
        self.tts_buffer.append(samples)
        self._buffer_condition.notify_all()
```

Use this helper everywhere `tts_buffer.append(...)` currently appears in production code.

Add a test that monkeypatches `time.sleep` to raise during an idle-to-buffered transition, then queues a chunk and verifies playback wakes by notification rather than by blind polling. Keep the test deterministic and avoid real audio devices by monkeypatching transport playback.

### Task 2.3: Run voice lane verification

Run:

```bash
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest tests/test_tts.py tests/test_tts_minimax.py tests/test_voice_profiles.py -q --tb=short
ruff check askme/voice/output/tts.py tests/test_tts.py
```

Expected: all tests and ruff PASS.

---

## P3: Later Maintainability Cleanup

Do this only after P0-P2 are complete and green.

Candidate tasks:

1. Split route registration functions that exceed 300 lines:
   - `askme/api/routes/skills.py:register_skill_routes`
   - `askme/api/routes/field_product_catalog.py:create_field_product_catalog_router`
   - `askme/api/routes/field_delivery_resources.py:register_delivery_resource_routes`
2. Split `askme/health_server.py:create_health_app` by moving route dependency assembly into focused helpers.
3. Split `askme/cli.py:build_parser` and `_handle_runtime_command` into subparser modules.

Rules:
- No business behavior changes in the same PR/commit as a large extraction.
- Use import/route compatibility tests first.
- Run boundary tests after any package move:

```bash
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest tests/test_six_layer_package_boundaries.py tests/test_package_migration_compat.py -q --tb=short
```

---

## Recommended Execution Sequence

1. Commit or explicitly preserve the already-completed vector store optimization separately.
2. Execute P0 first because it has measured threshold failure and smallest write scope.
3. Re-run quick benchmark; if it passes, continue to P1.
4. Execute P1 with field tests only.
5. Execute P2 only after current `audio_devices` work is either committed or explicitly out of scope.
6. Save broad extraction work for a separate cleanup round.

## Final Verification Before Hand-off

Run the target suite for all touched lanes:

```bash
ruff check askme/api/services/scenario_intent_payloads.py askme/api/services/conversation_service.py tests/test_conversation_service.py tests/test_conversation_http.py
ruff check askme/pipeline/field/field_operations.py tests/test_field_operations.py
ruff check askme/voice/output/tts.py tests/test_tts.py

env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python -m pytest \
  tests/test_conversation_service.py \
  tests/test_conversation_http.py \
  tests/test_field_operations.py \
  tests/test_tts.py \
  -q --tb=short

env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python scripts/benchmark_core_paths.py --quick --iterations 40 --concurrency 8 --output artifacts/perf/core_paths_latest.json
env -u PYTHONHOME -u UV_INTERNAL__PYTHONHOME python scripts/check_perf_thresholds.py --report artifacts/perf/core_paths_latest.json
```

Expected:
- lint passes;
- target tests pass;
- benchmark thresholds pass;
- `git status --short --untracked-files=all` shows only intentional source/test/doc changes.
