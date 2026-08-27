"""Tests for runtime diagnostic dialogue smoke pure functions."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from askme.runtime.diagnostics.dialogue_smoke import (
    _all_run_check,
    _build_checks,
    _burst_contract_checks,
    _burst_failure_reason,
    _compact_app_health,
    _duration_stats,
    _first_failed_check,
    _memory_health_summary,
    _percentile,
    _safe_run_id,
    _safe_token_prefix,
    _smoke_config,
    _token_in_evidence,
    _write_memory_seed,
    _write_report,
    print_dialogue_burst_summary,
    print_dialogue_smoke_summary,
    run_dialogue_smoke,
)

# ── _safe_run_id ─────────────────────────────────────────────────────────

class TestSafeRunId:
    def test_preserves_alphanumeric_lowercased(self):
        assert _safe_run_id("ASKME-LIVE-ABC123") == "askme-live-abc123"

    def test_replaces_special_chars_with_dash(self):
        result = _safe_run_id("hello world!@#")
        assert " " not in result
        assert "!" not in result
        assert "@" not in result

    def test_collapses_consecutive_dashes(self):
        assert _safe_run_id("a--b__c") == "a-b-c"

    def test_truncates_to_80_chars(self):
        long_input = "a" * 100
        assert len(_safe_run_id(long_input)) <= 80

    def test_empty_string_returns_fallback(self):
        result = _safe_run_id("")
        assert result.startswith("run-")
        assert len(result) > 4


# ── _safe_token_prefix ───────────────────────────────────────────────────

class TestSafeTokenPrefix:
    def test_uppercases_output(self):
        assert _safe_token_prefix("abc-def") == "ABC-DEF"

    def test_removes_special_chars(self):
        result = _safe_token_prefix("token!@#$%^")
        for ch in "!@#$%^":
            assert ch not in result

    def test_truncates_to_56_chars(self):
        long_input = "X" * 100
        assert len(_safe_token_prefix(long_input)) <= 56

    def test_empty_returns_fallback(self):
        result = _safe_token_prefix("")
        assert result.startswith("ASKME-BURST-")
        assert len(result) > 12


# ── _token_in_evidence ───────────────────────────────────────────────────

class TestTokenInEvidence:
    def test_finds_token_in_text(self):
        evidence = [{"text": "Thunder test id is TOKEN123", "metadata": {}}]
        assert _token_in_evidence("TOKEN123", evidence) is True

    def test_finds_token_in_metadata(self):
        evidence = [{"text": "", "metadata": {"token": "TOKEN123"}}]
        assert _token_in_evidence("TOKEN123", evidence) is True

    def test_returns_false_when_absent(self):
        evidence = [{"text": "other stuff", "metadata": {}}]
        assert _token_in_evidence("TOKEN123", evidence) is False

    def test_not_list_returns_false(self):
        assert _token_in_evidence("TOKEN", None) is False
        assert _token_in_evidence("TOKEN", "string") is False

    def test_empty_evidence_list_returns_false(self):
        assert _token_in_evidence("TOKEN", []) is False

    def test_non_dict_item_skipped(self):
        evidence = ["not a dict", {"text": "TOKEN123", "metadata": {}}]
        assert _token_in_evidence("TOKEN123", evidence) is True


# ── _first_failed_check ──────────────────────────────────────────────────

class TestFirstFailedCheck:
    def test_returns_first_failed_name(self):
        checks = {"a": True, "b": False, "c": False}
        assert _first_failed_check(checks) == "b"

    def test_returns_empty_when_all_pass(self):
        checks = {"a": True, "b": True}
        assert _first_failed_check(checks) == ""

    def test_returns_empty_for_empty_dict(self):
        assert _first_failed_check({}) == ""


# ── _all_run_check ───────────────────────────────────────────────────────

class TestAllRunCheck:
    def test_returns_true_when_all_pass(self):
        runs = [
            {"checks": {"memory_context_contains_token": True}},
            {"checks": {"memory_context_contains_token": True}},
        ]
        assert _all_run_check(runs, "memory_context_contains_token") is True

    def test_returns_false_when_any_fails(self):
        runs = [
            {"checks": {"memory_context_contains_token": True}},
            {"checks": {"memory_context_contains_token": False}},
        ]
        assert _all_run_check(runs, "memory_context_contains_token") is False

    def test_returns_false_for_empty_runs(self):
        assert _all_run_check([], "any_check") is False

    def test_handles_missing_checks_key(self):
        runs = [{"checks": {}}, {}]
        assert _all_run_check(runs, "missing_check") is False


# ── _burst_contract_checks ───────────────────────────────────────────────

class TestBurstContractChecks:
    def test_all_pass_when_every_run_passes(self):
        runs = [
            {
                "status": "passed",
                "checks": {
                    "memory_context_contains_token": True,
                    "chat_payload_has_rag": True,
                    "chat_evidence_contains_token": True,
                    "chat_reply_contains_token": True,
                },
            },
        ]
        result = _burst_contract_checks(runs, expected_runs=1)
        assert result["expected_run_count"] is True
        assert result["all_runs_passed"] is True

    def test_expected_run_count_fails_on_mismatch(self):
        result = _burst_contract_checks([], expected_runs=5)
        assert result["expected_run_count"] is False
        # all_runs_passed uses all([])→True (vacuous truth), but expected_run_count catches it

    def test_contract_has_all_keys(self):
        result = _burst_contract_checks([], expected_runs=0)
        expected_keys = {
            "expected_run_count",
            "all_runs_passed",
            "memory_context_all_passed",
            "chat_rag_all_passed",
            "chat_evidence_all_passed",
            "chat_reply_token_all_passed",
        }
        assert set(result.keys()) == expected_keys


# ── _burst_failure_reason ────────────────────────────────────────────────

class TestBurstFailureReason:
    def test_lists_failed_contracts_and_runs(self):
        checks = {"check_a": False, "check_b": True}
        runs = [
            {"kind": "real", "index": 1, "status": "failed", "failure_reason": "timeout"},
        ]
        reason = _burst_failure_reason(checks, runs)
        assert "check_a" in reason
        assert "real#1:timeout" in reason

    def test_returns_empty_for_all_pass(self):
        checks = {"a": True}
        runs = [{"kind": "fake", "index": 1, "status": "passed", "failure_reason": ""}]
        reason = _burst_failure_reason(checks, runs)
        assert reason == ""


# ── _duration_stats ──────────────────────────────────────────────────────

class TestDurationStats:
    def test_computes_correctly(self):
        values = [100.0, 200.0, 300.0, 400.0, 500.0]
        stats = _duration_stats(values)
        assert stats["min"] == 100.0
        assert stats["max"] == 500.0
        assert stats["avg"] == 300.0

    def test_empty_returns_zeros(self):
        stats = _duration_stats([])
        for key in ("min", "p50", "p95", "max", "avg"):
            assert stats[key] == 0.0

    def test_negative_values_excluded(self):
        stats = _duration_stats([100.0, -50.0])
        assert stats["min"] == 100.0


# ── _percentile ──────────────────────────────────────────────────────────

class TestPercentile:
    def test_p50_of_five_elements(self):
        values = sorted([100.0, 200.0, 300.0, 400.0, 500.0])
        assert _percentile(values, 50) == 300.0

    def test_p95_of_five_elements(self):
        values = sorted([100.0, 200.0, 300.0, 400.0, 500.0])
        assert _percentile(values, 95) == 500.0

    def test_empty_returns_zero(self):
        assert _percentile([], 50) == 0.0

    def test_single_element(self):
        assert _percentile([42.0], 50) == 42.0


# ── _memory_health_summary ───────────────────────────────────────────────

class TestMemoryHealthSummary:
    def test_extracts_known_keys_only(self):
        health = {
            "enabled": True,
            "backend": "vector",
            "configured_backend": "vector",
            "available": True,
            "selected_backend_ready": True,
            "vector_ready": True,
            "vector_store_path": "/tmp/store.json",
            "vector_size": 5,
            "last_backend": "vector",
            "last_retrieve_ms": 12.3,
            "last_retrieved_items": 2,
            "last_fallback_reason": "",
            "last_evidence": [],
            "last_dropped_evidence": [],
            "last_answer_policy": "",
            "extra_field": "should be excluded",
        }
        result = _memory_health_summary(health)
        assert "extra_field" not in result
        assert result["enabled"] is True
        assert result["vector_size"] == 5

    def test_handles_empty_dict(self):
        result = _memory_health_summary({})
        assert isinstance(result, dict)


# ── _compact_app_health ──────────────────────────────────────────────────

class TestCompactAppHealth:
    def test_compacts_module_keys(self):
        health = {
            "llm": {"status": "ok", "model": "gpt-4", "extra": "drop"},
            "memory": {
                "status": "ok",
                "enabled": True,
                "backend": "vector",
                "available": True,
            },
            "simple_module": "ok_string",
        }
        result = _compact_app_health(health)
        assert result["simple_module"] == "ok_string"
        assert result["llm"] == {"status": "ok", "model": "gpt-4"}
        assert "rag" in result["memory"]

    def test_non_dict_modules_preserved(self):
        health = {"simple": "ok", "also_simple": 42}
        result = _compact_app_health(health)
        assert result["simple"] == "ok"
        assert result["also_simple"] == 42


# ── _smoke_config ────────────────────────────────────────────────────────

class TestSmokeConfig:
    def test_sets_data_dir(self):
        cfg = _smoke_config(
            {"app": {}},
            data_dir=Path("/tmp/test"),
            memory_timeout_s=30.0,
            vector_min_similarity=0.1,
            fake_llm=False,
            token="T1",
        )
        assert cfg["app"]["data_dir"] == str(Path("/tmp/test"))

    def test_sets_memory_config(self):
        cfg = _smoke_config(
            {},
            data_dir=Path("/tmp/test"),
            memory_timeout_s=45.0,
            vector_min_similarity=0.3,
            fake_llm=False,
            token="T1",
        )
        assert cfg["memory"]["enabled"] is True
        assert cfg["memory"]["backend"] == "vector"
        assert cfg["memory"]["retrieve_timeout"] == 45.0
        assert cfg["memory"]["vector_min_similarity"] == 0.3

    def test_fake_llm_config(self):
        cfg = _smoke_config(
            {},
            data_dir=Path("/tmp/test"),
            memory_timeout_s=30.0,
            vector_min_similarity=0.1,
            fake_llm=True,
            token="MYTOKEN",
        )
        assert cfg["brain"]["provider"] == "fake"
        assert cfg["brain"]["model"] == "fake-dialogue-smoke"
        assert cfg["brain"]["provider_options"]["response_text"] == "MYTOKEN"

    def test_does_not_mutate_base_config(self):
        base = {"app": {"original": True}}
        cfg = _smoke_config(
            base,
            data_dir=Path("/tmp/test"),
            memory_timeout_s=30.0,
            vector_min_similarity=0.1,
            fake_llm=False,
            token="T1",
        )
        assert "data_dir" not in base["app"]


@pytest.mark.asyncio
async def test_fake_llm_smoke_runs_without_fastembed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from askme.memory.retrieval import vector_store

    monkeypatch.setattr(vector_store, "_FE_AVAILABLE", False)
    monkeypatch.setattr(vector_store, "_MODEL_CACHE", {})

    report = await run_dialogue_smoke(
        fake_llm=True,
        token="ASKME-OFFLINE-TEST",
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
        chat_timeout_s=15.0,
        memory_timeout_s=5.0,
    )

    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert report["config_overrides"]["memory.embedding_mode"] == "deterministic_offline"
    assert vector_store._FE_AVAILABLE is False
    assert vector_store._MODEL_CACHE == {}


# ── _build_checks ────────────────────────────────────────────────────────

class TestBuildChecks:
    def _base_payload(self) -> dict:
        return {
            "reply": "The test id is TOKEN999",
            "evidence": [{"text": "Thunder test id is TOKEN999", "metadata": {}}],
            "rag": {
                "enabled": True,
                "turn_scoped": True,
                "used_in_answer": True,
            },
        }

    def test_all_pass_for_valid_data(self):
        checks = _build_checks(
            token="TOKEN999",
            import_payload={"imported": 1, "errors": False},
            direct_context="context with TOKEN999",
            direct_health={"available": True},
            chat_payload=self._base_payload(),
            failure_reason="",
            require_reply_token=True,
        )
        assert all(checks.values())

    def test_runtime_completed_false_on_failure(self):
        checks = _build_checks(
            token="T", import_payload={}, direct_context="",
            direct_health={}, chat_payload={},
            failure_reason="crash", require_reply_token=True,
        )
        assert checks["runtime_completed"] is False

    def test_knowledge_not_imported_when_imported_zero(self):
        checks = _build_checks(
            token="T",
            import_payload={"imported": 0, "errors": False},
            direct_context="", direct_health={}, chat_payload={},
            failure_reason="", require_reply_token=True,
        )
        assert checks["knowledge_imported"] is False

    def test_chat_reply_nonempty_rejects_bracket_start(self):
        checks = _build_checks(
            token="T", import_payload={}, direct_context="",
            direct_health={},
            chat_payload={"reply": "[SILENT]"},
            failure_reason="", require_reply_token=False,
        )
        assert checks["chat_reply_nonempty"] is False

    def test_chat_reply_contains_token_respected(self):
        checks = _build_checks(
            token="TOKEN999",
            import_payload={"imported": 1, "errors": False},
            direct_context="", direct_health={},
            chat_payload=self._base_payload(),
            failure_reason="", require_reply_token=True,
        )
        assert checks["chat_reply_contains_token"] is True

    def test_chat_reply_token_not_required(self):
        checks = _build_checks(
            token="TOKEN999",
            import_payload={"imported": 1, "errors": False},
            direct_context="", direct_health={},
            chat_payload={"reply": "no token here", "rag": {}, "evidence": []},
            failure_reason="", require_reply_token=False,
        )
        assert checks["chat_reply_contains_token"] is True  # always True when not required

    def test_chat_rag_rejects_shared_backend_health_snapshot(self):
        payload = self._base_payload()
        payload["rag"] = {"enabled": True, "last_retrieved_items": 1}

        checks = _build_checks(
            token="TOKEN999",
            import_payload={"imported": 1, "errors": False},
            direct_context="context with TOKEN999",
            direct_health={"available": True},
            chat_payload=payload,
            failure_reason="",
            require_reply_token=True,
        )

        assert checks["chat_payload_has_rag"] is False


# ── _write_report ────────────────────────────────────────────────────────

class TestWriteReport:
    def test_writes_json_to_file(self, tmp_path):
        report = {"status": "passed", "token": "T1"}
        _write_report(tmp_path, report)
        written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert written["status"] == "passed"


# ── _write_memory_seed ───────────────────────────────────────────────────

class TestWriteMemorySeed:
    def test_writes_seed_with_correct_structure(self, tmp_path):
        path = _write_memory_seed(tmp_path, text="test fact", token="TOKEN1")
        assert path.name == "memory-seed.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["records"][0]["text"] == "test fact"
        assert payload["records"][0]["record_id"] == f"dialogue_smoke_{_safe_run_id('TOKEN1')}"


# ── print_dialogue_smoke_summary ─────────────────────────────────────────

class TestPrintDialogueSmokeSummary:
    def test_prints_status_and_token(self, caplog):
        caplog.set_level(logging.INFO)
        payload = {
            "status": "passed",
            "token": "MY-TOKEN",
            "checks": {"a": True, "b": True},
            "failure_reason": "",
            "chat": {"reply": "Hello world"},
            "paths": {"output_dir": "/tmp"},
        }
        print_dialogue_smoke_summary(payload)
        out = caplog.text
        assert "passed" in out
        assert "MY-TOKEN" in out

    def test_prints_failure_reason(self, caplog):
        caplog.set_level(logging.INFO)
        payload = {
            "status": "failed",
            "token": "X",
            "checks": {"a": False},
            "failure_reason": "timeout",
            "chat": {"reply": ""},
            "paths": {"output_dir": "/tmp"},
        }
        print_dialogue_smoke_summary(payload)
        out = caplog.text
        assert "failed" in out
        assert "timeout" in out

    def test_truncates_long_reply(self, caplog):
        caplog.set_level(logging.INFO)
        payload = {
            "status": "passed",
            "token": "X",
            "checks": {},
            "failure_reason": "",
            "chat": {"reply": "A" * 300},
            "paths": {"output_dir": "/tmp"},
        }
        print_dialogue_smoke_summary(payload)
        reply_msg = [r for r in caplog.records if "reply:" in r.getMessage()][0]
        assert len(reply_msg.getMessage()) < 200  # reply truncated to 160


# ── print_dialogue_burst_summary ─────────────────────────────────────────

class TestPrintDialogueBurstSummary:
    def test_prints_counts_and_timing(self, caplog):
        caplog.set_level(logging.INFO)
        payload = {
            "status": "passed",
            "counts": {"passed": 5, "total": 6, "fake": 5, "real": 1},
            "timing_ms": {"min": 100.0, "p50": 200.0, "p95": 500.0, "max": 600.0},
            "contract_checks": {"expected_run_count": True},
            "failure_reason": "",
            "paths": {"report": "/tmp/report.json"},
        }
        print_dialogue_burst_summary(payload)
        out = caplog.text
        assert "passed" in out
        assert "5/6" in out
        assert "fake=5" in out

    def test_prints_failure_reason(self, caplog):
        caplog.set_level(logging.INFO)
        payload = {
            "status": "failed",
            "counts": {"passed": 0, "total": 1, "fake": 0, "real": 1},
            "timing_ms": {"min": 0, "p50": 0, "p95": 0, "max": 0},
            "contract_checks": {"expected_run_count": False},
            "failure_reason": "expected_run_count",
            "paths": {"report": "/tmp"},
        }
        print_dialogue_burst_summary(payload)
        out = caplog.text
        assert "failed" in out
        assert "expected_run_count" in out
