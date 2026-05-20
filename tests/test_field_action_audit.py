from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from askme.pipeline.field.field_action_audit import (
    FIELD_ACTION_AUDIT_GENESIS,
    FIELD_ACTION_AUDIT_SIGNATURE_ALG,
    FieldActionAuditIntegrityError,
    audit_review_path_from_field_config,
    field_action_audit_counts_by_event,
    field_action_audit_hash,
    field_action_audit_signature,
    next_field_action_audit_checkpoint,
    strict_field_action_audit_checkpoint,
)


def test_field_action_audit_hash_and_signature_are_stable() -> None:
    record = {
        "sequence": 1,
        "prev_hash": FIELD_ACTION_AUDIT_GENESIS,
        "event_id": "field-1",
        "audit": {"action": "acknowledge", "operator_id": "ops-1"},
    }

    first_hash = field_action_audit_hash(record)
    record["record_hash"] = "ignored"
    record["record_signature"] = "ignored"

    assert field_action_audit_hash(record) == first_hash
    assert field_action_audit_signature(record, secret="secret") == field_action_audit_signature(
        record,
        secret="secret",
    )


def test_next_field_action_audit_checkpoint_skips_invalid_lines(tmp_path: Path) -> None:
    audit_path = tmp_path / "field-action-audit.jsonl"
    first = {"sequence": 1, "record_hash": "hash-1"}
    second = {"sequence": 2, "record_hash": "hash-2"}
    audit_path.write_text(
        "\n".join([
            json.dumps(first),
            "{not-json",
            "",
            json.dumps(second),
        ]),
        encoding="utf-8",
    )

    assert next_field_action_audit_checkpoint(tmp_path / "missing.jsonl") == (
        1,
        FIELD_ACTION_AUDIT_GENESIS,
    )
    assert next_field_action_audit_checkpoint(audit_path) == (3, "hash-2")


def test_strict_field_action_audit_checkpoint_rejects_invalid_json(tmp_path: Path) -> None:
    audit_path = tmp_path / "field-action-audit.jsonl"
    audit_path.write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(FieldActionAuditIntegrityError, match="invalid_json"):
        strict_field_action_audit_checkpoint(audit_path)


def test_strict_field_action_audit_checkpoint_rejects_hash_break(tmp_path: Path) -> None:
    audit_path = tmp_path / "field-action-audit.jsonl"
    record = {
        "kind": "field_event_action",
        "sequence": 1,
        "prev_hash": FIELD_ACTION_AUDIT_GENESIS,
        "event_id": "field-1",
        "audit": {"action": "acknowledge", "operator_id": "ops-1"},
        "hash_alg": "sha256",
    }
    record["record_hash"] = field_action_audit_hash(record)
    audit_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert strict_field_action_audit_checkpoint(audit_path) == (2, record["record_hash"])

    record["audit"]["operator_id"] = "attacker"
    audit_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FieldActionAuditIntegrityError, match="record_hash_mismatch"):
        strict_field_action_audit_checkpoint(audit_path)


def test_strict_field_action_audit_checkpoint_rejects_bad_signature(tmp_path: Path) -> None:
    audit_path = tmp_path / "field-action-audit.jsonl"
    record = {
        "kind": "field_event_action",
        "sequence": 1,
        "prev_hash": FIELD_ACTION_AUDIT_GENESIS,
        "event_id": "field-1",
        "audit": {"action": "acknowledge", "operator_id": "ops-1"},
        "hash_alg": "sha256",
        "signature_alg": FIELD_ACTION_AUDIT_SIGNATURE_ALG,
    }
    record["record_hash"] = field_action_audit_hash(record)
    record["record_signature"] = field_action_audit_signature(record, secret="secret")
    audit_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert strict_field_action_audit_checkpoint(audit_path, secret="secret") == (
        2,
        record["record_hash"],
    )

    record["record_signature"] = "bad"
    audit_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FieldActionAuditIntegrityError, match="record_signature_mismatch"):
        strict_field_action_audit_checkpoint(audit_path, secret="secret")


def test_field_action_audit_counts_by_event_ignores_invalid_records() -> None:
    assert field_action_audit_counts_by_event(
        [
            {"event_id": "field-1", "action_audit": [{}, {}]},
            {"event_id": "field-2", "action_audit": [{}]},
            {"event_id": "", "action_audit": [{}]},
            {"event_id": "field-3", "action_audit": "bad"},
        ]
    ) == {"field-1": 2, "field-2": 1}


def test_audit_review_path_from_field_config_supports_legacy_keys() -> None:
    assert audit_review_path_from_field_config({}) == Path("artifacts/audit/reviews.jsonl")
    assert audit_review_path_from_field_config({"audit_review_path": "out/reviews.jsonl"}) == Path(
        "out/reviews.jsonl"
    )
    assert audit_review_path_from_field_config(
        {"audit": {"review": {"jsonl_path": "review/from-nested.jsonl"}}}
    ) == Path("review/from-nested.jsonl")


def test_field_action_audit_kernel_is_leaf_and_field_operations_uses_it() -> None:
    helper_path = Path("askme/pipeline/field/field_action_audit.py")
    service_path = Path("askme/pipeline/field/field_operations.py")
    helper_tree = ast.parse(helper_path.read_text(encoding="utf-8"))
    service_tree = ast.parse(service_path.read_text(encoding="utf-8"))

    helper_imports = {
        node.module
        for node in ast.walk(helper_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(helper_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    service_imports = {
        node.module
        for node in ast.walk(service_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    service_defs = {
        node.name
        for node in ast.walk(service_tree)
        if isinstance(node, ast.FunctionDef)
    }

    assert "askme.pipeline.field.field_operations" not in helper_imports
    assert "askme.health_server" not in helper_imports
    assert "askme.pipeline.field.field_action_audit" in service_imports
    assert "_field_action_audit_hash" not in service_defs
    assert "_field_action_audit_signature" not in service_defs
    assert "_next_field_action_audit_checkpoint" not in service_defs
    assert "_strict_field_action_audit_checkpoint" not in service_defs
    assert "_field_action_audit_counts_by_event" not in service_defs
