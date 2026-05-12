from __future__ import annotations

from askme.memory.catalog import KnowledgeCatalog


def _catalog(tmp_path):
    return KnowledgeCatalog(path=tmp_path / "records.json")


def test_upsert_payloads_persists_records_and_returns_prompt_eligible_candidates(tmp_path):
    catalog = _catalog(tmp_path)

    result = catalog.upsert_payloads([{
        "record_id": "know_1",
        "text": "Restroom is east",
        "memory_text": "[location] Restroom is east",
        "category": "location",
        "approval_status": "published",
        "metadata": {"record_id": "know_1", "approval_status": "published"},
    }])

    assert result["indexed_candidates"][0]["record_id"] == "know_1"
    reloaded = _catalog(tmp_path)
    listed = reloaded.list_records()
    assert listed["total"] == 1
    assert listed["records"][0]["record_id"] == "know_1"


def test_upsert_payloads_marks_conflict_when_same_entity_fact_has_different_values(tmp_path):
    catalog = _catalog(tmp_path)

    result = catalog.upsert_payloads([
        {
            "record_id": "know_a",
            "text": "Device A is east",
            "memory_text": "[equipment] Device A is east",
            "approval_status": "published",
            "metadata": {
                "record_id": "know_a",
                "approval_status": "published",
                "entity_key": "device:a",
                "fact_key": "location",
                "value": "east",
            },
        },
        {
            "record_id": "know_b",
            "text": "Device A is west",
            "memory_text": "[equipment] Device A is west",
            "approval_status": "published",
            "metadata": {
                "record_id": "know_b",
                "approval_status": "published",
                "entity_key": "device:a",
                "fact_key": "location",
                "value": "west",
            },
        },
    ])

    assert result["indexed_candidates"] == []
    records = catalog.list_records()["records"]
    conflict_ids = {record["conflict_set_id"] for record in records}
    assert conflict_ids == {"conflict:device:a:location"}
    assert all(not catalog.is_prompt_eligible(record) for record in records)


def test_upsert_payloads_keeps_consistent_duplicate_records_prompt_eligible(tmp_path):
    catalog = _catalog(tmp_path)

    result = catalog.upsert_payloads([
        {
            "record_id": "know_a",
            "text": "Device A is east",
            "memory_text": "[equipment] Device A is east",
            "approval_status": "published",
            "metadata": {
                "record_id": "know_a",
                "approval_status": "published",
                "entity_key": "device:a",
                "fact_key": "location",
                "value": "east",
            },
        },
        {
            "record_id": "know_b",
            "text": "Device A is near east gate",
            "memory_text": "[equipment] Device A is near east gate",
            "approval_status": "published",
            "metadata": {
                "record_id": "know_b",
                "approval_status": "published",
                "entity_key": "device:a",
                "fact_key": "location",
                "value": "east",
            },
        },
    ])

    assert {record["record_id"] for record in result["indexed_candidates"]} == {
        "know_a",
        "know_b",
    }
    assert all(not record["conflict_set_id"] for record in catalog.list_records()["records"])


def test_update_metadata_delete_blocks_prompt_eligibility_but_keeps_admin_visible(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.upsert_payloads([{
        "record_id": "know_1",
        "text": "Restroom is east",
        "memory_text": "[location] Restroom is east",
        "approval_status": "published",
        "metadata": {"record_id": "know_1", "approval_status": "published"},
    }])

    result = catalog.update_metadata("know_1", {"approval_status": "deleted"})

    assert result["updated"] is True
    record = catalog.list_records()["records"][0]
    assert record["approval_status"] == "deleted"
    assert record["metadata"]["approval_status"] == "deleted"
    assert catalog.is_prompt_eligible(record) is False


def test_update_metadata_increments_evidence_version_for_prompt_affecting_change(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.upsert_payloads([{
        "record_id": "know_1",
        "text": "Restroom is east",
        "memory_text": "[location] Restroom is east",
        "approval_status": "published",
        "metadata": {"record_id": "know_1", "approval_status": "published"},
    }])

    result = catalog.update_metadata("know_1", {"expires_at": "2099-01-01T00:00:00+00:00"})

    assert result["record"]["evidence_version"] == 2
    assert result["record"]["metadata"]["evidence_version"] == 2


def test_evidence_drop_reason_rejects_stale_backend_version(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.upsert_payloads([{
        "record_id": "know_1",
        "text": "Restroom is east",
        "memory_text": "[location] Restroom is east",
        "approval_status": "published",
        "metadata": {"record_id": "know_1", "approval_status": "published"},
    }])
    catalog.update_metadata("know_1", {"expires_at": "2099-01-01T00:00:00+00:00"})

    reason = catalog.evidence_drop_reason({"record_id": "know_1", "evidence_version": 1})

    assert reason == "catalog_evidence_version:1->2"


def test_evidence_drop_reason_rejects_deleted_catalog_record(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.upsert_payloads([{
        "record_id": "know_1",
        "text": "Restroom is east",
        "memory_text": "[location] Restroom is east",
        "approval_status": "published",
        "metadata": {"record_id": "know_1", "approval_status": "published"},
    }])
    catalog.update_metadata("know_1", {"approval_status": "deleted"})

    reason = catalog.evidence_drop_reason({"record_id": "know_1", "evidence_version": 1})

    assert reason == "catalog_status:deleted"


def test_update_metadata_restore_recomputes_conflicts(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.upsert_payloads([
        {
            "record_id": "know_a",
            "text": "Device A is east",
            "memory_text": "[equipment] Device A is east",
            "approval_status": "published",
            "metadata": {
                "record_id": "know_a",
                "approval_status": "published",
                "entity_key": "device:a",
                "fact_key": "location",
                "value": "east",
            },
        },
        {
            "record_id": "know_b",
            "text": "Device A is west",
            "memory_text": "[equipment] Device A is west",
            "approval_status": "deleted",
            "metadata": {
                "record_id": "know_b",
                "approval_status": "deleted",
                "entity_key": "device:a",
                "fact_key": "location",
                "value": "west",
            },
        },
    ])

    catalog.update_metadata("know_b", {"approval_status": "published"})

    records = catalog.list_records()["records"]
    assert {record["conflict_set_id"] for record in records} == {
        "conflict:device:a:location"
    }


def test_update_metadata_many_patches_records_and_reports_missing_ids(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.upsert_payloads([
        {
            "record_id": "know_a",
            "text": "Restroom is east",
            "memory_text": "[location] Restroom is east",
            "approval_status": "published",
        },
        {
            "record_id": "know_b",
            "text": "Cafe is west",
            "memory_text": "[location] Cafe is west",
            "approval_status": "published",
        },
    ])

    result = catalog.update_metadata_many([
        {"record_id": "know_a", "patch": {"owner": "ops"}},
        {"record_id": "missing", "patch": {"owner": "ops"}},
        {"record_id": "know_b", "patch": {"approval_status": "deleted"}},
    ])

    assert result["updated"] == 2
    assert result["failed"] == 1
    assert result["errors"][0]["record_id"] == "missing"
    records = {record["record_id"]: record for record in catalog.list_records()["records"]}
    assert records["know_a"]["owner"] == "ops"
    assert records["know_b"]["approval_status"] == "deleted"


def test_records_for_rebuild_selects_only_prompt_eligible_records_by_default(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.upsert_payloads([
        {
            "record_id": "know_a",
            "text": "Restroom is east",
            "memory_text": "[location] Restroom is east",
            "approval_status": "published",
        },
        {
            "record_id": "know_b",
            "text": "Draft cafe note",
            "memory_text": "[location] Draft cafe note",
            "approval_status": "draft",
        },
    ])

    result = catalog.records_for_rebuild()

    assert result["total"] == 2
    assert result["eligible"] == 1
    assert result["skipped"] == 1
    assert result["record_ids"] == ["know_a"]


def test_list_records_exposes_customer_readable_lifecycle_state(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.upsert_payloads([
        {
            "record_id": "ready",
            "text": "Restroom is east",
            "memory_text": "[location] Restroom is east",
            "approval_status": "published",
        },
        {
            "record_id": "pending",
            "text": "Cafe is west",
            "memory_text": "[location] Cafe is west",
            "approval_status": "pending",
        },
        {
            "record_id": "old",
            "text": "Old route",
            "memory_text": "[route] Old route",
            "approval_status": "published",
            "expires_at": "2000-01-01T00:00:00+00:00",
        },
    ])

    records = {record["record_id"]: record for record in catalog.list_records()["records"]}
    health = catalog.health()

    assert records["ready"]["lifecycle_state"] == "needs_reindex"
    assert records["ready"]["lifecycle_label"] == "需重建索引"
    assert records["ready"]["prompt_eligible"] is True
    assert records["ready"]["needs_reindex"] is True
    assert records["pending"]["lifecycle_state"] == "pending_review"
    assert records["pending"]["prompt_eligible"] is False
    assert records["old"]["lifecycle_state"] == "expired"
    assert records["old"]["prompt_eligible"] is False
    assert health["prompt_eligible"] == 1
    assert health["needs_review"] == 1
    assert health["needs_reindex"] == 1
    assert health["expired"] == 1


def test_mark_indexed_tracks_evidence_version_and_reindex_need(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.upsert_payloads([{
        "record_id": "know_1",
        "text": "Restroom is east",
        "memory_text": "[location] Restroom is east",
        "approval_status": "published",
    }])

    catalog.mark_indexed("know_1")
    indexed = catalog.list_records()["records"][0]
    assert indexed["lifecycle_state"] == "ready"
    assert indexed["needs_reindex"] is False
    assert indexed["indexed_evidence_version"] == 1

    catalog.update_metadata("know_1", {"expires_at": "2099-01-01T00:00:00+00:00"})
    changed = catalog.list_records()["records"][0]
    assert changed["evidence_version"] == 2
    assert changed["indexed_evidence_version"] == 1
    assert changed["lifecycle_state"] == "needs_reindex"
    assert changed["needs_reindex"] is True


def test_update_metadata_records_approval_audit_event(tmp_path):
    catalog = _catalog(tmp_path)
    catalog.upsert_payloads([{
        "record_id": "know_1",
        "text": "Restroom is east",
        "memory_text": "[location] Restroom is east",
        "approval_status": "pending",
    }])

    result = catalog.update_metadata("know_1", {
        "approval_status": "approved",
        "approved_by": "ops.lead",
        "approved_at": "2026-05-11T08:00:00+00:00",
        "review_note": "verified on site",
    })

    record = result["record"]
    assert record["approved_by"] == "ops.lead"
    assert record["approved_at"] == "2026-05-11T08:00:00+00:00"
    assert record["review_note"] == "verified on site"
    assert record["events"][-1]["kind"] == "status:approved"
    assert record["events"][-1]["actor"] == "ops.lead"
    assert record["events"][-1]["note"] == "verified on site"
