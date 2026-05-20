from __future__ import annotations

import hashlib

from askme.pipeline.field import customer_project_evidence_inventory as inventory


def test_evidence_file_inventory_hashes_files_under_project_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(inventory, "PROJECT_ROOT", tmp_path)
    evidence = tmp_path / "artifacts" / "field" / "smoke.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("field evidence", encoding="utf-8")

    record = inventory._evidence_file_inventory(
        str(evidence),
        evidence_url=inventory._evidence_url(str(evidence)),
    )

    assert record["exists"] is True
    assert record["size_bytes"] == len(b"field evidence")
    assert record["sha256"] == hashlib.sha256(b"field evidence").hexdigest()
    assert record["evidence_url"] == "/api/field/evidence?path=artifacts/field/smoke.json"
    assert inventory._evidence_file_modified_at(str(evidence)) > 0


def test_evidence_url_encodes_project_relative_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(inventory, "PROJECT_ROOT", tmp_path)
    evidence = tmp_path / "artifacts" / "现场 证据.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("field evidence", encoding="utf-8")

    assert (
        inventory._evidence_url(str(evidence))
        == "/api/field/evidence?path=artifacts/%E7%8E%B0%E5%9C%BA%20%E8%AF%81%E6%8D%AE.json"
    )


def test_evidence_file_inventory_rejects_files_outside_project_root(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    outside = tmp_path / "outside.txt"
    project_root.mkdir()
    outside.write_text("outside", encoding="utf-8")
    monkeypatch.setattr(inventory, "PROJECT_ROOT", project_root)

    record = inventory._evidence_file_inventory(str(outside), evidence_url="")

    assert record["exists"] is False
    assert record["error"] == "outside_project"
    assert inventory._evidence_url(str(outside)) == ""
    assert inventory._evidence_file_modified_at(str(outside)) == 0.0


def test_customer_project_evidence_inventory_dedupes_and_marks_onsite_receipts(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(inventory, "PROJECT_ROOT", tmp_path)
    readiness = tmp_path / "readiness.json"
    onsite = tmp_path / "onsite.json"
    readiness.write_text("ready", encoding="utf-8")
    onsite.write_text("onsite", encoding="utf-8")

    rows = inventory._customer_project_evidence_inventory(
        {
            "field_readiness": {
                "evidence_reports": [
                    {"path": str(readiness), "evidence_url": "/existing"},
                    {"path": str(readiness), "evidence_url": "/duplicate"},
                ],
                "archive": {"path": str(onsite)},
            },
            "onsite_acceptance_evidence": {
                "receipts": [
                    {
                        "path": str(onsite),
                        "evidence_type": "notification_delivery",
                        "receipt_id": "receipt-1",
                    }
                ]
            },
        }
    )

    assert [row["path"] for row in rows] == [str(readiness), str(onsite)]
    assert rows[0]["evidence_url"] == "/existing"
    assert rows[0]["sha256"] == hashlib.sha256(b"ready").hexdigest()
    assert rows[1]["evidence_url"] == "/api/field/evidence?path=onsite.json"
    assert rows[1]["sha256"] == hashlib.sha256(b"onsite").hexdigest()
    assert rows[1]["evidence_type"] == "onsite_acceptance"
    assert rows[1]["onsite_evidence_type"] == "notification_delivery"
    assert rows[1]["receipt_id"] == "receipt-1"


def test_customer_project_evidence_inventory_merges_duplicate_onsite_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(inventory, "PROJECT_ROOT", tmp_path)
    evidence = tmp_path / "shared.json"
    evidence.write_text("shared-evidence", encoding="utf-8")

    rows = inventory._customer_project_evidence_inventory(
        {
            "field_readiness": {
                "evidence_reports": [{"path": str(evidence), "evidence_url": "/existing"}],
            },
            "onsite_acceptance_evidence": {
                "receipts": [
                    {
                        "path": str(evidence),
                        "evidence_type": "runtime_roundtrip",
                        "receipt_id": "receipt-shared",
                    }
                ]
            },
        }
    )

    assert len(rows) == 1
    assert rows[0]["path"] == str(evidence)
    assert rows[0]["evidence_url"] == "/existing"
    assert rows[0]["sha256"] == hashlib.sha256(b"shared-evidence").hexdigest()
    assert rows[0]["evidence_type"] == "onsite_acceptance"
    assert rows[0]["onsite_evidence_type"] == "runtime_roundtrip"
    assert rows[0]["receipt_id"] == "receipt-shared"


def test_customer_project_evidence_inventory_keeps_distinct_onsite_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(inventory, "PROJECT_ROOT", tmp_path)
    onsite = tmp_path / "onsite-only.json"
    onsite.write_text("onsite-only", encoding="utf-8")

    rows = inventory._customer_project_evidence_inventory(
        {
            "field_readiness": {},
            "onsite_acceptance_evidence": {
                "receipts": [
                    {
                        "path": str(onsite),
                        "evidence_type": "runtime_roundtrip",
                        "receipt_id": "receipt-2",
                    }
                ]
            },
        }
    )

    assert rows == [
        {
            "path": str(onsite),
            "evidence_url": "/api/field/evidence?path=onsite-only.json",
            "exists": True,
            "size_bytes": len(b"onsite-only"),
            "sha256": hashlib.sha256(b"onsite-only").hexdigest(),
            "evidence_type": "onsite_acceptance",
            "onsite_evidence_type": "runtime_roundtrip",
            "receipt_id": "receipt-2",
        }
    ]
