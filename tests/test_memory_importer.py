from __future__ import annotations

from pathlib import Path

import pytest
from askme.memory.importer import (
    import_knowledge_file,
    parse_knowledge_file,
    parse_knowledge_text,
)


def test_parse_markdown_records_with_heading(tmp_path: Path) -> None:
    path = tmp_path / "site.md"
    path.write_text("# 一楼\n- 洗手间在东侧\n服务台在入口旁", encoding="utf-8")

    records = parse_knowledge_file(path, category="location")

    assert len(records) == 2
    assert records[0].category == "location"
    assert records[0].source == "site.md"
    assert "一楼: 洗手间在东侧" == records[0].text


def test_parse_markdown_strips_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "site.md"
    path.write_text("# 一楼\n- 洗手间在东侧", encoding="utf-8-sig")

    records = parse_knowledge_file(path, category="location")

    assert len(records) == 1
    assert records[0].text == "一楼: 洗手间在东侧"


def test_parse_csv_records(tmp_path: Path) -> None:
    path = tmp_path / "equipment.csv"
    path.write_text(
        "name,description,category,owner\n3号空压机,位于B区冷却泵旁,equipment,ops\n",
        encoding="utf-8",
    )

    records = parse_knowledge_file(path)

    assert len(records) == 1
    assert records[0].category == "equipment"
    assert "位于B区冷却泵旁" in records[0].text
    assert records[0].owner == "ops"


def test_parse_knowledge_text_json_preview() -> None:
    records = parse_knowledge_text(
        '[{"question":"where","answer":"east gate","category":"location"}]',
        filename="site.json",
    )

    assert len(records) == 1
    assert records[0].category == "location"
    assert records[0].source == "site.json"
    assert "where" in records[0].text


def test_parse_knowledge_text_preserves_product_governance_fields() -> None:
    records = parse_knowledge_text(
        (
            '[{"text":"Fanmu coffee is on floor one","category":"merchant",'
            '"quality_status":"public","visibility":"external",'
            '"customer_id":"fanmu","project_id":"fanmu-phase-1",'
            '"product_area":"space","workstream":"wayfinding",'
            '"linked_object_type":"park_point","linked_object_id":"poi-fanmu-coffee"}]'
        ),
        filename="site.json",
    )

    metadata = records[0].to_metadata()
    assert metadata["quality_status"] == "public"
    assert metadata["visibility"] == "external"
    assert metadata["customer_id"] == "fanmu"
    assert metadata["project_id"] == "fanmu-phase-1"
    assert metadata["product_area"] == "space"
    assert metadata["workstream"] == "wayfinding"
    assert metadata["linked_object_type"] == "park_point"
    assert metadata["linked_object_id"] == "poi-fanmu-coffee"


def test_parse_knowledge_text_markdown_preview() -> None:
    records = parse_knowledge_text("# floor 1\n- restroom east", filename="site.md", category="faq")

    assert len(records) == 1
    assert records[0].category == "faq"
    assert records[0].text == "floor 1: restroom east"


def test_parse_knowledge_text_uses_product_taxonomy() -> None:
    records = parse_knowledge_text(
        "- 梵木咖啡在 2 号楼一层，靠近西门",
        filename="merchant.md",
        category="merchant",
    )

    assert len(records) == 1
    assert records[0].normalized_category() == "merchant"
    metadata = records[0].to_metadata()
    assert metadata["category"] == "merchant"
    assert metadata["category_label"] == "商户与服务"
    assert metadata["category_group"] == "visitor"


def test_parse_knowledge_text_maps_legacy_note_to_inspection() -> None:
    records = parse_knowledge_text("- 巡检时先拍摄设备铭牌", filename="sop.md", category="note")

    assert records[0].normalized_category() == "inspection"
    assert records[0].to_memory_text().startswith("[inspection]")


def test_parse_knowledge_text_unknown_category_becomes_general() -> None:
    records = parse_knowledge_text("- 交付现场补充说明", filename="misc.md", category="random")

    assert records[0].normalized_category() == "general"
    assert records[0].to_metadata()["category_label"] == "其他资料"


@pytest.mark.asyncio
async def test_import_knowledge_file_saves_facts(tmp_path: Path) -> None:
    path = tmp_path / "faq.json"
    path.write_text(
        '[{"question":"洗手间在哪","answer":"一楼东侧","category":"faq"}]',
        encoding="utf-8",
    )
    saved: list[tuple[str, dict]] = []

    class Bridge:
        async def save_fact(self, text: str, metadata: dict) -> None:
            saved.append((text, metadata))

    result = await import_knowledge_file(path, bridge=Bridge())

    assert result.imported == 1
    assert result.parsed == 1
    assert saved[0][0].startswith("[faq]")
    assert "洗手间在哪" in saved[0][0]
    assert saved[0][1]["category"] == "faq"


@pytest.mark.asyncio
async def test_import_knowledge_file_dry_run_does_not_save(tmp_path: Path) -> None:
    path = tmp_path / "site.md"
    path.write_text("- 配电室在二楼", encoding="utf-8")

    class Bridge:
        async def save_fact(self, text: str, metadata: dict) -> None:
            raise AssertionError("dry-run should not save")

    result = await import_knowledge_file(path, bridge=Bridge(), dry_run=True)

    assert result.imported == 1
    assert result.dry_run is True


@pytest.mark.asyncio
async def test_import_knowledge_file_missing_path_returns_error(tmp_path: Path) -> None:
    result = await import_knowledge_file(tmp_path / "missing.md", dry_run=True)

    assert result.imported == 0
    assert result.parsed == 0
    assert result.errors[0].startswith("file_not_found:")
