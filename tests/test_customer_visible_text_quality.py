from __future__ import annotations

from pathlib import Path
from typing import Any


CUSTOMER_VISIBLE_PATHS = [
    Path("README.md"),
    Path("config.yaml"),
    Path("askme/static/dashboard"),
    Path("askme/api/services/field_customer_project_workbench.py"),
    Path("askme/audit"),
    Path("askme/blueprints"),
    Path("askme/memory"),
    Path("askme/pipeline/field"),
    Path("askme/skills"),
    Path("askme/pipeline/core/persona.py"),
    Path("docs/PRODUCT.md"),
    Path("docs/OPERATIONS.md"),
]

TEXT_SUFFIXES = {".html", ".js", ".json", ".md", ".py", ".yaml", ".yml"}

MOJIBAKE_MARKERS = [
    "\u920d",  # 鈍
    "\u9225",  # 鈥
    "\u922b",  # 鈫
    "\ufffd",
    "\u941c",  # 鐜
    "\u934f",  # 鍏
    "\u6d63",  # 浣
    "\u7487",  # 璇
    "\u6d93",  # 涓
    "\u7ecc",  # 绌
    "\u8930",  # 褰
    "\u7039",  # 瀹
    "\u9356",  # 鍖
    "\u5a13",  # 娓
    "\u7d31",  # 绱
    "\u95c2",  # 闂
    "\u935a",  # 鍚
    "\u9359",  # 鍙
    "\u9352",  # 鍒
    "\u6434",  # 搴
    "\u7ecb",  # 绋
    "\u59dd",  # 姝
    "\u6783",  # 枃
    "\u6bb7",  # 殷
    "\u95ca",  # 闊
    "\u6f79",  # 澹
    "\u95ab",  # 閫
    "\u4e99",  # 亙
    "\u6086",  # 悆
    "\u6977",  # 楷
]

COMMON_CUSTOMER_TEXT_TOKENS = [
    "园区",
    "巡检",
    "机器人",
    "语音",
    "客户",
    "现场",
    "运行",
    "交付",
    "验证",
    "审计",
    "证据",
    "配置",
    "服务",
    "任务",
    "安全",
    "知识库",
    "能力中心",
]


HARDCODED_BRAND_TERMS = [
    "Thunder",
    "雷霆",
    "穹沛科技",
    "ThunderAgentShell",
]

RAW_CUSTOMER_PLACEHOLDERS = [
    "Must not be enabled",
    "Run site validation.",
    "Package can enter validation.",
    "Scenario can enter validation.",
    "Validate service point trigger.",
    "Lab rehearsal only; not production go-live evidence.",
    "Production launch claims require separate onsite acceptance",
    "Customer Delivery Audit Dossier",
    "Ready for acceptance",
    "Blocked before acceptance",
    "unattended production launch claim",
    "replacement for onsite acceptance result",
    "This report supports demo/trial acceptance review",
    "Project evidence is ready for onsite acceptance.",
    "Package has no open managed-object delivery actions.",
    "This gate controls customer handoff readiness only.",
    "Do not claim production launch.",
    "Acceptance evidence linked",
    "Acceptance test evidence missing",
    "Acceptance bindings incomplete",
    "Acceptance test evidence must stay inside the project repository.",
    "Only approved published template packages appear in these release notes.",
    "This proposal bundle is customer-facing planning material.",
    "This workflow supports delivery review and customer pilot handoff.",
    "Run live onsite smoke tests before any production launch claim.",
    "Managed object execution bindings",
    "Deployment credentials",
    "Onsite acceptance boundary",
    "required onsite evidence receipts passed",
    "required onsite evidence types passed",
    "Evidence and internal review are ready for customer signoff.",
]


def _iter_customer_visible_files() -> list[Path]:
    files: list[Path] = []
    for path in CUSTOMER_VISIBLE_PATHS:
        if path.is_file():
            if path.suffix.lower() in TEXT_SUFFIXES:
                files.append(path)
            continue
        if path.is_dir():
            files.extend(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in TEXT_SUFFIXES
            )
    return sorted(files)


def test_customer_visible_text_has_no_mojibake_markers() -> None:
    offenders: list[str] = []
    mojibake_markers = [*MOJIBAKE_MARKERS, *_generated_common_mojibake_markers()]
    for path in _iter_customer_visible_files():
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            if any(marker in line for marker in mojibake_markers):
                offenders.append(f"{path}:{line_no}: {line.strip()[:120]}")
            elif any("\ue000" <= char <= "\uf8ff" for char in line):
                offenders.append(f"{path}:{line_no}: private-use unicode in customer-visible text")

    assert offenders == []


def _generated_common_mojibake_markers() -> list[str]:
    markers: list[str] = []
    for token in COMMON_CUSTOMER_TEXT_TOKENS:
        marker = token.encode("utf-8").decode("gbk", errors="ignore")
        if marker and marker != token and marker not in markers:
            markers.append(marker)
    return markers


def test_customer_visible_text_has_no_hardcoded_robot_brand() -> None:
    offenders: list[str] = []
    for path in _iter_customer_visible_files():
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            for term in HARDCODED_BRAND_TERMS:
                if term in line:
                    offenders.append(f"{path}:{line_no}: hardcoded brand term {term!r}")

    assert offenders == []


def test_generated_customer_blueprint_payload_has_clean_text() -> None:
    from askme.blueprints import blueprint_delivery_package, catalog_payload
    from askme.api.services.capability_package_payloads import (
        capability_package_catalog,
        package_readiness_contract,
    )
    from askme.api.services.dashboard_pages import dashboard_pages_payload
    from askme.api.services.field_customer_project_workbench import (
        build_customer_project_workbench_payload,
    )
    from askme.api.routes.field_customer_project_execution import (
        _object_rehearsal_boundary,
        _rehearsal_onsite_evidence_rejection,
    )

    ready_config = {
        "voice": {},
        "perception": {},
        "field_operations": {
            "dingtalk_webhooks": {"security": "${ASKME_DINGTALK_SECURITY_WEBHOOK}"},
        },
        "runtime_handoff": {},
        "runtime": {"dog_control": {"base_url": "${DOG_CONTROL_SERVICE_URL}"}},
    }
    payloads = {
        "catalog": catalog_payload(config=ready_config),
        "park_delivery_package": blueprint_delivery_package("park", config=ready_config),
        "capability_package_contract": package_readiness_contract(),
        "capability_package_catalog": capability_package_catalog(
            {
                "capability_packages": {
                    "items": [
                        {
                            "package_id": "capability.answer_wayfinding",
                            "display_name": "问路回答",
                            "capability": "answer_wayfinding",
                            "status": "pilot",
                            "customer_visible_description": "回答园区路线问题。",
                        }
                    ],
                    "readiness": [
                        {
                            "package_id": "capability.answer_wayfinding",
                            "status": "ready",
                            "status_label": "Ready for site validation",
                            "customer_message": "Package can enter validation.",
                            "customer_next_step": "Run site validation.",
                        }
                    ],
                },
                "scenario_blueprints": {
                    "items": [
                        {
                            "scenario_id": "wayfinding_help_point",
                            "display_name": "问路服务点",
                            "package_manifest": {
                                "package_id": "scenario.wayfinding_help_point",
                                "scenario": "wayfinding_help_point",
                                "capability_packages": ["capability.answer_wayfinding"],
                            },
                            "package_readiness": {
                                "package_id": "scenario.wayfinding_help_point",
                                "status": "ready",
                                "customer_message": "Scenario can enter validation.",
                                "customer_next_step": "Validate service point trigger.",
                            },
                        }
                    ]
                },
            }
        ),
        "dashboard_pages": dashboard_pages_payload(),
        "customer_project_workbench": build_customer_project_workbench_payload(
            project_catalog={
                "summary": {
                    "delivery_acceptance_gate_status": "ready",
                    "project_count": 1,
                },
                "filters": {"industry": "park"},
            },
            template_catalog={
                "summary": {"overall_status": "ready", "template_count": 1},
                "templates": [],
            },
            resource_catalog={
                "summary": {"overall_status": "ready", "resource_count": 1},
                "resources": [],
            },
            object_summary={"overall_status": "ready", "object_count": 1},
            object_rows=[
                {
                    "project_id": "fanmu-park",
                    "object_id": "west-gate",
                    "delivery_status": "ready",
                    "bindings": {
                        "vision_models": ["people-detector"],
                        "sensor_protocols": ["camera-detection-json"],
                        "skill_packages": ["park-guide"],
                        "acceptance_tests": ["tests/test_field.py::test_west_gate"],
                    },
                }
            ],
            projects=[
                {
                    "customer_id": "fanmu",
                    "customer_name": "梵木创艺园",
                    "project_id": "fanmu-park",
                    "industry": "park",
                }
            ],
            readiness={
                "overall_status": "ready",
                "customer_status": "客户项目工作台已就绪",
                "release_claim": "仅声明可验收的交付范围。",
                "next_step": "核对对象目录。",
                "summary": {"project_count": 1},
            },
            scope_filtered=True,
        ),
        "object_rehearsal_boundary": _object_rehearsal_boundary("dry_run"),
        "object_rehearsal_evidence_rejection": _rehearsal_onsite_evidence_rejection("dry_run"),
    }
    mojibake_markers = [*MOJIBAKE_MARKERS, *_generated_common_mojibake_markers()]
    offenders: list[str] = []

    for label, payload in payloads.items():
        for path, value in _iter_payload_strings(payload):
            if any(marker in value for marker in mojibake_markers):
                offenders.append(f"{label}.{path}: mojibake: {value[:120]}")
            if any("\ue000" <= char <= "\uf8ff" for char in value):
                offenders.append(f"{label}.{path}: private-use unicode: {value[:120]}")
            for term in HARDCODED_BRAND_TERMS:
                if term in value:
                    offenders.append(f"{label}.{path}: hardcoded brand term {term!r}")
            for placeholder in RAW_CUSTOMER_PLACEHOLDERS:
                if placeholder in value:
                    offenders.append(f"{label}.{path}: raw customer placeholder {placeholder!r}")

    assert offenders == []


def test_dashboard_page_registry_is_customer_readable_and_evidence_backed() -> None:
    from askme.api.services.dashboard_pages import dashboard_pages_payload

    payload = dashboard_pages_payload()
    offenders: list[str] = []
    for page in payload["pages"]:
        prefix = f"{page['key']}:"
        if not page.get("label") or not page.get("title") or not page.get("description"):
            offenders.append(f"{prefix} missing customer-readable copy")
        if not page.get("primary_endpoint"):
            offenders.append(f"{prefix} missing primary endpoint")
        if len(page.get("evidence_promises") or []) < 2:
            offenders.append(f"{prefix} must promise at least two evidence points")
        if page.get("exposes_internal_runtime"):
            offenders.append(f"{prefix} exposes internal runtime as a customer page")
        if page.get("section") not in payload["sections"]:
            offenders.append(f"{prefix} section is not registered")

    assert offenders == []


def _iter_payload_strings(value: Any, path: str = "$") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        results: list[tuple[str, str]] = []
        for key, child in value.items():
            results.extend(_iter_payload_strings(child, f"{path}.{key}"))
        return results
    if isinstance(value, list):
        results = []
        for index, child in enumerate(value):
            results.extend(_iter_payload_strings(child, f"{path}[{index}]"))
        return results
    return []
