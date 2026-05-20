from __future__ import annotations

import ast
import importlib
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

PACKAGES_WITH_LEGACY_ALIASES = (
    "askme.blueprints",
    "askme.memory",
    "askme.pipeline",
    "askme.robot",
    "askme.runtime",
    "askme.skills",
    "askme.tools",
    "askme.voice",
)

PACKAGE_FACADE_FILES = tuple(
    Path(package_name.replace(".", "/")) / "__init__.py"
    for package_name in PACKAGES_WITH_LEGACY_ALIASES
)

EXPECTED_LEGACY_ALIAS_KEYS = {
    "askme.blueprints": (
        "askme.blueprints._runner",
        "askme.blueprints.edge_robot",
        "askme.blueprints.lingtu_voice",
        "askme.blueprints.mcp",
        "askme.blueprints.text",
        "askme.blueprints.voice",
        "askme.blueprints.voice_perception",
    ),
    "askme.memory": (
        "askme.memory.admission",
        "askme.memory.association",
        "askme.memory.bridge",
        "askme.memory.catalog",
        "askme.memory.conversation",
        "askme.memory.episode",
        "askme.memory.episodic_memory",
        "askme.memory.extraction_adapter",
        "askme.memory.importer",
        "askme.memory.index_jobs",
        "askme.memory.map_adapter",
        "askme.memory.mempalace_backend",
        "askme.memory.policies",
        "askme.memory.procedural",
        "askme.memory.robotmem_backend",
        "askme.memory.semantic_index",
        "askme.memory.service",
        "askme.memory.session",
        "askme.memory.site_knowledge",
        "askme.memory.strategy",
        "askme.memory.system",
        "askme.memory.taxonomy",
        "askme.memory.trend_analyzer",
        "askme.memory.vector_store",
    ),
    "askme.pipeline": (
        "askme.pipeline.alert_dispatcher",
        "askme.pipeline.brain_pipeline",
        "askme.pipeline.commands",
        "askme.pipeline.external_turns",
        "askme.pipeline.field_deployment_readiness",
        "askme.pipeline.field_ingest_adapters",
        "askme.pipeline.field_ingest_bridge",
        "askme.pipeline.field_operations",
        "askme.pipeline.field_scenarios",
        "askme.pipeline.field_site_profile",
        "askme.pipeline.frames",
        "askme.pipeline.hooks",
        "askme.pipeline.incident_alerts",
        "askme.pipeline.persona",
        "askme.pipeline.planner_agent",
        "askme.pipeline.proactive_agent",
        "askme.pipeline.product_launch_readiness",
        "askme.pipeline.prompt_builder",
        "askme.pipeline.protocols",
        "askme.pipeline.rag_policy",
        "askme.pipeline.reaction_engine",
        "askme.pipeline.skill_dispatcher",
        "askme.pipeline.skill_gate",
        "askme.pipeline.state_led_bridge",
        "askme.pipeline.stream_processor",
        "askme.pipeline.text_loop",
        "askme.pipeline.tool_executor",
        "askme.pipeline.trace",
        "askme.pipeline.turn_executor",
        "askme.pipeline.utils",
        "askme.pipeline.voice_loop",
    ),
    "askme.robot": (
        "askme.robot.arm_controller",
        "askme.robot.control_client",
        "askme.robot.direct_commands",
        "askme.robot.led_controller",
        "askme.robot.mock_pulse",
        "askme.robot.ota_bridge",
        "askme.robot.policy_runner",
        "askme.robot.pubsub",
        "askme.robot.pulse",
        "askme.robot.runtime_health",
        "askme.robot.safety",
        "askme.robot.safety_client",
        "askme.robot.serial_bridge",
        "askme.robot.state_led_bridge",
    ),
    "askme.runtime": (
        "askme.runtime.arbiter_client",
        "askme.runtime.audit",
        "askme.runtime.field_callbacks",
        "askme.runtime.handoff",
        "askme.runtime.mission",
        "askme.runtime.module",
        "askme.runtime.profiles",
        "askme.runtime.registry",
    ),
    "askme.skills": (
        "askme.skills.audit",
        "askme.skills.capability_center",
        "askme.skills.contracts_builtin",
        "askme.skills.field_capability_contracts",
        "askme.skills.growth_backlog",
        "askme.skills.packages",
        "askme.skills.skill_executor",
        "askme.skills.skill_manager",
        "askme.skills.skill_model",
        "askme.skills.validation",
    ),
    "askme.tools": (
        "askme.tools.builtin_tools",
        "askme.tools.execution_control",
        "askme.tools.field_event_tool",
        "askme.tools.move_tool",
        "askme.tools.robot_api_tool",
        "askme.tools.robot_tools",
        "askme.tools.scan_tool",
        "askme.tools.skill_tools",
        "askme.tools.space_tool",
        "askme.tools.temporal_query_tool",
        "askme.tools.tool_registry",
        "askme.tools.vision_tool",
        "askme.tools.voice_tools",
    ),
    "askme.voice": (
        "askme.voice.address_detector",
        "askme.voice.asr",
        "askme.voice.asr_manager",
        "askme.voice.audio_agent",
        "askme.voice.audio_devices",
        "askme.voice.audio_filter",
        "askme.voice.audio_processor",
        "askme.voice.audio_router",
        "askme.voice.cloud_asr",
        "askme.voice.generated_contracts",
        "askme.voice.health_check",
        "askme.voice.interaction_gate",
        "askme.voice.kws",
        "askme.voice.media_contracts",
        "askme.voice.mic_calibration",
        "askme.voice.mic_input",
        "askme.voice.minimax_hybrid",
        "askme.voice.noise_reduction",
        "askme.voice.online_smoke",
        "askme.voice.perception_context",
        "askme.voice.punctuation",
        "askme.voice.runtime_bridge",
        "askme.voice.s100p_readiness_bundle",
        "askme.voice.stream_splitter",
        "askme.voice.sunrise_audio_doctor",
        "askme.voice.sunrise_readiness",
        "askme.voice.tts",
        "askme.voice.turn_trace",
        "askme.voice.vad",
        "askme.voice.vad_controller",
        "askme.voice.voice_profiles",
    ),
}

PACKAGE_COLLISION_FACADE_MODULES = (
    "askme.blueprints.catalog",
    "askme.skills.contracts",
    "askme.skills.governance",
)

PACKAGE_LAYOUT_READMES = (
    "askme.blueprints",
    "askme.cognition",
    "askme.llm",
    "askme.memory",
    "askme.pipeline",
    "askme.robot",
    "askme.runtime",
    "askme.skills",
    "askme.tools",
    "askme.voice",
)

ROOT_LAZY_FACADE_PACKAGES = (
    "askme.cognition",
    "askme.llm",
    "askme.memory",
    "askme.pipeline",
    "askme.runtime",
    "askme.skills",
    "askme.tools",
    "askme.voice",
)

LEGACY_LLM_FACADE_MODULES = {
    "askme.llm.client",
    "askme.llm.config",
    "askme.llm.conversation",
    "askme.llm.factory",
    "askme.llm.gateway",
    "askme.llm.intent_router",
    "askme.llm.model_policy",
}

LEGACY_LLM_FACADE_FILES = {
    Path("askme/llm/client.py"),
    Path("askme/llm/config.py"),
    Path("askme/llm/contracts.py"),
    Path("askme/llm/conversation.py"),
    Path("askme/llm/factory.py"),
    Path("askme/llm/gateway.py"),
    Path("askme/llm/intent_router.py"),
    Path("askme/llm/model_policy.py"),
}

TEMPLATE_RELEASE_PUBLIC_NAMES = {
    "create_customer_project_template_release_request",
    "customer_project_template_release_notes",
    "export_customer_project_template_release_notes_bundle",
    "list_customer_project_template_release_requests",
    "list_customer_project_template_revisions",
    "review_customer_project_template_release_request",
    "update_customer_project_template_release",
}

TEMPLATE_RELEASE_PRIVATE_HELPERS = {
    "_customer_project_template_release_request_dir",
    "_customer_project_template_revision_dir",
    "_find_customer_project_template_release_request",
    "_iter_customer_project_template_release_requests",
    "_load_customer_project_template_revisions",
    "_read_customer_project_template_release_request_file",
    "_read_customer_project_template_revision_file",
    "_release_notes_bundle_slug",
    "_release_notes_customer_context",
    "_snapshot_customer_project_template_revision",
    "_template_release_note_delivery_details",
    "_template_release_notes_bundle_html",
    "_template_release_notes_proposal_insert",
    "_template_release_payload",
    "_template_release_request_public_payload",
    "_template_revision_public_payload",
    "_write_json_atomic",
    "_write_release_request_file",
}

TEMPLATE_SUPPORT_HELPERS = {
    "load_field_site_profile",
    "_clean_mapping",
    "_clean_nested_mapping",
    "_dedupe_env_references",
    "_delivery_namespace",
    "_delivery_tenant_id",
    "_env_reference",
    "_find_template_path",
    "_is_semver",
    "_non_empty_text",
    "_sha256_json",
    "_site_profile_paths",
    "_slug",
    "_stable_json",
    "_string_list",
    "_write_yaml",
    "site_profile_env_references",
}

TEMPLATE_DELIVERY_HELPERS = {
    "_customer_delivery_applicability_scope",
    "_customer_delivery_dependency_matrix",
    "_customer_delivery_out_of_scope",
    "_customer_delivery_prerequisites",
    "_customer_delivery_scenario_acceptance_criteria",
    "_customer_delivery_surface",
    "_template_delivery_checklist",
    "_template_delivery_summary",
    "_template_package_summary",
    "_unique_template_binding_values",
    "_unique_template_object_values",
}

SOLUTION_DELIVERY_READINESS_HELPERS = {
    "_delivery_gate_rollup_status",
    "_solution_delivery_customer_project_gate",
    "_solution_delivery_resource_binding_gate",
    "_solution_delivery_resource_governance_gate",
    "_solution_delivery_template_market_gate",
}

CUSTOMER_PROJECT_IMPLEMENTATION_HANDOFF_HELPERS = {
    "_customer_project_implementation_handoff",
}

CUSTOMER_PROJECT_ARTIFACT_MANIFEST_HELPERS = {
    "_customer_project_acceptance_dossier_manifest",
    "_customer_project_acceptance_dossier_payload_sha256",
    "_customer_project_package_manifest",
    "_customer_project_package_payload_sha256",
    "_customer_project_proposal_bundle_payload_sha256",
}

CUSTOMER_PROJECT_EVIDENCE_INVENTORY_HELPERS = {
    "_customer_project_evidence_inventory",
    "_evidence_file_inventory",
    "_evidence_file_modified_at",
    "_evidence_url",
}

CUSTOMER_PROJECT_PACKAGE_HTML_HELPERS = {
    "_dossier_evidence_row",
    "_dossier_gate_row",
    "_dossier_workflow_row",
    "_h",
    "_metric",
    "_render_customer_project_acceptance_dossier_html",
    "_render_customer_project_proposal_bundle_html",
    "_status_class",
}

CUSTOMER_PROJECT_PACKAGE_RULE_HELPERS = {
    "_customer_project_package_action_plan",
    "_customer_project_package_delivery_gate",
    "_customer_project_package_import_gate_result",
}

CUSTOMER_PROJECT_PACKAGE_ASSESSMENT_HELPERS = {
    "_customer_project_package_acceptance_summary",
    "_customer_project_package_reuse_assessment",
    "_customer_project_reuse_dependencies",
    "_managed_object_acceptance_summary",
    "_managed_object_binding_readiness_summary",
}

CUSTOMER_PROJECT_SCOPE_HELPERS = {
    "_customer_delivery_filename_parts",
    "_customer_project_profile_diff",
    "_delivery_scope_payload",
    "_delivery_scope_payload_from_customer_site",
    "_get_nested",
    "_same_customer_project_identity",
    "_same_delivery_project_scope",
}

CUSTOMER_PROJECT_PROFILE_STORE_HELPERS = {
    "archive_customer_project_profile",
    "customer_project_catalog_acceptance_gate",
    "customer_project_catalog_summary_from_projects",
    "find_site_profile_path",
    "list_customer_project_revisions",
    "_append_object_change_log",
    "_customer_project_catalog_delivery_acceptance_gate",
    "_customer_project_catalog_filters",
    "_customer_payload",
    "_customer_profile_path",
    "_customer_profile_target",
    "_customer_project_collision_candidates",
    "_customer_project_delivery_status",
    "_customer_project_matches_filters",
    "_customer_project_product_acceptance_gate",
    "_customer_project_revision_dir",
    "_customer_rows",
    "_delivery_identifier_candidates",
    "_find_customer_project_profile_path",
    "_find_customer_project_revision",
    "_load_customer_project_revisions",
    "_normalize_customer_project_profile",
    "_object_change_log_payload",
    "_object_change_summary",
    "_read_customer_project_revision_file",
    "_revision_public_payload",
    "_snapshot_customer_project_revision",
    "_text_filter_matches",
}

CUSTOMER_PROJECT_MANAGED_OBJECT_HELPERS = {
    "managed_object_catalog_from_site_profile",
    "_ACCEPTANCE_TEST_ALIASES",
    "_acceptance_node_match",
    "_acceptance_resource_bucket",
    "_acceptance_test_check",
    "_delivery_resource_link_status",
    "_managed_object_acceptance_summary",
    "_managed_object_acceptance_status",
    "_managed_object_binding_readiness_summary",
    "_managed_object_binding_missing_count",
    "_managed_object_binding_payload",
    "_managed_object_payload",
    "_managed_object_resource_binding_status",
}

CUSTOMER_PROJECT_EXECUTION_BINDING_HELPERS = {
    "build_customer_project_execution_bindings",
    "_SCENARIO_REQUIRED_INPUTS",
    "_execution_binding_customer_claim",
    "_execution_binding_next_step",
    "_execution_binding_summary",
    "_execution_check_status",
    "_execution_resource_ref",
    "_field_devices_by_source",
    "_field_ingest_adapter_contract",
    "_first_zone_for_object",
    "_managed_object_bridge_contract",
    "_managed_object_execution_binding_plan",
    "_managed_object_ingest_contract",
    "_scenario_required_inputs_for_object",
    "_sensor_protocol_adapter_name",
    "_sensor_protocol_execution_sources",
}

CUSTOMER_PROJECT_ACCEPTANCE_REGISTRY_HELPERS = {
    "build_customer_project_acceptance_registry",
    "_acceptance_registry_consumers_from_profile",
    "_acceptance_registry_next_step",
    "_acceptance_registry_references",
    "_acceptance_registry_status_bucket",
    "_acceptance_registry_summary",
    "_merge_acceptance_registry_status",
}

FIELD_SITE_RUNTIME_CONFIG_HELPERS = {
    "field_operations_config_from_site_profile",
    "render_site_profile_env_template",
    "_device_registry_entry",
    "_env_placeholder",
    "_field_threshold_config",
}

FIELD_SITE_VALIDATION_HELPERS = {
    "REQUIRED_DEVICE_SOURCES",
    "REQUIRED_RESPONDER_GROUPS",
    "validate_field_site_profile",
    "_require_env_reference",
    "_validate_customer_project",
    "_validate_managed_object_bindings",
    "_validate_managed_objects",
    "_validate_thresholds",
    "_zones_by_type",
}

FIELD_SITE_CATALOG_HELPERS = {
    "build_customer_project_catalog",
    "build_site_profile_catalog",
    "build_site_profile_report",
    "_customer_project_delivery_workflow",
    "_customer_project_summary",
    "_site_catalog_next_step",
    "_site_customer_status",
    "_site_deployment_stage",
    "_site_profile_catalog_item",
    "_site_profile_next_step",
}

CUSTOMER_PROJECT_ACCEPTANCE_HELPERS = {
    "_FIELD_READINESS_EVIDENCE_DEFAULTS",
    "ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES",
    "ONSITE_ACCEPTANCE_EVIDENCE_TYPES",
    "ONSITE_ACCEPTANCE_STATUSES",
    "ACCEPTANCE_REVIEW_DECISIONS",
    "CUSTOMER_SIGNOFF_DECISIONS",
    "list_customer_project_onsite_evidence",
    "register_customer_project_onsite_evidence",
    "customer_project_acceptance_closure",
    "register_customer_project_acceptance_review",
    "list_customer_project_customer_signoffs",
    "register_customer_project_customer_signoff",
    "customer_project_acceptance_report",
    "_customer_project_launch_readiness",
    "_launch_readiness_gate",
    "_customer_project_launch_gate_status",
    "_customer_project_launch_gate_next_step",
    "_customer_project_launch_rollup_status",
    "_execution_binding_report_contracts",
    "_customer_project_field_readiness",
    "_compact_field_readiness",
    "_compact_evidence_report",
    "_customer_project_raw_onsite_evidence",
    "_customer_project_onsite_evidence_payload",
    "_customer_project_onsite_evidence_payload_from_receipts",
    "_customer_project_auto_onsite_evidence_receipts",
    "_customer_project_auto_onsite_evidence_receipt",
    "_customer_project_onsite_evidence_receipts",
    "_customer_project_onsite_evidence_summary",
    "_customer_project_raw_acceptance_reviews",
    "_customer_project_acceptance_reviews",
    "_customer_project_acceptance_review_gate",
    "_customer_project_raw_customer_signoffs",
    "_customer_project_customer_signoffs",
    "_customer_project_customer_signoff_gate_snapshot",
    "_customer_project_customer_signoff_handoff_materials",
    "_customer_project_customer_signoff_payload_sha256",
    "_customer_project_customer_signoff_gate",
    "_customer_project_acceptance_evidence_timeline",
    "_customer_project_acceptance_closure_next_step",
    "_customer_project_latest_proposal_verification",
    "_customer_project_latest_audit_export",
    "_recent_json_files",
    "_read_json_file",
    "_audit_manifest_matches_scope",
    "_audit_records_hash_matches",
    "_normalize_onsite_evidence_type",
    "_normalize_onsite_evidence_status",
    "_normalize_acceptance_review_decision",
    "_normalize_customer_signoff_decision",
    "_normalize_sha256_hex",
    "_float_value",
    "_customer_project_field_readiness_gates",
    "_readiness_status",
    "_boolean_gate_status",
    "_reports_evidence",
    "_customer_project_site_acceptance_checklist",
    "_latest_onsite_receipts_by_type",
    "_onsite_acceptance_checklist_item",
    "_build_customer_project_acceptance_dossier",
    "_customer_project_acceptance_dossier_verification",
    "verify_customer_project_proposal_bundle",
    "verify_customer_project_acceptance_dossier",
}

CUSTOMER_PROJECT_PROFILE_OPERATION_HELPERS = {
    "create_customer_project_from_template",
    "upsert_customer_project_profile",
    "get_customer_project_profile",
    "upsert_managed_object",
    "delete_managed_object",
    "rollback_customer_project_profile",
}

CUSTOMER_PROJECT_PUBLIC_NAMES = {
    "archive_customer_project_profile",
    "build_customer_project_acceptance_registry",
    "build_customer_project_catalog",
    "build_customer_project_execution_bindings",
    "build_customer_project_resource_catalog",
    "build_site_profile_catalog",
    "build_site_profile_report",
    "build_solution_delivery_readiness",
    "customer_project_acceptance_closure",
    "customer_project_acceptance_report",
    "customer_project_catalog_acceptance_gate",
    "customer_project_catalog_summary_from_projects",
    "delete_managed_object",
    "get_customer_project_profile",
    "list_customer_project_customer_signoffs",
    "list_customer_project_onsite_evidence",
    "list_customer_project_revisions",
    "register_customer_project_acceptance_review",
    "register_customer_project_customer_signoff",
    "register_customer_project_onsite_evidence",
    "rollback_customer_project_profile",
    "upsert_customer_project_profile",
    "upsert_managed_object",
}

CUSTOMER_PROJECT_ARTIFACT_PUBLIC_NAMES = {
    "diff_customer_project_package",
    "export_customer_project_acceptance_dossier",
    "export_customer_project_package",
    "export_customer_project_proposal_bundle",
    "export_customer_project_template_release_notes_bundle",
    "import_customer_project_package",
    "verify_customer_project_acceptance_dossier",
    "verify_customer_project_package",
    "verify_customer_project_proposal_bundle",
}

CUSTOMER_PROJECT_ARTIFACT_KERNEL_HELPERS = {
    "diff_customer_project_package",
    "export_customer_project_acceptance_dossier",
    "export_customer_project_package",
    "export_customer_project_proposal_bundle",
    "import_customer_project_package",
    "verify_customer_project_package",
}


def _assert_facade_exports(facade, canonical) -> None:
    assert facade.__all__, facade.__name__
    for name in facade.__all__:
        assert getattr(facade, name) is getattr(canonical, name), name


def test_package_facades_are_utf8_without_bom() -> None:
    for path in PACKAGE_FACADE_FILES:
        data = path.read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf"), str(path)
        ast.parse(data.decode("utf-8"), filename=str(path))


def test_package_readmes_document_current_owner_subpackages() -> None:
    for package_name in PACKAGE_LAYOUT_READMES:
        package_path = Path(package_name.replace(".", "/"))
        readme = (package_path / "README.md").read_text(encoding="utf-8")
        subpackages = {
            path.name
            for path in package_path.iterdir()
            if path.is_dir() and path.name != "__pycache__"
        }

        assert subpackages, package_name
        for subpackage in sorted(subpackages):
            documented = f"`{subpackage}`" in readme or f"`{subpackage}/`" in readme
            assert documented, f"{package_path / 'README.md'} missing {subpackage}/"


def test_legacy_alias_package_readmes_document_compatibility_contract() -> None:
    migration_terms = (
        "legacy imports",
        "historical imports",
        "compatibility aliases",
        "compatibility facades",
    )

    for package_name in PACKAGES_WITH_LEGACY_ALIASES:
        readme_path = Path(package_name.replace(".", "/")) / "README.md"
        readme = readme_path.read_text(encoding="utf-8").lower()

        assert any(term in readme for term in migration_terms), (
            f"{readme_path} must document the legacy import compatibility contract"
        )


def test_multi_agent_docs_reference_existing_verification_targets() -> None:
    docs = (
        Path("docs/MULTI_AGENT_WORKFLOW.md"),
        Path("docs/MODULE_OWNERSHIP.md"),
    )
    required_lanes = (
        "Runtime / blueprints",
        "Voice gateway / interaction",
        "API / MCP / tools",
        "Providers / ports",
        "Product workflows",
        "Migration compatibility",
        "Test hardening",
    )

    for doc_path in docs:
        text = doc_path.read_text(encoding="utf-8")
        assert "pytest " in text, f"{doc_path} must provide copyable pytest commands"
        for test_path in sorted(set(re.findall(r"tests[\\/][A-Za-z0-9_./\\-]+\.py", text))):
            normalized = test_path.replace("\\", "/")
            assert Path(normalized).is_file(), f"{doc_path} references missing {test_path}"

    ownership_text = Path("docs/MODULE_OWNERSHIP.md").read_text(encoding="utf-8")
    workflow_text = Path("docs/MULTI_AGENT_WORKFLOW.md").read_text(encoding="utf-8")
    for lane in required_lanes:
        assert lane in workflow_text, f"workflow doc missing lane {lane}"
        assert lane in ownership_text, f"ownership doc missing lane {lane}"


def test_product_package_roots_do_not_regrow_script_piles() -> None:
    """Product packages must keep implementation ownership inside subpackages."""

    ignored_roots = {
        "__pycache__",
        "data",
        "static",
    }
    violations: list[str] = []
    for package_path in sorted(Path("askme").iterdir()):
        if not package_path.is_dir() or package_path.name in ignored_roots:
            continue
        implementation_files = sorted(
            path.name
            for path in package_path.glob("*.py")
            if path.name != "__init__.py"
        )
        if len(implementation_files) > 8:
            violations.append(
                f"{package_path}: {len(implementation_files)} root implementation files "
                f"({', '.join(implementation_files)}); split by owner subpackage"
            )

    assert violations == []


def test_lower_level_capability_packages_do_not_import_blueprints() -> None:
    """Blueprints are the composition root; lower-level capability packages stay reusable."""

    lower_level_roots = (
        Path("askme/cognition"),
        Path("askme/llm"),
        Path("askme/memory"),
        Path("askme/perception"),
        Path("askme/ports"),
        Path("askme/providers"),
        Path("askme/robot"),
        Path("askme/robot_interaction"),
        Path("askme/runtime"),
        Path("askme/skills"),
        Path("askme/tools"),
        Path("askme/voice"),
        Path("askme/voice_gateway"),
    )
    violations: list[str] = []
    for root in lower_level_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    if module_name == "askme.blueprints" or module_name.startswith(
                        "askme.blueprints."
                    ):
                        violations.append(f"{path}:{node.lineno} imports {module_name}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "askme.blueprints" or alias.name.startswith(
                            "askme.blueprints."
                        ):
                            violations.append(f"{path}:{node.lineno} imports {alias.name}")

    assert violations == []


def test_llm_root_files_are_compat_facades_not_new_implementation() -> None:
    allowed_root_files = {
        Path("askme/llm/__init__.py"),
        *LEGACY_LLM_FACADE_FILES,
    }
    current_root_files = set(Path("askme/llm").glob("*.py"))

    assert current_root_files == allowed_root_files
    for path in sorted(LEGACY_LLM_FACADE_FILES):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        function_defs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        assert function_defs == [], f"{path} should stay a compatibility facade"
        assert "__all__" in source or "sys.modules[__name__]" in source


def test_root_lazy_facades_export_and_resolve_public_contracts() -> None:
    for package_name in ROOT_LAZY_FACADE_PACKAGES:
        package = importlib.import_module(package_name)
        lazy_exports = getattr(package, "_LAZY_EXPORTS")
        assert package.__all__ == sorted(lazy_exports)
        for public_name, (module_name, attr_name) in lazy_exports.items():
            canonical = getattr(importlib.import_module(module_name), attr_name)
            assert getattr(package, public_name) is canonical


def test_owner_subpackage_lazy_facades_resolve_public_contracts() -> None:
    facade_names = (
        "askme.blueprints.catalog",
        "askme.blueprints.presets",
        "askme.blueprints.runner",
        "askme.cognition.memory",
        "askme.cognition.perception",
        "askme.cognition.planning",
        "askme.cognition.world",
        "askme.memory.core",
        "askme.memory.retrieval",
        "askme.memory.intelligence",
        "askme.memory.backends",
        "askme.robot.arm",
        "askme.robot.dog",
        "askme.robot.indicators",
        "askme.robot.telemetry",
        "askme.runtime.core",
        "askme.runtime.task",
        "askme.skills.core",
        "askme.skills.catalog",
        "askme.skills.contracts",
        "askme.skills.governance",
        "askme.tools.core",
        "askme.tools.field",
        "askme.tools.robot",
        "askme.tools.skills",
        "askme.tools.spatial",
        "askme.tools.voice",
        "askme.voice.core",
        "askme.voice.diagnostics",
        "askme.voice.input",
        "askme.voice.interaction",
        "askme.voice.orchestration",
        "askme.voice.output",
    )

    for facade_name in facade_names:
        facade = importlib.import_module(facade_name)
        lazy_exports = getattr(facade, "_LAZY_EXPORTS")
        assert facade.__all__ == sorted(lazy_exports)
        for public_name, (module_name, attr_name) in lazy_exports.items():
            canonical = getattr(importlib.import_module(module_name), attr_name)
            assert getattr(facade, public_name) is canonical


def test_owner_subpackage_facades_do_not_eagerly_import_heavy_backends() -> None:
    code = """
import importlib
import json
import sys

for module_name in (
    "askme.blueprints.catalog",
    "askme.blueprints.presets",
    "askme.blueprints.runner",
    "askme.cognition.memory",
    "askme.cognition.perception",
    "askme.cognition.planning",
    "askme.cognition.world",
    "askme.memory.core",
    "askme.memory.retrieval",
    "askme.memory.intelligence",
    "askme.memory.backends",
    "askme.robot.arm",
    "askme.robot.dog",
    "askme.robot.indicators",
    "askme.robot.telemetry",
    "askme.runtime.core",
    "askme.runtime.task",
    "askme.skills.core",
    "askme.skills.catalog",
    "askme.skills.contracts",
    "askme.skills.governance",
    "askme.tools.core",
    "askme.tools.field",
    "askme.tools.robot",
    "askme.tools.skills",
    "askme.tools.spatial",
    "askme.tools.voice",
    "askme.voice.core",
    "askme.voice.diagnostics",
    "askme.voice.input",
    "askme.voice.interaction",
    "askme.voice.orchestration",
    "askme.voice.output",
):
    importlib.import_module(module_name)

forbidden = {
    "cyclonedds",
    "askme.memory.backends.robotmem_backend",
    "askme.memory.backends.mempalace_backend",
    "onnxruntime",
    "sentence_transformers",
    "sherpa_onnx",
    "sounddevice",
}
loaded = sorted(name for name in forbidden if name in sys.modules)
print(json.dumps({"loaded": loaded}, ensure_ascii=False))
raise SystemExit(1 if loaded else 0)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_legacy_alias_manifest_is_locked_to_migration_scope() -> None:
    assert set(EXPECTED_LEGACY_ALIAS_KEYS) == set(PACKAGES_WITH_LEGACY_ALIASES)

    for package_name, expected_keys in EXPECTED_LEGACY_ALIAS_KEYS.items():
        package = importlib.import_module(package_name)
        aliases = getattr(package, "_LEGACY_MODULE_ALIASES")
        assert set(aliases) == set(expected_keys), package_name


def test_legacy_aliases_import_in_cold_python_process() -> None:
    code = """
import importlib
import json
import sys

manifest = json.loads(sys.argv[1])
for package_name, expected_keys in manifest["aliases"].items():
    package = importlib.import_module(package_name)
    aliases = getattr(package, "_LEGACY_MODULE_ALIASES")
    if set(aliases) != set(expected_keys):
        raise SystemExit(f"{package_name} alias manifest drifted")
    for legacy_name in expected_keys:
        canonical_name = aliases[legacy_name]
        canonical = importlib.import_module(canonical_name)
        legacy = importlib.import_module(legacy_name)
        if legacy is not canonical:
            raise SystemExit(f"{legacy_name} did not alias {canonical_name}")

for module_name in manifest["collision_facades"]:
    module = importlib.import_module(module_name)
    if not getattr(module, "__all__", None):
        raise SystemExit(f"{module_name} has no public facade exports")
"""
    manifest = {
        "aliases": EXPECTED_LEGACY_ALIAS_KEYS,
        "collision_facades": PACKAGE_COLLISION_FACADE_MODULES,
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, json.dumps(manifest)],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_legacy_module_aliases_import_old_and_new_paths() -> None:
    for package_name in PACKAGES_WITH_LEGACY_ALIASES:
        package = importlib.import_module(package_name)
        aliases = getattr(package, "_LEGACY_MODULE_ALIASES")
        assert aliases, package_name

        for legacy_name, canonical_name in aliases.items():
            canonical = importlib.import_module(canonical_name)
            legacy = importlib.import_module(legacy_name)
            assert legacy is canonical, f"{legacy_name} should alias {canonical_name}"
            assert legacy.__spec__ is not None
            assert legacy.__spec__.name == canonical_name


def test_legacy_facade_registry_documents_confusing_entrypoints() -> None:
    from askme.compat import LEGACY_FACADES, legacy_facade_for

    required_paths = {
        "askme.voice.runtime_bridge",
        "askme.voice.orchestration.runtime_bridge",
        "askme.voice_gateway.runtime_bridge",
        "askme.voice.input.address_detector",
        "askme.voice.interaction.interaction_gate",
        "askme.voice.interaction.perception_context",
        "askme.interaction.intent_router",
        "askme.pipeline.reactions.state_led_bridge",
        "askme.robot.telemetry.ota_bridge",
        "askme.robot.telemetry.pubsub",
    }
    paths = {item.legacy_path for item in LEGACY_FACADES}

    assert required_paths <= paths
    assert len(paths) == len(LEGACY_FACADES)

    for item in LEGACY_FACADES:
        assert legacy_facade_for(item.legacy_path) == item
        assert item.canonical_path
        assert item.new_code_import
        assert item.owner
        assert item.reason
        importlib.import_module(item.legacy_path)
        importlib.import_module(item.canonical_path)


def test_legacy_facade_modules_stay_thin() -> None:
    from askme.compat import LEGACY_FACADES

    violations: list[str] = []
    for item in LEGACY_FACADES:
        spec = importlib.util.find_spec(item.legacy_path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            continue
        path = Path(spec.origin)
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        implementation_nodes = (
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.FunctionDef,
        )
        for node in ast.walk(tree):
            if isinstance(node, implementation_nodes):
                if isinstance(node, ast.FunctionDef) and node.name in {
                    "__dir__",
                    "__getattr__",
                }:
                    continue
                violations.append(
                    f"{path.relative_to(Path.cwd())}:{node.lineno} defines {node.name}"
                )

    assert violations == []


def test_blueprint_operational_legacy_aliases_keep_module_entrypoints() -> None:
    for module_name, label in (
        ("askme.blueprints.edge_robot", "园区巡检机器人运行时"),
        ("askme.blueprints.voice", "语音任务中心"),
    ):
        result = subprocess.run(
            [sys.executable, "-B", "-m", module_name, "--help"],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )

        assert result.returncode == 0, result.stderr or result.stdout
        assert label in result.stdout
        assert "用途：启动一个 askme 产品蓝图" in result.stdout
        assert "启动命令：" in result.stdout
        assert "交付检查：" in result.stdout
        assert "Usage:" not in result.stdout
        assert "Options:" not in result.stdout
        assert "python -m askme runtime blueprints --help" in result.stdout


def test_legacy_module_aliases_are_available_as_package_attributes() -> None:
    for package_name in PACKAGES_WITH_LEGACY_ALIASES:
        package = importlib.import_module(package_name)
        aliases = getattr(package, "_LEGACY_MODULE_ALIASES")

        for legacy_name, canonical_name in aliases.items():
            attribute = legacy_name.rsplit(".", 1)[-1]
            canonical = importlib.import_module(canonical_name)
            assert getattr(package, attribute) is canonical


def test_root_lazy_exports_preserve_optional_dependency_fallback(monkeypatch) -> None:
    import askme.pipeline as pipeline_pkg
    import askme.voice as voice_pkg

    sentinel = object()
    old_asr = voice_pkg.__dict__.pop("ASREngine", sentinel)
    old_voice_loop = pipeline_pkg.__dict__.pop("VoiceLoop", sentinel)

    def voice_import_module(name: str):
        if name == "askme.voice.input.asr":
            raise ModuleNotFoundError(
                "No module named 'sherpa_onnx'",
                name="sherpa_onnx",
            )
        return importlib.import_module(name)

    def pipeline_import_module(name: str):
        if name == "askme.pipeline.channels.voice_loop":
            raise ModuleNotFoundError(
                "No module named 'sounddevice'",
                name="sounddevice",
            )
        return importlib.import_module(name)

    try:
        monkeypatch.setattr(voice_pkg, "import_module", voice_import_module)
        monkeypatch.setattr(pipeline_pkg, "import_module", pipeline_import_module)

        assert voice_pkg.ASREngine is None
        assert pipeline_pkg.VoiceLoop is None
    finally:
        voice_pkg.__dict__.pop("ASREngine", None)
        pipeline_pkg.__dict__.pop("VoiceLoop", None)
        if old_asr is not sentinel:
            voice_pkg.ASREngine = old_asr
        if old_voice_loop is not sentinel:
            pipeline_pkg.VoiceLoop = old_voice_loop


def test_root_lazy_exports_raise_unknown_nested_import_failures(monkeypatch) -> None:
    import askme.pipeline as pipeline_pkg
    import askme.voice as voice_pkg

    sentinel = object()
    old_asr = voice_pkg.__dict__.pop("ASREngine", sentinel)
    old_voice_loop = pipeline_pkg.__dict__.pop("VoiceLoop", sentinel)

    def voice_import_module(name: str):
        if name == "askme.voice.input.asr":
            raise ModuleNotFoundError(
                "No module named 'askme.voice.input.internal_missing'",
                name="askme.voice.input.internal_missing",
            )
        return importlib.import_module(name)

    def pipeline_import_module(name: str):
        if name == "askme.pipeline.channels.voice_loop":
            raise ModuleNotFoundError(
                "No module named 'askme.pipeline.channels.internal_missing'",
                name="askme.pipeline.channels.internal_missing",
            )
        return importlib.import_module(name)

    try:
        monkeypatch.setattr(voice_pkg, "import_module", voice_import_module)
        monkeypatch.setattr(pipeline_pkg, "import_module", pipeline_import_module)

        with pytest.raises(ModuleNotFoundError):
            _ = voice_pkg.ASREngine
        with pytest.raises(ModuleNotFoundError):
            _ = pipeline_pkg.VoiceLoop
    finally:
        voice_pkg.__dict__.pop("ASREngine", None)
        pipeline_pkg.__dict__.pop("VoiceLoop", None)
        if old_asr is not sentinel:
            voice_pkg.ASREngine = old_asr
        if old_voice_loop is not sentinel:
            pipeline_pkg.VoiceLoop = old_voice_loop


def test_tool_type_checking_imports_use_canonical_owner_paths() -> None:
    expected_imports = {
        Path("askme/tools/robot/robot_tools.py"): {
            "askme.ports",
        },
        Path("askme/tools/skills/skill_tools.py"): {
            "askme.robot_interaction",
            "askme.skills.core.skill_manager",
        },
        Path("askme/tools/voice/voice_tools.py"): {
            "askme.ports",
        },
    }

    for path, modules in expected_imports.items():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        relative_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level > 0
        ]
        assert modules <= imported
        assert relative_imports == []


def test_product_code_uses_canonical_import_paths_after_package_split() -> None:
    legacy_aliases: dict[str, str] = {}
    for package_name in PACKAGES_WITH_LEGACY_ALIASES:
        package = importlib.import_module(package_name)
        legacy_aliases.update(getattr(package, "_LEGACY_MODULE_ALIASES"))

    violations: list[str] = []
    for path in Path("askme").rglob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                for legacy_name, canonical_name in legacy_aliases.items():
                    if node.module == legacy_name or node.module.startswith(f"{legacy_name}."):
                        imported_names = ", ".join(alias.name for alias in node.names)
                        violations.append(
                            f"{path}:{node.lineno} imports {imported_names} "
                            f"from legacy {node.module}; use {canonical_name}"
                        )
                        break
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for legacy_name, canonical_name in legacy_aliases.items():
                        if alias.name == legacy_name or alias.name.startswith(f"{legacy_name}."):
                            violations.append(
                                f"{path}:{node.lineno} imports legacy {alias.name}; "
                                f"use {canonical_name}"
                            )
                            break

    assert violations == []


def test_product_code_does_not_import_legacy_pubsub_facade() -> None:
    forbidden = "askme.robot.telemetry.pubsub"
    allowed_files = {
        Path("askme/compat/legacy_facades.py"),
        Path("askme/robot/__init__.py"),
        Path("askme/robot/telemetry/__init__.py"),
        Path("askme/robot/telemetry/pubsub.py"),
    }

    violations: list[str] = []
    for path in Path("askme").rglob("*.py"):
        if path in allowed_files:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module == forbidden or node.module.startswith(f"{forbidden}."):
                    imported_names = ", ".join(alias.name for alias in node.names)
                    violations.append(
                        f"{path}:{node.lineno} imports {imported_names} "
                        f"from legacy {node.module}; use askme.interfaces.bus.BusBackend"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == forbidden or alias.name.startswith(f"{forbidden}."):
                        violations.append(
                            f"{path}:{node.lineno} imports legacy {alias.name}; "
                            "use askme.interfaces.bus.BusBackend"
                        )

    assert violations == []


def test_public_catalog_and_governance_exports_are_stable() -> None:
    catalog = importlib.import_module("askme.blueprints.catalog")
    governance = importlib.import_module("askme.skills.governance")

    for name in (
        "ALIASES",
        "TEXT_MODULES",
        "VOICE_MODULES",
        "VOICE_PERCEPTION_MODULES",
        "EDGE_ROBOT_MODULES",
        "MCP_MODULES",
        "LINGTU_VOICE_MODULES",
    ):
        assert hasattr(catalog, name), name

    assert hasattr(governance, "SkillGovernanceRecord")
    assert hasattr(governance, "SkillGovernanceStore")


def test_package_collision_legacy_paths_export_public_symbols() -> None:
    matrix = {
        "askme.blueprints.catalog": (
            "ALIASES",
            "BlueprintSpec",
            "catalog_payload",
            "list_blueprints",
        ),
        "askme.skills.contracts": (
            "SkillContract",
            "SkillContractRegistry",
            "registered_skill_contracts",
            "skill_contract",
        ),
        "askme.skills.governance": (
            "APPROVED",
            "SkillGovernanceRecord",
            "SkillGovernanceStore",
        ),
    }

    for module_name, symbols in matrix.items():
        module = importlib.import_module(module_name)
        exported = set(getattr(module, "__all__", ()))
        assert exported, module_name
        for symbol in symbols:
            namespace: dict[str, object] = {}
            exec(f"from {module_name} import {symbol} as imported", namespace)
            assert symbol in exported
            assert namespace["imported"] is getattr(module, symbol)


def test_pipeline_owner_subpackage_facades_export_expected_entrypoints() -> None:
    matrix = {
        "askme.pipeline.channels": {
            "CommandHandler": ("askme.pipeline.channels.commands", "CommandHandler"),
            "TextLoop": ("askme.pipeline.channels.text_loop", "TextLoop"),
            "VoiceLoop": ("askme.pipeline.channels.voice_loop", "VoiceLoop"),
            "record_external_turn": (
                "askme.pipeline.channels.external_turns",
                "record_external_turn",
            ),
        },
        "askme.pipeline.core": {
            "BrainPipeline": ("askme.pipeline.core.brain_pipeline", "BrainPipeline"),
            "PipelineHooks": ("askme.pipeline.core.hooks", "PipelineHooks"),
            "PromptBuilder": ("askme.pipeline.core.prompt_builder", "PromptBuilder"),
            "StreamProcessor": ("askme.pipeline.core.stream_processor", "StreamProcessor"),
            "ToolExecutor": ("askme.pipeline.core.tool_executor", "ToolExecutor"),
            "TurnContext": ("askme.pipeline.core.protocols", "TurnContext"),
            "TurnExecutor": ("askme.pipeline.core.turn_executor", "TurnExecutor"),
            "get_tracer": ("askme.pipeline.core.trace", "get_tracer"),
            "strip_think_blocks": ("askme.pipeline.core.utils", "strip_think_blocks"),
        },
        "askme.pipeline.skills": {
            "MissionContext": ("askme.pipeline.skills.skill_dispatcher", "MissionContext"),
            "MissionState": ("askme.pipeline.skills.skill_dispatcher", "MissionState"),
            "PlanStep": ("askme.pipeline.skills.planner_agent", "PlanStep"),
            "PlannerAgent": ("askme.pipeline.skills.planner_agent", "PlannerAgent"),
            "SkillDispatcher": ("askme.pipeline.skills.skill_dispatcher", "SkillDispatcher"),
            "SkillGate": ("askme.pipeline.skills.skill_gate", "SkillGate"),
        },
        "askme.pipeline.reactions": {
            "HybridReaction": ("askme.pipeline.reactions.reaction_engine", "HybridReaction"),
            "LLMReaction": ("askme.pipeline.reactions.reaction_engine", "LLMReaction"),
            "ProactiveAgent": ("askme.pipeline.reactions.proactive_agent", "ProactiveAgent"),
            "RuleBasedReaction": (
                "askme.pipeline.reactions.reaction_engine",
                "RuleBasedReaction",
            ),
            "StateLedBridge": (
                "askme.pipeline.reactions.state_led_bridge",
                "StateLedBridge",
            ),
            "evaluate_rules": ("askme.pipeline.reactions.reaction_engine", "evaluate_rules"),
        },
    }

    for facade_name, symbols in matrix.items():
        facade = importlib.import_module(facade_name)
        exported = set(getattr(facade, "__all__", ()))
        assert set(symbols) <= exported
        for symbol, (module_name, attr_name) in symbols.items():
            canonical = getattr(importlib.import_module(module_name), attr_name)
            assert getattr(facade, symbol) is canonical


def test_brain_pipeline_core_does_not_import_skill_implementation() -> None:
    tree = ast.parse(
        Path("askme/pipeline/core/brain_pipeline.py").read_text(encoding="utf-8-sig")
    )
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)

    assert not any(
        module.startswith("askme.pipeline.skills") for module in imports
    ), "pipeline.core must receive skill gates by protocol injection"


def test_default_interface_registration_delegates_provider_backends() -> None:
    tree = ast.parse(
        Path("askme/interfaces/register_defaults.py").read_text(encoding="utf-8")
    )
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)

    assert "askme.voice.asr" not in imports
    assert "askme.voice.input.asr" not in imports
    assert "askme.voice.output.tts" not in imports
    assert "askme.providers.register_defaults" in imports

    provider_source = Path("askme/providers/register_defaults.py").read_text(
        encoding="utf-8"
    )
    assert "askme.voice.asr" not in provider_source
    assert "askme.voice.input.asr" in provider_source
    assert "askme.voice.output.tts" in provider_source


def test_api_package_exports_product_app_factory() -> None:
    api = importlib.import_module("askme.api")
    server = importlib.import_module("askme.api.server")

    assert api.create_api_app is server.create_api_app
    assert api.create_health_app is server.create_api_app
    assert api.create_product_app is server.create_product_app
    assert api.create_product_app is server.create_api_app


def test_api_mission_routes_live_outside_health_server() -> None:
    health_source = Path("askme/health_server.py").read_text(encoding="utf-8")
    composition_source = Path("askme/api/composition.py").read_text(encoding="utf-8")
    product_surface_source = Path("askme/api/product/routes.py").read_text(encoding="utf-8")
    route_source = Path("askme/api/routes/mission.py").read_text(encoding="utf-8")

    assert "from askme.api.composition import ApiRouteDependencies, register_api_routes" in health_source
    assert "register_api_routes(" in health_source
    assert "register_mission_routes(" not in health_source
    assert 'registrar="register_product_routes"' in composition_source
    assert "register_mission_routes(" in product_surface_source
    assert '@app.post("/api/missions' not in health_source
    assert '@app.get("/api/missions' not in health_source
    assert '@app.options("/api/missions' not in health_source
    assert 'app.post("/api/missions/draft"' in route_source
    assert 'app.get("/api/missions/{mission_id}/report"' in route_source


def test_api_route_composition_owns_product_route_registration() -> None:
    health_source = Path("askme/health_server.py").read_text(encoding="utf-8")
    composition_source = Path("askme/api/composition.py").read_text(encoding="utf-8")
    product_surface_source = Path("askme/api/product/routes.py").read_text(encoding="utf-8")
    admin_surface_source = Path("askme/api/admin/routes.py").read_text(encoding="utf-8")
    internal_surface_source = Path("askme/api/internal/routes.py").read_text(encoding="utf-8")
    platform_surface_source = Path("askme/api/platform/routes.py").read_text(encoding="utf-8")

    assert "from askme.api.routes." not in health_source
    assert "register_api_routes(" in health_source
    assert 'registrar="register_platform_routes"' in composition_source
    assert 'registrar="register_product_routes"' in composition_source
    assert 'registrar="register_admin_routes"' in composition_source
    assert 'registrar="register_internal_routes"' in composition_source

    route_surface_sources = {
        "register_system_routes": platform_surface_source,
        "register_monitor_routes": platform_surface_source,
        "register_memory_routes": product_surface_source,
        "register_voice_routes": product_surface_source,
        "register_space_routes": product_surface_source,
        "register_field_surface_routes": product_surface_source,
        "register_dashboard_routes": product_surface_source,
        "register_capability_routes": product_surface_source,
        "register_conversation_routes": product_surface_source,
        "register_mission_routes": product_surface_source,
        "register_governance_routes": admin_surface_source,
        "register_audit_routes": admin_surface_source,
        "register_skill_routes": admin_surface_source,
        "register_agent_profile_routes": admin_surface_source,
        "register_cognition_routes": internal_surface_source,
        "register_runtime_routes": internal_surface_source,
        "register_vision_routes": internal_surface_source,
    }
    for route_registrar, surface_source in route_surface_sources.items():
        assert route_registrar not in health_source
        assert route_registrar not in composition_source
        assert route_registrar in surface_source


def test_api_routes_package_does_not_export_surface_bypasses() -> None:
    routes_package = importlib.import_module("askme.api.routes")
    source = Path("askme/api/routes/__init__.py").read_text(encoding="utf-8")

    assert routes_package.__all__ == []
    assert "internal implementation files" in source


def test_api_route_json_body_reads_are_object_validated() -> None:
    """All route JSON bodies must fail as 400 before business dispatch."""
    route_root = Path("askme/api/routes")
    offenders: list[str] = []
    for path in sorted(route_root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        optional_reads = source.count("optional_json_body(request)")
        value_error_handlers = source.count("except ValueError as exc:")
        if value_error_handlers < optional_reads:
            offenders.append(
                f"{path}: {optional_reads} optional_json_body reads but "
                f"{value_error_handlers} ValueError handlers"
            )

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            function_source = ast.get_source_segment(source, node) or ""
            if "await request.json()" not in function_source:
                continue
            if node.name.startswith("_"):
                continue
            if "require_json_object(await request.json())" not in function_source:
                offenders.append(f"{path}:{node.lineno} {node.name} reads raw request.json")
            if "except ValueError as exc:" not in function_source:
                offenders.append(f"{path}:{node.lineno} {node.name} lacks ValueError -> 400")

    assert offenders == []


def test_api_surface_manifest_is_product_boundary_contract() -> None:
    composition_module = importlib.import_module("askme.api.composition")
    manifest = composition_module.api_surface_manifest()

    assert [item["name"] for item in manifest] == ["platform", "product", "admin", "internal"]
    assert [item["registrar"] for item in manifest] == [
        "register_platform_routes",
        "register_product_routes",
        "register_admin_routes",
        "register_internal_routes",
    ]

    by_name = {item["name"]: item for item in manifest}
    assert by_name["product"]["audience"] == "customer dashboard and operator workflows"
    assert by_name["product"]["customer_visible"] is True
    assert by_name["product"]["hardware_authority_allowed"] is False
    assert by_name["product"]["production_claim_allowed"] is False
    assert "不能直接暴露硬件控制" in by_name["product"]["customer_boundary"]
    assert "customer knowledge" in by_name["product"]["owns"]
    assert "field events" in by_name["product"]["owns"]
    assert "direct hardware authority" in by_name["product"]["must_not_expose"]
    assert "raw handoff internals" in by_name["product"]["must_not_expose"]
    assert by_name["internal"]["customer_visible"] is False
    assert by_name["internal"]["hardware_authority_allowed"] is True
    assert by_name["internal"]["production_claim_allowed"] is False
    assert "runtime callbacks" in by_name["internal"]["owns"]
    assert "device onboarding evidence" in by_name["internal"]["owns"]
    assert "robot bridges" in by_name["internal"]["owns"]
    assert "skill governance" in by_name["admin"]["owns"]
    assert "health" in by_name["platform"]["owns"]
    assert "askme.api.routes.memory" in by_name["product"]["route_modules"]
    assert "askme.api.routes.runtime" in by_name["internal"]["route_modules"]
    assert "askme.api.routes.field_internal" in by_name["internal"]["route_modules"]
    assert "askme.api.routes.audit" in by_name["admin"]["route_modules"]
    assert "askme.api.routes.system" in by_name["platform"]["route_modules"]


def test_every_route_module_with_fastapi_decorators_is_surface_classified() -> None:
    composition_module = importlib.import_module("askme.api.composition")
    module_map = composition_module.api_surface_module_map()
    route_methods = {"get", "post", "put", "patch", "delete", "options"}
    route_modules_with_decorators: set[str] = set()

    for path in sorted(Path("askme/api/routes").glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        module_name = ".".join(path.with_suffix("").parts)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr in route_methods
                    and isinstance(decorator.func.value, ast.Name)
                ):
                    continue
                route_modules_with_decorators.add(module_name)

    assert route_modules_with_decorators == set(module_map)


def test_agent_profile_route_uses_tool_catalog_boundary() -> None:
    route_source = Path("askme/api/routes/agent_profiles.py").read_text(encoding="utf-8")
    service_source = Path("askme/api/services/agent_profile_tools.py").read_text(
        encoding="utf-8"
    )

    forbidden_route_imports = (
        "askme.tools.core.builtin_tools",
        "askme.tools.core.tool_registry",
        "askme.tools.robot",
        "askme.tools.spatial",
    )

    for forbidden in forbidden_route_imports:
        assert forbidden not in route_source

    assert "agent_profile_known_tools" in route_source
    assert "ToolRegistry(" not in service_source
    assert "BaseTool" not in service_source
    assert "RobotApiTool" not in service_source
    assert "SpeakProgressTool" not in service_source


def test_api_route_composition_invokes_every_route_registrar_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from askme.api import composition as composition_module

    surface_registrars = (
        "register_platform_routes",
        "register_product_routes",
        "register_admin_routes",
        "register_internal_routes",
    )
    calls: list[tuple[str, FastAPI, object]] = []

    def recorder(name: str):
        def _record(app: FastAPI, deps: object) -> None:
            calls.append((name, app, deps))

        return _record

    for name in surface_registrars:
        monkeypatch.setattr(composition_module, name, recorder(name))

    async def optional_json_body(_request: object) -> dict[str, object]:
        return {}

    async def passthrough_result(result: dict[str, object], **_kwargs: object) -> dict[str, object]:
        return result

    deps = composition_module.ApiRouteDependencies(
        health_provider=lambda: {},
        metrics_provider=lambda: {},
        render_prometheus_metrics=lambda _metrics: "",
        json_snapshot_response=lambda payload: JSONResponse(dict(payload)),
        snapshot_payload=lambda: {},
        prometheus_content_type="text/plain; version=0.0.4",
        governance_payload=lambda: {},
        identity_readiness_payload=lambda: {},
        current_operator_payload=lambda _request: {},
        authorization_payload=lambda _request, _body: {},
        mission_json=lambda payload, status_code=200: JSONResponse(payload, status_code=status_code),
        cors_options_response=lambda methods: Response(
            headers={"Access-Control-Allow-Methods": methods}
        ),
        dispatch_memory=lambda _action, _payload: {},
        logger=logging.getLogger("tests.api_composition"),
        authorize=lambda _request, _body, _permission: None,
        dispatch_cognition=lambda *_args, **_kwargs: {},
        json_error=lambda message, status_code=500: JSONResponse(
            {"error": message},
            status_code=status_code,
        ),
        cors_headers={"Access-Control-Allow-Origin": "*"},
        dispatch_runtime=lambda *_args, **_kwargs: {},
        optional_json_body=optional_json_body,
        operator_action_kwargs=lambda _request, _body: {},
        dispatch_voice=lambda *_args, **_kwargs: {},
        dispatch_space=lambda _action, _payload: {},
        dispatch_field_operations=lambda *_args, **_kwargs: {},
        field_manual_trigger_body=lambda _request, body: body,
        looks_like_device_ingest_without_scenario=lambda _body: False,
        dispatch_field_voice_directive=passthrough_result,
        dispatch_field_runtime_policy=passthrough_result,
        runtime_callback_trust=lambda *_args, **_kwargs: {"trusted": True},
        runtime_callback_delivery_body=lambda body, **_kwargs: dict(body),
        runtime_callback_secret=None,
        runtime_callback_max_age_s=30.0,
        field_path_roots={
            "site_profile_root": tmp_path / "site-profiles",
            "customer_project_template_root": tmp_path / "templates",
        },
        config_provider=lambda: {},
        dashboard_html="<html></html>",
        dashboard_asset_dir=tmp_path,
        dashboard_pages={},
        capabilities_provider=lambda: {},
        blueprints_provider=lambda: {},
        operator_id_from_request=lambda _request, _body: "test.operator",
        conversation_service=object(),
        runtime_available=False,
        runtime_voice_turn_timeout_s=0.1,
        monitor_service=object(),
        dispatch_mission=lambda *_args, **_kwargs: {},
        request_has_control_auth=lambda _request: False,
        skill_growth_candidate_prompt=lambda candidate: str(candidate),
        vision_snapshot_handler=None,
        vision_analyze_handler=None,
        archive_snapshot_handler=None,
        archive_list_handler=None,
        archive_get_handler=None,
        archive_delete_handler=None,
    )
    app = FastAPI()

    composition_module.register_api_routes(app, deps)

    assert [name for name, _app, _deps in calls] == list(surface_registrars)
    assert all(called_app is app for _name, called_app, _deps in calls)
    assert all(called_deps is deps for _name, _app, called_deps in calls)


def test_field_route_registration_delegates_to_split_route_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from askme.api.routes import field as field_routes

    child_registrars = (
        "register_field_internal_routes",
        "register_field_admin_routes",
        "register_customer_project_template_routes",
        "register_delivery_resource_routes",
        "register_customer_project_acceptance_routes",
        "register_customer_project_execution_routes",
        "register_customer_project_artifact_routes",
        "register_customer_project_profile_routes",
    )
    calls: list[tuple[str, FastAPI, dict[str, object]]] = []

    def recorder(name: str):
        def _record(app: FastAPI, **kwargs: object) -> None:
            calls.append((name, app, kwargs))

        return _record

    for name in child_registrars:
        monkeypatch.setattr(field_routes, name, recorder(name))

    async def dispatch_field_operations(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {}

    async def optional_json_body(_request: object) -> dict[str, object]:
        return {}

    async def result_hook(result: dict[str, object], **_kwargs: object) -> dict[str, object]:
        return result

    app = FastAPI()

    field_routes.register_field_routes(
        app,
        dispatch_field_operations=dispatch_field_operations,
        mission_json=lambda payload, status_code=200: JSONResponse(payload, status_code=status_code),
        optional_json_body=optional_json_body,
        cors_options_response=lambda methods: Response(
            headers={"Access-Control-Allow-Methods": methods}
        ),
        logger=logging.getLogger("tests.field_routes"),
        authorize=lambda _request, _body, _permission: None,
        field_manual_trigger_body=lambda _request, body: body,
        looks_like_device_ingest_without_scenario=lambda _body: False,
        dispatch_field_voice_directive=result_hook,
        dispatch_field_runtime_policy=result_hook,
        runtime_callback_trust=lambda *_args, **_kwargs: {"trusted": True},
        runtime_callback_delivery_body=lambda body, **_kwargs: dict(body),
        runtime_callback_secret=None,
        runtime_callback_max_age_s=30.0,
        cors_headers={"Access-Control-Allow-Origin": "*"},
        identity_readiness_payload=lambda: {},
        site_profile_root=tmp_path / "site-profiles",
        customer_project_template_root=tmp_path / "templates",
        delivery_resource_root=tmp_path / "delivery-resources",
        customer_project_package_root=tmp_path / "packages",
        customer_project_acceptance_dossier_root=tmp_path / "dossiers",
        customer_project_proposal_root=tmp_path / "proposals",
        config_provider=lambda: {},
    )

    assert [name for name, _app, _kwargs in calls] == list(child_registrars)
    assert all(called_app is app for _name, called_app, _kwargs in calls)
    by_name = {name: kwargs for name, _app, kwargs in calls}
    assert by_name["register_field_internal_routes"]["dispatch_field_operations"] is (
        dispatch_field_operations
    )
    assert "runtime_callback_trust" in by_name["register_field_internal_routes"]
    assert by_name["register_field_admin_routes"]["authorize"] is not None
    assert "template_root" in by_name["register_field_admin_routes"]
    assert "scope_item_from_create_body" in by_name["register_field_admin_routes"]
    assert "template_root" in by_name["register_customer_project_template_routes"]
    assert "delivery_resource_root" in by_name["register_delivery_resource_routes"]
    assert "site_profile_root" in by_name["register_customer_project_acceptance_routes"]
    assert by_name["register_customer_project_execution_routes"]["dispatch_field_operations"] is (
        dispatch_field_operations
    )
    assert "package_output_root" in by_name["register_customer_project_artifact_routes"]
    assert "scope_item_from_profile" in by_name["register_customer_project_profile_routes"]


def test_health_server_does_not_declare_inline_route_decorators() -> None:
    tree = ast.parse(Path("askme/health_server.py").read_text(encoding="utf-8"))
    route_methods = {"get", "post", "put", "patch", "delete", "options"}
    inline_routes: set[tuple[str, str]] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr in route_methods
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "app"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
            ):
                continue
            inline_routes.add((decorator.func.attr.upper(), decorator.args[0].value))

    assert inline_routes == set()


def test_llm_root_exports_are_lazy_product_facades() -> None:
    llm = importlib.import_module("askme.llm")
    conversation = importlib.import_module("askme.memory.core.conversation")
    intent_router = importlib.import_module("askme.interaction.intent_router")

    assert llm.ConversationManager is conversation.ConversationManager
    assert llm.IntentRouter is intent_router.IntentRouter
    assert llm.LLMClient is importlib.import_module("askme.llm.core.client").LLMClient


def test_internal_code_uses_canonical_llm_modules() -> None:
    offenders: list[str] = []
    for path in Path("askme").rglob("*.py"):
        if path in LEGACY_LLM_FACADE_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in LEGACY_LLM_FACADE_MODULES:
                offenders.append(f"{path}:{node.lineno}: from {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in LEGACY_LLM_FACADE_MODULES:
                        offenders.append(f"{path}:{node.lineno}: import {alias.name}")
            if (
                isinstance(node, ast.Call)
                and (
                    (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "import_module"
                    )
                    or (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "import_module"
                    )
                )
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in LEGACY_LLM_FACADE_MODULES
            ):
                offenders.append(f"{path}:{node.lineno}: import_module {node.args[0].value}")

    assert offenders == []


def test_field_route_imports_product_facades_not_site_profile_monolith() -> None:
    route_path = Path("askme/api/routes/field.py")
    artifact_route_path = Path("askme/api/routes/field_customer_project_artifacts.py")
    product_route_path = Path("askme/api/routes/field_product_catalog.py")
    delivery_resource_route_path = Path("askme/api/routes/field_delivery_resources.py")
    tree = ast.parse(route_path.read_text(encoding="utf-8"))
    artifact_tree = ast.parse(artifact_route_path.read_text(encoding="utf-8"))
    product_tree = ast.parse(product_route_path.read_text(encoding="utf-8"))
    delivery_resource_tree = ast.parse(delivery_resource_route_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    artifact_imports = {
        node.module
        for node in ast.walk(artifact_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    product_imports = {
        node.module
        for node in ast.walk(product_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    delivery_resource_imports = {
        node.module
        for node in ast.walk(delivery_resource_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "askme.pipeline.field.field_site_profile" not in imports
    assert "askme.pipeline.field_site_profile" not in imports
    assert "askme.pipeline.field.field_site_profile" not in product_imports
    assert "askme.pipeline.field_site_profile" not in product_imports
    assert "askme.pipeline.field.customer_projects" in product_imports
    assert "askme.pipeline.field.customer_project_templates" in product_imports
    assert "askme.api.routes.field_product_catalog" in imports
    assert "askme.api.routes.field_customer_project_artifacts" in imports
    assert "askme.pipeline.field.customer_project_artifacts" not in imports
    assert "askme.pipeline.field.customer_project_artifacts" in artifact_imports
    assert "askme.pipeline.field.delivery_resources" not in imports
    assert "askme.pipeline.field.delivery_resources" in delivery_resource_imports


def test_field_operations_imports_product_facade_not_site_profile_monolith() -> None:
    route_path = Path("askme/pipeline/field/field_operations.py")
    tree = ast.parse(route_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "askme.pipeline.field.field_site_profile" not in imports
    assert "askme.pipeline.field_site_profile" not in imports
    assert "askme.pipeline.field.customer_projects" in imports


def test_customer_project_public_facades_export_expected_contracts() -> None:
    customer_projects = importlib.import_module("askme.pipeline.field.customer_projects")
    customer_project_acceptance = importlib.import_module(
        "askme.pipeline.field.customer_project_acceptance"
    )
    customer_project_acceptance_registry = importlib.import_module(
        "askme.pipeline.field.customer_project_acceptance_registry"
    )
    customer_project_artifacts = importlib.import_module(
        "askme.pipeline.field.customer_project_artifacts"
    )
    customer_project_evidence_inventory = importlib.import_module(
        "askme.pipeline.field.customer_project_evidence_inventory"
    )
    customer_project_execution_bindings = importlib.import_module(
        "askme.pipeline.field.customer_project_execution_bindings"
    )
    customer_project_profile_operations = importlib.import_module(
        "askme.pipeline.field.customer_project_profile_operations"
    )
    customer_project_package_assessment = importlib.import_module(
        "askme.pipeline.field.customer_project_package_assessment"
    )
    customer_project_package_html = importlib.import_module(
        "askme.pipeline.field.customer_project_package_html"
    )
    customer_project_package_rules = importlib.import_module(
        "askme.pipeline.field.customer_project_package_rules"
    )
    customer_project_managed_objects = importlib.import_module(
        "askme.pipeline.field.customer_project_managed_objects"
    )
    customer_project_profiles = importlib.import_module(
        "askme.pipeline.field.customer_project_profiles"
    )
    customer_project_scope = importlib.import_module("askme.pipeline.field.customer_project_scope")
    field_site_runtime_config = importlib.import_module(
        "askme.pipeline.field.field_site_runtime_config"
    )
    field_site_validation = importlib.import_module(
        "askme.pipeline.field.field_site_validation"
    )
    field_site_catalog = importlib.import_module(
        "askme.pipeline.field.field_site_catalog"
    )

    assert set(customer_projects.__all__) == CUSTOMER_PROJECT_PUBLIC_NAMES
    assert set(customer_project_acceptance.__all__) == CUSTOMER_PROJECT_ACCEPTANCE_HELPERS
    assert set(customer_project_acceptance_registry.__all__) == (
        CUSTOMER_PROJECT_ACCEPTANCE_REGISTRY_HELPERS
    )
    assert set(customer_project_artifacts.__all__) == CUSTOMER_PROJECT_ARTIFACT_PUBLIC_NAMES
    assert set(customer_project_evidence_inventory.__all__) == CUSTOMER_PROJECT_EVIDENCE_INVENTORY_HELPERS
    assert set(customer_project_execution_bindings.__all__) == CUSTOMER_PROJECT_EXECUTION_BINDING_HELPERS
    assert set(customer_project_profile_operations.__all__) == CUSTOMER_PROJECT_PROFILE_OPERATION_HELPERS
    assert set(customer_project_package_assessment.__all__) == CUSTOMER_PROJECT_PACKAGE_ASSESSMENT_HELPERS
    assert set(customer_project_package_html.__all__) == CUSTOMER_PROJECT_PACKAGE_HTML_HELPERS
    assert set(customer_project_package_rules.__all__) == CUSTOMER_PROJECT_PACKAGE_RULE_HELPERS
    assert set(customer_project_managed_objects.__all__) == CUSTOMER_PROJECT_MANAGED_OBJECT_HELPERS
    assert set(customer_project_profiles.__all__) == CUSTOMER_PROJECT_PROFILE_STORE_HELPERS
    assert set(customer_project_scope.__all__) == CUSTOMER_PROJECT_SCOPE_HELPERS
    assert set(field_site_runtime_config.__all__) == FIELD_SITE_RUNTIME_CONFIG_HELPERS
    assert set(field_site_validation.__all__) == FIELD_SITE_VALIDATION_HELPERS
    assert set(field_site_catalog.__all__) == FIELD_SITE_CATALOG_HELPERS


def test_customer_project_public_calls_use_product_facades() -> None:
    allowed_files = {
        Path("askme/pipeline/field/customer_project_artifacts.py"),
        Path("askme/pipeline/field/customer_projects.py"),
        Path("askme/pipeline/field/customer_project_templates.py"),
        Path("askme/pipeline/field/field_site_profile.py"),
    }
    forbidden_modules = {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field_site_profile",
    }
    public_names = CUSTOMER_PROJECT_PUBLIC_NAMES | CUSTOMER_PROJECT_ARTIFACT_PUBLIC_NAMES
    violations: list[str] = []

    for path in Path("askme").rglob("*.py"):
        normalized = Path(path.as_posix())
        if normalized in allowed_files:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module not in forbidden_modules:
                continue
            imported_public_names = {
                alias.name for alias in node.names if alias.name in public_names
            }
            if imported_public_names:
                violations.append(
                    f"{path}:{node.lineno} imports {sorted(imported_public_names)} "
                    f"from {node.module}"
                )

    assert violations == []


def test_field_package_lists_public_delivery_boundaries() -> None:
    field_package = importlib.import_module("askme.pipeline.field")

    assert {
        "customer_project_acceptance",
        "customer_project_acceptance_registry",
        "customer_project_artifact_manifests",
        "customer_project_evidence_inventory",
        "customer_project_execution_bindings",
        "customer_project_package_assessment",
        "customer_project_package_html",
        "customer_project_package_rules",
        "customer_project_managed_objects",
        "customer_project_profile_operations",
        "customer_project_profiles",
        "customer_projects",
        "customer_project_resource_catalog",
        "customer_project_scope",
        "customer_project_implementation_handoff",
        "customer_project_template_catalog",
        "customer_project_template_delivery",
        "customer_project_template_release",
        "customer_project_template_support",
        "customer_project_templates",
        "customer_project_artifacts",
        "delivery_resource_governance",
        "delivery_resource_registry",
        "delivery_resources",
        "field_site_catalog",
        "field_site_runtime_config",
        "field_site_validation",
        "paths",
        "solution_delivery_readiness",
    } <= set(field_package.__all__)


def test_delivery_resource_registry_kernel_is_physically_split() -> None:
    route_path = Path("askme/pipeline/field/field_site_profile.py")
    tree = ast.parse(route_path.read_text(encoding="utf-8"))
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert not {
        "load_delivery_resource_registry",
        "list_delivery_resource_registry",
        "upsert_delivery_resource",
        "list_delivery_resource_revisions",
        "disable_delivery_resource",
        "rollback_delivery_resource_registry",
        "_delivery_resource_catalog",
        "_delivery_resource_rows",
        "_delivery_resource_registry_path",
    } & function_names


def test_delivery_resource_governance_kernel_is_physically_split() -> None:
    route_path = Path("askme/pipeline/field/field_site_profile.py")
    tree = ast.parse(route_path.read_text(encoding="utf-8"))
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert not {
        "create_delivery_resource_governance_request",
        "list_delivery_resource_governance_requests",
        "review_delivery_resource_governance_request",
        "escalate_overdue_delivery_resource_governance_requests",
        "_delivery_resource_governance_operation_payload",
        "_preview_delivery_resource_governance_operation",
        "_delivery_resource_governance_impact",
        "_delivery_resource_governance_request_public_payload",
    } & function_names


def test_customer_project_resource_catalog_kernel_is_physically_split() -> None:
    route_path = Path("askme/pipeline/field/field_site_profile.py")
    tree = ast.parse(route_path.read_text(encoding="utf-8"))
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert not {
        "build_customer_project_resource_catalog",
        "_delivery_resource_consumers_from_profile",
    } & function_names


def test_customer_project_template_catalog_kernel_is_physically_split() -> None:
    route_path = Path("askme/pipeline/field/field_site_profile.py")
    tree = ast.parse(route_path.read_text(encoding="utf-8"))
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert not {
        "list_customer_project_templates",
        "customer_project_template_summary_from_items",
        "_customer_project_template_filters",
        "_customer_project_template_matches_filters",
    } & function_names


def test_customer_project_template_catalog_stays_read_only() -> None:
    catalog_path = Path("askme/pipeline/field/customer_project_template_catalog.py")
    tree = ast.parse(catalog_path.read_text(encoding="utf-8"))
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert not {
        "create_customer_project_from_template",
        "create_customer_project_template_release_request",
        "customer_project_template_release_notes",
        "export_customer_project_template_release_notes_bundle",
        "review_customer_project_template_release_request",
        "update_customer_project_template_release",
    } & function_names
    assert not {
        "mkdir",
        "rename",
        "replace",
        "rmdir",
        "unlink",
        "write_bytes",
        "write_text",
    } & called_attributes
    assert not {"open", "shutil"} & called_names


def test_customer_project_template_release_governance_kernel_is_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    release_path = Path("askme/pipeline/field/customer_project_template_release.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    release_tree = ast.parse(release_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    release_functions = {
        node.name
        for node in ast.walk(release_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    migrated_names = TEMPLATE_RELEASE_PUBLIC_NAMES | TEMPLATE_RELEASE_PRIVATE_HELPERS
    assert not migrated_names & monolith_functions
    assert TEMPLATE_RELEASE_PUBLIC_NAMES <= release_functions
    assert TEMPLATE_RELEASE_PRIVATE_HELPERS <= release_functions


def test_customer_project_template_support_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    support_path = Path("askme/pipeline/field/customer_project_template_support.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    support_tree = ast.parse(support_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    support_functions = {
        node.name
        for node in ast.walk(support_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    support_imports = {
        node.module
        for node in ast.walk(support_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not TEMPLATE_SUPPORT_HELPERS & monolith_functions
    assert TEMPLATE_SUPPORT_HELPERS <= support_functions
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_project_template_release",
        "askme.pipeline.field.customer_project_templates",
    } & support_imports


def test_customer_project_template_delivery_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    delivery_path = Path("askme/pipeline/field/customer_project_template_delivery.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    delivery_tree = ast.parse(delivery_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    delivery_functions = {
        node.name
        for node in ast.walk(delivery_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    delivery_imports = {
        node.module
        for node in ast.walk(delivery_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(delivery_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not TEMPLATE_DELIVERY_HELPERS & monolith_functions
    assert TEMPLATE_DELIVERY_HELPERS <= delivery_functions
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_project_template_release",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_artifacts",
    } & delivery_imports


def test_solution_delivery_readiness_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    readiness_path = Path("askme/pipeline/field/solution_delivery_readiness.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    readiness_tree = ast.parse(readiness_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    readiness_functions = {
        node.name
        for node in ast.walk(readiness_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    readiness_imports = {
        node.module
        for node in ast.walk(readiness_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(readiness_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    migrated_names = SOLUTION_DELIVERY_READINESS_HELPERS | {
        "build_solution_delivery_readiness",
    }
    assert not migrated_names & monolith_functions
    assert migrated_names <= readiness_functions
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_template_catalog",
        "askme.pipeline.field.customer_project_resource_catalog",
        "askme.pipeline.field.delivery_resource_governance",
    } & readiness_imports


def test_customer_project_implementation_handoff_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    handoff_path = Path("askme/pipeline/field/customer_project_implementation_handoff.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    handoff_tree = ast.parse(handoff_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    handoff_functions = {
        node.name
        for node in ast.walk(handoff_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    handoff_imports = {
        node.module
        for node in ast.walk(handoff_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(handoff_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not CUSTOMER_PROJECT_IMPLEMENTATION_HANDOFF_HELPERS & monolith_functions
    assert CUSTOMER_PROJECT_IMPLEMENTATION_HANDOFF_HELPERS <= handoff_functions
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
    } & handoff_imports


def test_customer_project_artifact_manifest_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    manifest_path = Path("askme/pipeline/field/customer_project_artifact_manifests.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    manifest_tree = ast.parse(manifest_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    manifest_functions = {
        node.name
        for node in ast.walk(manifest_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    manifest_imports = {
        node.module
        for node in ast.walk(manifest_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(manifest_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not CUSTOMER_PROJECT_ARTIFACT_MANIFEST_HELPERS & monolith_functions
    assert CUSTOMER_PROJECT_ARTIFACT_MANIFEST_HELPERS <= manifest_functions
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & manifest_imports


def test_customer_project_artifact_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    artifact_path = Path("askme/pipeline/field/customer_project_artifacts.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    artifact_tree = ast.parse(artifact_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    artifact_functions = {
        node.name
        for node in ast.walk(artifact_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    artifact_imports = {
        node.module
        for node in ast.walk(artifact_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(artifact_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not CUSTOMER_PROJECT_ARTIFACT_KERNEL_HELPERS & monolith_functions
    assert CUSTOMER_PROJECT_ARTIFACT_KERNEL_HELPERS <= artifact_functions
    assert set(importlib.import_module("askme.pipeline.field.customer_project_artifacts").__all__) == (
        CUSTOMER_PROJECT_ARTIFACT_PUBLIC_NAMES
    )
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
    } & artifact_imports


def test_customer_project_evidence_inventory_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    inventory_path = Path("askme/pipeline/field/customer_project_evidence_inventory.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    inventory_tree = ast.parse(inventory_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    inventory_functions = {
        node.name
        for node in ast.walk(inventory_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    inventory_imports = {
        node.module
        for node in ast.walk(inventory_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(inventory_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not CUSTOMER_PROJECT_EVIDENCE_INVENTORY_HELPERS & monolith_functions
    assert CUSTOMER_PROJECT_EVIDENCE_INVENTORY_HELPERS <= inventory_functions
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & inventory_imports


def test_customer_project_package_assessment_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    assessment_path = Path("askme/pipeline/field/customer_project_package_assessment.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    assessment_tree = ast.parse(assessment_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assessment_functions = {
        node.name
        for node in ast.walk(assessment_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assessment_imports = {
        node.module
        for node in ast.walk(assessment_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(assessment_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assessment_function_helpers = CUSTOMER_PROJECT_PACKAGE_ASSESSMENT_HELPERS - {
        "_managed_object_acceptance_summary",
        "_managed_object_binding_readiness_summary",
    }
    assert not CUSTOMER_PROJECT_PACKAGE_ASSESSMENT_HELPERS & monolith_functions
    assert assessment_function_helpers <= assessment_functions
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & assessment_imports


def test_customer_project_package_html_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    html_path = Path("askme/pipeline/field/customer_project_package_html.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    html_tree = ast.parse(html_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    html_functions = {
        node.name
        for node in ast.walk(html_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    html_imports = {
        node.module
        for node in ast.walk(html_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(html_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not CUSTOMER_PROJECT_PACKAGE_HTML_HELPERS & monolith_functions
    assert CUSTOMER_PROJECT_PACKAGE_HTML_HELPERS <= html_functions
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & html_imports


def test_customer_project_package_rules_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    rules_path = Path("askme/pipeline/field/customer_project_package_rules.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    rules_tree = ast.parse(rules_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    rules_functions = {
        node.name
        for node in ast.walk(rules_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    rules_imports = {
        node.module
        for node in ast.walk(rules_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(rules_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not CUSTOMER_PROJECT_PACKAGE_RULE_HELPERS & monolith_functions
    assert CUSTOMER_PROJECT_PACKAGE_RULE_HELPERS <= rules_functions
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & rules_imports


def test_customer_project_scope_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    scope_path = Path("askme/pipeline/field/customer_project_scope.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    scope_tree = ast.parse(scope_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    scope_functions = {
        node.name
        for node in ast.walk(scope_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    scope_imports = {
        node.module
        for node in ast.walk(scope_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(scope_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not CUSTOMER_PROJECT_SCOPE_HELPERS & monolith_functions
    assert CUSTOMER_PROJECT_SCOPE_HELPERS <= scope_functions
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & scope_imports


def test_customer_project_profile_store_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    profiles_path = Path("askme/pipeline/field/customer_project_profiles.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    profiles_tree = ast.parse(profiles_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    profiles_functions = {
        node.name
        for node in ast.walk(profiles_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    profiles_imports = {
        node.module
        for node in ast.walk(profiles_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(profiles_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not CUSTOMER_PROJECT_PROFILE_STORE_HELPERS & monolith_functions
    assert CUSTOMER_PROJECT_PROFILE_STORE_HELPERS <= profiles_functions
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & profiles_imports


def test_customer_project_managed_objects_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    managed_objects_path = Path("askme/pipeline/field/customer_project_managed_objects.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    managed_objects_tree = ast.parse(managed_objects_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    managed_object_functions = {
        node.name
        for node in ast.walk(managed_objects_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    managed_object_imports = {
        node.module
        for node in ast.walk(managed_objects_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(managed_objects_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    managed_object_function_helpers = (
        CUSTOMER_PROJECT_MANAGED_OBJECT_HELPERS - {"_ACCEPTANCE_TEST_ALIASES"}
    )
    assert not managed_object_function_helpers & monolith_functions
    assert managed_object_function_helpers <= managed_object_functions
    assert set(importlib.import_module("askme.pipeline.field.customer_project_managed_objects").__all__) == (
        CUSTOMER_PROJECT_MANAGED_OBJECT_HELPERS
    )
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & managed_object_imports


def test_customer_project_execution_bindings_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    execution_path = Path("askme/pipeline/field/customer_project_execution_bindings.py")
    customer_projects_path = Path("askme/pipeline/field/customer_projects.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    execution_tree = ast.parse(execution_path.read_text(encoding="utf-8"))
    customer_projects_tree = ast.parse(customer_projects_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    execution_functions = {
        node.name
        for node in ast.walk(execution_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    execution_imports = {
        node.module
        for node in ast.walk(execution_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(execution_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    customer_project_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(customer_projects_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    execution_function_helpers = (
        CUSTOMER_PROJECT_EXECUTION_BINDING_HELPERS - {"_SCENARIO_REQUIRED_INPUTS"}
    )
    assert not execution_function_helpers & monolith_functions
    assert execution_function_helpers <= execution_functions
    assert set(importlib.import_module("askme.pipeline.field.customer_project_execution_bindings").__all__) == (
        CUSTOMER_PROJECT_EXECUTION_BINDING_HELPERS
    )
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & execution_imports
    assert (
        "askme.pipeline.field.customer_project_execution_bindings",
        ("build_customer_project_execution_bindings",),
    ) in customer_project_imports


def test_customer_project_acceptance_registry_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    registry_path = Path("askme/pipeline/field/customer_project_acceptance_registry.py")
    customer_projects_path = Path("askme/pipeline/field/customer_projects.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    registry_tree = ast.parse(registry_path.read_text(encoding="utf-8"))
    customer_projects_tree = ast.parse(customer_projects_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    registry_functions = {
        node.name
        for node in ast.walk(registry_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    registry_imports = {
        node.module
        for node in ast.walk(registry_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(registry_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    customer_project_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(customer_projects_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not CUSTOMER_PROJECT_ACCEPTANCE_REGISTRY_HELPERS & monolith_functions
    assert CUSTOMER_PROJECT_ACCEPTANCE_REGISTRY_HELPERS <= registry_functions
    assert set(importlib.import_module("askme.pipeline.field.customer_project_acceptance_registry").__all__) == (
        CUSTOMER_PROJECT_ACCEPTANCE_REGISTRY_HELPERS
    )
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & registry_imports
    assert (
        "askme.pipeline.field.customer_project_acceptance_registry",
        ("build_customer_project_acceptance_registry",),
    ) in customer_project_imports


def test_field_site_runtime_config_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    runtime_config_path = Path("askme/pipeline/field/field_site_runtime_config.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    runtime_config_tree = ast.parse(runtime_config_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    runtime_config_functions = {
        node.name
        for node in ast.walk(runtime_config_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    runtime_config_imports = {
        node.module
        for node in ast.walk(runtime_config_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(runtime_config_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not FIELD_SITE_RUNTIME_CONFIG_HELPERS & monolith_functions
    assert FIELD_SITE_RUNTIME_CONFIG_HELPERS <= runtime_config_functions
    assert set(importlib.import_module("askme.pipeline.field.field_site_runtime_config").__all__) == (
        FIELD_SITE_RUNTIME_CONFIG_HELPERS
    )
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & runtime_config_imports


def test_field_site_validation_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    validation_path = Path("askme/pipeline/field/field_site_validation.py")
    catalog_path = Path("askme/pipeline/field/customer_project_template_catalog.py")
    release_path = Path("askme/pipeline/field/customer_project_template_release.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    validation_tree = ast.parse(validation_path.read_text(encoding="utf-8"))
    catalog_tree = ast.parse(catalog_path.read_text(encoding="utf-8"))
    release_tree = ast.parse(release_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    validation_functions = {
        node.name
        for node in ast.walk(validation_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    validation_imports = {
        node.module
        for node in ast.walk(validation_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(validation_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    catalog_imports = {
        node.module
        for node in ast.walk(catalog_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    release_imports = {
        node.module
        for node in ast.walk(release_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    validation_function_helpers = FIELD_SITE_VALIDATION_HELPERS - {
        "REQUIRED_DEVICE_SOURCES",
        "REQUIRED_RESPONDER_GROUPS",
    }
    assert not validation_function_helpers & monolith_functions
    assert validation_function_helpers <= validation_functions
    assert set(importlib.import_module("askme.pipeline.field.field_site_validation").__all__) == (
        FIELD_SITE_VALIDATION_HELPERS
    )
    assert "askme.pipeline.field.field_site_validation" in catalog_imports
    assert "askme.pipeline.field.field_site_validation" in release_imports
    assert "askme.pipeline.field.field_site_profile" not in catalog_imports
    assert "askme.pipeline.field.field_site_profile" not in release_imports
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & validation_imports


def test_field_site_catalog_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    catalog_path = Path("askme/pipeline/field/field_site_catalog.py")
    customer_projects_path = Path("askme/pipeline/field/customer_projects.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    catalog_tree = ast.parse(catalog_path.read_text(encoding="utf-8"))
    customer_projects_tree = ast.parse(customer_projects_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    catalog_functions = {
        node.name
        for node in ast.walk(catalog_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    catalog_imports = {
        node.module
        for node in ast.walk(catalog_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(catalog_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    customer_project_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(customer_projects_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not FIELD_SITE_CATALOG_HELPERS & monolith_functions
    assert FIELD_SITE_CATALOG_HELPERS <= catalog_functions
    assert set(importlib.import_module("askme.pipeline.field.field_site_catalog").__all__) == (
        FIELD_SITE_CATALOG_HELPERS
    )
    assert (
        "askme.pipeline.field.field_site_catalog",
        (
            "build_customer_project_catalog",
            "build_site_profile_catalog",
            "build_site_profile_report",
        ),
    ) in customer_project_imports
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & catalog_imports


def test_customer_project_acceptance_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    acceptance_path = Path("askme/pipeline/field/customer_project_acceptance.py")
    customer_projects_path = Path("askme/pipeline/field/customer_projects.py")
    artifacts_path = Path("askme/pipeline/field/customer_project_artifacts.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    acceptance_tree = ast.parse(acceptance_path.read_text(encoding="utf-8"))
    customer_projects_tree = ast.parse(customer_projects_path.read_text(encoding="utf-8"))
    artifacts_tree = ast.parse(artifacts_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    acceptance_functions = {
        node.name
        for node in ast.walk(acceptance_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    acceptance_imports = {
        node.module
        for node in ast.walk(acceptance_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(acceptance_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    customer_project_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(customer_projects_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    artifact_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(artifacts_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    acceptance_constant_names = {
        "_FIELD_READINESS_EVIDENCE_DEFAULTS",
        "ONSITE_ACCEPTANCE_REQUIRED_EVIDENCE_TYPES",
        "ONSITE_ACCEPTANCE_EVIDENCE_TYPES",
        "ONSITE_ACCEPTANCE_STATUSES",
        "ACCEPTANCE_REVIEW_DECISIONS",
        "CUSTOMER_SIGNOFF_DECISIONS",
    }
    acceptance_function_helpers = CUSTOMER_PROJECT_ACCEPTANCE_HELPERS - acceptance_constant_names
    assert not acceptance_function_helpers & monolith_functions
    assert acceptance_function_helpers <= acceptance_functions
    assert set(importlib.import_module("askme.pipeline.field.customer_project_acceptance").__all__) == (
        CUSTOMER_PROJECT_ACCEPTANCE_HELPERS
    )
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
        "askme.pipeline.field.customer_project_template_release",
    } & acceptance_imports
    assert (
        "askme.pipeline.field.customer_project_acceptance",
        (
            "customer_project_acceptance_closure",
            "customer_project_acceptance_report",
            "list_customer_project_customer_signoffs",
            "list_customer_project_onsite_evidence",
            "register_customer_project_acceptance_review",
            "register_customer_project_customer_signoff",
            "register_customer_project_onsite_evidence",
        ),
    ) in customer_project_imports
    assert any(
        module == "askme.pipeline.field.customer_project_acceptance"
        and {
            "verify_customer_project_acceptance_dossier",
            "verify_customer_project_proposal_bundle",
        }
        <= set(names)
        for module, names in artifact_imports
    )


def test_customer_project_profile_operations_kernel_is_leaf_and_physically_split() -> None:
    monolith_path = Path("askme/pipeline/field/field_site_profile.py")
    operations_path = Path("askme/pipeline/field/customer_project_profile_operations.py")
    customer_projects_path = Path("askme/pipeline/field/customer_projects.py")
    customer_templates_path = Path("askme/pipeline/field/customer_project_templates.py")
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    operations_tree = ast.parse(operations_path.read_text(encoding="utf-8"))
    customer_projects_tree = ast.parse(customer_projects_path.read_text(encoding="utf-8"))
    customer_templates_tree = ast.parse(customer_templates_path.read_text(encoding="utf-8"))
    monolith_functions = {
        node.name
        for node in ast.walk(monolith_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    operation_functions = {
        node.name
        for node in ast.walk(operations_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    operation_imports = {
        node.module
        for node in ast.walk(operations_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(operations_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    customer_project_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(customer_projects_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    template_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(customer_templates_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not CUSTOMER_PROJECT_PROFILE_OPERATION_HELPERS & monolith_functions
    assert CUSTOMER_PROJECT_PROFILE_OPERATION_HELPERS <= operation_functions
    assert set(importlib.import_module("askme.pipeline.field.customer_project_profile_operations").__all__) == (
        CUSTOMER_PROJECT_PROFILE_OPERATION_HELPERS
    )
    assert not {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field_site_profile",
        "askme.pipeline.field.customer_projects",
        "askme.pipeline.field.customer_project_templates",
        "askme.pipeline.field.customer_project_artifacts",
    } & operation_imports
    assert (
        "askme.pipeline.field.customer_project_profile_operations",
        (
            "delete_managed_object",
            "get_customer_project_profile",
            "rollback_customer_project_profile",
            "upsert_customer_project_profile",
            "upsert_managed_object",
        ),
    ) in customer_project_imports
    assert (
        "askme.pipeline.field.customer_project_profile_operations",
        ("create_customer_project_from_template",),
    ) in template_imports


def test_customer_project_template_release_public_calls_use_product_facade() -> None:
    allowed_files = {
        Path("askme/pipeline/field/customer_project_artifacts.py"),
        Path("askme/pipeline/field/customer_project_template_release.py"),
        Path("askme/pipeline/field/customer_project_templates.py"),
        Path("askme/pipeline/field/field_site_profile.py"),
    }
    forbidden_modules = {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field_site_profile",
    }
    violations: list[str] = []

    for path in Path("askme").rglob("*.py"):
        normalized = Path(path.as_posix())
        if normalized in allowed_files:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module not in forbidden_modules:
                continue
            imported_public_names = {
                alias.name for alias in node.names if alias.name in TEMPLATE_RELEASE_PUBLIC_NAMES
            }
            if imported_public_names:
                violations.append(
                    f"{path}:{node.lineno} imports {sorted(imported_public_names)} "
                    f"from {node.module}"
                )

    assert violations == []


def test_delivery_resource_public_calls_use_product_facade() -> None:
    public_names = {
        "load_delivery_resource_registry",
        "list_delivery_resource_registry",
        "upsert_delivery_resource",
        "list_delivery_resource_revisions",
        "disable_delivery_resource",
        "rollback_delivery_resource_registry",
        "create_delivery_resource_governance_request",
        "list_delivery_resource_governance_requests",
        "review_delivery_resource_governance_request",
        "escalate_overdue_delivery_resource_governance_requests",
    }
    allowed_files = {
        Path("askme/pipeline/field/delivery_resources.py"),
        Path("askme/pipeline/field/delivery_resource_governance.py"),
        Path("askme/pipeline/field/delivery_resource_registry.py"),
        Path("askme/pipeline/field/field_site_profile.py"),
    }
    forbidden_modules = {
        "askme.pipeline.field.field_site_profile",
        "askme.pipeline.field_site_profile",
        "askme.pipeline.field.delivery_resource_governance",
        "askme.pipeline.field.delivery_resource_registry",
    }
    violations: list[str] = []

    for path in Path("askme").rglob("*.py"):
        normalized = Path(path.as_posix())
        if normalized in allowed_files:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module not in forbidden_modules:
                continue
            imported_public_names = {
                alias.name for alias in node.names if alias.name in public_names
            }
            if imported_public_names:
                violations.append(
                    f"{path}:{node.lineno} imports {sorted(imported_public_names)} "
                    f"from {node.module}"
                )

    assert violations == []


def test_moved_packages_resolve_repo_root_paths() -> None:
    from askme.config import project_root
    from askme.pipeline.field import (
        customer_project_acceptance,
        customer_project_acceptance_registry,
        customer_project_artifact_manifests,
        customer_project_artifacts,
        customer_project_evidence_inventory,
        customer_project_execution_bindings,
        customer_project_implementation_handoff,
        customer_project_managed_objects,
        customer_project_package_assessment,
        customer_project_package_html,
        customer_project_package_rules,
        customer_project_profile_operations,
        customer_project_profiles,
        customer_project_resource_catalog,
        customer_project_scope,
        customer_project_template_catalog,
        customer_project_template_delivery,
        customer_project_template_release,
        customer_project_template_support,
        customer_project_templates,
        customer_projects,
        delivery_resource_governance,
        delivery_resource_registry,
        delivery_resources,
        field_site_catalog,
        field_site_profile,
        field_site_runtime_config,
        field_site_validation,
        paths,
        solution_delivery_readiness,
    )
    from askme.pipeline.field.field_site_profile import (
        DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT,
        DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT,
        DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT,
        DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
        DEFAULT_DELIVERY_RESOURCE_ROOT,
        DEFAULT_SITE_PROFILE_ROOT,
    )
    from askme.skills.governance.audit import default_skill_audit_path
    from askme.skills.governance.growth_backlog import default_skill_growth_state_path
    from askme.voice.input.mic_input import MicInput

    root = project_root()

    assert paths.PROJECT_ROOT == root
    assert field_site_profile.PROJECT_ROOT == root
    assert field_site_profile.DEFAULT_SITE_PROFILE_ROOT == paths.DEFAULT_SITE_PROFILE_ROOT
    assert DEFAULT_SITE_PROFILE_ROOT == root / "deploy" / "site-profiles"
    assert DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT == (
        root / "deploy" / "customer-project-templates"
    )
    assert DEFAULT_DELIVERY_RESOURCE_ROOT == root / "deploy" / "delivery-resources"
    assert DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT == (
        root / "artifacts" / "customer-project-packages"
    )
    assert DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT == (
        root / "artifacts" / "customer-project-acceptance-dossiers"
    )
    assert DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT == (
        root / "artifacts" / "customer-project-proposals"
    )
    assert paths.project_path("deploy/site-profiles") == DEFAULT_SITE_PROFILE_ROOT
    assert paths.project_path(DEFAULT_SITE_PROFILE_ROOT) == DEFAULT_SITE_PROFILE_ROOT
    for name in FIELD_SITE_CATALOG_HELPERS:
        assert getattr(field_site_profile, name) is getattr(field_site_catalog, name)
    for name in (
        "build_customer_project_catalog",
        "build_site_profile_catalog",
        "build_site_profile_report",
    ):
        assert getattr(customer_projects, name) is getattr(field_site_catalog, name)
    assert field_site_profile.load_delivery_resource_registry is (
        delivery_resource_registry.load_delivery_resource_registry
    )
    assert delivery_resources.load_delivery_resource_registry is (
        delivery_resource_registry.load_delivery_resource_registry
    )
    assert delivery_resources.create_delivery_resource_governance_request is (
        delivery_resource_governance.create_delivery_resource_governance_request
    )
    assert field_site_profile.create_delivery_resource_governance_request is (
        delivery_resource_governance.create_delivery_resource_governance_request
    )
    assert customer_projects.build_customer_project_resource_catalog is (
        customer_project_resource_catalog.build_customer_project_resource_catalog
    )
    assert field_site_profile.build_customer_project_resource_catalog is (
        customer_project_resource_catalog.build_customer_project_resource_catalog
    )
    assert customer_project_templates.list_customer_project_templates is (
        customer_project_template_catalog.list_customer_project_templates
    )
    assert field_site_profile.list_customer_project_templates is (
        customer_project_template_catalog.list_customer_project_templates
    )
    assert customer_project_templates.customer_project_template_summary_from_items is (
        customer_project_template_catalog.customer_project_template_summary_from_items
    )
    assert field_site_profile.customer_project_template_summary_from_items is (
        customer_project_template_catalog.customer_project_template_summary_from_items
    )
    for name in TEMPLATE_SUPPORT_HELPERS:
        assert getattr(field_site_profile, name) is getattr(customer_project_template_support, name)
    for name in TEMPLATE_DELIVERY_HELPERS:
        assert getattr(field_site_profile, name) is getattr(customer_project_template_delivery, name)
    for name in SOLUTION_DELIVERY_READINESS_HELPERS:
        assert getattr(field_site_profile, name) is getattr(solution_delivery_readiness, name)
    for name in CUSTOMER_PROJECT_IMPLEMENTATION_HANDOFF_HELPERS:
        assert getattr(field_site_profile, name) is getattr(
            customer_project_implementation_handoff,
            name,
        )
    for name in CUSTOMER_PROJECT_ARTIFACT_MANIFEST_HELPERS:
        assert getattr(field_site_profile, name) is getattr(
            customer_project_artifact_manifests,
            name,
        )
    for name in CUSTOMER_PROJECT_EVIDENCE_INVENTORY_HELPERS:
        assert getattr(field_site_profile, name) is getattr(
            customer_project_evidence_inventory,
            name,
        )
    for name in CUSTOMER_PROJECT_PACKAGE_ASSESSMENT_HELPERS:
        assert getattr(field_site_profile, name) is getattr(
            customer_project_package_assessment,
            name,
        )
    for name in CUSTOMER_PROJECT_PACKAGE_HTML_HELPERS:
        assert getattr(field_site_profile, name) is getattr(
            customer_project_package_html,
            name,
        )
    for name in CUSTOMER_PROJECT_PACKAGE_RULE_HELPERS:
        assert getattr(field_site_profile, name) is getattr(
            customer_project_package_rules,
            name,
        )
    for name in CUSTOMER_PROJECT_PROFILE_STORE_HELPERS:
        assert getattr(field_site_profile, name) is getattr(customer_project_profiles, name)
    for name in CUSTOMER_PROJECT_MANAGED_OBJECT_HELPERS:
        assert getattr(field_site_profile, name) is getattr(customer_project_managed_objects, name)
    for name in CUSTOMER_PROJECT_ACCEPTANCE_REGISTRY_HELPERS:
        assert getattr(field_site_profile, name) is getattr(customer_project_acceptance_registry, name)
    for name in CUSTOMER_PROJECT_ACCEPTANCE_HELPERS:
        assert getattr(field_site_profile, name) is getattr(customer_project_acceptance, name)
    for name in CUSTOMER_PROJECT_EXECUTION_BINDING_HELPERS:
        assert getattr(field_site_profile, name) is getattr(customer_project_execution_bindings, name)
    for name in CUSTOMER_PROJECT_PROFILE_OPERATION_HELPERS:
        assert getattr(field_site_profile, name) is getattr(customer_project_profile_operations, name)
    for name in FIELD_SITE_RUNTIME_CONFIG_HELPERS:
        assert getattr(field_site_profile, name) is getattr(field_site_runtime_config, name)
    for name in FIELD_SITE_VALIDATION_HELPERS:
        assert getattr(field_site_profile, name) is getattr(field_site_validation, name)
    assert customer_projects.build_customer_project_acceptance_registry is (
        customer_project_acceptance_registry.build_customer_project_acceptance_registry
    )
    for name in (
        "customer_project_acceptance_closure",
        "customer_project_acceptance_report",
        "list_customer_project_customer_signoffs",
        "list_customer_project_onsite_evidence",
        "register_customer_project_acceptance_review",
        "register_customer_project_customer_signoff",
        "register_customer_project_onsite_evidence",
    ):
        assert getattr(customer_projects, name) is getattr(customer_project_acceptance, name)
    for name in (
        "verify_customer_project_acceptance_dossier",
        "verify_customer_project_proposal_bundle",
    ):
        assert getattr(customer_project_artifacts, name) is getattr(customer_project_acceptance, name)
    for name in CUSTOMER_PROJECT_ARTIFACT_KERNEL_HELPERS:
        assert getattr(field_site_profile, name) is getattr(customer_project_artifacts, name)
    assert customer_projects.build_customer_project_execution_bindings is (
        customer_project_execution_bindings.build_customer_project_execution_bindings
    )
    for name in (
        "archive_customer_project_profile",
        "customer_project_catalog_acceptance_gate",
        "customer_project_catalog_summary_from_projects",
        "list_customer_project_revisions",
    ):
        assert getattr(customer_projects, name) is getattr(customer_project_profiles, name)
    for name in (
        "upsert_customer_project_profile",
        "get_customer_project_profile",
        "upsert_managed_object",
        "delete_managed_object",
        "rollback_customer_project_profile",
    ):
        assert getattr(customer_projects, name) is getattr(customer_project_profile_operations, name)
    assert customer_project_templates.create_customer_project_from_template is (
        customer_project_profile_operations.create_customer_project_from_template
    )
    for name in CUSTOMER_PROJECT_SCOPE_HELPERS:
        assert getattr(field_site_profile, name) is getattr(customer_project_scope, name)
    assert field_site_profile.build_solution_delivery_readiness is (
        solution_delivery_readiness.build_solution_delivery_readiness
    )
    assert customer_projects.build_solution_delivery_readiness is (
        solution_delivery_readiness.build_solution_delivery_readiness
    )
    assert field_site_profile.DEFAULT_DELIVERY_NAMESPACE is (
        customer_project_template_support.DEFAULT_DELIVERY_NAMESPACE
    )
    assert field_site_profile.TEMPLATE_PUBLISH_STATUSES is (
        customer_project_template_support.TEMPLATE_PUBLISH_STATUSES
    )
    assert field_site_profile.TEMPLATE_RELEASE_FIELDS is (
        customer_project_template_support.TEMPLATE_RELEASE_FIELDS
    )
    for name in TEMPLATE_RELEASE_PUBLIC_NAMES:
        assert getattr(customer_project_templates, name) is getattr(
            customer_project_template_release,
            name,
        )
        assert getattr(field_site_profile, name) is getattr(
            customer_project_template_release,
            name,
        )
    assert customer_project_artifacts.export_customer_project_template_release_notes_bundle is (
        customer_project_template_release.export_customer_project_template_release_notes_bundle
    )
    assert delivery_resources.rollback_delivery_resource_registry is (
        delivery_resource_registry.rollback_delivery_resource_registry
    )
    _assert_facade_exports(customer_project_resource_catalog, field_site_profile)
    _assert_facade_exports(customer_project_template_catalog, field_site_profile)
    _assert_facade_exports(customer_project_template_release, field_site_profile)
    _assert_facade_exports(delivery_resources, field_site_profile)
    _assert_facade_exports(customer_project_artifacts, field_site_profile)
    _assert_facade_exports(customer_project_templates, field_site_profile)
    _assert_facade_exports(customer_projects, field_site_profile)
    assert default_skill_audit_path() == root / "data" / "skill_audit.jsonl"
    assert default_skill_growth_state_path() == root / "data" / "skill_growth_backlog.json"
    assert MicInput(input_transport="usb_direct")._usb_direct_source_path() == (
        root / "scripts" / "bench" / "mcp01_usb_audio_libusb.c"
    )
    assert Path(root / "tests").is_dir()
