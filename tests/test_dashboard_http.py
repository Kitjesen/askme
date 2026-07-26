"""HTTP and static asset tests for the dashboard surface."""

from fastapi.testclient import TestClient

from askme.api.schemas.monitor import DashboardPageRegistryResponse
from askme.health_server import create_health_app
from tests.support.health_snapshots import minimal_runtime_snapshot as _runtime_snapshot


class TestDashboardHttp:
    def test_dashboard_contains_cognition_planning_controls(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/dashboard")

        assert response.status_code == 200
        assert 'id="dashboard-nav"' in response.text
        assert 'id="app-page"' in response.text
        retired_projects = client.get("/dashboard/projects", follow_redirects=False)
        assert retired_projects.status_code == 307
        assert retired_projects.headers["location"] == "/dashboard"
        for page in (
            "/dashboard/conversation",
            "/dashboard/scenarios",
            "/dashboard/field",
            "/dashboard/space",
            "/dashboard/knowledge",
            "/dashboard/capabilities",
            "/dashboard/voice",
            "/dashboard/delivery",
            "/dashboard/audit",
        ):
            page_response = client.get(page)
            assert page_response.status_code == 200
            assert 'id="app-page"' in page_response.text

        js_response = client.get("/dashboard/app.js")
        css_response = client.get("/dashboard/app.css")
        voice_css_response = client.get("/dashboard/voice.css")
        product_css_response = client.get("/dashboard/product.css")

        assert js_response.status_code == 200
        assert css_response.status_code == 200
        assert voice_css_response.status_code == 200
        assert product_css_response.status_code == 200
        assert "/dashboard/voice.css?v=20260712-5" in response.text
        assert "/dashboard/product.css?v=20260713-16" in response.text
        assert "/dashboard/app.js?v=20260713-6" in response.text
        assert 'class="workspace-bar"' in response.text
        assert 'id="nav-toggle"' in response.text
        assert 'id="nav-drawer"' in response.text
        assert 'id="nav-backdrop"' in response.text
        assert "body.nav-open .nav-rail" in product_css_response.text
        assert "body.voice-page-active .voice-command-bar" in product_css_response.text
        assert ".voice-workbench" in voice_css_response.text
        assert ".voice-command-title h1" in voice_css_response.text
        assert 'class="voice-command-route"' not in js_response.text
        assert "askme.voice / runtime" not in js_response.text
        assert "/api/governance/current-operator" in js_response.text
        assert "/api/dashboard/pages" in js_response.text
        assert "fallbackPages" in js_response.text
        assert "loadDashboardPageRegistry" in js_response.text
        assert "dashboardPageRegistryReady" in js_response.text
        assert "data-dashboard-page-section" in js_response.text
        assert "nav-section" in css_response.text
        assert "nav-section-title" in css_response.text
        assert "/dashboard/conversation" in js_response.text
        assert "/dashboard/projects" not in js_response.text
        assert "/dashboard/projects" not in response.text
        assert "/dashboard/scenarios" in js_response.text
        assert "/dashboard/field" in js_response.text
        assert "/dashboard/space" in js_response.text
        assert "/dashboard/knowledge" in js_response.text
        assert "/dashboard/capabilities" in js_response.text
        assert "/dashboard/voice" in js_response.text
        assert "/dashboard/delivery" in js_response.text
        assert "renderOverview" in js_response.text
        assert "需要关注的三件事" in js_response.text
        assert "dashboard-status-grid" in js_response.text
        assert "dashboard-event-grid" in js_response.text
        assert "dashboard-readiness-card" in js_response.text
        assert "dashboard-shortcuts" not in js_response.text
        assert "服务在线" in js_response.text
        assert "renderConversation" in js_response.text
        assert "现场上下文" in js_response.text
        assert "chat-service-point" in js_response.text
        assert "chat-location-summary" in js_response.text
        assert "loadConversationSpaceContext" in js_response.text
        assert "chatSpaceContextPayload" in js_response.text
        assert "service_point_id" in js_response.text
        assert "current_point_id" in js_response.text
        assert "咖啡店在哪" in js_response.text
        assert "chat-evidence-panel" in js_response.text
        assert "回答依据" in js_response.text
        assert "证据策略" in js_response.text
        assert "renderChatSpacePolicy" in js_response.text
        assert "只回答，不启动带路" in js_response.text
        assert "renderProjects" not in js_response.text
        assert "renderScenarios" in js_response.text
        assert "renderField" in js_response.text
        assert "renderKnowledge" in js_response.text
        assert "renderCapabilities" in js_response.text
        assert "renderVoice" in js_response.text
        assert "renderDelivery" in js_response.text
        assert "renderAudit" in js_response.text
        assert "renderAuditWorkspace" in js_response.text
        assert "/api/chat" in js_response.text
        assert "/api/knowledge/preview" in js_response.text
        assert "/api/knowledge/import" in js_response.text
        assert "/api/knowledge/list" in js_response.text
        assert "/api/memory/health" in js_response.text
        assert "/api/memory/search" in js_response.text
        assert "operator-card-head" in js_response.text
        assert "切换演示身份" in js_response.text
        assert "knowledge-command" in js_response.text
        assert "knowledge-workbar" in js_response.text
        assert "knowledge-workbench" in js_response.text
        assert "memory-search-result" in js_response.text
        assert "memory-health" in js_response.text
        assert "客户知识库" in js_response.text
        assert "上传说明" in js_response.text
        assert "knowledge-upload-guide" in js_response.text
        assert "knowledge-jump-upload" in js_response.text
        assert "已有知识" in js_response.text
        assert "新增知识" in js_response.text
        assert "knowledge-category-picker" in js_response.text
        assert "data-knowledge-category" in js_response.text
        assert "路线与带路" in js_response.text
        assert "商户与服务" in js_response.text
        assert "异常处置" in js_response.text
        assert "安防应急" in js_response.text
        assert "通知联系人" in js_response.text
        assert "传感器与协议" in js_response.text
        assert "knowledge-readable-title" in js_response.text
        assert "来源文件" in js_response.text
        assert "知识进入回答的规则" in js_response.text
        assert "先结构化" in js_response.text
        assert "再治理" in js_response.text
        assert "最后引用" in js_response.text
        assert "回答依据" in js_response.text
        assert js_response.text.count("function renderKnowledgeSummary") == 1
        assert js_response.text.count("function renderMemoryHealth") == 1
        assert js_response.text.count("function renderKnowledgeOperations") == 1
        assert js_response.text.count("function renderKnowledgeServiceError") == 1
        assert "knowledge-command" in css_response.text
        assert "knowledge-upload-guide" in css_response.text
        assert "knowledge-category-field" in css_response.text
        assert "knowledge-category-picker" in css_response.text
        assert "knowledge-kind" in css_response.text
        assert "knowledge-flow-card" in css_response.text
        assert "knowledge-flow-steps" in css_response.text
        assert "knowledge-search-result" in css_response.text
        assert "/api/space/service-point-trigger" in js_response.text
        assert "/api/space/guide" in js_response.text
        assert "/api/space/manage" in js_response.text
        assert "/api/space/rollback" in js_response.text
        assert "/api/space/history" in js_response.text
        assert "/api/space/proposals" in js_response.text
        assert "/api/space/proposals/review" in js_response.text
        assert "/api/runtime/handoff" in js_response.text
        assert "/api/surfaces" in js_response.text
        assert "renderApiSurfaceMap" in js_response.text
        assert "renderDashboardPageContracts" in js_response.text
        assert "primary_endpoint_status" in js_response.text
        assert "internal_surface_must_not_drive_customer_ui" in js_response.text
        assert "客户说明页依赖客户可见接口" in js_response.text
        assert "上层页面必须依赖客户可见接口" not in js_response.text
        assert "/api/blueprints" in js_response.text
        assert "/api/blueprints/park" in js_response.text
        assert "产品运行包和验收边界" in js_response.text
        assert "推荐园区运行包" in js_response.text
        assert "不能声明无人值守上线" in js_response.text
        assert "renderSpace" in js_response.text
        assert "renderSpaceChanges" in js_response.text
        assert "renderSpaceProposals" in js_response.text
        assert "renderSpaceGuideResult" in js_response.text
        assert "renderSpaceRuntimeSubmission" in js_response.text
        assert "renderBlueprintReadiness" in js_response.text
        assert "runtime_handoff_ready" in js_response.text
        assert "runtime_handoff_plan" in js_response.text
        assert "space-submit-runtime" in js_response.text
        assert "space-confirmed" in js_response.text
        assert "space-save-service-point" in js_response.text
        assert "space-save-route" in js_response.text
        assert "space-rollback" in js_response.text
        assert "space-propose-point" in js_response.text
        assert "space-form" in css_response.text
        assert "space-handoff" in css_response.text
        assert "/api/capability-center" in js_response.text
        assert "/api/scenario-intents" in js_response.text
        assert "/api/scenario-intents/preview" in js_response.text
        assert "/api/skill-audit" in js_response.text
        assert "/api/audit/events" in js_response.text
        assert "/api/audit/reviews" in js_response.text
        assert "/api/audit/export" in js_response.text
        assert "/api/audit/exports" in js_response.text
        assert "product_summary" in js_response.text
        assert "review_queue" in js_response.text
        assert 'id="audit-since"' in js_response.text
        assert "auditSeverityClass" in js_response.text
        assert 'id="audit-review-panel"' in js_response.text
        assert "renderAuditReviewPanel" in js_response.text
        assert "renderAuditReviewIntegrity" in js_response.text
        assert "renderAuditSourceHealth" in js_response.text
        assert "renderAuditExportHistory" in js_response.text
        assert "renderAuditDeliveryDossier" in js_response.text
        assert "delivery_dossier" in js_response.text
        assert "客户交付审计材料" in js_response.text
        assert "禁止声明" in js_response.text
        assert "无人值守生产上线声明" in js_response.text
        assert "source_health" in js_response.text
        assert "invalid_record_count" in js_response.text
        assert "renderAuditExportResult" in js_response.text
        assert "askme.audit_last_export" in js_response.text
        assert "evidence_summary" in js_response.text
        assert "renderAuditEvidenceRefs" in js_response.text
        assert "auditEvidenceHref" in js_response.text
        assert "isImageEvidence" in js_response.text
        assert "renderAuditReviewHistory" in js_response.text
        assert "wireAuditReviewPanelControls" in js_response.text
        assert "audit-source-health" in css_response.text
        assert "audit-source-grid" in css_response.text
        assert "audit-evidence-thumb" in css_response.text
        assert "audit-export-result" in css_response.text
        assert "audit-export-history" in css_response.text
        assert "audit-delivery-dossier" in css_response.text
        assert "/api/agent-profiles" in js_response.text
        assert "/api/skills/generated" in js_response.text
        assert "/api/skill-packages" in js_response.text
        assert "/api/skill-growth/backlog" in js_response.text
        assert 'id="agent-profile-name"' in js_response.text
        assert "data-agent-preview" in js_response.text
        assert "保存 Agent Profile" in js_response.text
        assert "renderAgentProfilePreview" in js_response.text
        assert "scenario_blueprints" in js_response.text
        assert "场景能力蓝图" in js_response.text
        assert "renderScenarioBlueprint" in js_response.text
        assert "runtime_blueprint_binding" in js_response.text
        assert "renderProjectRuntimeBlueprintBinding" in js_response.text
        assert "renderTemplateRuntimeBlueprintBinding" in js_response.text
        assert "template-runtime-binding" in css_response.text
        assert "/api/capability-packages" in js_response.text
        assert "renderCapabilityPackageItem" in js_response.text
        assert "renderScenarioPackageItem" in js_response.text
        assert "runtime_blueprints" in js_response.text
        assert "runtimeBlueprintsFromCatalog" in js_response.text
        assert "runtimeBlueprintNextStep" in js_response.text
        assert "renderRuntimeBlueprints" in js_response.text
        assert "renderRuntimeBlueprintItem" in js_response.text
        assert "运行方案和交付状态" in js_response.text
        assert "客户可验证能力包" in js_response.text
        assert "客户可启用能力包" not in js_response.text
        assert "enablement_decision" in js_response.text
        assert "release_summary" in js_response.text
        assert "生产声明" in js_response.text
        assert "生产上线声明必须有现场验收" in js_response.text
        assert "交付声明：" in js_response.text
        assert "package_readiness" in js_response.text
        assert "启用准入" in js_response.text
        assert "重新检查" in js_response.text
        assert "customer_next_step" in js_response.text
        assert "下一步：" in js_response.text
        assert "runtime-blueprints-panel" in css_response.text
        assert "dashboard-page-contract-card" in css_response.text
        assert "dashboard-page-contract-grid" in css_response.text
        assert "runtime-blueprint-grid" in css_response.text
        assert "runtime-blueprint-card" in css_response.text
        assert "api-surface-grid" in css_response.text
        assert "api-surface-item" in css_response.text
        assert "project-runtime-binding" in css_response.text
        assert "/draft" in js_response.text
        assert 'id="skill-package-id"' in js_response.text
        assert "data-skill-package" in js_response.text
        assert "data-growth-action" in js_response.text
        assert "生成草稿" in js_response.text
        assert "/preview" in js_response.text
        assert "预检" in js_response.text
        assert "知识管理" in js_response.text
        assert "导入并发布" in js_response.text
        assert "重建索引" in js_response.text
        assert "现场运行总览" in js_response.text
        assert "/api/governance/operator-directory" in js_response.text
        assert "/api/governance/identity-readiness" in js_response.text
        assert "data-identity-gateway-readiness" in js_response.text
        assert "企业身份准入" in js_response.text
        assert "只能演示或试点" in js_response.text
        assert "identity-header-grid" in css_response.text
        assert "knowledge-operations" in js_response.text
        assert "/api/field/scenarios" in js_response.text
        assert "/api/field/scenario-acceptance" in js_response.text
        assert "scenario-acceptance-strip" in js_response.text
        assert "scenario-acceptance-metrics" in css_response.text
        assert "renderFieldAdmissionDecision" in js_response.text
        assert "renderFieldIngestScopeContract" in js_response.text
        assert "field-admission-card" in js_response.text
        assert "admission_decision" in js_response.text
        assert "ingest_scope_contract" in js_response.text
        assert "field-admission-facts" in css_response.text
        assert "field-ingest-scope-card" in css_response.text
        assert "field-ingest-scope-grid" in css_response.text
        assert "/api/field/site-profiles" in js_response.text
        assert "/api/field/customer-projects" in js_response.text
        assert "/api/field/customer-project-workbench" in js_response.text
        assert "/api/field/customer-projects/managed-object-directory" in js_response.text
        assert "/api/field/customer-project-templates" in js_response.text
        assert "/api/field/customer-project-template-release-requests" in js_response.text
        assert "/api/field/customer-project-template-release-notes" in js_response.text
        assert "/api/field/customer-projects/import" in js_response.text
        assert "/api/field/customer-projects/package/verify" in js_response.text
        assert "/api/field/customer-projects/package/diff" in js_response.text
        assert "data-project-package-verify" in js_response.text
        assert "data-project-package-diff" in js_response.text
        assert "renderProjectPackageScopeEvidence" in js_response.text
        assert "当前账号可操作范围" in js_response.text
        assert "交付包归属" in js_response.text
        assert "导入演练" in js_response.text
        assert "project-scope-evidence" in css_response.text
        assert "/api/field/customer-projects/proposal-bundle/verify" in js_response.text
        assert "/api/field/customer-projects/acceptance-dossier/verify" in js_response.text
        assert "onsite-evidence" in js_response.text
        assert "include_readiness_auto" in js_response.text
        assert "acceptance-closure" in js_response.text
        assert "acceptance-review" in js_response.text
        assert "/api/field/customer-project-acceptance-registry" in js_response.text
        assert "/api/field/customer-project-resource-catalog" in js_response.text
        assert "/api/field/solution-delivery-readiness" in js_response.text
        assert "/api/field/product-launch-readiness" in js_response.text
        assert "/api/field/delivery-resource-registry" in js_response.text
        assert "/api/field/delivery-resource-registry/history" in js_response.text
        assert "/api/field/delivery-resource-registry/rollback" in js_response.text
        assert "/api/field/delivery-resource-governance-requests" in js_response.text
        assert "fieldDeliveryResourceRegistry" in js_response.text
        assert "fieldDeliveryResourceRegistryHistory" in js_response.text
        assert "fieldDeliveryResourceRegistryRollback" in js_response.text
        assert "fieldDeliveryResourceGovernanceRequests" in js_response.text
        assert "产品验收准入" in js_response.text
        assert "data-delivery-acceptance-gate" in js_response.text
        assert "data-project-product-acceptance-gate" in js_response.text
        assert "data-project-filter-apply" in js_response.text
        assert "customerProjectFilterQuery" in js_response.text
        assert "data-template-filter-apply" in js_response.text
        assert "customerProjectTemplateFilterQuery" in js_response.text
        assert "客户交付总门禁" in js_response.text
        assert "renderSolutionDeliveryReadiness" in js_response.text
        assert "客户上线准入总览" in js_response.text
        assert "renderProductLaunchReadiness" in js_response.text
        assert "data-product-launch-readiness" in js_response.text
        assert "product-launch-readiness" in css_response.text
        assert "验证：" in js_response.text
        assert "边界：" in js_response.text
        assert "外部服务：" in js_response.text
        assert "product-launch-gates" in css_response.text
        assert "验收引用登记" in js_response.text
        assert "renderAcceptanceRegistrySummary" in js_response.text
        assert "renderProjectResourceCatalogSummary" in js_response.text
        assert "resource_binding_status" in js_response.text
        assert "renderProjectBindingReadiness" in js_response.text
        assert "resource_binding_overall_status" in js_response.text
        assert "action_label" in js_response.text
        assert "owner_label" in js_response.text
        assert "customer_next_step" in js_response.text
        assert "交付资源登记" in js_response.text
        assert "共享交付资源登记表" in js_response.text
        assert "data-resource-register" in js_response.text
        assert "data-resource-history" in js_response.text
        assert "data-resource-disable" in js_response.text
        assert "data-resource-rollback" in js_response.text
        assert "data-resource-governance-requests" in js_response.text
        assert "data-resource-governance-review" in js_response.text
        assert "overdue_only=true" in js_response.text
        assert "review_sla" in js_response.text
        assert "due_at" in js_response.text
        assert "复核时限" in js_response.text
        assert "只看逾期" in js_response.text
        assert "升级逾期" in js_response.text
        assert "escalate-overdue" in js_response.text
        assert "renderDeliveryResourceGovernanceEscalation" in js_response.text
        assert "renderDeliveryResourceGovernanceEscalationResult" in js_response.text
        assert "renderDeliveryResourceGovernanceSla" in js_response.text
        assert "资源治理影响" in js_response.text
        assert "preview?.impact" in js_response.text
        assert "affected_consumers" in js_response.text
        assert "affected_customer_project_count" in js_response.text
        assert "affected_template_count" in js_response.text
        assert 'id="resource-project-id"' in js_response.text
        assert 'id="resource-type"' in js_response.text
        assert "registerDeliveryResourceFromForm" in js_response.text
        assert "renderDeliveryResourceGovernancePanel" in js_response.text
        assert "loadDeliveryResourceHistory" in js_response.text
        assert "disableDeliveryResource" in js_response.text
        assert "requestDeliveryResourceDisable" in js_response.text
        assert "reviewDeliveryResourceGovernanceRequest" in js_response.text
        assert "rollbackDeliveryResourceRegistry" in js_response.text
        assert "delivery-resource-form" in css_response.text
        assert "delivery-resource-governance" in css_response.text
        assert "resource-impact" in css_response.text
        assert "resource-sla" in css_response.text
        assert "resource-escalation" in css_response.text
        assert "资源绑定行动计划" in js_response.text
        assert "renderDeliveryResourceActionPlan" in js_response.text
        assert "data-delivery-resource-action-plan" in js_response.text
        assert "delivery-resource-action-plan" in css_response.text
        assert "incoming_binding_readiness_summary" in js_response.text
        assert "unregistered_resources" in js_response.text
        assert "导入预检" in js_response.text
        assert "renderProjectPackageDeliveryGate" in js_response.text
        assert "data-project-package-delivery-gate" in js_response.text
        assert "package_delivery_gate" in js_response.text
        assert "incoming_delivery_gate" in js_response.text
        assert "project-package-delivery-gate" in css_response.text
        assert "project-page-nav" not in js_response.text
        assert "polishProjectWorkspaceCopy" in js_response.text
        assert "data-product-workbench" not in js_response.text
        assert "renderProjectAcceptanceSnapshot" in js_response.text
        assert "customer_acceptance_snapshot" in js_response.text
        assert "data-project-acceptance-snapshot" in js_response.text
        assert "project-section-acceptance-summary" in js_response.text
        assert "project-acceptance-snapshot" in css_response.text
        assert "project-acceptance-grid" in css_response.text
        assert "renderProjectGoldenPathWorkbench" in js_response.text
        assert "data-project-golden-path" in js_response.text
        assert "方案商交付路径" in js_response.text
        assert "从行业模板到客户交付包，按验收节点推进" in js_response.text
        assert "renderProjectDeliveryChain" in js_response.text
        assert "data-project-delivery-chain" in js_response.text
        assert "project-delivery-chain" in css_response.text
        assert "project-golden-path" in css_response.text
        assert "客户项目不是配置文件，是交付产品" not in js_response.text
        assert "行业模板市场" in js_response.text
        assert "project-section-projects" in js_response.text
        assert "project-section-templates" in js_response.text
        assert "project-section-objects" in js_response.text
        assert "project-section-package" in js_response.text
        assert "project-section-acceptance" in js_response.text
        assert "project-section-resources" in js_response.text
        assert "project-section-events" in js_response.text
        assert "project-section-sites" in js_response.text
        assert "project-package-workbench" in css_response.text
        assert 'id="project-proposal-json"' in js_response.text
        assert 'id="project-dossier-json"' in js_response.text
        assert 'document.getElementById("project-proposal-json")' in js_response.text
        assert 'document.getElementById("project-dossier-json")' in js_response.text
        assert "客户项目交付包已生成" in js_response.text
        assert "客户方案包校验" in js_response.text
        assert "验收材料校验" in js_response.text
        assert "验收报告" in js_response.text
        assert "客户现场验收清单" in js_response.text
        assert "现场验收证据" in js_response.text
        assert "验收闭环" in js_response.text
        assert "客户签收" in js_response.text
        assert "customer-signoff" in js_response.text
        assert "fieldCustomerProjectCustomerSignoffSuffix" in js_response.text
        assert "fieldCustomerProjectExecutionBindingsSuffix" in js_response.text
        assert "data-project-execution-bindings" in js_response.text
        assert "执行接入计划" in js_response.text
        assert "renderCustomerProjectExecutionBindings" in js_response.text
        assert "renderExecutionScopeConstraints" in js_response.text
        assert "范围约束" in js_response.text
        assert "loadSelectedCustomerProjectExecutionBindings" in js_response.text
        assert "approval_policy" in js_response.text
        assert "output_contract" in js_response.text
        assert "hardware_boundary" in js_response.text
        assert "renderExecutionAdapterContracts" in js_response.text
        assert "adapter_contract" in js_response.text
        assert "field-ingest-bridge" in js_response.text
        assert "dry_run_command" in js_response.text
        assert "live_command" in js_response.text
        assert "data-object-rehearsal" in js_response.text
        assert "rehearseCustomerProjectObject" in js_response.text
        assert "renderObjectExecutionRehearsalResult" in js_response.text
        assert "实验室演示证据，不能作为生产上线验收依据。" in js_response.text
        assert "register_onsite_evidence" in js_response.text
        assert "renderEvidenceBoundaryTags" in js_response.text
        assert "production_eligible" in js_response.text
        assert "\\u9a8c\\u6536\\u5019\\u9009\\u8bc1\\u636e" in js_response.text
        assert "data-project-customer-signoff-load" in js_response.text
        assert "data-project-customer-signoff-submit" in js_response.text
        assert "project-customer-signatory-name" in js_response.text
        assert "project-customer-signoff-credential-ref" in js_response.text
        assert "project-customer-signoff-credential-sha256" in js_response.text
        assert "签收凭证 SHA-256" in js_response.text
        assert "customer_signoff" in js_response.text
        assert "ready_for_customer_signoff" in js_response.text
        assert "accepted_by_customer" in js_response.text
        assert "renderCustomerProjectCustomerSignoff" in js_response.text
        assert "loadSelectedCustomerProjectCustomerSignoff" in js_response.text
        assert "registerSelectedCustomerProjectCustomerSignoff" in js_response.text
        assert "客户验收材料" in js_response.text
        assert "系统自动采信" in js_response.text
        assert "managed-objects" in js_response.text
        assert "acceptance_checks" in js_response.text
        assert "field_changes" in js_response.text
        assert "data-managed-object-summary" in js_response.text
        assert "data-managed-object-export" in js_response.text
        assert "fieldCustomerProjectManagedObjectDirectory" in js_response.text
        assert "当前可见" in js_response.text
        assert "managedObjectDirectorySummary" in js_response.text
        assert "managedObjectExportRows" in js_response.text
        assert "managedObjectDirectoryKey" in js_response.text
        assert "renderManagedObjectActionPlan" in js_response.text
        assert "data-managed-object-action-plan" in js_response.text
        assert "action_plan" in js_response.text
        assert "managed-object-actions" in css_response.text
        assert "导出对象目录" in js_response.text
        assert "导出可交付对象清单" in js_response.text
        assert "managed-object-checks" in css_response.text
        assert "field_readiness" in js_response.text
        assert "evidence_reports" in js_response.text
        assert "onsite_acceptance_evidence" in js_response.text
        assert "data-acceptance-gates" in js_response.text
        assert "data-project-acceptance-report" in js_response.text
        assert "data-project-acceptance-dossier" in js_response.text
        assert "data-project-lifecycle-onsite-evidence" in js_response.text
        assert "data-project-lifecycle-onsite-load" in js_response.text
        assert "data-project-lifecycle-closure" in js_response.text
        assert "data-project-lifecycle-review" in js_response.text
        assert 'id="project-acceptance-evidence-refs"' in js_response.text
        assert 'id="project-acceptance-evidence-picker"' in js_response.text
        assert "data-acceptance-evidence-add" in js_response.text
        assert "setAcceptanceEvidenceOptions" in js_response.text
        assert "addSelectedAcceptanceEvidenceRef" in js_response.text
        assert "onsiteReceiptEvidenceRef" in js_response.text
        assert "acceptance-evidence-picker" in css_response.text
        assert (
            "evidence_refs: commaList(document.getElementById(\"project-acceptance-evidence-refs\")"
            in js_response.text
        )
        assert "renderCustomerProjectOnsiteEvidence" in js_response.text
        assert "renderSiteAcceptanceChecklist" in js_response.text
        assert "site_acceptance_checklist" in js_response.text
        assert "客户现场验收清单" in js_response.text
        assert "onsiteReceiptSourceLabel" in js_response.text
        assert "field_readiness_auto_backfill" in js_response.text
        assert "系统自动采信" in js_response.text
        assert "renderCustomerProjectAcceptanceClosure" in js_response.text
        assert "registerSelectedCustomerProjectOnsiteEvidence" in js_response.text
        assert "registerSelectedCustomerProjectAcceptanceReview" in js_response.text
        assert "data-project-proposal" in js_response.text
        assert "data-project-proposal-verify" in js_response.text
        assert "data-project-dossier-verify" in js_response.text
        assert "data-project-lifecycle-proposal" in js_response.text
        assert "renderCustomerProjectAcceptanceDossier" in js_response.text
        assert "renderCustomerProjectProposalBundle" in js_response.text
        assert "verifyCustomerProjectProposalBundle" in js_response.text
        assert "verifyCustomerProjectAcceptanceDossier" in js_response.text
        assert "renderProjectDossierVerifyResult" in js_response.text
        assert "renderProjectProposalVerifyResult" in js_response.text
        assert "acceptance-dossier" in js_response.text
        assert "proposal-bundle" in js_response.text
        assert "exportCustomerProjectProposalBundle" in js_response.text
        assert "downloadCustomerProjectProposalBundle" in js_response.text
        assert "html_path" in js_response.text
        assert "Printable HTML" in js_response.text
        assert "renderCustomerProjectAcceptanceReport" in js_response.text
        assert "客户项目与对象目录" not in js_response.text
        assert "项目交付包导入" in js_response.text
        assert "事件归属检查" in js_response.text
        assert "data-project-lifecycle-export" in js_response.text
        assert "data-project-lifecycle-history" in js_response.text
        assert "data-project-lifecycle-rollback-dry" in js_response.text
        assert "data-project-lifecycle-rollback" in js_response.text
        assert "project-rollback-revision" in js_response.text
        assert "loadSelectedCustomerProjectHistory" in js_response.text
        assert "rollbackSelectedCustomerProject" in js_response.text
        assert "renderProjectRevisionHistory" in js_response.text
        assert "renderProjectRollbackResult" in js_response.text
        assert "data-project-lifecycle-archive" in js_response.text
        assert "data-object-delete" in js_response.text
        assert "renderManagedObjectWriteResult" in js_response.text
        assert "data-managed-object-write-result" in js_response.text
        assert "对象变更后的实施步骤" in js_response.text
        assert "对象已保存" in js_response.text
        assert "对象已下线" in js_response.text
        assert "renderProjectExportResult" in js_response.text
        assert "renderProjectScopeLabel" in js_response.text
        assert "renderProjectCollisionCandidates" in js_response.text
        assert "交付包冲突项" in js_response.text
        assert 'id="project-tenant-id"' in js_response.text
        assert 'id="project-delivery-namespace"' in js_response.text
        assert "tenant_id: document.getElementById(\"project-tenant-id\")" in js_response.text
        assert (
            "delivery_namespace: document.getElementById(\"project-delivery-namespace\")"
            in js_response.text
        )
        assert "项目基础信息" in js_response.text
        assert 'id="project-edit-id"' in js_response.text
        assert "data-project-edit-load" in js_response.text
        assert "data-project-edit-save" in js_response.text
        assert "loadProjectProfileForEdit" in js_response.text
        assert "saveProjectProfileMetadata" in js_response.text
        assert "currentProjectEditProfile" in js_response.text
        assert "对象目录" in js_response.text
        assert "data-managed-object-directory" in js_response.text
        assert "renderManagedObjectDirectory" in js_response.text
        assert "renderManagedObjectCard" in js_response.text
        assert "对象变更记录" in js_response.text
        assert "data-object-change-log" in js_response.text
        assert "renderObjectChangeLog" in js_response.text
        assert "object-change-log" in css_response.text
        assert "交付流程" in js_response.text
        assert "data-project-delivery-workflow" in js_response.text
        assert "renderProjectDeliveryWorkflow" in js_response.text
        assert "project-delivery-workflow" in css_response.text
        assert "project-delivery-step" in css_response.text
        assert "data-object-load" in js_response.text
        assert "loadManagedObjectIntoEditor" in js_response.text
        assert "currentCustomerProjectItems" in js_response.text
        assert "managed-object-directory" in css_response.text
        assert "managed-object-card" in css_response.text
        assert "managed-object-bindings" in css_response.text
        assert "object-delete-impact" in js_response.text
        assert "object-offline-impact" in css_response.text
        assert "renderManagedObjectOfflineImpact" in js_response.text
        assert "updateManagedObjectDeleteImpact" in js_response.text
        assert "移除现场对象前必须填写下线原因" in js_response.text
        assert "data-managed-object-editor" in js_response.text
        assert "基础对象" in js_response.text
        assert "识别范围" in js_response.text
        assert "能力配置" in js_response.text
        assert "绑定交付资源" in js_response.text
        assert "加入绑定" in js_response.text
        assert "data-object-resource-picker" in js_response.text
        assert "data-object-resource-add" in js_response.text
        assert 'id="object-resource-picker"' in js_response.text
        assert "addSelectedObjectResourceBinding" in js_response.text
        assert "objectBindingInputId" in js_response.text
        assert "object-resource-picker" in css_response.text
        assert 'id="object-labels"' in js_response.text
        assert 'id="object-responder-group"' in js_response.text
        assert 'id="object-evidence-required"' in js_response.text
        assert 'id="object-vision-models"' in js_response.text
        assert 'id="object-sensor-protocols"' in js_response.text
        assert "setObjectEditInput(\"object-vision-models\"" in js_response.text
        assert "setObjectEditInput(\"object-sensor-protocols\"" in js_response.text
        assert 'id="object-project-ids"' in js_response.text
        assert 'id="object-site-ids"' in js_response.text
        assert "客户范围保护" in js_response.text
        assert "object-project-ids" in js_response.text
        assert "tenant_ids: commaList(document.getElementById(\"object-tenant-ids\")" in js_response.text
        assert "project_ids: commaList(document.getElementById(\"object-project-ids\")" in js_response.text
        assert (
            "vision_models: commaList(document.getElementById(\"object-vision-models\")"
            in js_response.text
        )
        assert (
            "sensor_protocols: commaList(document.getElementById(\"object-sensor-protocols\")"
            in js_response.text
        )
        assert "验收证据" in js_response.text
        assert "managed-object-editor" in css_response.text
        assert "object-editor-section" in css_response.text
        assert "incoming_delivery_scope" in js_response.text
        assert "collision_candidates" in js_response.text
        assert "delivery_namespace_count" in js_response.text
        assert "project-collision-warning" in css_response.text
        assert "deleteJson" in js_response.text
        assert "多现场交付" in js_response.text
        assert "多现场交付" in js_response.text
        assert "客户项目目录" in js_response.text
        assert "行业模板市场" in js_response.text
        assert "从模板创建项目" in js_response.text
        assert "data-project-template-create-readiness" in js_response.text
        assert "renderProjectTemplateCreateReadiness" in js_response.text
        assert "renderCustomerProjectCreateResult" in js_response.text
        assert "implementation_handoff" in js_response.text
        assert "renderProjectImplementationHandoff" in js_response.text
        assert "data-project-implementation-handoff" in js_response.text
        assert "data-project-detail-handoff" in js_response.text
        assert "项目详情加载后的实施步骤" in js_response.text
        assert "导入演练后的实施步骤" in js_response.text
        assert "保存后的实施步骤" in js_response.text
        assert "项目信息已保存" in js_response.text
        assert "data-created-project-load" in js_response.text
        assert "data-created-object-guide" in js_response.text
        assert "客户项目已创建" in js_response.text
        assert "补齐对象绑定" in js_response.text
        assert "创建准入" in js_response.text
        assert "客户配合项" in js_response.text
        assert "暂不承诺" in js_response.text
        assert "project-create-readiness" in css_response.text
        assert "project-create-result-card" in css_response.text
        assert "project-create-next-steps" in css_response.text
        assert "template-market-card" in css_response.text
        assert "template-delivery-checklist" in css_response.text
        assert "template_package" in js_response.text
        assert "renderTemplatePackageReadiness" in js_response.text
        assert "renderTemplateApplicabilityScope" in js_response.text
        assert "renderTemplateCustomerPrerequisites" in js_response.text
        assert "renderTemplateScenarioAcceptanceCriteria" in js_response.text
        assert "renderTemplateOutOfScope" in js_response.text
        assert "模板交付包" in js_response.text
        assert "renderTemplateDeliveryChecklist" in js_response.text
        assert "renderTemplateObjectPreview" in js_response.text
        assert "data-template-select" in js_response.text
        assert "data-template-release" in js_response.text
        assert "data-template-history" in js_response.text
        assert "data-template-release-request" in js_response.text
        assert "data-template-release-requests" in js_response.text
        assert "updateTemplateRelease" in js_response.text
        assert "createTemplateReleaseRequest" in js_response.text
        assert "reviewTemplateReleaseRequest" in js_response.text
        assert "project-section-template-governance" in js_response.text
        assert "renderTemplateReleaseGovernance" in js_response.text
        assert "renderTemplateReleaseGovernanceRequest" in js_response.text
        assert "template-governance-result" in js_response.text
        assert "templateReleaseGovernanceResultEl" in js_response.text
        assert "wireTemplateReleaseReviewControls" in js_response.text
        assert "template-governance-board" in css_response.text
        assert "renderTemplateReleaseRequests" in js_response.text
        assert "renderTemplateReleaseNotes" in js_response.text
        assert "exportTemplateReleaseNotesBundle" in js_response.text
        assert "downloadTemplateReleaseNotesBundle" in js_response.text
        assert "downloadTextFile" in js_response.text
        assert "URL.createObjectURL" in js_response.text
        assert "proposal_insert" in js_response.text
        assert "data-template-release-notes-export" in js_response.text
        assert "/api/field/customer-project-template-release-notes/export" in js_response.text
        assert "renderTemplateReleaseHistory" in js_response.text
        assert "请输入发布治理原因" in js_response.text
        assert "请输入模板发布申请原因" in js_response.text
        assert "模板发布说明" in js_response.text
        assert "selectTemplateForCreate" in js_response.text
        assert "对象快速维护" in js_response.text
        assert "/api/field/events" in js_response.text
        assert "/api/field/notification-preflight?status_as_200=true" in js_response.text
        assert "renderFieldEventDetail" in js_response.text
        assert "incident_workflow" in js_response.text
        assert "action_audit" in js_response.text
        assert "runtime_delivery" in js_response.text
        assert "resend-notification" in js_response.text
        assert "request-close" in js_response.text
        assert "/api/field/readiness" in js_response.text
        assert "/api/field/devices" in js_response.text
        assert "/api/field/device-onboarding" in js_response.text
        assert "renderDeviceOnboarding" in js_response.text
        assert "device-onboarding-card" in css_response.text
        assert "device-onboarding-metrics" in css_response.text
        assert "暂无现场事件" in js_response.text
        assert "保安群" in js_response.text
        assert "/api/voice/profiles" in js_response.text
        assert "/api/voice/profile" in js_response.text
        assert "asr_final_ms" in js_response.text
        assert "llm_ttft_ms" in js_response.text
        assert "tts_first_audio_ms" in js_response.text
        assert "playback_start_ms" in js_response.text
        assert "speak: true" in js_response.text
        assert "play_audio: true" in js_response.text


    def test_dashboard_pages_endpoint_returns_product_page_map(self):
        client = TestClient(create_health_app(lambda: _runtime_snapshot()))

        response = client.get("/api/dashboard/pages")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        data = response.json()
        DashboardPageRegistryResponse.model_validate(data)
        assert data["ok"] is True
        assert data["summary"]["page_count"] == 10
        assert data["summary"]["internal_page_count"] == 0
        assert data["summary"]["primary_endpoint_missing_count"] == 0
        assert data["summary"]["primary_endpoint_internal_count"] == 0
        assert data["summary"]["primary_endpoint_unclassified_count"] == 0
        assert data["summary"]["section_counts"] == {
            "customer": 6,
            "governance": 2,
            "operations": 2,
        }
        assert data["policy"]["internal_runtime_is_not_a_customer_page"] is True
        assert data["policy"]["dashboard_shell_uses_registered_pages"] is True
        assert data["policy"]["primary_endpoints_must_exist_in_route_inventory"] is True
        assert (
            data["policy"]["customer_pages_must_not_point_to_internal_or_unclassified_routes"]
            is True
        )

        by_key = {page["key"]: page for page in data["pages"]}
        assert set(by_key) == {
            "overview",
            "conversation",
            "scenarios",
            "field",
            "space",
            "knowledge",
            "capabilities",
            "voice",
            "delivery",
            "audit",
        }
        assert by_key["overview"]["section"] == "customer"
        assert by_key["knowledge"]["title"] == "客户知识库"
        assert by_key["knowledge"]["primary_endpoint"] == "/api/knowledge/list"
        assert by_key["delivery"]["section"] == "governance"
        assert by_key["audit"]["primary_endpoint"] == "/api/audit/events"
        assert all(page["primary_endpoint"] for page in data["pages"])
        assert all(page["evidence_promises"] for page in data["pages"])
        assert all(page["primary_endpoint_status"]["available"] for page in data["pages"])
        assert all(page["primary_endpoint_status"]["customer_safe"] for page in data["pages"])
        assert by_key["overview"]["primary_endpoint_status"]["surfaces"] == ["platform"]
        assert by_key["audit"]["primary_endpoint_status"]["surfaces"] == ["admin"]
        assert by_key["knowledge"]["primary_endpoint_status"]["surfaces"] == ["product"]

        surfaces = client.get("/api/surfaces").json()["route_inventory"]["routes"]
        page_route = next(route for route in surfaces if route["path"] == "/api/dashboard/pages")
        assert page_route["surface"] == "product"
