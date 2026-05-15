const BASE = location.origin;
const app = document.getElementById("app-page");
const nav = document.getElementById("dashboard-nav");
const pageTitle = document.getElementById("page-title");
const pageKicker = document.getElementById("page-kicker");
const pageDescription = document.getElementById("page-description");
const globalStatusDot = document.getElementById("global-status-dot");
const globalStatusText = document.getElementById("global-status-text");
const ENDPOINTS = {
  governance: "/api/governance/operator-directory",
  identityReadiness: "/api/governance/identity-readiness",
  currentOperator: "/api/governance/current-operator",
  authorize: "/api/governance/authorize",
  auditEvents: "/api/audit/events",
  auditReviews: "/api/audit/reviews",
  auditExport: "/api/audit/export",
  auditExports: "/api/audit/exports",
  auditExportRetry: "/api/audit/export/retry",
  knowledgePreview: "/api/knowledge/preview",
  knowledgeImport: "/api/knowledge/import",
  knowledgeList: "/api/knowledge/list",
  knowledgeUpdate: "/api/knowledge/update",
  memorySearch: "/api/memory/search",
  spaceHealth: "/api/space/health",
  spacePoints: "/api/space/points",
  spaceServicePoints: "/api/space/service-points",
  spaceRoutes: "/api/space/routes",
  spaceHistory: "/api/space/history",
  spaceResolveDestination: "/api/space/resolve-destination",
  spaceGuide: "/api/space/guide",
  spaceServicePointTrigger: "/api/space/service-point-trigger",
  spaceManage: "/api/space/manage",
  spaceRollback: "/api/space/rollback",
  spaceProposals: "/api/space/proposals",
  spaceProposalReview: "/api/space/proposals/review",
  fieldSiteProfiles: "/api/field/site-profiles",
  fieldCustomerProjects: "/api/field/customer-projects",
  fieldCustomerProjectWorkbench: "/api/field/customer-project-workbench",
  fieldCustomerProjectManagedObjectDirectory: "/api/field/customer-projects/managed-object-directory",
  fieldCustomerProjectTemplates: "/api/field/customer-project-templates",
  fieldCustomerProjectTemplateReleaseRequests: "/api/field/customer-project-template-release-requests",
  fieldCustomerProjectTemplateReleaseNotes: "/api/field/customer-project-template-release-notes",
  fieldCustomerProjectTemplateReleaseNotesExport: "/api/field/customer-project-template-release-notes/export",
  fieldCustomerProjectAcceptanceRegistry: "/api/field/customer-project-acceptance-registry",
  fieldCustomerProjectResourceCatalog: "/api/field/customer-project-resource-catalog",
  fieldSolutionDeliveryReadiness: "/api/field/solution-delivery-readiness",
  fieldProductLaunchReadiness: "/api/field/product-launch-readiness",
  fieldDeliveryResourceRegistry: "/api/field/delivery-resource-registry",
  fieldDeliveryResourceRegistryHistory: "/api/field/delivery-resource-registry/history",
  fieldDeliveryResourceRegistryRollback: "/api/field/delivery-resource-registry/rollback",
  fieldDeliveryResourceGovernanceRequests: "/api/field/delivery-resource-governance-requests",
  fieldCustomerProjectImport: "/api/field/customer-projects/import",
  fieldCustomerProjectPackageVerify: "/api/field/customer-projects/package/verify",
  fieldCustomerProjectPackageDiff: "/api/field/customer-projects/package/diff",
  fieldCustomerProjectProposalBundleSuffix: "proposal-bundle",
  fieldCustomerProjectOnsiteEvidenceSuffix: "onsite-evidence",
  fieldCustomerProjectAcceptanceClosureSuffix: "acceptance-closure",
  fieldCustomerProjectAcceptanceReviewSuffix: "acceptance-review",
  fieldCustomerProjectCustomerSignoffSuffix: "customer-signoff",
  fieldCustomerProjectExecutionBindingsSuffix: "execution-bindings",
  fieldCustomerProjectProposalBundleVerify: "/api/field/customer-projects/proposal-bundle/verify",
  fieldCustomerProjectAcceptanceDossierVerify: "/api/field/customer-projects/acceptance-dossier/verify",
  runtimeHandoff: "/api/runtime/handoff",
  capabilityCenter: "/api/capability-center",
  capabilityPackages: "/api/capability-packages",
  capabilityPackageReadiness: "/api/capability-packages/readiness",
  skillAudit: "/api/skill-audit",
  agentProfiles: "/api/agent-profiles",
  generatedSkills: "/api/skills/generated",
  skillPackages: "/api/skill-packages",
  skillGrowthBacklog: "/api/skill-growth/backlog",
  blueprints: "/api/blueprints",
};

const pages = [
  { path: "/dashboard", key: "overview", label: "总览", hint: "功能地图", title: "现场任务平台", kicker: "产品总览", desc: "给客户和交付团队看的功能地图：语音入口、客户项目、现场事件、知识库、能力中心和交付检查分开验收。" },
  { path: "/dashboard/conversation", key: "conversation", label: "对话", hint: "语音文本", title: "语音和文本对话", kicker: "真实交互", desc: "用于输入任务、问路、知识问答和安全确认。回答需要展示证据、任务状态和拒答原因。" },
  { path: "/dashboard/projects", key: "projects", label: "客户项目", hint: "对象目录", title: "客户项目与对象目录", kicker: "方案商交付", desc: "为每个客户项目维护园区、厂区、仓储或景区对象目录，并绑定视觉模型、传感器协议、技能包、验收用例和交付包。" },
  { path: "/dashboard/field", key: "field", label: "现场事件", hint: "安防巡检", title: "现场事件处置", kicker: "园区场景", desc: "覆盖摔倒、卡住、陌生人拍照、违停、烟雾火灾、垃圾桶满溢、人群聚集、游客问路和带路。" },
  { path: "/dashboard/space", key: "space", label: "空间认知", hint: "问路带路", title: "园区空间认知", kicker: "访客服务", desc: "管理园区点位、别名、问询服务点和带路路线，模拟访客停留后的主动问候、目的地解析和带路决策。" },
  { path: "/dashboard/knowledge", key: "knowledge", label: "知识库", hint: "上传审批", title: "知识管理", kicker: "可审计回答", desc: "上传、预览、审批、检索和重建索引。过期、冲突或未审批知识不能直接进入回答。" },
  { path: "/dashboard/capabilities", key: "capabilities", label: "能力中心", hint: "技能增长", title: "机器人能力中心", kicker: "客户可见能力", desc: "按巡检、异常处置、安防、访客服务、空间认知和在线增长展示机器人当前能做什么、缺什么、哪些能力需要审批。" },
  { path: "/dashboard/voice", key: "voice", label: "语音音色", hint: "播报策略", title: "语音音色和实时链路", kicker: "声音系统", desc: "按巡检、访客、安防、紧急告警、夜间低扰等场景切换音色和提示音，并查看端到端延迟。" },
  { path: "/dashboard/delivery", key: "delivery", label: "交付检查", hint: "可验收", title: "交付检查", kicker: "上线门禁", desc: "把演示、试点、真实硬件和外部通知的缺口拆成清晰门禁，避免把实验室能力说成生产上线。" },
  { path: "/dashboard/audit", key: "audit", label: "审计", hint: "证据包", title: "审计证据包", kicker: "交付证据", desc: "查看客户可读的事件证据、复核状态、导出历史和交付声明边界。" },
];
let health = {};
let governance = { operators: [] };
let operatorSession = null;
let identityReadiness = null;
let liveBaseline = null;
let chatStarted = false;
let chatRenderedCount = 0;
let selectedFieldEventId = null;
let fieldActionResult = null;
let selectedGeneratedSkillPreview = null;
let selectedAgentProfilePreview = null;
let auditRecordCache = [];
let selectedAuditReview = null;
let latestSpaceGuidePayload = null;
let capabilityScenarioItems = {};
let currentProjectEditProfile = null;
let currentCustomerProjectItems = [];
let currentCustomerProjectTemplateItems = [];
let currentDeliveryResourceItems = [];
const DELIVERY_RESOURCE_TYPES = [
  "vision_models",
  "sensor_protocols",
  "skill_packages",
  "acceptance_tests",
];

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function safeDomId(value) {
  return String(value || "item").replace(/[^A-Za-z0-9_-]/g, "-");
}

function currentPage() {
  const path = location.pathname.replace(/\/$/, "") || "/dashboard";
  return pages.find((page) => page.path === path) || pages[0];
}

function setHeader(page) {
  pageTitle.textContent = page.title;
  pageKicker.textContent = page.kicker;
  pageDescription.textContent = page.desc;
}

function renderNav(activePage) {
  nav.innerHTML = pages.map((page) => `
    <a class="nav-link ${page.key === activePage.key ? "active" : ""}" href="${page.path}">
      <span>${page.label}</span><small>${page.hint}</small>
    </a>
  `).join("");
}

function routeTo(path) {
  history.pushState({}, "", path);
  render();
}

async function getJson(path, fallback = null) {
  try {
    const response = await fetch(BASE + path, {
      headers: {
        "X-Askme-Operator-Id": operatorId(),
      },
    });
    if (!response.ok) return fallback;
    return await response.json();
  } catch {
    return fallback;
  }
}

async function postJson(path, body, fallback = null) {
  try {
    const response = await fetch(BASE + path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Askme-Operator-Id": operatorId(),
      },
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => ({}));
    return { ok: response.ok, payload };
  } catch (error) {
    return fallback ?? { ok: false, payload: { error: error.message } };
  }
}

async function deleteJson(path, body, fallback = null) {
  try {
    const response = await fetch(BASE + path, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
        "X-Askme-Operator-Id": operatorId(),
      },
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => ({}));
    return { ok: response.ok, payload };
  } catch (error) {
    return fallback ?? { ok: false, payload: { error: error.message } };
  }
}

function downloadTextFile(filename, content, mimeType = "text/plain;charset=utf-8") {
  if (!filename || content === undefined || content === null) return false;
  const blob = new Blob([String(content)], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  return true;
}

function operatorId() {
  return localStorage.getItem("askme.operator_id") || "dashboard.operator";
}

function setOperatorId(value) {
  const clean = String(value || "").trim() || "dashboard.operator";
  localStorage.setItem("askme.operator_id", clean);
}

function currentOperator() {
  const id = operatorId();
  if (operatorSession?.operator?.operator_id === id) return operatorSession.operator;
  const operators = Array.isArray(governance.operators) ? governance.operators : [];
  return operators.find((operator) => operator.operator_id === id) || { operator_id: id, roles: [], source: "unresolved", known: false, authenticated: false };
}

function operatorRolesText(operator = currentOperator()) {
  const roles = Array.isArray(operator.roles) ? operator.roles : [];
  return roles.length ? roles.join(" / ") : "无权限";
}

function currentOperatorPermissions() {
  return Array.isArray(operatorSession?.permissions) ? operatorSession.permissions : [];
}

function renderOperatorCard() {
  const operators = Array.isArray(governance.operators) ? governance.operators : [];
  const active = currentOperator();
  const activeKnown = active.known !== false;
  const readiness = governance.readiness || operatorSession?.readiness || {};
  const findings = Array.isArray(readiness.findings) ? readiness.findings : [];
  const warnings = Array.isArray(operatorSession?.warnings) ? operatorSession.warnings : [];
  const options = operators.length
    ? `${activeKnown ? "" : `<option value="${esc(active.operator_id)}" selected>${esc(active.operator_id)}（未登记）</option>`}${operators.map((operator) => `<option value="${esc(operator.operator_id)}" ${operator.operator_id === active.operator_id ? "selected" : ""}>${esc(operator.display_name || operator.operator_id)} (${esc(operatorRolesText(operator))})</option>`).join("")}`
    : `<option value="${esc(active.operator_id)}">${esc(active.operator_id)} (${esc(operatorRolesText(active))})</option>`;
  return `
    <div class="operator-card">
      <div>
        <strong>当前操作人</strong>
        <p>${esc(active.display_name || active.operator_id)} · ${esc(operatorRolesText(active))}</p>
      </div>
      <select id="operator-select">${options}</select>
      <div class="operator-meta">
        ${badge(activeKnown ? "已在目录" : "未登记", activeKnown ? "ok" : "err")}
        ${badge(active.authenticated ? "企业身份" : "本地演示身份", active.authenticated ? "ok" : "warn")}
        ${badge(`${currentOperatorPermissions().length} 项权限`, currentOperatorPermissions().length ? "ok" : "err")}
      </div>
      <p>${esc(governance.mode || "demo_config")} / ${esc(governance.identity_provider || "local_config")}；${esc(readiness.status || "demo_or_trial_only")}</p>
      ${warnings.length ? `<div class="operator-warnings">${warnings.slice(0, 2).map((item) => `<span>${esc(item)}</span>`).join("")}</div>` : ""}
      ${findings.length ? `<div class="operator-warnings">${findings.slice(0, 2).map((item) => `<span>${esc(item.message || item.code)}</span>`).join("")}</div>` : ""}
    </div>
  `;
}

function renderIdentityGatewayCard(readiness = identityReadiness || governance.identity_gateway_readiness || {}) {
  const blockers = Array.isArray(readiness.blockers) ? readiness.blockers : [];
  const warnings = Array.isArray(readiness.warnings) ? readiness.warnings : [];
  const contract = readiness.trusted_gateway_contract || {};
  const requiredHeaders = Array.isArray(contract.required_headers) ? contract.required_headers : [];
  const scopeHeaders = Array.isArray(contract.scope_headers) ? contract.scope_headers : [];
  const status = readiness.status || "blocked";
  return `
    <div class="card identity-gateway-card" data-identity-gateway-readiness>
      <div class="section-title-row">
        <div>
          <h2>企业身份准入</h2>
          <p>给方案商交付团队判断：当前账号体系能否支撑多客户、多项目、租户隔离和高风险操作审计。</p>
        </div>
        ${badge(status, statusClass(status) || acceptanceGateClass(status))}
      </div>
      <div class="metric"><b>身份模式</b><span>${esc(readiness.identity_mode || "demo_operator_directory")}</span></div>
      <div class="metric"><b>身份来源</b><span>${esc(readiness.identity_provider || governance.identity_provider || "local_config")}</span></div>
      <div class="metric"><b>生产声明</b><span>${esc(readiness.production_ready ? "可进入生产准入测试" : "只能演示或试点")}</span></div>
      <div class="metric"><b>受信网关</b><span>${esc(readiness.trusted_identity_headers_enabled ? "已启用" : "未启用")}</span></div>
      <p>${esc(readiness.customer_status || "当前只能用于演示、实验室或客户试点，不能声明无人值守生产上线。")}</p>
      <div class="operator-warnings light">
        ${blockers.slice(0, 3).map((item) => `<span>${esc(item.message || item.code)}</span>`).join("")}
        ${!blockers.length && warnings.slice(0, 2).map((item) => `<span>${esc(item.message || item.code)}</span>`).join("")}
      </div>
      <details>
        <summary>查看网关 header 合约</summary>
        <div class="identity-header-grid">
          ${requiredHeaders.concat(scopeHeaders).map((item) => `
            <span class="${item.configured ? "ok" : (item.required ? "err" : "warn")}">
              <b>${esc(item.claim)}</b>
              <small>${esc(item.header || "未配置")}${item.required ? " / 必填" : ""}</small>
            </span>
          `).join("")}
        </div>
      </details>
    </div>
  `;
}

function wireOperatorControls() {
  const select = document.getElementById("operator-select");
  if (!select) return;
  select.addEventListener("change", () => {
    setOperatorId(select.value);
    render();
  });
}

function badge(text, cls = "") {
  return `<span class="badge ${cls}">${esc(text)}</span>`;
}

function statusClass(value) {
  const text = String(value || "").toLowerCase();
  if (["ok", "ready", "healthy", "normal", "production_ready", "passed"].includes(text)) return "ok";
  if (["degraded", "warning", "disabled", "ready_for_lab", "insufficient_evidence"].includes(text)) return "warn";
  if (["error", "failed", "unhealthy", "missing", "blocked"].includes(text)) return "err";
  return "";
}

function productStatus() {
  const components = health.components || {};
  const field = health.field_operations || {};
  const voice = health.voice_pipeline_status || {};
  const readinessStatus = health.field_readiness?.status || "unknown";
  return [
    { name: "对话入口", value: voice.pipeline_ok ? "可说话" : "需检查", cls: voice.pipeline_ok ? "ok" : "warn" },
    { name: "模型大脑", value: health.model_name || components.llm?.model || "-", cls: components.llm?.status === "ok" ? "ok" : "warn" },
    { name: "现场场景", value: `${field.passed ?? 0}/${field.scenario_count ?? 0} 通过`, cls: field.status === "passed" ? "ok" : "warn" },
    { name: "交付门禁", value: readinessStatus, cls: statusClass(readinessStatus) || "warn" },
  ];
}

async function refreshGlobalStatus() {
  const currentOperatorPath = `${ENDPOINTS.currentOperator}?operator_id=${encodeURIComponent(operatorId())}`;
  const [healthPayload, governancePayload, identityPayload, operatorPayload] = await Promise.all([
    getJson("/health", {}),
    getJson(ENDPOINTS.governance, governance),
    getJson(ENDPOINTS.identityReadiness, identityReadiness),
    getJson(currentOperatorPath, operatorSession),
  ]);
  health = healthPayload || {};
  governance = governancePayload || governance || { operators: [] };
  identityReadiness = identityPayload || governance.identity_gateway_readiness || identityReadiness;
  operatorSession = operatorPayload || operatorSession;
  const ok = health.status === "ok";
  globalStatusDot.className = `status-dot ${ok ? "ok" : "err"}`;
  globalStatusText.textContent = ok ? "服务在线" : "服务异常";
}

async function renderOverview() {
  const [eventsPayload, scenariosPayload, readiness, notification, siteProfiles, customerProjects] = await Promise.all([
    getJson("/api/field/events?limit=6&needs_attention=true", { events: [] }),
    getJson("/api/field/scenarios", { scenarios: [] }),
    getJson("/api/field/readiness", {}),
    getJson("/api/field/notification-preflight?status_as_200=true", {}),
    getJson(ENDPOINTS.fieldSiteProfiles, { sites: [], summary: {} }),
    getJson(ENDPOINTS.fieldCustomerProjects, { projects: [], customers: [], summary: {} }),
  ]);
  const events = eventsPayload.events || eventsPayload.items || [];
  const scenarios = scenariosPayload.scenarios || scenariosPayload.items || [];
  const projectSummary = customerProjects.summary || {};
  app.innerHTML = `
    <section class="ops-hero">
      <div>
        <p class="page-kicker">客户验收视角</p>
        <h2>机器人现场任务运营台</h2>
        <p>客户需要一眼看懂：当前服务哪些项目，覆盖哪些场景，有没有待处理事件，通知链路是否可用，以及这套系统现在能不能进入试点或交付验收。</p>
        <p class="muted-line">客户验收视角：现场任务、客户项目、知识库、语音配置和交付检查。</p>
      </div>
      <div class="ops-summary">
        <div><b>${esc(events.length)}</b><span>待关注事件</span></div>
        <div><b>${esc(scenarios.length || 8)}</b><span>覆盖场景</span></div>
        <div><b>${esc(projectSummary.project_count ?? 0)}</b><span>客户项目</span></div>
        <div><b>${esc(readiness.status || "unknown")}</b><span>交付门禁</span></div>
      </div>
    </section>
    <section class="grid three">
      ${renderOperatorCard()}
      ${renderIdentityGatewayCard()}
      ${renderReadinessCard(readiness, notification)}
      ${renderSiteProfileSummaryCard(siteProfiles)}
      <div class="card">
        <h2>客户现在能验收什么</h2>
        <div class="metric"><b>现场事件</b><span>场景、地点、风险、状态</span></div>
        <div class="metric"><b>证据链路</b><span>照片、传感器、运行记录</span></div>
        <div class="metric"><b>处置闭环</b><span>通知对象、负责人、下一步</span></div>
        <div class="metric"><b>项目边界</b><span>${esc(projectSummary.customer_count ?? 0)} 个客户 / ${esc(projectSummary.managed_object_type_count ?? 0)} 类对象</span></div>
      </div>
    </section>
    <section class="grid two">
      <div class="card">
        <h2>场景覆盖</h2>
        <div class="scenario-lanes">
          ${renderScenarioLanes(scenarios)}
        </div>
      </div>
      <div class="card">
        <h2>最近需要处理</h2>
        <div class="table-list">${renderCustomerEvents(events)}</div>
      </div>
    </section>
  `;
  wireOperatorControls();
}

function renderReadinessCard(readiness = {}, notification = {}) {
  const ready = notification.ready === true || notification.status === "ready";
  const blockerCount = ((readiness.blockers || []).length) + ((notification.blockers || []).length);
  return `
    <div class="card">
      <h2>交付门禁</h2>
      <div class="metric"><b>现场配置</b>${badge(readiness.status || "unknown", statusClass(readiness.status))}</div>
      <div class="metric"><b>通知链路</b>${badge(ready ? "可通知" : "需配置", ready ? "ok" : "warn")}</div>
      <div class="metric"><b>阻塞项</b><span>${esc(blockerCount)}</span></div>
      <p>${esc((readiness.blockers || [])[0] || (notification.blockers || [])[0] || "当前没有阻塞项，可继续做现场试点验证。")}</p>
    </div>
  `;
}

function renderScenarioLanes(scenarios = []) {
  const fallback = [
    { title: "机器人摔倒或卡住", notification_group: "保安群", priority: "P1", evidence: "位置、照片、运行状态" },
    { title: "夜间陌生人拍照", notification_group: "保安群", priority: "P1", evidence: "照片、地点、时间" },
    { title: "车辆违停", notification_group: "保安群", priority: "P2", evidence: "车牌/照片、区域、停留时长" },
    { title: "烟雾火灾", notification_group: "保安群", priority: "P0", evidence: "烟感/温度、照片" },
    { title: "垃圾桶满溢", notification_group: "保洁群", priority: "P3", evidence: "定点照片、满溢比例" },
    { title: "访客问路和带路", notification_group: "运营组", priority: "P4", evidence: "服务点、目的地、路线结果" },
  ];
  const rows = scenarios.length ? scenarios : fallback;
  return rows.slice(0, 8).map((scenario) => {
    const name = scenario.title || scenario.name || scenario.label || scenario.scenario_id || "现场场景";
    const group = scenario.notification_group || scenario.notify_group || scenario.notification || "按规则通知";
    const risk = scenario.priority || scenario.risk_level || scenario.severity || "P2";
    const evidence = scenario.evidence || scenario.required_evidence || scenario.evidence_required || "照片/位置/时间";
    return `
      <div class="scenario-lane">
        <div><strong>${esc(name)}</strong><span>${esc(evidence)}</span></div>
        <div>${badge(risk, String(risk).toUpperCase() === "P0" ? "err" : "warn")}${badge(group)}</div>
      </div>
    `;
  }).join("");
}

function renderCustomerEvents(events = []) {
  if (!events.length) {
    return `<div class="row-item"><strong>暂无待关注事件</strong><p>接入摄像头、传感器或手动创建事件后，这里会显示证据、通知对象和负责人。</p></div>`;
  }
  return events.map((event) => {
    const workflow = event.incident_workflow || {};
    const stages = Array.isArray(workflow.stages) ? workflow.stages : [];
    const ownerStage = stages.find((stage) => stage.owner) || {};
    const evidence = event.evidence_media || event.evidence || [];
    const evidenceCount = Array.isArray(evidence) ? evidence.length : (evidence ? 1 : 0);
    return `
      <div class="row-item">
        <strong>${esc(event.title || event.scenario_id || event.event_type || "现场事件")}</strong>
        <p>${esc(event.narrative || event.summary || event.location || "等待现场证据")}</p>
        <div class="row-meta">
          <span>地点 ${esc(event.location || event.location_name || "-")}</span>
          <span>风险 ${esc(event.priority || event.severity || "-")}</span>
          <span>状态 ${esc(event.status || "-")}</span>
          <span>负责人 ${esc(ownerStage.owner || event.owner || "-")}</span>
          <span>通知 ${esc(event.notification_group || event.notification?.group || "-")}</span>
          <span>证据 ${esc(evidenceCount)}</span>
        </div>
      </div>
    `;
  }).join("");
}
function renderConversation() {
  app.innerHTML = `
    <section class="split-view">
      <div>
        <div id="chat-box" class="chat-window"><div class="empty-state">输入或说出任务，例如：巡检 A 区</div></div>
        <div class="quick-actions">
          <button data-fill="巡检 A 区">巡检 A 区</button>
          <button data-fill="发现陌生人拍照">陌生人拍照</button>
          <button data-fill="垃圾桶满溢">垃圾桶满溢</button>
          <button data-fill="去 3 号楼怎么走">访客问路</button>
        </div>
        <div class="chat-input">
          <input id="chat-input" placeholder="输入任务或问题，例如：巡检 A 区">
          <button id="chat-send" class="primary-button">发送</button>
        </div>
      </div>
      <aside class="card">
        <h2>对话页面验收点</h2>
        <div class="metric"><b>游客问路</b><span>只回答路线，不误触发机器人任务</span></div>
        <div class="metric"><b>巡检任务</b><span>生成计划，等待确认后进入运行</span></div>
        <div class="metric"><b>知识回答</b><span>气泡展示证据，过期或冲突时拒答</span></div>
        <div class="metric"><b>语音状态</b><span id="voice-state-text">读取中</span></div>
      </aside>
    </section>
  `;
  document.querySelectorAll("[data-fill]").forEach((button) => {
    button.addEventListener("click", () => {
      document.getElementById("chat-input").value = button.dataset.fill || "";
      document.getElementById("chat-input").focus();
    });
  });
  document.getElementById("chat-send").addEventListener("click", sendChat);
  document.getElementById("chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") sendChat();
  });
  renderVoiceState();
  pollLive();
}

function addChatMessage(text, role = "system") {
  const box = document.getElementById("chat-box");
  if (!box) return;
  if (box.querySelector(".empty-state")) box.innerHTML = "";
  const div = document.createElement("div");
  div.className = `chat-message ${role}`;
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function sendChat() {
  const input = document.getElementById("chat-input");
  const text = (input?.value || "").trim();
  if (!text) return;
  chatStarted = true;
  input.value = "";
  addChatMessage(text, "user");
  const response = await postJson("/api/chat", { text, speak: true, play_audio: true });
  const payload = response.payload || {};
  if (payload.reply) addChatMessage(payload.reply, "assistant");
  else addChatMessage(payload.error || "服务没有返回可展示内容", "system");
}

async function pollLive() {
  const box = document.getElementById("chat-box");
  if (!box) return;
  const payload = await getJson("/api/live", { messages: [] });
  const messages = payload.messages || [];
  if (liveBaseline === null) liveBaseline = messages.length;
  if (!chatStarted) return;
  const visible = messages.slice(liveBaseline);
  if (visible.length === chatRenderedCount) return;
  chatRenderedCount = visible.length;
  box.innerHTML = "";
  visible.forEach((message) => addChatMessage(message.content, message.role === "user" ? "user" : "assistant"));
}

function renderVoiceState() {
  const voice = health.voice_pipeline_status || {};
  const interaction = voice.interaction || {};
  const el = document.getElementById("voice-state-text");
  if (el) el.textContent = interaction.can_talk ? "可以说话" : interaction.hint || voice.agent_state || "未知";
}

async function renderField() {
  const [scenarios, eventsPayload] = await Promise.all([
    getJson("/api/field/scenarios", { scenarios: [] }),
    getJson("/api/field/events?limit=20", { events: [] }),
  ]);
  const scenarioRows = scenarios.scenarios || scenarios.items || [];
  const events = eventsPayload.events || eventsPayload.items || [];
  if (!selectedFieldEventId && events.length) selectedFieldEventId = fieldEventId(events[0]);
  const detailPayload = selectedFieldEventId
    ? await getJson(`/api/field/events/${encodeURIComponent(selectedFieldEventId)}`, null)
    : null;
  const selectedEvent = detailPayload?.event || events.find((event) => fieldEventId(event) === selectedFieldEventId) || null;
  const summary = eventsPayload.summary || {};
  app.innerHTML = `
    <section class="grid two">
      <div class="card">
        <h2>触发现场事件</h2>
        <div class="field-form">
          <select id="field-scenario">
            ${scenarioRows.map((item) => `<option value="${esc(item.scenario_id || item.id || "")}">${esc(item.title || item.label || item.scenario_id || item.id)}</option>`).join("")}
          </select>
          <input id="field-location" placeholder="地点，例如：A 区北门">
          <textarea id="field-note" placeholder="补充描述，例如：夜间窗边有人拍照"></textarea>
          <div class="panel-actions"><button id="field-submit" class="primary-button">记录并处置</button></div>
        </div>
      </div>
      <div class="card">
        <h2>事件处置概况</h2>
        <div class="metric"><b>总事件</b><span>${esc(summary.total ?? eventsPayload.total ?? events.length)}</span></div>
        <div class="metric"><b>待关注</b><span>${esc(summary.needs_attention ?? "-")}</span></div>
        <div class="metric"><b>SLA 风险</b><span>${esc((summary.overdue ?? 0) + (summary.due_soon ?? 0))}</span></div>
        <div class="metric"><b>当前操作员</b><span>${esc(operatorId())} / ${esc(operatorRolesText())}</span></div>
      </div>
    </section>
    <section class="field-workbench">
      <div class="card">
        <div class="section-title-row">
          <h2>最近现场事件</h2>
          ${badge(`${esc(eventsPayload.filtered_total ?? events.length)} 条`, "")}
        </div>
        <div id="field-events" class="table-list">${renderFieldEvents(events, selectedFieldEventId)}</div>
      </div>
      ${renderFieldEventDetail(selectedEvent)}
    </section>
  `;
  wireFieldControls();
}

function renderSiteProfileSummaryCard(payload = {}) {
  const summary = payload.summary || {};
  return `
    <div class="card">
      <h2>多现场交付</h2>
      <div class="metric"><b>${esc(summary.site_count ?? 0)}</b><span>现场档案</span></div>
      <div class="metric"><b>${esc(summary.configured_count ?? 0)}</b><span>配置有效</span></div>
      <div class="metric"><b>${esc(summary.env_missing_count ?? 0)}</b><span>缺少凭据</span></div>
      <p>${esc(payload.next_step || "新增客户项目时先完成 site profile，再接设备、通知和验收用例。")}</p>
    </div>
  `;
}

function fieldEventId(event = {}) {
  return String(event.event_id || event.id || "");
}

function renderFieldEvents(payload, selectedId = "") {
  const rows = Array.isArray(payload) ? payload : payload.events || payload.items || [];
  if (!rows.length) return `<div class="row-item"><strong>暂无现场事件</strong><p>可以从上方选择场景创建一条演示事件。</p></div>`;
  return rows.map((event) => `
    <div class="row-item field-event-row ${fieldEventId(event) === selectedId ? "active" : ""}">
      <div class="field-row-head">
        <strong>${esc(event.title || event.scenario_name || event.scenario_id || event.event_type || "现场事件")}</strong>
        <button class="ghost-button field-event-select" data-field-event-id="${esc(fieldEventId(event))}">查看</button>
      </div>
      <p>${esc(event.narrative || event.summary || event.location || "-")}</p>
      <div class="row-meta">
        <span>状态 ${esc(event.status || "-")}</span>
        <span>地点 ${esc(event.location || "-")}</span>
        <span>SLA ${esc(event.sla?.state || "-")}</span>
        <span>通知 ${esc(event.notification_group || event.notification?.group || "-")}</span>
      </div>
    </div>
  `).join("");
}

function renderFieldEventDetail(event) {
  if (!event) {
    return `
      <div class="card field-detail-card">
        <h2>事件详情</h2>
        <p>选择左侧事件后查看证据、SLA、流程、操作审计和处置动作。</p>
      </div>
    `;
  }
  const workflow = event.incident_workflow || {};
  const stages = Array.isArray(workflow.stages) ? workflow.stages : [];
  const evidence = Array.isArray(event.evidence_media) ? event.evidence_media : [];
  const audit = Array.isArray(event.action_audit) ? event.action_audit : [];
  const delivery = Array.isArray(event.delivery_report) ? event.delivery_report : [];
  const runtime = event.runtime_delivery || {};
  const closed = event.status === "closed" || event.status === "duplicate";
  return `
    <div class="card field-detail-card">
      <div class="field-detail-head">
        <div>
          <h2>${esc(event.scenario_name || event.title || event.scenario_id || "现场事件")}</h2>
          <p>${esc(event.narrative || event.operator_action || event.voice || event.location || "")}</p>
        </div>
        <div class="field-detail-badges">
          ${badge(event.status || "-", statusClass(event.status))}
          ${badge(event.priority || event.severity || "-", statusClass(event.priority))}
        </div>
      </div>
      <div class="grid four field-metrics">
        <div><b>事件号</b><span>${esc(fieldEventId(event))}</span></div>
        <div><b>地点</b><span>${esc(event.location || "-")}</span></div>
        <div><b>当前阶段</b><span>${esc(workflow.stage || event.incident_stage || "-")}</span></div>
        <div><b>SLA</b><span>${esc(event.sla?.state || "-")} / ${esc(event.sla?.remaining_s ?? "-")}s</span></div>
      </div>
      ${renderFieldActionResult(fieldEventId(event))}
      <div class="field-action-panel">
        <textarea id="field-action-note" placeholder="处置备注，例如：已联系保安到场，等待主管审批"></textarea>
        <div class="field-supervisor-row">
          <input id="field-supervisor-id" placeholder="主管 ID，用于高风险事件关闭审批">
          <input id="field-approval-note" placeholder="审批备注">
        </div>
        <div class="panel-actions">
          <button class="primary-button" data-field-action="acknowledge" data-field-event-id="${esc(fieldEventId(event))}" ${closed ? "disabled" : ""}>确认</button>
          <button class="ghost-button" data-field-action="resend" data-field-event-id="${esc(fieldEventId(event))}" ${closed ? "disabled" : ""}>补发通知</button>
          <button class="ghost-button" data-field-action="request_close" data-field-event-id="${esc(fieldEventId(event))}" ${closed ? "disabled" : ""}>申请关闭</button>
          <button class="danger-button" data-field-action="close" data-field-event-id="${esc(fieldEventId(event))}" ${closed ? "disabled" : ""}>关闭事件</button>
          <button class="ghost-button" data-field-action="report" data-field-event-id="${esc(fieldEventId(event))}">生成报告</button>
        </div>
      </div>
      <div class="grid two field-detail-sections">
        <div>
          <h3>证据</h3>
          ${renderFieldEvidence(evidence)}
        </div>
        <div>
          <h3>送达</h3>
          ${renderFieldDelivery(delivery, runtime)}
        </div>
      </div>
      <div class="grid two field-detail-sections">
        <div>
          <h3>处置流程</h3>
          ${renderFieldWorkflow(stages)}
        </div>
        <div>
          <h3>操作审计</h3>
          ${renderFieldAudit(audit)}
        </div>
      </div>
    </div>
  `;
}

function renderFieldActionResult(eventId) {
  if (!fieldActionResult || fieldActionResult.eventId !== eventId) return "";
  const payload = fieldActionResult.payload || {};
  if (fieldActionResult.action === "report") {
    return `
      <div class="notice-card">
        <strong>${fieldActionResult.ok ? "报告已生成" : "报告生成失败"}</strong>
        <div class="mono">${esc(payload.markdown || payload.reason || payload.error || "无报告内容")}</div>
      </div>
    `;
  }
  return `
    <div class="notice-card">
      <strong>${fieldActionResult.ok ? "操作已提交" : "操作失败"}</strong>
      <p>${esc(payload.reason || payload.message || payload.error || fieldActionResult.action)}</p>
    </div>
  `;
}

function renderFieldEvidence(evidence = []) {
  if (!evidence.length) return `<div class="mini-list-empty">暂无证据媒体</div>`;
  return `<div class="field-evidence-grid">${evidence.map((item) => {
    const preview = item.preview_url || item.url || "";
    const isImage = (item.media_type || item.type || "").includes("image");
    return `
      <a class="field-evidence-item" href="${esc(preview || "#")}" target="_blank" rel="noreferrer">
        ${preview && isImage ? `<img src="${esc(preview)}" alt="">` : `<span>${esc(item.media_type || item.type || "file")}</span>`}
        <small>${esc(item.label || item.source_key || item.path || preview || "证据")}</small>
      </a>
    `;
  }).join("")}</div>`;
}

function renderFieldDelivery(delivery = [], runtime = {}) {
  const rows = delivery.length
    ? delivery.map((item) => `<div class="mini-row"><b>${esc(item.channel || "-")}</b><span>${esc(item.status || item.reason || "-")}</span></div>`).join("")
    : `<div class="mini-list-empty">暂无外部通知记录</div>`;
  const runtimeText = runtime.status ? `<div class="mini-row"><b>runtime</b><span>${esc(runtime.status)} ${esc(runtime.reason || "")}</span></div>` : "";
  return `<div class="mini-list">${rows}${runtimeText}</div>`;
}

function renderFieldWorkflow(stages = []) {
  if (!stages.length) return `<div class="mini-list-empty">暂无流程记录</div>`;
  return `<div class="field-stage-list">${stages.map((item) => `
    <div class="field-stage">
      <span class="badge ${statusClass(item.status)}">${esc(item.status || "-")}</span>
      <strong>${esc(fieldStageLabel(item.stage))}</strong>
      <small>${esc(item.owner || "-")} ${item.detail ? " / " + esc(item.detail) : ""}</small>
    </div>
  `).join("")}</div>`;
}

function fieldStageLabel(stage) {
  return ({
    admission: "准入",
    assessment: "判定",
    notification: "通知",
    voice: "语音",
    robot_motion: "机器人",
    operator: "人工处置",
    archive: "归档",
    memory: "记忆",
  })[stage] || stage || "-";
}

function renderFieldAudit(audit = []) {
  if (!audit.length) return `<div class="mini-list-empty">暂无操作审计</div>`;
  return `<div class="mini-list">${audit.slice(-8).reverse().map((item) => `
    <div class="mini-row">
      <b>${esc(item.action || "-")} / ${esc(item.outcome || "-")}</b>
      <span>${esc(item.operator_id || "-")} ${item.reason ? " / " + esc(item.reason) : ""}</span>
    </div>
  `).join("")}</div>`;
}

function wireFieldControls() {
  const submit = document.getElementById("field-submit");
  if (submit) submit.addEventListener("click", submitFieldEvent);
  document.querySelectorAll(".field-event-select").forEach((button) => {
    button.addEventListener("click", async () => {
      selectedFieldEventId = button.dataset.fieldEventId || null;
      fieldActionResult = null;
      await renderField();
    });
  });
  document.querySelectorAll("[data-field-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      await handleFieldAction(button.dataset.fieldAction, button.dataset.fieldEventId);
    });
  });
}

function fieldActionBody(action) {
  const note = document.getElementById("field-action-note")?.value || "";
  const supervisorId = document.getElementById("field-supervisor-id")?.value || "";
  const approvalNote = document.getElementById("field-approval-note")?.value || "";
  const body = { operator_id: operatorId(), note };
  if (action === "close" && supervisorId.trim()) {
    body.supervisor_approved = true;
    body.supervisor_id = supervisorId.trim();
    body.approval_note = approvalNote.trim();
  }
  return body;
}

async function handleFieldAction(action, eventId) {
  if (!eventId) return;
  const base = `/api/field/events/${encodeURIComponent(eventId)}`;
  if (action === "report") {
    const payload = await getJson(`${base}/report`, { found: false, reason: "request_failed" });
    fieldActionResult = { eventId, action, ok: payload.found !== false, payload };
    await renderField();
    return;
  }
  const endpoint = {
    acknowledge: "acknowledge",
    resend: "resend-notification",
    request_close: "request-close",
    close: "close",
  }[action];
  if (!endpoint) return;
  const response = await postJson(`${base}/${endpoint}`, fieldActionBody(action));
  selectedFieldEventId = eventId;
  fieldActionResult = { eventId, action, ok: response.ok, payload: response.payload || {} };
  await renderField();
}

async function submitFieldEvent() {
  const scenario = document.getElementById("field-scenario").value || "manual_event";
  const location = document.getElementById("field-location").value || "未填写地点";
  const note = document.getElementById("field-note").value || "Dashboard 手动触发";
  const response = await postJson("/api/field/events", {
    scenario_id: scenario,
    source: "dashboard",
    trigger_source: "dashboard",
    operator_id: operatorId(),
    observed_at: Date.now() / 1000,
    location,
    description: note,
    ...fieldScenarioDemoEvidence(scenario, location, note),
  });
  const box = document.getElementById("field-events");
  if (response.ok) {
    const event = response.payload.event || response.payload;
    selectedFieldEventId = fieldEventId(event);
    fieldActionResult = {
      eventId: selectedFieldEventId,
      action: "create",
      ok: true,
      payload: response.payload,
    };
    await renderField();
    return;
  }
  box.innerHTML = `<div class="row-item"><strong>提交失败</strong><p>${esc(response.payload.error || response.payload.message || "未知错误")}</p></div>`;
}

function fieldScenarioDemoEvidence(scenario, location, note) {
  const baseImage = "artifacts/evidence/dashboard-field-demo.jpg";
  const common = { zone_name: location, image_path: baseImage };
  const payloads = {
    fire_or_smoke: {
      ...common,
      temperature_c: 72,
      smoke_level: 0.86,
      sensor: { temperature_c: 72, smoke_level: 0.86 },
    },
    illegal_parking: {
      ...common,
      plate_number: "DEMO-123",
      duration_s: 180,
    },
    night_stranger_photo: {
      ...common,
      person_count: 1,
      dwell_seconds: 45,
      confidence: 0.91,
    },
    robot_abnormal_incident: {
      fault_type: "immobilized",
      task_id: "dashboard-task",
      image_path: baseImage,
    },
    urgent_patrol_dispatch: {
      target_location: location,
      mission_reason: note,
      current_task_id: "dashboard-task",
    },
    crowd_gathering: {
      ...common,
      person_count: 8,
      duration_min: 35,
    },
    trash_bin_full: {
      bin_id: "trash-bin-demo",
      fill_ratio: 0.92,
      image_path: baseImage,
    },
    wayfinding_help_point: {
      help_point_id: "guide-point-01",
      question: note || "请问服务中心怎么走？",
      requested_destination: "服务中心",
      map_version: "demo-map-v1",
    },
    visitor_escort: {
      destination: "服务中心",
      route_id: "demo-route-01",
      map_version: "demo-map-v1",
    },
  };
  return payloads[scenario] || {};
}

async function renderSpace() {
  const [healthPayload, pointsPayload, servicePointsPayload, routesPayload, historyPayload, proposalsPayload] = await Promise.all([
    getJson(ENDPOINTS.spaceHealth, {}),
    getJson(ENDPOINTS.spacePoints, { points: [] }),
    getJson(ENDPOINTS.spaceServicePoints, { service_points: [] }),
    getJson(ENDPOINTS.spaceRoutes, { routes: [] }),
    getJson(ENDPOINTS.spaceHistory, { changes: [] }),
    getJson(ENDPOINTS.spaceProposals, { proposals: [] }),
  ]);
  const points = Array.isArray(pointsPayload.points) ? pointsPayload.points : [];
  const servicePoints = Array.isArray(servicePointsPayload.service_points) ? servicePointsPayload.service_points : [];
  const routes = Array.isArray(routesPayload.routes) ? routesPayload.routes : [];
  const changes = Array.isArray(historyPayload.changes) ? historyPayload.changes : [];
  const proposals = Array.isArray(proposalsPayload.proposals) ? proposalsPayload.proposals : [];
  app.innerHTML = `
    <section class="ops-hero space-hero">
      <div>
        <p class="page-kicker">\u7a7a\u95f4\u8bed\u4e49\u5730\u56fe</p>
        <h2>\u628a\u56ed\u533a\u5730\u70b9\u53d8\u6210\u673a\u5668\u72d7\u542c\u5f97\u61c2\u7684\u670d\u52a1</h2>
        <p>\u56fa\u5b9a\u95ee\u8be2\u70b9\u4e3b\u52a8\u95ee\u5019\uff0c\u8bc6\u522b\u8bbf\u5ba2\u76ee\u7684\u5730\uff0c\u7ed9\u51fa\u8bed\u97f3\u6307\u8def\uff0c\u5fc5\u8981\u65f6\u751f\u6210\u5e26\u8def\u4efb\u52a1\u3002</p>
      </div>
      <div class="ops-summary">
        <div><b>${esc(points.length)}</b><span>\u70b9\u4f4d</span></div>
        <div><b>${esc(servicePoints.length)}</b><span>\u95ee\u8be2\u70b9</span></div>
        <div><b>${esc(routes.length)}</b><span>\u8def\u7ebf</span></div>
        <div><b>v${esc(healthPayload.revision || historyPayload.revision || 0)}</b><span>\u7a7a\u95f4\u7248\u672c</span></div>
      </div>
    </section>
    <section class="grid two">
      <div class="card">
        <h2>\u6a21\u62df\u8bbf\u5ba2\u5728\u95ee\u8be2\u70b9\u505c\u7559</h2>
        <div class="space-form">
          <label>\u95ee\u8be2\u670d\u52a1\u70b9
            <select id="space-service-point">
              ${servicePoints.map((item) => `<option value="${esc(item.service_point_id)}">${esc(item.service_point_name || item.service_point_id)}</option>`).join("")}
            </select>
          </label>
          <label>\u505c\u7559\u79d2\u6570<input id="space-dwell" type="number" min="0" step="0.5" value="4"></label>
          <label>\u8bbf\u5ba2\u76ee\u7684\u5730<input id="space-query" value="\u5496\u5561\u5e97\u5728\u54ea" placeholder="\u4f8b\u5982\uff1a\u6700\u8fd1\u7684\u5395\u6240\u5728\u54ea"></label>
          <label><input id="space-confirmed" type="checkbox"> \u8bbf\u5ba2\u5df2\u786e\u8ba4\u76ee\u7684\u5730\uff0c\u5141\u8bb8\u751f\u6210\u5e26\u8def\u4efb\u52a1</label>
          <div class="panel-actions">
            <button id="space-trigger" class="ghost-button">\u68c0\u67e5\u662f\u5426\u4e3b\u52a8\u95ee\u5019</button>
            <button id="space-guide" class="primary-button">\u89e3\u6790\u5e76\u751f\u6210\u6307\u8def</button>
          </div>
        </div>
        <div id="space-result" class="result-box">\u9009\u62e9\u95ee\u8be2\u70b9\u540e\u53ef\u4ee5\u6a21\u62df\u8bbf\u5ba2\u505c\u7559\u548c\u95ee\u8def\u3002</div>
      </div>
      <div class="card">
        <h2>\u5feb\u901f\u7ef4\u62a4\u56ed\u533a\u70b9\u4f4d</h2>
        <div class="space-form">
          <label>\u70b9\u4f4dID<input id="space-new-point-id" value="poi-new-place" placeholder="poi-new-place"></label>
          <label>\u70b9\u4f4d\u540d\u79f0<input id="space-new-point-name" value="\u65b0\u589e\u70b9\u4f4d" placeholder="\u4f8b\u5982\uff1a\u68b5\u6728\u4e66\u5e97"></label>
          <label>\u70b9\u4f4d\u7c7b\u578b
            <select id="space-new-point-type">
              <option value="shop">\u5546\u6237</option>
              <option value="restaurant">\u9910\u996e</option>
              <option value="restroom">\u536b\u751f\u95f4</option>
              <option value="exit">\u51fa\u5165\u53e3</option>
              <option value="service">\u95ee\u8be2\u70b9</option>
              <option value="place">\u666e\u901a\u70b9\u4f4d</option>
            </select>
          </label>
          <label>\u5e38\u7528\u522b\u540d<input id="space-new-point-aliases" value="\u522b\u540d1\uff0c\u522b\u540d2" placeholder="\u4f8b\u5982\uff1a\u4e66\u5e97\uff0c\u4e70\u4e66\u7684\u5730\u65b9"></label>
          <label>\u670d\u52a1\u65b9\u5f0f
            <select id="space-new-point-guide-mode">
              <option value="voice">\u8bed\u97f3\u6307\u8def</option>
              <option value="escort">\u53ef\u5e26\u8def</option>
            </select>
          </label>
          <div class="panel-actions">
            <button id="space-propose-point" class="ghost-button">\u63d0\u4ea4\u5ba1\u6279</button>
            <button id="space-save-point" class="primary-button">\u76f4\u63a5\u53d1\u5e03\u70b9\u4f4d</button>
          </div>
        </div>
        <div id="space-manage-result" class="result-box">\u9700\u8981\u4e3b\u7ba1\u6216\u7ba1\u7406\u5458\u6743\u9650\u3002\u4fdd\u5b58\u540e\u4f1a\u5199\u5165\u7a7a\u95f4\u8ba4\u77e5\u5b58\u50a8\u6587\u4ef6\u3002</div>
      </div>
    </section>
    <section class="grid three">
      <div class="card"><div class="section-title-row"><h2>\u56ed\u533a\u70b9\u4f4d</h2>${badge(`${points.length}`)}</div><div class="space-list">${renderSpacePoints(points)}</div></div>
      <div class="card"><div class="section-title-row"><h2>\u95ee\u8be2\u670d\u52a1\u70b9</h2>${badge(`${servicePoints.length}`)}</div><div class="space-list">${renderSpaceServicePoints(servicePoints)}</div></div>
      <div class="card"><div class="section-title-row"><h2>\u53ef\u7528\u8def\u7ebf</h2>${badge(`${routes.length}`)}</div><div class="space-list">${renderSpaceRoutes(routes)}</div></div>
    </section>
    <section class="card">
      <div class="section-title-row"><h2>\u7a7a\u95f4\u5e93\u53d8\u66f4\u8bb0\u5f55</h2>${badge(`${changes.length}`)}</div>
      <div class="space-form inline-form">
        <label>\u56de\u6eda\u5230\u7248\u672c<input id="space-rollback-revision" type="number" min="0" step="1" value="0"></label>
        <label>\u56de\u6eda\u539f\u56e0<input id="space-rollback-reason" value="\u73b0\u573a\u8bef\u6539\u6062\u590d"></label>
        <button id="space-rollback" class="ghost-button">\u56de\u6eda\u7a7a\u95f4\u5e93</button>
      </div>
      <div class="space-list">${renderSpaceChanges(changes)}</div>
    </section>
    <section class="card">
      <div class="section-title-row"><h2>\u5f85\u5ba1\u6279\u7a7a\u95f4\u53d8\u66f4</h2>${badge(`${proposals.filter((item) => item.status === "pending").length}`)}</div>
      <div class="space-list">${renderSpaceProposals(proposals)}</div>
    </section>
    <section class="grid two">
      <div class="card">
        <h2>\u914d\u7f6e\u95ee\u8be2\u670d\u52a1\u70b9</h2>
        <div class="space-form">
          <label>\u670d\u52a1\u70b9ID<input id="space-new-service-id" value="guide-new-point" placeholder="guide-new-point"></label>
          <label>\u7ed1\u5b9a\u70b9\u4f4d
            <select id="space-new-service-point-id">
              ${points.map((point) => `<option value="${esc(point.point_id)}">${esc(point.point_name || point.point_id)}</option>`).join("")}
            </select>
          </label>
          <label>\u670d\u52a1\u70b9\u540d\u79f0<input id="space-new-service-name" value="\u65b0\u589e\u95ee\u8be2\u70b9"></label>
          <label>\u505c\u7559\u89e6\u53d1\u79d2\u6570<input id="space-new-service-dwell" type="number" min="1" step="0.5" value="3"></label>
          <label>\u4e3b\u52a8\u95ee\u5019\u8bdd\u672f<input id="space-new-service-greeting" value="\u4f60\u597d\uff0c\u8bf7\u95ee\u9700\u8981\u6307\u8def\u5417\uff1f"></label>
          <button id="space-save-service-point" class="primary-button">\u4fdd\u5b58\u95ee\u8be2\u70b9</button>
        </div>
      </div>
      <div class="card">
        <h2>\u914d\u7f6e\u6307\u8def\u548c\u5e26\u8def\u8def\u7ebf</h2>
        <div class="space-form">
          <label>\u8def\u7ebfID<input id="space-new-route-id" value="route-new-guide" placeholder="route-new-guide"></label>
          <label>\u8d77\u70b9
            <select id="space-new-route-from">
              ${points.map((point) => `<option value="${esc(point.point_id)}">${esc(point.point_name || point.point_id)}</option>`).join("")}
            </select>
          </label>
          <label>\u7ec8\u70b9
            <select id="space-new-route-to">
              ${points.map((point) => `<option value="${esc(point.point_id)}">${esc(point.point_name || point.point_id)}</option>`).join("")}
            </select>
          </label>
          <label>\u8def\u7ebf\u8bdd\u672f<input id="space-new-route-instructions" value="\u8bf7\u6cbf\u4e3b\u901a\u9053\u524d\u884c\uff0c\u6309\u73b0\u573a\u6807\u8bc6\u5230\u8fbe\u76ee\u7684\u5730\u3002"></label>
          <label>\u670d\u52a1\u65b9\u5f0f
            <select id="space-new-route-guide-mode">
              <option value="voice">\u8bed\u97f3\u6307\u8def</option>
              <option value="escort">\u673a\u5668\u72d7\u5e26\u8def</option>
            </select>
          </label>
          <label><input id="space-new-route-passable" type="checkbox"> \u673a\u5668\u72d7\u53ef\u901a\u884c</label>
          <button id="space-save-route" class="primary-button">\u4fdd\u5b58\u8def\u7ebf</button>
        </div>
      </div>
    </section>
  `;
  wireSpaceControls();
}

function renderSpacePoints(points = []) {
  if (!points.length) return `<div class="mini-list-empty">\u8fd8\u6ca1\u6709\u914d\u7f6e\u56ed\u533a\u70b9\u4f4d</div>`;
  return points.map((point) => `
    <div class="row-item">
      <strong>${esc(point.point_name || point.point_id)}</strong>
      <p>${esc((point.aliases || []).join(" / ") || "\u6682\u65e0\u522b\u540d")}</p>
      <div class="row-meta">${badge(point.point_type || "place")}${badge(point.guide_mode || "voice")}<span>${esc(point.building || "-")} ${esc(point.floor || "")}</span></div>
    </div>
  `).join("");
}

function renderSpaceServicePoints(items = []) {
  if (!items.length) return `<div class="mini-list-empty">\u8fd8\u6ca1\u6709\u95ee\u8be2\u670d\u52a1\u70b9</div>`;
  return items.map((item) => `
    <div class="row-item">
      <strong>${esc(item.service_point_name || item.service_point_id)}</strong>
      <p>${esc(item.greeting_prompt || "-")}</p>
      <div class="row-meta">${badge(`${item.dwell_seconds || 0}s dwell`, "ok")}${(item.supported_intents || []).map((intent) => badge(intent)).join("")}</div>
    </div>
  `).join("");
}

function renderSpaceRoutes(routes = []) {
  if (!routes.length) return `<div class="mini-list-empty">\u8fd8\u6ca1\u6709\u914d\u7f6e\u53ef\u9a8c\u6536\u8def\u7ebf</div>`;
  return routes.map((route) => `
    <div class="row-item">
      <strong>${esc(route.from_point_id || "-")} \u2192 ${esc(route.to_point_id || "-")}</strong>
      <p>${esc(route.instructions || "\u672a\u914d\u7f6e\u8bed\u97f3\u6307\u8def\u8bdd\u672f")}</p>
      <div class="row-meta">${badge(route.guide_mode || "voice", route.robot_passable ? "ok" : "warn")}${badge(route.robot_passable ? "\u53ef\u5e26\u8def" : "\u4ec5\u6307\u8def", route.robot_passable ? "ok" : "warn")}<span>${esc(route.distance_m || "-")}m</span></div>
    </div>
  `).join("");
}

function renderSpaceChanges(changes = []) {
  if (!changes.length) return `<div class="mini-list-empty">\u8fd8\u6ca1\u6709\u7a7a\u95f4\u5e93\u53d8\u66f4\u8bb0\u5f55</div>`;
  return changes.map((change) => `
    <div class="row-item">
      <strong>v${esc(change.revision || "-")} ${esc(change.entity || "-")} / ${esc(change.action || "-")}</strong>
      <p>${esc(change.item_id || "-")} ${change.reason ? `- ${esc(change.reason)}` : ""}</p>
      <div class="row-meta">${badge(change.status || "applied", "ok")}<span>${esc(change.operator_id || "unknown")}</span></div>
    </div>
  `).join("");
}

function renderSpaceProposals(proposals = []) {
  if (!proposals.length) return `<div class="mini-list-empty">\u6ca1\u6709\u5f85\u5ba1\u6279\u7684\u7a7a\u95f4\u53d8\u66f4</div>`;
  return proposals.map((proposal) => `
    <div class="row-item">
      <strong>${esc(proposal.entity || "-")} / ${esc(proposal.action || "-")} ${badge(proposal.status || "pending")}</strong>
      <p>${esc(proposal.proposal_id || "-")} ${proposal.reason ? `- ${esc(proposal.reason)}` : ""}</p>
      <div class="row-meta">
        <span>${esc(proposal.operator_id || "unknown")}</span>
        ${proposal.status === "pending" ? `<button class="ghost-button mini-button" data-space-approve="${esc(proposal.proposal_id)}">\u6279\u51c6</button><button class="ghost-button mini-button" data-space-reject="${esc(proposal.proposal_id)}">\u9a73\u56de</button>` : ""}
      </div>
    </div>
  `).join("");
}

function renderSpaceGuideResult(payload = {}) {
  if (!payload.guide_ready) {
    return `<pre>${esc(JSON.stringify(payload, null, 2))}</pre>`;
  }
  const handoff = payload.runtime_handoff || payload.runtime_handoff_preview || {};
  const steps = Array.isArray(handoff.steps) ? handoff.steps : [];
  const validation = Array.isArray(payload.runtime_handoff_validation) ? payload.runtime_handoff_validation : [];
  const ready = payload.runtime_handoff_ready === true;
  return `
    <div class="space-handoff ${ready ? "ready" : "pending"}">
      <div class="section-title-row compact">
        <div>
          <h3>${ready ? "\u5e26\u8def\u4efb\u52a1\u53ef\u4ea4\u7ed9\u6267\u884c\u670d\u52a1" : "\u7b49\u5f85\u8bbf\u5ba2\u786e\u8ba4"}</h3>
          <p>${esc(payload.speech_text || payload.reply || "-")}</p>
        </div>
        ${badge(ready ? "\u53ef\u63d0\u4ea4" : "\u4ec5\u9884\u89c8", ready ? "ok" : "warn")}
      </div>
      <div class="row-meta">
        ${badge(esc(payload.mode || "voice"), payload.mode === "escort" ? "ok" : "")}
        ${badge(esc(handoff.task_type || "-"))}
        ${badge(esc(handoff.risk_level || "-"), statusClass(handoff.risk_level))}
        ${badge(esc(payload.runtime_handoff_reason || "-"), ready ? "ok" : "warn")}
      </div>
      ${ready ? `<div class="panel-actions"><button id="space-submit-runtime" class="primary-button">\u63d0\u4ea4\u8fd0\u884c\u4efb\u52a1</button></div>` : ""}
      <div class="space-step-list">
        ${steps.map((step) => `
          <div>
            <b>${esc(step.sequence || "-")}. ${esc(step.skill_name || "-")}</b>
            <span>${esc(JSON.stringify(step.parameters || {}))}</span>
          </div>
        `).join("") || `<div class="mini-list-empty">\u6682\u65e0\u6267\u884c\u6b65\u9aa4</div>`}
      </div>
      ${validation.length ? `<div class="skill-validation">${validation.map((item) => `<span class="warn">${esc(item)}</span>`).join("")}</div>` : ""}
      <details>
        <summary>\u67e5\u770b\u539f\u59cb\u4ea4\u63a5\u5951\u7ea6</summary>
        <pre>${esc(JSON.stringify(handoff, null, 2))}</pre>
      </details>
    </div>
  `;
}

function renderSpaceRuntimeSubmission(payload = {}) {
  const run = payload.run || {};
  const report = run.report || payload.report || {};
  const handoff = payload.handoff || run.handoff || {};
  const completed = Array.isArray(report.completed_steps) ? report.completed_steps : [];
  const events = Array.isArray(run.events) ? run.events : [];
  return `
    <div class="space-handoff ready">
      <div class="section-title-row compact">
        <div>
          <h3>\u5e26\u8def\u4efb\u52a1\u5df2\u8fdb\u5165\u6267\u884c\u670d\u52a1</h3>
          <p>\u8fd0\u884c\u53f7 ${esc(run.run_id || "-")}\uff0c\u72b6\u6001 ${esc(run.current_state || "-")}\uff0c\u6863\u4f4d ${esc(payload.profile || run.profile || "-")}</p>
        </div>
        ${badge(payload.accepted === false ? "\u672a\u901a\u8fc7" : "\u5df2\u63a5\u6536", payload.accepted === false ? "warn" : "ok")}
      </div>
      <div class="row-meta">
        ${badge(esc(handoff.task_type || "-"))}
        ${badge(esc(handoff.risk_level || "-"), statusClass(handoff.risk_level))}
        ${badge(esc(payload.reason || "\u6267\u884c\u524d\u68c0\u67e5\u5b8c\u6210"), payload.accepted === false ? "warn" : "ok")}
      </div>
      <div class="space-step-list">
        ${completed.map((step, index) => `<div><b>${index + 1}. ${esc(step)}</b><span>\u5df2\u5b8c\u6210</span></div>`).join("") || `<div class="mini-list-empty">\u8fd0\u884c\u4efb\u52a1\u5df2\u521b\u5efa\uff0c\u7b49\u5f85\u6267\u884c\u670d\u52a1\u56de\u4f20\u3002</div>`}
      </div>
      <details>
        <summary>\u67e5\u770b\u6267\u884c\u4e8b\u4ef6\u548c\u62a5\u544a</summary>
        <pre>${esc(JSON.stringify({ report, events }, null, 2))}</pre>
      </details>
    </div>
  `;
}

function wireSpaceRuntimeSubmit(result) {
  document.getElementById("space-submit-runtime")?.addEventListener("click", async () => {
    if (!latestSpaceGuidePayload?.runtime_handoff_plan) {
      if (result) result.innerHTML += `<div class="skill-validation"><span class="warn">\u7f3a\u5c11\u53ef\u63d0\u4ea4\u7684\u6267\u884c\u8ba1\u5212</span></div>`;
      return;
    }
    if (result) result.innerHTML = `<div class="result-box">\u6b63\u5728\u8fdb\u884c\u6267\u884c\u524d\u68c0\u67e5...</div>`;
    const response = await postJson(ENDPOINTS.runtimeHandoff, {
      operator_id: operatorId(),
      runtime_handoff_plan: latestSpaceGuidePayload.runtime_handoff_plan,
    });
    const payload = response.payload || response;
    if (result) result.innerHTML = renderSpaceRuntimeSubmission(payload);
  });
}

function wireSpaceControls() {
  const result = document.getElementById("space-result");
  const selectedServicePoint = () => document.getElementById("space-service-point")?.value || "";
  const dwellSeconds = () => Number(document.getElementById("space-dwell")?.value || 0);
  const queryText = () => document.getElementById("space-query")?.value || "";
  const pointItem = () => ({
    point_id: document.getElementById("space-new-point-id")?.value || "",
    point_name: document.getElementById("space-new-point-name")?.value || "",
    point_type: document.getElementById("space-new-point-type")?.value || "place",
    aliases: (document.getElementById("space-new-point-aliases")?.value || "")
      .split(/[,\uFF0C\u3001;]/)
      .map((item) => item.trim())
      .filter(Boolean),
    guide_mode: document.getElementById("space-new-point-guide-mode")?.value || "voice",
  });
  document.getElementById("space-trigger")?.addEventListener("click", async () => {
    result.textContent = "\u6b63\u5728\u5224\u65ad\u662f\u5426\u5e94\u8be5\u4e3b\u52a8\u95ee\u5019...";
    const response = await postJson(ENDPOINTS.spaceServicePointTrigger, {
      operator_id: operatorId(),
      service_point_id: selectedServicePoint(),
      person_present: true,
      dwell_seconds: dwellSeconds(),
    });
    result.textContent = JSON.stringify(response.payload || response, null, 2);
  });
  document.getElementById("space-guide")?.addEventListener("click", async () => {
    result.textContent = "\u6b63\u5728\u89e3\u6790\u76ee\u7684\u5730\u5e76\u751f\u6210\u6307\u8def...";
    const response = await postJson(ENDPOINTS.spaceGuide, {
      operator_id: operatorId(),
      service_point_id: selectedServicePoint(),
      current_point_id: "sp-west-gate",
      query: queryText(),
      guide_mode: "escort",
      visitor_confirmed: Boolean(document.getElementById("space-confirmed")?.checked),
      operator_roles: currentOperator().roles || ["operator"],
    });
    latestSpaceGuidePayload = response.payload || response;
    result.innerHTML = renderSpaceGuideResult(latestSpaceGuidePayload);
    wireSpaceRuntimeSubmit(result);
  });
  document.getElementById("space-save-point")?.addEventListener("click", async () => {
    const manageResult = document.getElementById("space-manage-result");
    if (manageResult) manageResult.textContent = "\u6b63\u5728\u4fdd\u5b58\u70b9\u4f4d...";
    const response = await postJson(ENDPOINTS.spaceManage, {
      operator_id: operatorId(),
      entity: "point",
      action: "upsert",
      item: pointItem(),
    });
    const payload = response.payload || response;
    if (manageResult) manageResult.textContent = JSON.stringify(payload, null, 2);
    if (payload.ok) await renderSpace();
  });
  document.getElementById("space-propose-point")?.addEventListener("click", async () => {
    const manageResult = document.getElementById("space-manage-result");
    if (manageResult) manageResult.textContent = "\u6b63\u5728\u63d0\u4ea4\u7a7a\u95f4\u53d8\u66f4\u5ba1\u6279...";
    const response = await postJson(ENDPOINTS.spaceProposals, {
      operator_id: operatorId(),
      entity: "point",
      action: "upsert",
      item: pointItem(),
      reason: "\u73b0\u573a\u70b9\u4f4d\u7ef4\u62a4",
    });
    const payload = response.payload || response;
    if (manageResult) manageResult.textContent = JSON.stringify(payload, null, 2);
    if (payload.ok) await renderSpace();
  });
  document.getElementById("space-save-service-point")?.addEventListener("click", async () => {
    const manageResult = document.getElementById("space-manage-result");
    if (manageResult) manageResult.textContent = "\u6b63\u5728\u4fdd\u5b58\u95ee\u8be2\u70b9...";
    const response = await postJson(ENDPOINTS.spaceManage, {
      operator_id: operatorId(),
      entity: "service_point",
      action: "upsert",
      item: {
        service_point_id: document.getElementById("space-new-service-id")?.value || "",
        point_id: document.getElementById("space-new-service-point-id")?.value || "",
        service_point_name: document.getElementById("space-new-service-name")?.value || "",
        dwell_seconds: Number(document.getElementById("space-new-service-dwell")?.value || 3),
        greeting_prompt: document.getElementById("space-new-service-greeting")?.value || "",
        supported_intents: ["wayfinding", "escort"],
      },
    });
    const payload = response.payload || response;
    if (manageResult) manageResult.textContent = JSON.stringify(payload, null, 2);
    if (payload.ok) await renderSpace();
  });
  document.getElementById("space-save-route")?.addEventListener("click", async () => {
    const manageResult = document.getElementById("space-manage-result");
    if (manageResult) manageResult.textContent = "\u6b63\u5728\u4fdd\u5b58\u8def\u7ebf...";
    const response = await postJson(ENDPOINTS.spaceManage, {
      operator_id: operatorId(),
      entity: "route",
      action: "upsert",
      item: {
        route_id: document.getElementById("space-new-route-id")?.value || "",
        from_point_id: document.getElementById("space-new-route-from")?.value || "",
        to_point_id: document.getElementById("space-new-route-to")?.value || "",
        instructions: document.getElementById("space-new-route-instructions")?.value || "",
        guide_mode: document.getElementById("space-new-route-guide-mode")?.value || "voice",
        robot_passable: Boolean(document.getElementById("space-new-route-passable")?.checked),
      },
    });
    const payload = response.payload || response;
    if (manageResult) manageResult.textContent = JSON.stringify(payload, null, 2);
    if (payload.ok) await renderSpace();
  });
  document.getElementById("space-rollback")?.addEventListener("click", async () => {
    const manageResult = document.getElementById("space-manage-result");
    if (manageResult) manageResult.textContent = "\u6b63\u5728\u56de\u6eda\u7a7a\u95f4\u5e93...";
    const response = await postJson(ENDPOINTS.spaceRollback, {
      operator_id: operatorId(),
      revision: Number(document.getElementById("space-rollback-revision")?.value || 0),
      reason: document.getElementById("space-rollback-reason")?.value || "",
    });
    const payload = response.payload || response;
    if (manageResult) manageResult.textContent = JSON.stringify(payload, null, 2);
    if (payload.ok) await renderSpace();
  });
  document.querySelectorAll("[data-space-approve]").forEach((button) => {
    button.addEventListener("click", async () => {
      const manageResult = document.getElementById("space-manage-result");
      if (manageResult) manageResult.textContent = "\u6b63\u5728\u6279\u51c6\u7a7a\u95f4\u53d8\u66f4...";
      const response = await postJson(ENDPOINTS.spaceProposalReview, {
        operator_id: operatorId(),
        proposal_id: button.dataset.spaceApprove,
        decision: "approve",
      });
      const payload = response.payload || response;
      if (manageResult) manageResult.textContent = JSON.stringify(payload, null, 2);
      if (payload.ok) await renderSpace();
    });
  });
  document.querySelectorAll("[data-space-reject]").forEach((button) => {
    button.addEventListener("click", async () => {
      const manageResult = document.getElementById("space-manage-result");
      if (manageResult) manageResult.textContent = "\u6b63\u5728\u9a73\u56de\u7a7a\u95f4\u53d8\u66f4...";
      const response = await postJson(ENDPOINTS.spaceProposalReview, {
        operator_id: operatorId(),
        proposal_id: button.dataset.spaceReject,
        decision: "reject",
        reason: "\u4e0d\u7b26\u5408\u5f53\u524d\u5730\u56fe\u914d\u7f6e",
      });
      const payload = response.payload || response;
      if (manageResult) manageResult.textContent = JSON.stringify(payload, null, 2);
      if (payload.ok) await renderSpace();
    });
  });
}

async function renderVoice() {
  const profiles = await getJson("/api/voice/profiles", { profiles: [] });
  const voice = health.voice_pipeline_status || {};
  const latency = voice.voice_turn?.latency_summary || {};
  app.innerHTML = `
    <section class="voice-state">
      <div class="card">
        <h2>当前语音状态</h2>
        <div class="wave">${Array.from({ length: 22 }).map(() => "<i></i>").join("")}</div>
        <div class="metric"><b>是否可对话</b><span>${esc(voice.interaction?.can_talk ? "可以说话" : voice.interaction?.hint || "未知")}</span></div>
        <div class="metric"><b>ASR</b><span>${esc(voice.asr?.provider || "cloud+local")}</span></div>
        <div class="metric"><b>TTS</b><span>${esc(voice.tts?.backend || voice.tts_backend || "-")}</span></div>
      </div>
      <div class="card">
        <h2>音色选择</h2>
        <div class="knowledge-form">
          <select id="voice-profile-select">${(profiles.profiles || []).map((p) => `<option value="${esc(p.profile_id)}">${esc(p.label)} / ${esc(p.category || "general")}</option>`).join("")}</select>
          <div class="panel-actions">
            <button id="voice-apply" class="primary-button">应用音色</button>
            <button id="voice-sample" class="ghost-button">播放样例</button>
          </div>
        </div>
      </div>
    </section>
    <section class="card">
      <h2>端到端延迟证据</h2>
      <div class="grid four">
        ${["asr_final_ms", "llm_ttft_ms", "tts_first_audio_ms", "playback_start_ms"].map((key) => {
          const bucket = latency.buckets?.[key] || {};
          return `<div class="metric"><b>${esc(key)}</b><span>${esc(bucket.latest_ms ?? "-")} ms</span></div>`;
        }).join("")}
      </div>
    </section>
  `;
  document.getElementById("voice-apply").addEventListener("click", () => applyVoice(false));
  document.getElementById("voice-sample").addEventListener("click", () => applyVoice(true));
}

async function applyVoice(speakSample) {
  const select = document.getElementById("voice-profile-select");
  const response = await postJson("/api/voice/profile", { profile_id: select.value, speak_sample: speakSample });
  alert(response.ok ? "音色已应用" : (response.payload.error || "音色切换失败"));
}

function renderKnowledge() {
  app.innerHTML = `
    <section class="grid four">
      ${renderOperatorCard()}
      <div class="card knowledge-step">
        <span class="badge ok">1 上传</span>
        <h3>把园区路线、SOP、设备说明粘贴进来</h3>
        <p>先预览解析结果，确认一条条知识是否正确。</p>
      </div>
      <div class="card knowledge-step">
        <span class="badge warn">2 审批</span>
        <h3>发布前检查过期、冲突和重复</h3>
        <p>只有可回答的知识才会进入问答证据。</p>
      </div>
      <div class="card knowledge-step">
        <span class="badge ok">3 使用</span>
        <h3>语音或文本提问时自动检索</h3>
        <p>回答气泡会展示引用依据；没有证据时要求确认或拒答。</p>
      </div>
    </section>
    <section class="grid two">
      <div class="card">
        <h2>上传知识</h2>
        <div class="knowledge-form">
          <input id="knowledge-title" placeholder="知识标题或来源文件，例如：site-routes.md">
          <input id="knowledge-owner" placeholder="负责人，例如：交付工程师 / 客户管理员">
          <select id="knowledge-category">
            <option value="route">路线和点位</option>
            <option value="sop">巡检 SOP</option>
            <option value="device">设备说明</option>
            <option value="policy">制度规则</option>
            <option value="faq">常见问答</option>
          </select>
          <input id="knowledge-file" type="file" accept=".txt,.md,.csv,.json">
          <textarea id="knowledge-content" placeholder="粘贴知识内容。示例：&#10;- 3 号楼在主路尽头左转 80 米。&#10;- A 区消防栓每晚 22:00 巡检一次。"></textarea>
          <div class="panel-actions">
            <button id="knowledge-preview" class="ghost-button">预览解析</button>
            <button id="knowledge-import" class="primary-button">导入并发布</button>
            <button id="knowledge-rebuild" class="ghost-button">重建索引</button>
          </div>
        </div>
      </div>
      <div class="card">
        <h2>问答怎么使用知识</h2>
        <div class="metric"><b>上传后怎么用</b><span>直接在“对话”页问，系统会自动检索知识库</span></div>
        <div class="metric"><b>回答哪里看依据</b><span>回答气泡会显示引用证据和拒答原因</span></div>
        <div class="metric"><b>哪些不能回答</b><span>已删除、过期、冲突、未审批或未命中的知识</span></div>
        <div class="chat-input">
          <input id="knowledge-query" placeholder="测试检索，例如：3 号楼怎么走">
          <button id="knowledge-search" class="primary-button">测试检索</button>
        </div>
      </div>
    </section>
    <section class="card">
      <div class="section-title-row">
        <h2>已有知识</h2>
        <div class="panel-actions compact"><button id="knowledge-list" class="ghost-button">刷新列表</button></div>
      </div>
      <div id="knowledge-summary" class="knowledge-summary-panel">正在读取知识库...</div>
      <div id="knowledge-operations" class="knowledge-ops-panel">正在读取运营队列...</div>
      <div id="knowledge-results" class="knowledge-record-grid">正在读取知识库...</div>
    </section>
  `;
  wireOperatorControls();
  document.getElementById("knowledge-file").addEventListener("change", loadKnowledgeFile);
  document.getElementById("knowledge-preview").addEventListener("click", () => knowledgeAction("preview"));
  document.getElementById("knowledge-import").addEventListener("click", () => knowledgeAction("import"));
  document.getElementById("knowledge-rebuild").addEventListener("click", rebuildKnowledge);
  document.getElementById("knowledge-list").addEventListener("click", () => listKnowledge());
  document.getElementById("knowledge-search").addEventListener("click", searchKnowledge);
  listKnowledge();
}

function knowledgePayload() {
  const title = document.getElementById("knowledge-title").value || "dashboard-knowledge.txt";
  return {
    filename: title,
    source: title,
    category: document.getElementById("knowledge-category").value,
    content: document.getElementById("knowledge-content").value,
    owner: document.getElementById("knowledge-owner").value,
    operator_id: operatorId(),
  };
}

async function loadKnowledgeFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  document.getElementById("knowledge-title").value = file.name;
  document.getElementById("knowledge-content").value = await file.text();
}

async function knowledgeAction(action) {
  const result = document.getElementById("knowledge-results");
  result.innerHTML = `<div class="loading-card">正在${action === "preview" ? "预览解析" : "导入知识"}...</div>`;
  const response = await postJson(action === "preview" ? ENDPOINTS.knowledgePreview : ENDPOINTS.knowledgeImport, knowledgePayload());
  if (action === "preview") {
    renderKnowledgeSummary(response.payload);
    result.innerHTML = renderKnowledgePreview(response.payload);
    wireKnowledgeActions();
    return;
  }
  renderKnowledgeSummary(response.payload);
  result.innerHTML = renderKnowledgeImport(response.payload);
  if (action === "diff") {
    document.getElementById("knowledge-results").innerHTML = renderKnowledgeDiff(response.payload);
    wireKnowledgeActions();
    return;
  }
  await listKnowledge(false);
}

async function listKnowledge(showLoading = true) {
  const result = document.getElementById("knowledge-results");
  if (showLoading) result.innerHTML = `<div class="loading-card">正在读取知识库...</div>`;
  const response = await postJson(ENDPOINTS.knowledgeList, { limit: 100 });
  renderKnowledgeSummary(response.payload);
  renderKnowledgeOperations(response.payload);
  result.innerHTML = renderKnowledgeList(response.payload);
  wireKnowledgeActions();
}

async function searchKnowledge() {
  const query = document.getElementById("knowledge-query").value || "";
  const result = document.getElementById("knowledge-results");
  result.innerHTML = `<div class="loading-card">正在检索证据...</div>`;
  const response = await postJson(ENDPOINTS.memorySearch, { query });
  renderKnowledgeSummary(response.payload);
  result.innerHTML = renderKnowledgeSearch(response.payload);
}

async function updateKnowledgeRecord(recordId, action) {
  const response = await postJson(ENDPOINTS.knowledgeUpdate, {
    record_id: recordId,
    action,
    operator_id: operatorId(),
    reason: `dashboard:${action}`,
  });
  if (!response.ok || response.payload.error) {
    document.getElementById("knowledge-results").innerHTML = `<div class="row-item"><strong>操作失败</strong><p>${esc(response.payload.error || response.payload.reason || "未知错误")}</p></div>`;
    return;
  }
  await listKnowledge(false);
}

async function rebuildKnowledge() {
  const result = document.getElementById("knowledge-results");
  result.innerHTML = `<div class="loading-card">正在重建可回答知识索引...</div>`;
  const response = await postJson(ENDPOINTS.knowledgeUpdate, {
    action: "rebuild_index",
    operator_id: operatorId(),
  });
  renderKnowledgeSummary(response.payload);
  result.innerHTML = renderKnowledgeImport(response.payload);
  await listKnowledge(false);
}

function wireKnowledgeActions() {
  document.querySelectorAll("[data-knowledge-action]").forEach((button) => {
    button.addEventListener("click", () => updateKnowledgeRecord(button.dataset.recordId, button.dataset.knowledgeAction));
  });
}

function renderKnowledgeSummary(payload = {}) {
  const target = document.getElementById("knowledge-summary");
  if (!target) return;
  const catalog = payload.catalog || {};
  const rag = payload.rag || {};
  target.innerHTML = `
    <div class="knowledge-summary-grid">
      <span>${badge(`总数 ${catalog.total ?? payload.total ?? 0}`)}</span>
      <span>${badge(`可回答 ${catalog.prompt_eligible ?? 0}`, "ok")}</span>
      <span>${badge(`待复核 ${catalog.needs_review ?? 0}`, catalog.needs_review ? "warn" : "")}</span>
      <span>${badge(`冲突 ${catalog.conflicted ?? 0}`, catalog.conflicted ? "err" : "")}</span>
      <span>${badge(`过期 ${catalog.expired ?? 0}`, catalog.expired ? "err" : "")}</span>
      <span>${badge(`已删除 ${catalog.deleted ?? 0}`, catalog.deleted ? "warn" : "")}</span>
    </div>
    <p>当前检索后端：${esc(rag.last_backend || rag.backend || payload.backend || "-")}；只有“可回答”的知识会进入对话证据。</p>
  `;
}

function renderKnowledgeOperations(payload = {}) {
  const target = document.getElementById("knowledge-operations");
  if (!target) return;
  const operations = payload.operations || {};
  const approval = operations.approval_queue || [];
  const conflicts = operations.conflict_queue || [];
  const expiry = operations.expiry_queue || [];
  const reindex = operations.reindex_queue || [];
  target.innerHTML = `
    <div class="knowledge-ops-grid">
      <div><strong>${esc(approval.length)}</strong><span>待审批</span></div>
      <div><strong>${esc(conflicts.length)}</strong><span>待处理冲突</span></div>
      <div><strong>${esc(expiry.length)}</strong><span>过期提醒</span></div>
      <div><strong>${esc(reindex.length)}</strong><span>待重建索引</span></div>
      <div><strong>${esc(operations.release_cadence?.mode || "manual")}</strong><span>发布节奏</span></div>
    </div>
    <p>未完成产品能力：版本回滚、字段级变更对比、定时过期提醒、发布日历。当前先把队列和风险显性化。</p>
  `;
}

function knowledgeStateLabel(record = {}) {
  const state = String(record.lifecycle_state || "").toLowerCase();
  const status = String(record.approval_status || "").toLowerCase();
  if (state === "ready" || record.prompt_eligible) return ["可回答", "ok"];
  if (state === "deleted" || status === "deleted") return ["已删除", "warn"];
  if (state === "expired") return ["已过期", "err"];
  if (state === "conflicted" || record.conflict_set_id) return ["有冲突", "err"];
  if (record.needs_reindex) return ["需重建索引", "warn"];
  if (status === "rejected") return ["已拒绝", "err"];
  if (status === "draft" || status === "pending_review") return ["待审批", "warn"];
  return [status || state || "不可回答", "warn"];
}

function renderKnowledgeList(payload = {}) {
  const records = Array.isArray(payload.records) ? payload.records : [];
  if (!records.length) return `<div class="row-item"><strong>还没有知识</strong><p>从上方上传路线、SOP、设备说明或常见问答。</p></div>`;
  return records.map((record) => {
    const [label, cls] = knowledgeStateLabel(record);
    const canDelete = String(record.approval_status || "").toLowerCase() !== "deleted";
    const text = record.text || record.memory_text || "";
    return `
      <article class="knowledge-card">
        <div class="knowledge-card-head">
          <strong>${esc(record.source || record.record_id || "知识记录")}</strong>
          ${badge(label, cls)}
        </div>
        <p>${esc(text)}</p>
        <div class="row-meta">
          <span>分类 ${esc(record.category || "-")}</span>
          <span>负责人 ${esc(record.owner || "-")}</span>
          <span>版本 ${esc(record.evidence_version || "-")}</span>
          <span>更新 ${esc(record.updated_at || "-")}</span>
          <span>过期 ${esc(record.expires_at || "无")}</span>
        </div>
        <div class="panel-actions compact">
          <button class="ghost-button" data-knowledge-action="publish" data-record-id="${esc(record.record_id)}">发布</button>
          <button class="ghost-button" data-knowledge-action="approve" data-record-id="${esc(record.record_id)}">审批</button>
          <button class="ghost-button" data-knowledge-action="diff" data-record-id="${esc(record.record_id)}">对比</button>
          ${(record.revisions || []).length ? `<button class="ghost-button" data-knowledge-action="rollback" data-record-id="${esc(record.record_id)}">回滚</button>` : ""}
          ${canDelete
            ? `<button class="ghost-button" data-knowledge-action="delete" data-record-id="${esc(record.record_id)}">删除</button>`
            : `<button class="ghost-button" data-knowledge-action="restore" data-record-id="${esc(record.record_id)}">恢复</button>`}
        </div>
      </article>
    `;
  }).join("");
}

function renderKnowledgePreview(payload = {}) {
  const records = Array.isArray(payload.records) ? payload.records : [];
  if (payload.errors?.length) return `<div class="row-item"><strong>解析失败</strong><p>${esc(payload.errors.join("；"))}</p></div>`;
  if (!records.length) return `<div class="row-item"><strong>没有解析到知识</strong><p>请检查文本内容。</p></div>`;
  return `<div class="notice-card">${badge(`预览 ${records.length} 条`, "ok")} 这些内容还没有进入问答，确认后点击“导入并发布”。</div>${renderKnowledgeList({ records })}`;
}

function renderKnowledgeDiff(payload = {}) {
  const changes = Array.isArray(payload.changes) ? payload.changes : [];
  if (!payload.found) return `<div class="row-item"><strong>没有找到版本记录</strong><p>${esc(payload.error || "record not found")}</p></div>`;
  if (!changes.length) return `<div class="row-item"><strong>没有发现差异</strong><p>当前记录和所选版本一致。</p></div>`;
  return `
    <div class="notice-card">${badge(`字段变更 ${changes.length}`, "warn")} 对比版本：${esc(payload.revision_id || "-")}</div>
    ${changes.map((change) => `
      <article class="knowledge-card">
        <div class="knowledge-card-head"><strong>${esc(change.field)}</strong>${badge("变更")}</div>
        <div class="metric"><b>之前</b><span>${esc(change.before || "-")}</span></div>
        <div class="metric"><b>现在</b><span>${esc(change.after || "-")}</span></div>
      </article>
    `).join("")}
    <div class="panel-actions compact">
      <button class="ghost-button" data-knowledge-action="rollback" data-record-id="${esc(payload.record_id)}">回滚到这个版本</button>
    </div>
  `;
}

function renderKnowledgeImport(payload = {}) {
  const errors = Array.isArray(payload.errors) ? payload.errors : [];
  return `
    <div class="notice-card">
      ${badge(errors.length ? "导入有问题" : "导入完成", errors.length ? "warn" : "ok")}
      解析 ${esc(payload.parsed ?? payload.scanned ?? 0)} 条，导入 ${esc(payload.imported ?? payload.indexed ?? 0)} 条，跳过 ${esc(payload.skipped ?? 0)} 条。
    </div>
    ${errors.map((error) => `<div class="row-item"><strong>错误</strong><p>${esc(error)}</p></div>`).join("")}
  `;
}

function renderKnowledgeSearch(payload = {}) {
  const results = Array.isArray(payload.results) ? payload.results : Array.isArray(payload.evidence) ? payload.evidence : [];
  const rag = payload.rag || {};
  const dropped = Array.isArray(rag.dropped_evidence) ? rag.dropped_evidence : [];
  const policy = rag.answer_policy || payload.answer_policy || {};
  if (!results.length && !dropped.length) return `<div class="row-item"><strong>没有可引用证据</strong><p>${esc(policy.message || "系统会要求用户补充信息或拒答。")}</p></div>`;
  return `
    <div class="notice-card">${badge(`可引用 ${results.length}`, results.length ? "ok" : "warn")} ${badge(`已拦截 ${dropped.length}`, dropped.length ? "warn" : "")}</div>
    ${results.map((item) => `<article class="knowledge-card"><div class="knowledge-card-head"><strong>${esc(item.source || item.record_id || "证据")}</strong>${badge("可引用", "ok")}</div><p>${esc(item.text || item.memory_text || item.content || "")}</p></article>`).join("")}
    ${dropped.map((item) => `<article class="knowledge-card blocked"><div class="knowledge-card-head"><strong>${esc(item.source || item.record_id || "被拦截证据")}</strong>${badge("已拦截", "warn")}</div><p>${esc(item.text || item.reason || "")}</p></article>`).join("")}
  `;
}

async function renderCapabilities() {
  const [
    center,
    capabilityPackagePayload,
    auditPayload,
    agentPayload,
    generatedPayload,
    packagePayload,
    growthPayload,
  ] = await Promise.all([
    getJson(ENDPOINTS.capabilityCenter, {}),
    getJson(ENDPOINTS.capabilityPackages, { capability_packages: [], scenario_packages: [], summary: {} }),
    getJson(`${ENDPOINTS.skillAudit}?limit=8`, { records: [] }),
    getJson(ENDPOINTS.agentProfiles, { profiles: [] }),
    getJson(ENDPOINTS.generatedSkills, { records: [], summary: {} }),
    getJson(ENDPOINTS.skillPackages, { packages: [], summary: {} }),
    getJson(`${ENDPOINTS.skillGrowthBacklog}?min_occurrences=1&limit=8`, { candidates: [], summary: {} }),
  ]);
  const summary = center.summary || {};
  const groups = Array.isArray(center.groups) ? center.groups : [];
  const missing = Array.isArray(center.missing_recommended) ? center.missing_recommended : [];
  const scenarioBlueprints = center.scenario_blueprints || {};
  const scenarioItems = Array.isArray(scenarioBlueprints.items) ? scenarioBlueprints.items : [];
  capabilityScenarioItems = Object.fromEntries(
    scenarioItems.map((item) => [String(item.scenario_id || ""), item]),
  );
  const scenarioSummary = scenarioBlueprints.summary || {};
  const capabilityPackages = Array.isArray(capabilityPackagePayload.capability_packages)
    ? capabilityPackagePayload.capability_packages
    : [];
  const scenarioPackages = Array.isArray(capabilityPackagePayload.scenario_packages)
    ? capabilityPackagePayload.scenario_packages
    : [];
  const packageSummary = capabilityPackagePayload.summary || {};
  const releaseSummary = capabilityPackagePayload.release_summary || {};
  const growth = center.online_growth || {};
  const audit = Array.isArray(auditPayload.records) ? auditPayload.records : [];
  const profiles = Array.isArray(agentPayload.profiles) ? agentPayload.profiles : [];
  const generatedSkills = Array.isArray(generatedPayload.records) ? generatedPayload.records : [];
  const generatedSummary = generatedPayload.summary || {};
  const skillPackages = Array.isArray(packagePayload.packages) ? packagePayload.packages : [];
  const growthCandidates = Array.isArray(growthPayload.candidates) ? growthPayload.candidates : [];
  const growthSummary = growthPayload.summary || {};
  app.innerHTML = `
    <section class="ops-hero capability-hero">
      <div>
        <p class="page-kicker">能力目录</p>
        <h2>${esc(center.title || "园区巡检机器人能力中心")}</h2>
        <p>客户需要看到的是机器人能完成哪些业务动作、哪些动作需要审批、哪些能力还没有接入真实传感器或机器人执行器。这里把底层 skills 翻译成产品能力。</p>
      </div>
      <div class="ops-summary">
        <div><b>${esc(summary.group_count ?? groups.length)}</b><span>能力分组</span></div>
        <div><b>${esc(summary.enabled_count ?? 0)}</b><span>已启用</span></div>
        <div><b>${esc(summary.available_count ?? 0)}</b><span>已安装</span></div>
        <div><b>${esc(scenarioSummary.ready_count ?? summary.scenario_ready_count ?? 0)}/${esc(scenarioSummary.scenario_count ?? summary.scenario_count ?? 0)}</b><span>场景就绪</span></div>
      </div>
    </section>
    <section class="grid two">
      <div class="card">
        <h2>在线增长机制</h2>
        <div class="metric"><b>当前机制</b><span>${esc(growth.mechanism || "generated skill + hot reload")}</span></div>
        <div class="metric"><b>上线流程</b><span>${esc((growth.recommended_lifecycle || []).join(" → ") || "draft → review → approve → enable → audit")}</span></div>
        <div class="metric"><b>控制方式</b><span>${esc((growth.claude_inspired_controls || []).join(" / ") || "文件化能力 / 工具边界 / 审计钩子")}</span></div>
        <p>新增能力默认应该先进入草稿和审批，而不是直接可执行。审批通过后按客户项目启用，并记录每一次触发、阻断、执行和结果。</p>
      </div>
      <div class="card">
        <h2>下一批必须补齐</h2>
        <div class="mini-list">
          ${missing.slice(0, 10).map((item) => `
            <div class="mini-row">
              <b>${esc(item.display_name || item.skill_name || "待补齐能力")}</b>
              <span>${esc(item.reason || item.group || "需要产品化接入")}</span>
            </div>
          `).join("") || `<div class="mini-list-empty">当前没有待补齐能力。</div>`}
        </div>
      </div>
      <div class="card">
        <h2>生成技能审批队列</h2>
        <div class="metric"><b>待审批</b><span>${esc(generatedSummary.pending_approval ?? 0)}</span></div>
        <div class="metric"><b>已批准</b><span>${esc(generatedSummary.approved ?? 0)}</span></div>
        <div class="metric"><b>已拒绝/停用</b><span>${esc((generatedSummary.rejected ?? 0) + (generatedSummary.disabled ?? 0))}</span></div>
        <p>所有在线生成的技能默认不能触发语音命令，必须审批后才进入客户项目能力包。</p>
      </div>
      <div class="card">
        <h2>增长候选池</h2>
        <div class="metric"><b>候选</b><span>${esc(growthSummary.candidate_count ?? growthCandidates.length)}</span></div>
        <div class="metric"><b>待产品判断</b><span>${esc(growthSummary.open_count ?? 0)}</span></div>
        <div class="metric"><b>已转入规划</b><span>${esc(growthSummary.promoted_count ?? 0)}</span></div>
        <p>候选来自真实调用审计里的失败、阻断和未命中请求，只作为产品输入，不会自动创建或启用技能。</p>
      </div>
    </section>
    <section class="card">
      <div class="section-title-row">
        <div>
          <h2>客户可启用能力包</h2>
          <p>把底层技能整理成客户可验收的能力范围，展示启用状态、缺失依赖和下一步动作。</p>
        </div>
        ${badge(`${packageSummary.ready_count ?? 0} 就绪 / ${(packageSummary.capability_package_count ?? 0) + (packageSummary.scenario_package_count ?? 0)} 个能力包`, (packageSummary.blocked_count ?? 0) ? "warn" : "ok")}
      </div>
      <div class="grid four">
        <div class="metric"><b>${esc(releaseSummary.controlled_demo_allowed_count ?? 0)}</b><span>受控演示</span></div>
        <div class="metric"><b>${esc(releaseSummary.customer_pilot_allowed_count ?? 0)}</b><span>客户试点</span></div>
        <div class="metric ${Number(releaseSummary.production_claim_allowed_count || 0) ? "ok" : "warn"}"><b>${esc(releaseSummary.production_claim_allowed_count ?? 0)}</b><span>生产声明</span></div>
        <div class="metric"><b>${esc((releaseSummary.blocked_count ?? 0) + (releaseSummary.manual_acceptance_required_count ?? 0))}</b><span>阻断/待复核</span></div>
      </div>
      <p class="muted-line">发布声明规则：${esc(releaseSummary.claim_policy || "生产上线声明必须有现场验收和人工接管审批。")}</p>
      <div class="grid two">
        <div>
          <h3>能力包</h3>
          <div class="capability-list">
            ${capabilityPackages.slice(0, 12).map(renderCapabilityPackageItem).join("") || `<div class="mini-list-empty">暂无能力包。</div>`}
          </div>
        </div>
        <div>
          <h3>场景包</h3>
          <div class="capability-list">
            ${scenarioPackages.slice(0, 12).map(renderScenarioPackageItem).join("") || `<div class="mini-list-empty">暂无场景包。</div>`}
          </div>
        </div>
      </div>
    </section>
    <section class="card">
      <div class="section-title-row">
        <div>
          <h2>场景能力蓝图</h2>
          <p>按客户场景展示需要哪些技能、传感器/数据依赖、通知归档和验收标准，避免只看散乱的技能清单。</p>
        </div>
        ${badge(`${scenarioSummary.ready_count ?? 0} 就绪 / ${scenarioSummary.scenario_count ?? scenarioItems.length} 个场景`, (scenarioSummary.blocked_count ?? 0) ? "warn" : "ok")}
      </div>
      <div class="capability-list">
        ${scenarioItems.map(renderScenarioBlueprint).join("") || `<div class="mini-list-empty">暂无场景蓝图。</div>`}
      </div>
    </section>
    <section class="card">
      <div class="section-title-row">
        <div>
          <h2>在线增长候选</h2>
          <p>产品经理在这里判断哪些重复需求值得沉淀成新技能，哪些只是客户个性化或噪声。</p>
        </div>
        ${badge(`${growthCandidates.length} 个候选`)}
      </div>
      <div class="capability-list">
        ${growthCandidates.map(renderGrowthCandidate).join("") || `<div class="mini-list-empty">还没有足够证据形成增长候选。</div>`}
      </div>
    </section>
    <section class="card">
      <div class="section-title-row">
        <div>
          <h2>生成技能治理</h2>
          <p>这里处理“机器人自己学会的新能力”：先审查提示词、工具边界和安全等级，再决定是否启用。</p>
        </div>
        ${badge(`${generatedSkills.length} 个草稿`)}
      </div>
      <div class="capability-list">
        ${generatedSkills.map(renderGeneratedSkill).join("") || `<div class="mini-list-empty">还没有在线生成的技能草稿。</div>`}
      </div>
    </section>
    <section class="card">
      <div class="section-title-row">
        <div>
          <h2>Agent 分工体系</h2>
          <p>不同任务使用不同 Agent Profile，限制工具边界，避免一个大脑同时负责巡检、问路、知识运营和技能生成。</p>
        </div>
        ${badge(`${profiles.length} 个角色`)}
      </div>
      <div class="knowledge-form agent-profile-form">
        <input id="agent-profile-name" placeholder="Profile ID，例如 parking_detection_pm">
        <input id="agent-profile-display" placeholder="客户可见名称，例如 违停检测产品代理">
        <input id="agent-profile-description" placeholder="触发场景：这个 Agent 什么时候该被使用">
        <input id="agent-profile-tools" placeholder="允许工具，逗号分隔，例如 read_file,create_skill">
        <input id="agent-profile-spawnable" placeholder="可派生 Agent，例如 safety_reviewer">
        <input id="agent-profile-skills" placeholder="预加载技能，例如 detect_illegal_parking">
        <select id="agent-profile-risk">
          <option value="low">Low risk</option>
          <option value="medium" selected>Medium risk</option>
          <option value="high">High risk</option>
          <option value="critical">Critical risk</option>
        </select>
        <textarea id="agent-profile-instructions" placeholder="写清楚角色边界、不能做什么、何时转交人工、何时必须要求证据。"></textarea>
        <button id="agent-profile-save" class="primary-button">保存 Agent Profile</button>
      </div>
      <div class="capability-list">
        ${profiles.map(renderAgentProfile).join("") || `<div class="mini-list-empty">暂无 Agent Profile。</div>`}
      </div>
    </section>
    <section class="card">
      <div class="section-title-row">
        <div>
          <h2>最近技能调用审计</h2>
          <p>用于客户试点复盘和生产追责：每次技能触发都要能说明来源、结果、耗时和被阻断原因。</p>
        </div>
        ${badge(`${audit.length} 条`)}
      </div>
      <div class="mini-list">
        ${audit.map((record) => `
          <div class="mini-row">
            <b>${esc(record.skill_name || "unknown")} ${badge(record.status || "-", statusClass(record.status))}</b>
            <span>${esc(record.timestamp || "-")} / ${esc(record.source || "-")} / ${esc(record.reason || record.result_preview || "-")}</span>
          </div>
        `).join("") || `<div class="mini-list-empty">还没有技能调用审计记录。</div>`}
      </div>
    </section>
    <section class="card">
      <div class="section-title-row">
        <div>
          <h2>客户能力包</h2>
          <p>已审批的生成技能只随客户或现场能力包启用，不会全局启用。</p>
        </div>
        ${badge(`${skillPackages.length} 个能力包`)}
      </div>
      <div class="knowledge-form skill-package-form">
        <input id="skill-package-id" placeholder="能力包编号，例如 fanmu-phase-1">
        <input id="skill-package-name" placeholder="能力包名称，例如梵木试点包">
        <input id="skill-package-site" placeholder="现场编号，例如 fanmu-park">
        <input id="skill-package-customer" placeholder="客户名称">
        <select id="skill-package-enabled">
          <option value="true">启用</option>
          <option value="false">停用</option>
        </select>
        <select id="skill-package-channel">
          <option value="draft">草稿</option>
          <option value="pilot">试点</option>
          <option value="prod">正式</option>
        </select>
        <input id="skill-package-rollout" type="number" min="0" max="100" value="100" placeholder="启用比例 %">
        <textarea id="skill-package-description" placeholder="能力范围、验收边界、启用说明"></textarea>
        <button id="skill-package-save" class="primary-button">保存能力包</button>
      </div>
      <div class="capability-list">
        ${skillPackages.map(renderSkillPackage).join("") || `<div class="mini-list-empty">暂无客户能力包。</div>`}
      </div>
    </section>
    <section class="capability-grid">
      ${groups.map(renderCapabilityGroup).join("") || `<div class="loading-card">能力中心暂无数据。</div>`}
    </section>
  `;
  wireGeneratedSkillReview();
}

function renderCapabilityPackageItem(item = {}) {
  const readiness = item.readiness && typeof item.readiness === "object" ? item.readiness : {};
  const decision = item.enablement_decision && typeof item.enablement_decision === "object"
    ? item.enablement_decision
    : readiness.enablement_decision || {};
  const status = readiness.status || "unknown";
  return `
    <div class="capability-item">
      <div>
        <strong>${esc(item.display_name || item.package_id || "能力包")}</strong>
        <p>${esc(item.summary || item.customer_message || "客户项目可启用的机器人能力。")}</p>
        <div class="row-meta">
          <span>${esc(item.package_id || "-")}</span>
          <span>能力 ${esc(item.capability || "-")}</span>
          <span>风险 ${esc(item.risk_level || "-")}</span>
          <span>状态 ${esc(item.status || "-")}</span>
        </div>
        ${decision.release_claim ? `<p class="small-note">交付声明：${esc(decision.release_claim)}</p>` : ""}
        ${item.customer_next_step ? `<p class="small-note">下一步：${esc(item.customer_next_step)}</p>` : ""}
      </div>
      <div class="capability-badges">
        ${badge(item.customer_status || status, status === "ready" ? "ok" : status === "manual_check" ? "warn" : "err")}
        ${badge(decision.decision || "enablement")}
        ${badge(item.kind || "capability_package")}
      </div>
    </div>
  `;
}

function renderScenarioPackageItem(item = {}) {
  const readiness = item.readiness && typeof item.readiness === "object" ? item.readiness : {};
  const decision = item.enablement_decision && typeof item.enablement_decision === "object"
    ? item.enablement_decision
    : readiness.enablement_decision || {};
  const status = readiness.status || "unknown";
  const packages = Array.isArray(item.required_capability_packages)
    ? item.required_capability_packages
    : [];
  return `
    <div class="capability-item">
      <div>
        <strong>${esc(item.display_name || item.package_id || "场景包")}</strong>
        <p>${esc(item.customer_message || "面向客户验收的场景交付范围。")}</p>
        <div class="row-meta">
          <span>${esc(item.scenario_id || "-")}</span>
          <span>${esc(item.package_id || "-")}</span>
          <span>风险 ${esc(item.risk_level || "-")}</span>
          <span>依赖 ${esc(packages.join(" / ") || "-")}</span>
        </div>
        ${decision.release_claim ? `<p class="small-note">交付声明：${esc(decision.release_claim)}</p>` : ""}
        ${item.customer_next_step ? `<p class="small-note">下一步：${esc(item.customer_next_step)}</p>` : ""}
      </div>
      <div class="capability-badges">
        ${badge(item.customer_status || status, status === "ready" ? "ok" : status === "manual_check" ? "warn" : "err")}
        ${badge(decision.decision || "enablement")}
        ${badge(item.coverage_status || "scenario")}
      </div>
    </div>
  `;
}

function renderScenarioBlueprint(item = {}) {
  const skills = Array.isArray(item.required_skills) ? item.required_skills : [];
  const dependencies = Array.isArray(item.dependencies) ? item.dependencies : [];
  const evidence = Array.isArray(item.required_evidence) ? item.required_evidence : [];
  const acceptance = Array.isArray(item.acceptance_criteria) ? item.acceptance_criteria : [];
  const status = item.coverage_status || "blocked";
  const packageReadiness = item.package_readiness && typeof item.package_readiness === "object"
    ? item.package_readiness
    : {};
  const readinessStatus = packageReadiness.status || status;
  const missingPackages = Array.isArray(packageReadiness.customer_missing_dependencies)
    ? packageReadiness.customer_missing_dependencies
    : Array.isArray(packageReadiness.missing_required_dependencies)
      ? packageReadiness.missing_required_dependencies
      : [];
  const readinessMessage = packageReadiness.customer_message || item.next_action || "";
  const readinessNextStep = packageReadiness.customer_next_step || item.next_action || "";
  return `
    <div class="capability-item scenario-blueprint">
      <div>
        <strong>${esc(item.display_name || item.scenario_id || "场景")}</strong>
        <p>${esc(item.trigger_rule || "暂无触发规则")}</p>
        <div class="row-meta">
          <span>${esc(item.scenario_id || "-")}</span>
          <span>${esc(item.priority || "P2")}</span>
          <span>通知 ${esc(item.notification_group || "none")}</span>
          <span>${esc(item.archive_required ? "需归档" : "不归档")}</span>
          <span>${esc(item.requires_operator_approval ? "需审批" : "自动/低风险")}</span>
        </div>
        <div class="skill-validation">
          ${skills.map((skill) => `
            <span class="${skill.enabled ? "ok" : skill.installed ? "warn" : "err"}">${esc(skill.display_name || skill.skill_name)} · ${esc(skill.status || "-")}</span>
          `).join("") || `<span class="err">未配置必需技能</span>`}
        </div>
        <div class="skill-validation">
          <span class="${readinessStatus === "ready" ? "ok" : readinessStatus === "manual_check" || readinessStatus === "partial" ? "warn" : "err"}">启用准入 · ${esc(readinessStatus)}</span>
          <span>缺失 ${esc(missingPackages.join(" / ") || "无")}</span>
        </div>
        <p class="small-note">${esc(readinessMessage)}</p>
        ${readinessNextStep ? `<p class="small-note">下一步：${esc(readinessNextStep)}</p>` : ""}
        <div class="row-meta">
          <span>依赖 ${esc(dependencies.join(" / ") || "-")}</span>
          <span>证据 ${esc(evidence.join(" / ") || "-")}</span>
        </div>
        <details>
          <summary>验收标准</summary>
          <ul>${acceptance.map((text) => `<li>${esc(text)}</li>`).join("") || `<li>待补充客户验收标准</li>`}</ul>
        </details>
      </div>
      <div class="capability-badges">
        ${badge(status, status === "ready" ? "ok" : status === "partial" ? "warn" : "err")}
        ${badge(`交付包 ${readinessStatus}`, readinessStatus === "ready" ? "ok" : readinessStatus === "manual_check" ? "warn" : "err")}
        ${badge(`skills ${esc(item.enabled_count ?? 0)}/${esc(item.required_skill_count ?? skills.length)}`)}
        ${badge(item.runtime_entry || "field_event_trigger")}
        <button class="ghost-button" data-scenario-readiness="${esc(item.scenario_id || "")}">重新检查</button>
        <span class="small-note">${esc(item.next_action || "")}</span>
      </div>
    </div>
  `;
}

function scenarioReadinessManifest(item = {}) {
  if (item.package_manifest && typeof item.package_manifest === "object") {
    return item.package_manifest;
  }
  const skills = Array.isArray(item.required_skills) ? item.required_skills : [];
  return {
    package_id: `scenario.${item.scenario_id || "unknown"}`,
    display_name: item.display_name || item.scenario_id || "场景包",
    scenario: item.scenario_id || "scenario_operation",
    capability_packages: skills
      .map((skill) => `capability.${skill.skill_name}`)
      .filter(Boolean),
    inputs: ["site_event", "operator_policy", "runtime_context"],
    outputs: ["customer_visible_response", "audit_record"],
    dependencies: skills.map((skill) => ({
      name: skill.skill_name,
      kind: "skill",
      required: true,
      reason: `Scenario ${item.scenario_id || ""} requires ${skill.display_name || skill.skill_name}.`,
      customer_visible: true,
    })),
    risk_level: item.requires_operator_approval ? "high" : "medium",
    risk_controls: item.requires_operator_approval
      ? ["客户可见启用前需要主管审批。"]
      : ["记录每次触发，并保留人工接管能力。"],
    customer_visible_name: item.display_name || item.scenario_id || "场景包",
    customer_visible_description: item.trigger_rule || "客户场景就绪检查。",
    customer_visible_steps: Array.isArray(item.robot_behavior)
      ? item.robot_behavior.slice(0, 4)
      : [],
    customer_visible_outputs: Array.isArray(item.acceptance_criteria)
      ? item.acceptance_criteria.slice(0, 4)
      : [],
  };
}

function scenarioReadinessInventory(item = {}) {
  const skills = Array.isArray(item.required_skills) ? item.required_skills : [];
  return {
    skills: skills
      .filter((skill) => skill.installed && skill.enabled)
      .map((skill) => skill.skill_name)
      .filter(Boolean),
    capability_packages: skills
      .filter((skill) => skill.installed && skill.enabled)
      .map((skill) => `capability.${skill.skill_name}`)
      .filter(Boolean),
  };
}

function renderScenarioReadinessResult(payload = {}) {
  const readiness = payload.readiness || payload;
  const missing = Array.isArray(readiness.customer_missing_dependencies)
    ? readiness.customer_missing_dependencies
    : Array.isArray(readiness.missing_required_dependencies)
      ? readiness.missing_required_dependencies
      : [];
  const manual = Array.isArray(readiness.manual_check_dependencies)
    ? readiness.manual_check_dependencies
    : [];
  const checks = Array.isArray(readiness.dependency_checks) ? readiness.dependency_checks : [];
  const status = readiness.status || "unknown";
  const decision = readiness.enablement_decision || {};
  return `
    <div class="skill-preview-panel scenario-readiness-panel">
      <h3>启用检查 ${badge(readiness.status_label || status, status === "ready" ? "ok" : status === "manual_check" ? "warn" : "err")}</h3>
      <p>${esc(readiness.customer_message || "暂无检查结论")}</p>
      ${decision.release_claim ? `<p>${esc(`交付声明：${decision.release_claim}`)}</p>` : ""}
      ${readiness.customer_next_step ? `<p>${esc(`下一步：${readiness.customer_next_step}`)}</p>` : ""}
      <div class="row-meta">
        <span>缺失 ${esc(missing.join(" / ") || "无")}</span>
        <span>人工复核 ${esc(manual.join(" / ") || "无")}</span>
      </div>
      <div class="skill-validation">
        ${checks.map((check) => `<span class="${check.status === "available" ? "ok" : check.status === "missing" ? "err" : "warn"}">${esc(check.name)} · ${esc(check.kind)} · ${esc(check.status)}</span>`).join("") || `<span class="warn">没有依赖检查项</span>`}
      </div>
    </div>
  `;
}

function renderGeneratedSkill(skill = {}) {
  const status = skill.status || "pending_approval";
  const enabled = skill.enabled === true;
  const packageIds = Array.isArray(skill.package_ids) ? skill.package_ids : [];
  const inDefaultPackage = packageIds.includes("default-demo");
  const packageAction = inDefaultPackage ? "unassign" : "assign";
  const packageLabel = inDefaultPackage ? "移出能力包" : "加入能力包";
  return `
    <div class="capability-item">
      <div>
        <strong>${esc(skill.skill_name || "generated_skill")}</strong>
        <p>${esc(skill.description || "暂无说明")}</p>
        <div class="row-meta">
          <span>${esc(skill.voice_trigger || "无语音触发词")}</span>
          <span>${esc(skill.safety_level || "normal")}</span>
          <span>package ${esc(packageIds.join(" / ") || "none")}</span>
          <span>预检 ${esc(skill.validation?.ok ? "通过" : "未通过")}</span>
          <span>${esc(skill.path || "-")}</span>
        </div>
        ${renderGeneratedSkillValidation(skill.validation)}
      </div>
      <div class="capability-badges">
        ${badge(status, status === "approved" ? "ok" : status === "pending_approval" ? "warn" : "err")}
        <button class="ghost-button" data-skill-preview="${esc(skill.skill_name)}">Preview</button>
        ${status === "approved" ? `<button class="ghost-button" data-skill-package="${packageAction}" data-skill-name="${esc(skill.skill_name)}">${packageLabel}</button>` : ""}
        ${badge(enabled ? "已启用" : "未启用", enabled ? "ok" : "warn")}
        <button class="ghost-button" data-skill-review="approve" data-skill-name="${esc(skill.skill_name)}">批准</button>
        <button class="ghost-button" data-skill-review="reject" data-skill-name="${esc(skill.skill_name)}">拒绝</button>
        <button class="ghost-button" data-skill-review="disable" data-skill-name="${esc(skill.skill_name)}">停用</button>
      </div>
    </div>
  `;
}

function renderSkillPackage(item = {}) {
  const skills = Array.isArray(item.skill_names) ? item.skill_names : [];
  const active = Array.isArray(item.active_skill_names) ? item.active_skill_names : [];
  const missing = Array.isArray(item.missing_skill_names) ? item.missing_skill_names : [];
  const version = item.release_version ?? 0;
  const rollout = item.rollout_percent ?? 100;
  return `
    <div class="capability-item">
      <div>
        <strong>${esc(item.display_name || item.package_id || "能力包")}</strong>
        <p>${esc(item.description || "客户或现场范围内的能力包。")}</p>
        <div class="row-meta">
          <span>${esc(item.package_id || "-")}</span>
          <span>site ${esc(item.site_id || "-")}</span>
          <span>v${esc(version)}</span>
          <span>${esc(item.release_channel || "draft")} / ${esc(rollout)}%</span>
          <span>skills ${esc(skills.length)}</span>
          <span>active ${esc(active.length)}</span>
          <span>history ${esc(item.history_count ?? 0)}</span>
        </div>
        <div class="skill-validation">
          ${(skills.length ? skills : ["未分配技能"]).map((name) => `
            <span class="${missing.includes(name) ? "warn" : ""}">${esc(name)}</span>
          `).join("")}
        </div>
      </div>
      <div class="capability-badges">
        ${badge(item.enabled ? "enabled" : "disabled", item.enabled ? "ok" : "warn")}
        ${badge(rollout > 0 ? `${rollout}% rollout` : "rollout paused", rollout > 0 ? "ok" : "warn")}
        <button class="ghost-button" data-package-release="pilot" data-package-id="${esc(item.package_id || "default-demo")}" data-package-rollout="25">试点 25%</button>
        <button class="ghost-button" data-package-release="prod" data-package-id="${esc(item.package_id || "default-demo")}" data-package-rollout="100">正式 100%</button>
        <button class="ghost-button" data-package-rollback="${esc(item.package_id || "default-demo")}">回滚</button>
        ${badge(item.customer_name || "客户范围")}
      </div>
    </div>
  `;
}

function renderAgentProfile(profile = {}) {
  const tools = Array.isArray(profile.allowed_tools) ? profile.allowed_tools : [];
  const denied = Array.isArray(profile.disallowed_tools) ? profile.disallowed_tools : [];
  const spawnable = Array.isArray(profile.spawnable_profiles) ? profile.spawnable_profiles : [];
  const skills = Array.isArray(profile.preloaded_skills) ? profile.preloaded_skills : [];
  return `
    <div class="capability-item">
      <div>
        <strong>${esc(profile.display_name || profile.name)}</strong>
        <p>${esc(profile.description || "")}</p>
        <div class="row-meta">
          <span>${esc(profile.name || "-")}</span>
          <span>来源 ${esc(profile.source || "builtin")}</span>
          <span>允许工具 ${esc(tools.length ? tools.join(" / ") : "继承或未展示")}</span>
          <span>禁用工具 ${esc(denied.join(" / ") || "无")}</span>
          <span>可派生 ${esc(spawnable.join(" / ") || "无")}</span>
          <span>预加载 ${esc(skills.join(" / ") || "无")}</span>
        </div>
      </div>
      <div class="capability-badges">
        ${badge(profile.risk_level || "medium", statusClass(profile.risk_level))}
        ${badge(profile.customer_visible === false ? "内部" : "客户可见")}
        ${badge(profile.disabled ? "已禁用" : "可用", profile.disabled ? "warn" : "ok")}
        <button class="ghost-button" data-agent-preview="${esc(profile.name || "")}">Preview</button>
      </div>
    </div>
  `;
}

function renderAgentProfilePreview(preview = null) {
  if (!preview || preview.ok !== true) return "";
  const profile = preview.profile || {};
  return `
    <div class="skill-preview-panel">
      <div class="section-title-row">
        <div>
          <h3>Agent Profile: ${esc(profile.display_name || profile.name || "-")}</h3>
          <p>${esc(profile.description || "")}</p>
        </div>
        ${badge(profile.risk_level || "medium", statusClass(profile.risk_level))}
      </div>
      <div class="grid two">
        <div>
          <h4>Allowed / Denied Tools</h4>
          <pre>${esc(`allowed: ${(profile.allowed_tools || []).join(", ") || "(inherit/none)"}\ndisallowed: ${(profile.disallowed_tools || []).join(", ") || "(none)"}`)}</pre>
        </div>
        <div>
          <h4>Spawn / Skills</h4>
          <pre>${esc(`spawnable: ${(profile.spawnable_profiles || []).join(", ") || "(none)"}\nskills: ${(profile.preloaded_skills || []).join(", ") || "(none)"}`)}</pre>
        </div>
      </div>
      <details open>
        <summary>Raw profile Markdown</summary>
        <pre>${esc(preview.raw_body || "内置画像没有项目 Markdown 内容。")}</pre>
      </details>
    </div>
  `;
}

function renderGrowthCandidate(item = {}) {
  const examples = Array.isArray(item.examples) ? item.examples : [];
  const reasons = Array.isArray(item.reasons) ? item.reasons : [];
  const status = item.status || "candidate";
  return `
    <div class="capability-item">
      <div>
        <strong>${esc(item.summary || item.suggested_skill_name || "增长候选")}</strong>
        <p>${esc(examples[0] || "来自审计日志的重复失败/阻断请求。")}</p>
        <div class="row-meta">
          <span>${esc(item.candidate_id || "-")}</span>
          <span>建议技能 ${esc(item.suggested_skill_name || "-")}</span>
          <span>触发词 ${esc(item.suggested_voice_trigger || "-")}</span>
          <span>证据 ${esc(item.evidence_count ?? 0)}</span>
          <span>原因 ${esc(reasons.join(" / ") || "-")}</span>
        </div>
      </div>
      <div class="capability-badges">
        ${badge(status, status === "candidate" ? "warn" : status === "promoted" ? "ok" : "")}
        ${badge(item.priority || "P2", item.priority === "P1" ? "warn" : "")}
        ${badge(item.risk_level || "normal", statusClass(item.risk_level))}
        <button class="primary-button" data-growth-action="draft" data-growth-candidate="${esc(item.candidate_id)}">生成草稿</button>
        <button class="ghost-button" data-growth-action="promote" data-growth-candidate="${esc(item.candidate_id)}">转入规划</button>
        <button class="ghost-button" data-growth-action="dismiss" data-growth-candidate="${esc(item.candidate_id)}">忽略</button>
        ${status !== "candidate" ? `<button class="ghost-button" data-growth-action="reopen" data-growth-candidate="${esc(item.candidate_id)}">重新观察</button>` : ""}
      </div>
    </div>
  `;
}

function renderGeneratedSkillPreview(preview = null) {
  if (!preview || preview.ok !== true) return "";
  const validation = preview.validation || {};
  return `
    <div class="skill-preview-panel">
      <div class="section-title-row">
        <div>
          <h3>Skill Preview: ${esc(preview.skill_name || "-")}</h3>
          <p>${esc(preview.description || "")}</p>
        </div>
        ${badge(validation.ok ? "校验通过" : "校验失败", validation.ok ? "ok" : "err")}
      </div>
      <div class="grid two">
        <div>
          <h4>Prompt</h4>
          <pre>${esc(preview.prompt || "")}</pre>
        </div>
        <div>
          <h4>Tools</h4>
          <pre>${esc(preview.tools || "(none)")}</pre>
        </div>
      </div>
      <details>
        <summary>Raw SKILL.md</summary>
        <pre>${esc(preview.raw_body || "")}</pre>
      </details>
    </div>
  `;
}

function renderGeneratedSkillValidation(validation = {}) {
  const issues = Array.isArray(validation.issues) ? validation.issues : [];
  if (!issues.length) return "";
  return `
    <div class="skill-validation">
      ${issues.slice(0, 4).map((issue) => `
        <span class="${issue.severity === "error" ? "err" : "warn"}">${esc(issue.message || issue.code || "validation issue")}</span>
      `).join("")}
    </div>
  `;
}

function wireGeneratedSkillReview() {
  document.querySelectorAll("[data-scenario-readiness]").forEach((button) => {
    button.addEventListener("click", async () => {
      const scenarioId = button.dataset.scenarioReadiness || "";
      const item = capabilityScenarioItems[scenarioId] || {};
      button.disabled = true;
      document.querySelectorAll(".scenario-readiness-panel").forEach((panel) => panel.remove());
      const response = await postJson(ENDPOINTS.capabilityPackageReadiness, {
        kind: "scenario_package",
        manifest: scenarioReadinessManifest(item),
        inventory: scenarioReadinessInventory(item),
      });
      const payload = response.payload || response;
      button.closest(".capability-item")?.insertAdjacentHTML(
        "afterend",
        renderScenarioReadinessResult(payload),
      );
      button.disabled = false;
    });
  });
  const agentSave = document.getElementById("agent-profile-save");
  if (agentSave) {
    agentSave.addEventListener("click", async () => {
      agentSave.disabled = true;
      const response = await postJson(ENDPOINTS.agentProfiles, {
        operator_id: operatorId(),
        name: document.getElementById("agent-profile-name")?.value || "",
        display_name: document.getElementById("agent-profile-display")?.value || "",
        description: document.getElementById("agent-profile-description")?.value || "",
        instructions: document.getElementById("agent-profile-instructions")?.value || "",
        tools: document.getElementById("agent-profile-tools")?.value || "",
        spawnable_profiles: document.getElementById("agent-profile-spawnable")?.value || "",
        skills: document.getElementById("agent-profile-skills")?.value || "",
        risk_level: document.getElementById("agent-profile-risk")?.value || "medium",
      });
      if (!response.ok) {
        alert(response.payload?.error || response.payload?.reason || "Agent 画像保存失败");
      }
      await renderCapabilities();
    });
  }
  document.querySelectorAll("[data-agent-preview]").forEach((button) => {
    button.addEventListener("click", async () => {
      const profileName = button.dataset.agentPreview || "";
      button.disabled = true;
      selectedAgentProfilePreview = await getJson(
        `/api/agent-profiles/${encodeURIComponent(profileName)}/preview`,
        null,
      );
      document.querySelectorAll(".skill-preview-panel.agent-profile-preview").forEach((panel) => panel.remove());
      const html = renderAgentProfilePreview(selectedAgentProfilePreview).replace(
        "skill-preview-panel",
        "skill-preview-panel agent-profile-preview",
      );
      button.closest(".capability-item")?.insertAdjacentHTML("afterend", html);
      button.disabled = false;
    });
  });
  const packageSave = document.getElementById("skill-package-save");
  if (packageSave) {
    packageSave.addEventListener("click", async () => {
      packageSave.disabled = true;
      const response = await postJson(ENDPOINTS.skillPackages, {
        package_id: document.getElementById("skill-package-id")?.value || "default-demo",
        display_name: document.getElementById("skill-package-name")?.value || "",
        site_id: document.getElementById("skill-package-site")?.value || "demo",
        customer_name: document.getElementById("skill-package-customer")?.value || "",
        description: document.getElementById("skill-package-description")?.value || "",
        enabled: (document.getElementById("skill-package-enabled")?.value || "true") === "true",
        release_channel: document.getElementById("skill-package-channel")?.value || "draft",
        rollout_percent: Number(document.getElementById("skill-package-rollout")?.value || 100),
        operator_id: operatorId(),
      });
      if (!response.ok) {
        alert(response.payload?.error || response.payload?.reason || "能力包保存失败");
      }
      await renderCapabilities();
    });
  }
  document.querySelectorAll("[data-package-release]").forEach((button) => {
    button.addEventListener("click", async () => {
      const packageId = button.dataset.packageId || "default-demo";
      button.disabled = true;
      const response = await postJson(`/api/skill-packages/${encodeURIComponent(packageId)}/release`, {
        release_channel: button.dataset.packageRelease || "pilot",
        rollout_percent: Number(button.dataset.packageRollout || 100),
        operator_id: operatorId(),
        note: "dashboard-package-release",
      });
      if (!response.ok) {
        alert(response.payload?.error || response.payload?.reason || "能力包发布失败");
      }
      await renderCapabilities();
    });
  });
  document.querySelectorAll("[data-package-rollback]").forEach((button) => {
    button.addEventListener("click", async () => {
      const packageId = button.dataset.packageRollback || "default-demo";
      button.disabled = true;
      const response = await postJson(`/api/skill-packages/${encodeURIComponent(packageId)}/rollback`, {
        operator_id: operatorId(),
        note: "dashboard-package-rollback",
      });
      if (!response.ok) {
        alert(response.payload?.error || response.payload?.reason || "能力包回滚失败");
      }
      await renderCapabilities();
    });
  });
  document.querySelectorAll("[data-skill-preview]").forEach((button) => {
    button.addEventListener("click", async () => {
      const skillName = button.dataset.skillPreview || "";
      button.disabled = true;
      selectedGeneratedSkillPreview = await getJson(
        `/api/skills/generated/${encodeURIComponent(skillName)}/preview`,
        null,
      );
      document.querySelectorAll(".skill-preview-panel").forEach((panel) => panel.remove());
      button.closest(".capability-item")?.insertAdjacentHTML(
        "afterend",
        renderGeneratedSkillPreview(selectedGeneratedSkillPreview),
      );
      button.disabled = false;
    });
  });
  document.querySelectorAll("[data-skill-review]").forEach((button) => {
    button.addEventListener("click", async () => {
      const skillName = button.dataset.skillName || "";
      const action = button.dataset.skillReview || "request_review";
      button.disabled = true;
      const response = await postJson(`/api/skills/generated/${encodeURIComponent(skillName)}/review`, {
        action,
        operator_id: operatorId(),
        note: "dashboard-review",
      });
      if (!response.ok) {
        alert(response.payload?.error || response.payload?.reason || "Skill review failed");
      }
      await renderCapabilities();
    });
  });
  document.querySelectorAll("[data-skill-package]").forEach((button) => {
    button.addEventListener("click", async () => {
      const skillName = button.dataset.skillName || "";
      const action = button.dataset.skillPackage || "assign";
      button.disabled = true;
      const response = await postJson(`/api/skill-packages/default-demo/skills/${encodeURIComponent(skillName)}`, {
        action,
        operator_id: operatorId(),
      });
      if (!response.ok) {
        alert(response.payload?.error || response.payload?.reason || "Skill package update failed");
      }
      await renderCapabilities();
    });
  });
  document.querySelectorAll("[data-growth-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      const candidateId = button.dataset.growthCandidate || "";
      const action = button.dataset.growthAction || "observe";
      button.disabled = true;
      const path = action === "draft"
        ? `/api/skill-growth/backlog/${encodeURIComponent(candidateId)}/draft`
        : `/api/skill-growth/backlog/${encodeURIComponent(candidateId)}`;
      const response = await postJson(path, {
        action,
        operator_id: operatorId(),
        note: "dashboard-growth-review",
      });
      if (!response.ok) {
        alert(response.payload?.error || response.payload?.reason || "Skill growth update failed");
      }
      await renderCapabilities();
    });
  });
}

function renderCapabilityGroup(group = {}) {
  const skills = Array.isArray(group.skills) ? group.skills : [];
  return `
    <article class="card capability-group">
      <div class="section-title-row">
        <div>
          <h2>${esc(group.display_name || group.group || "能力分组")}</h2>
          <p>${esc(group.description || "")}</p>
        </div>
        ${badge(`${skills.length} 项`)}
      </div>
      <div class="capability-list">
        ${skills.map(renderCapabilitySkill).join("") || `<div class="mini-list-empty">暂无已安装能力。</div>`}
      </div>
    </article>
  `;
}

function renderCapabilitySkill(skill = {}) {
  const installed = skill.installed !== false;
  const enabled = skill.enabled !== false;
  const status = skill.status || (installed ? (enabled ? "enabled" : "disabled") : "missing");
  const statusCls = installed && enabled ? "ok" : installed ? "warn" : "err";
  const approval = skill.requires_approval ? "需审批" : "可直接处理";
  return `
    <div class="capability-item ${installed ? "" : "missing"}">
      <div>
        <strong>${esc(skill.display_name || skill.skill_name || "未命名能力")}</strong>
        <p>${esc(skill.customer_description || skill.description || "")}</p>
        <div class="row-meta">
          <span>${esc(skill.skill_name || "-")}</span>
          <span>${esc(skill.source || "local")}</span>
          <span>${esc(skill.execution || "llm")}</span>
        </div>
      </div>
      <div class="capability-badges">
        ${badge(status, statusCls)}
        ${badge(skill.safety_level || "medium", statusClass(skill.safety_level))}
        ${badge(approval, skill.requires_approval ? "warn" : "")}
      </div>
    </div>
  `;
}

function renderBlueprintReadiness(payload = {}) {
  const items = Array.isArray(payload.items) ? payload.items : [];
  const visible = items.filter((item) => item.customer_visible).slice(0, 6);
  const summary = payload.summary || {};
  return `
    <section class="card">
      <div class="section-title-row">
        <div>
          <h2>\u4ea7\u54c1\u5305\u4ea4\u4ed8\u95e8\u7981</h2>
          <p>\u6309\u6267\u884c\u84dd\u56fe\u68c0\u67e5\u6a21\u5757\u7ec4\u5408\u3001\u5fc5\u9700\u914d\u7f6e\u3001\u5916\u90e8\u670d\u52a1\u548c\u9a8c\u6536\u547d\u4ee4\uff0c\u907f\u514d\u628a\u672a\u914d\u7f6e\u7684\u80fd\u529b\u8bf4\u6210\u53ef\u4ea4\u4ed8\u3002</p>
        </div>
        ${badge(`${summary.ready_for_validation_count ?? 0} ready / ${summary.blueprint_count ?? items.length} packages`, (summary.configuration_incomplete_count ?? 0) ? "warn" : "ok")}
      </div>
      <div class="capability-list">
        ${visible.map(renderBlueprintReadinessItem).join("") || `<div class="mini-list-empty">\u6682\u65e0\u5ba2\u6237\u53ef\u89c1\u4ea7\u54c1\u5305\u3002</div>`}
      </div>
    </section>
  `;
}

function renderBlueprintReadinessItem(item = {}) {
  const readiness = item.readiness || {};
  const missing = Array.isArray(readiness.missing_config) ? readiness.missing_config : [];
  const gates = Array.isArray(readiness.gates) ? readiness.gates : [];
  const status = readiness.status || "unknown";
  return `
    <div class="capability-item">
      <div>
        <strong>${esc(item.title || item.name || "-")}</strong>
        <p>${esc(readiness.customer_claim || item.description || "")}</p>
        <div class="row-meta">
          <span>${esc(item.name || "-")}</span>
          <span>${esc(item.product_stage || "-")}</span>
          <span>${esc((item.modules || []).length)} modules</span>
        </div>
        ${missing.length ? `<div class="skill-validation">${missing.slice(0, 6).map((value) => `<span class="warn">${esc(value)}</span>`).join("")}</div>` : ""}
      </div>
      <div class="capability-badges">
        ${badge(status, status === "ready_for_validation" ? "ok" : "warn")}
        ${badge(`${gates.length} gates`)}
        ${badge(readiness.production_ready ? "\u751f\u4ea7\u53ef\u7528" : "\u9700\u73b0\u573a\u9a8c\u8bc1", readiness.production_ready ? "ok" : "warn")}
      </div>
    </div>
  `;
}

async function renderDelivery() {
  const auditWindow = getAuditWindow();
  const canExportAudit = currentOperatorPermissions().includes("audit:export");
  const auditPath = `${ENDPOINTS.auditEvents}?actor_id=${encodeURIComponent(operatorId())}&limit=12${auditWindowToQuery(auditWindow)}`;
  const retryPath = `${ENDPOINTS.auditExportRetry}?actor_id=${encodeURIComponent(operatorId())}&limit=8`;
  const reviewsPath = `${ENDPOINTS.auditReviews}?actor_id=${encodeURIComponent(operatorId())}&limit=100`;
  const exportsPath = `${ENDPOINTS.auditExports}?actor_id=${encodeURIComponent(operatorId())}&limit=5`;
  const [
    readiness,
    devices,
    runtime,
    blueprints,
    siteProfiles,
    customerProjects,
    projectTemplates,
    solutionDeliveryReadiness,
    productLaunchReadiness,
    audit,
    retry,
    reviews,
    exportsPayload,
  ] = await Promise.all([
    getJson("/api/field/readiness", {}),
    getJson("/api/field/devices", {}),
    getJson("/api/runtime/context", {}),
    getJson(ENDPOINTS.blueprints, { items: [], summary: {} }),
    getJson(`${ENDPOINTS.fieldSiteProfiles}?check_env=true`, { sites: [], summary: {} }),
    getJson(`${ENDPOINTS.fieldCustomerProjects}?check_env=true`, { projects: [], customers: [], summary: {} }),
    getJson(ENDPOINTS.fieldCustomerProjectTemplates, { templates: [], summary: {} }),
    getJson(`${ENDPOINTS.fieldSolutionDeliveryReadiness}?check_env=true`, { gates: [], summary: {} }),
    getJson(`${ENDPOINTS.fieldProductLaunchReadiness}?check_env=true`, { gates: [], summary: {} }),
    getJson(auditPath, { records: [], summary: {} }),
    canExportAudit ? getJson(retryPath, { pending: 0, invalid: 0, items: [] }) : Promise.resolve({ pending: 0, invalid: 0, items: [], skipped: "audit_export_permission_required" }),
    getJson(reviewsPath, { records: [] }),
    canExportAudit ? getJson(exportsPath, { exports: [] }) : Promise.resolve({ exports: [], skipped: "audit_export_permission_required" }),
  ]);
  auditRecordCache = auditRecordsForReview(audit, reviews);
  if (selectedAuditReview?.record_id && !auditRecordCache.some((item) => item.record_id === selectedAuditReview.record_id)) {
    selectedAuditReview = null;
  }
  app.innerHTML = `
    ${renderProductLaunchReadiness(productLaunchReadiness)}
    ${renderBlueprintReadiness(blueprints)}
    ${renderSolutionDeliveryReadiness(solutionDeliveryReadiness)}
    ${renderIndustryTemplateCatalog(projectTemplates)}
    ${renderCustomerProjectCatalog(customerProjects)}
    ${renderSiteProfileCatalog(siteProfiles)}
    <section class="readiness-grid">
      <div class="card">
        <h2>上线门禁</h2>
        <div class="metric"><b>状态</b>${badge(readiness.status || "missing", statusClass(readiness.status))}</div>
        <div class="metric"><b>阻塞项</b><span>${esc((readiness.blockers || []).length)}</span></div>
        <div class="metric"><b>提醒项</b><span>${esc((readiness.warnings || []).length)}</span></div>
        <p>${esc((readiness.blockers || [])[0] || (readiness.warnings || [])[0] || "现场运行门禁已通过")}</p>
      </div>
      <div class="card">
        <h2>设备接入</h2>
        <div class="metric"><b>在线设备</b><span>${esc(devices.summary?.online ?? 0)}</span></div>
        <div class="metric"><b>已注册设备</b><span>${esc(devices.summary?.registered ?? 0)}</span></div>
        <div class="metric"><b>未注册事件</b><span>${esc(devices.summary?.unregistered_observed ?? 0)}</span></div>
      </div>
    </section>
    <section class="card">
      <h2>运行闭环</h2>
      <div class="metric"><b>运行档位</b><span>${esc(runtime.current_profile || runtime.profile || "fake")}</span></div>
      <div class="metric"><b>活跃任务</b><span>${esc(runtime.active_run ? "有" : "无")}</span></div>
      <div class="metric"><b>硬件下发</b><span>${esc(runtime.hardware_dispatch ? "允许" : "未允许")}</span></div>
    </section>
    <section class="card">
      <div class="section-title-row">
        <div>
          <h2>统一审计</h2>
          <p>汇总现场处置、技能增长和运行控制，用于客户验收和事后追溯。</p>
        </div>
        ${badge(`${audit.filtered_total ?? audit.count ?? 0} 条`)}
      </div>
      ${renderAuditWindowControls(auditWindow, customerProjects)}
      ${renderAuditProductSummary(audit)}
      ${renderAuditReviewIntegrity(audit)}
      ${renderAuditSourceHealth(audit)}
      ${renderAuditExportHistory(exportsPayload)}
      <div id="audit-review-panel">${selectedAuditReview ? renderAuditReviewPanel(selectedAuditReview) : ""}</div>
      ${renderUnifiedAudit(audit)}
      ${renderAuditRetryStatus(retry)}
      <div class="panel-actions">
        <button class="ghost-button" data-audit-retry>重试失败投递</button>
        <button data-audit-export="local">生成审计包</button>
        <button class="ghost-button" data-audit-export="deliver">生成并投递</button>
      </div>
    </section>
    <section class="card"><h2>原始门禁证据</h2><div class="mono">${esc(JSON.stringify(readiness, null, 2))}</div></section>
  `;
  wireAuditWindowControls();
  wireAuditReviewOpenControls();
  wireAuditReviewPanelControls();
  wireAuditExportControls();
  wireAuditRetryControls();
  wireCustomerProjectControls();
  wireCustomerProjectTemplateFilterControls();
}

async function renderProjects() {
  const projectQuery = customerProjectFilterQuery();
  const templateQuery = customerProjectTemplateFilterQuery();
  const [
    projectWorkbench,
    solutionDeliveryReadiness,
    customerProjects,
    managedObjectDirectory,
    projectTemplates,
    releaseRequests,
    releaseNotes,
    acceptanceRegistry,
    resourceCatalog,
    siteProfiles,
  ] = await Promise.all([
    getJson(`${ENDPOINTS.fieldCustomerProjectWorkbench}?${projectQuery}`, { delivery_surfaces: [], overall_status: "unknown" }),
    getJson(`${ENDPOINTS.fieldSolutionDeliveryReadiness}?check_env=true`, { gates: [], summary: {} }),
    getJson(`${ENDPOINTS.fieldCustomerProjects}?${projectQuery}`, { projects: [], customers: [], summary: {} }),
    getJson(`${ENDPOINTS.fieldCustomerProjectManagedObjectDirectory}?${projectQuery}`, { objects: [], summary: {} }),
    getJson(`${ENDPOINTS.fieldCustomerProjectTemplates}?${templateQuery}`, { templates: [], summary: {} }),
    getJson(ENDPOINTS.fieldCustomerProjectTemplateReleaseRequests, { requests: [], summary: {}, request_count: 0 }),
    getJson(ENDPOINTS.fieldCustomerProjectTemplateReleaseNotes, { notes: [], summary: {} }),
    getJson(ENDPOINTS.fieldCustomerProjectAcceptanceRegistry, { references: [], consumers: [], summary: {} }),
    getJson(ENDPOINTS.fieldCustomerProjectResourceCatalog, { resources: [], consumers: [], summary: {} }),
    getJson(`${ENDPOINTS.fieldSiteProfiles}?check_env=true`, { sites: [], summary: {} }),
  ]);
  app.innerHTML = `
    <section class="project-hero">
      <div>
        <h2>客户项目不是配置文件，是交付产品</h2>
        <p>每个客户可以有不同的园区、对象、场景、设备、通知组和验收用例。这里用于交付团队复制项目、维护对象目录、导入导出交付包，并检查事件是否正确归属到客户项目。</p>
      </div>
      <div class="project-hero-metrics">
        <div><b>${esc(customerProjects.summary?.tenant_count ?? 0)}</b><span>客户空间</span></div>
        <div><b>${esc(customerProjects.summary?.delivery_namespace_count ?? 0)}</b><span>交付空间</span></div>
        <div><b>${esc(customerProjects.summary?.customer_count ?? 0)}</b><span>客户</span></div>
        <div><b>${esc(customerProjects.summary?.project_count ?? 0)}</b><span>项目</span></div>
        <div><b>${esc(customerProjects.summary?.managed_object_type_count ?? 0)}</b><span>对象类型</span></div>
        <div><b>${esc(projectTemplates.summary?.template_count ?? 0)}</b><span>行业模板</span></div>
      </div>
    </section>
    <nav class="project-page-nav" aria-label="客户项目功能导航">
      <a href="#project-section-readiness">交付门禁</a>
      <a href="#project-section-projects">项目目录</a>
      <a href="#project-section-templates">模板市场</a>
      <a href="#project-section-template-governance">模板发布</a>
      <a href="#project-section-objects">对象目录</a>
      <a href="#project-section-package">导入导出</a>
      <a href="#project-section-acceptance">验收证据</a>
      <a href="#project-section-resources">资源绑定</a>
      <a href="#project-section-events">事件归属</a>
      <a href="#project-section-sites">多现场</a>
    </nav>
    ${renderProjectGoldenPathWorkbench(projectWorkbench)}
    ${renderSolutionDeliveryReadiness(solutionDeliveryReadiness)}
    <div class="project-console-grid">
      <div>
        ${renderCustomerProjectCatalog(customerProjects, resourceCatalog, managedObjectDirectory)}
        ${renderProjectPackageImportPanel()}
      </div>
      <div>
        ${renderProjectScopedEventPanel(customerProjects)}
        ${renderProjectResourceCatalogSummary(resourceCatalog, customerProjects)}
        ${renderAcceptanceRegistrySummary(acceptanceRegistry)}
        ${renderSiteProfileCatalog(siteProfiles)}
      </div>
    </div>
    ${renderTemplateReleaseGovernance(releaseRequests)}
    ${renderTemplateReleaseNotes(releaseNotes)}
    ${renderIndustryTemplateCatalog(projectTemplates)}
  `;
  polishProjectWorkspaceCopy({
    solutionDeliveryReadiness,
    projectWorkbench,
    customerProjects,
    managedObjectDirectory,
    projectTemplates,
    resourceCatalog,
  });
  wireProjectConsoleControls();
}

function renderProjectGoldenPathWorkbench(workbench = {}) {
  const surfaces = Array.isArray(workbench.delivery_surfaces) ? workbench.delivery_surfaces : [];
  const visibleSurfaces = surfaces.length ? surfaces : [
    { surface_id: "customer_projects", customer_label: "客户项目目录", customer_description: "按客户、项目、现场和交付阶段管理项目范围。", status: "unknown", count: 0, customer_count_label: "项目" },
    { surface_id: "template_market", customer_label: "行业模板市场", customer_description: "提供厂区、园区、仓储、景区等可复用方案模板。", status: "unknown", count: 0, customer_count_label: "模板" },
    { surface_id: "managed_objects", customer_label: "对象目录", customer_description: "展示现场对象及能力配置。", status: "unknown", count: 0, customer_count_label: "对象" },
    { surface_id: "delivery_resources", customer_label: "交付资源", customer_description: "检查模型、设备、能力和验收项是否可交付。", status: "unknown", count: 0, customer_count_label: "资源" },
    { surface_id: "package_delivery_gate", customer_label: "交付包准入", customer_description: "导出、导入前统一检查交付风险。", status: "unknown", count: 0, customer_count_label: "项目" },
  ];
  return `
    <section class="project-golden-path" data-project-golden-path>
      <div class="project-golden-path-head">
        <div>
          <p class="page-kicker">方案商交付路径</p>
          <h2>从行业模板到客户交付包，按验收节点推进</h2>
          <p>${esc(workbench.customer_status || "按客户项目、行业模板、对象目录、交付资源和交付包准入拆开检查，避免功能揉在一起。")}</p>
        </div>
        <div>
          ${badge(workbench.overall_status || "unknown", acceptanceGateClass(workbench.overall_status))}
          <span>${esc(workbench.scope_filtered ? "已按当前操作员项目范围过滤" : "当前可见全部授权项目")}</span>
        </div>
      </div>
      <div class="project-golden-path-steps">
        ${visibleSurfaces.map((surface, index) => `
          <div class="${acceptanceGateClass(surface.status)}">
            <b>${esc(index + 1)}. ${esc(surface.customer_label || surface.label || surface.surface_id)}</b>
            <span>${esc(surface.customer_description || surface.customer_action || "")}</span>
            <small>${esc(surface.count ?? 0)} ${esc(surface.customer_count_label || "项")}</small>
            ${badge(surface.status || "unknown", acceptanceGateClass(surface.status))}
          </div>
        `).join("")}
      </div>
      <p class="muted-line">下一步：${esc(workbench.next_step || "选择一个客户项目，补齐对象绑定、现场证据和交付包准入。")}</p>
    </section>
  `;
}

function polishProjectWorkspaceCopy(context = {}) {
  const setText = (selector, text) => {
    const element = app.querySelector(selector);
    if (element) element.textContent = text;
  };
  const setPlaceholder = (selector, text) => {
    const element = app.querySelector(selector);
    if (element) element.setAttribute("placeholder", text);
  };
  const setButtonText = (selector, text) => setText(selector, text);
  const metrics = app.querySelectorAll(".project-hero-metrics span");
  [
    "租户",
    "交付空间",
    "客户",
    "项目",
    "对象类型",
    "行业模板",
  ].forEach((label, index) => {
    if (metrics[index]) metrics[index].textContent = label;
  });
  const navLabels = [
    "交付总览",
    "项目目录",
    "模板市场",
    "模板发布",
    "对象目录",
    "导入导出",
    "验收证据",
    "资源绑定",
    "事件归属",
    "多现场",
  ];
  app.querySelectorAll(".project-page-nav a").forEach((item, index) => {
    if (navLabels[index]) item.textContent = navLabels[index];
  });
  setText(".project-hero h2", "客户项目不是配置文件，是交付产品");
  setText(
    ".project-hero p",
    "面向不同园区、厂区、仓储和景区客户，维护项目边界、对象目录、行业模板、交付资源、验收证据和可复用交付包。交付团队可以在这里判断一个项目是否能复制、能演示、能验收、能交给客户。",
  );
  const hero = app.querySelector(".project-hero");
  if (hero && !app.querySelector("[data-product-workbench]")) {
    const readinessStatus = context.solutionDeliveryReadiness?.overall_status || "unknown";
    const projectStatus = context.customerProjects?.summary?.delivery_acceptance_gate_status || "unknown";
    const objectStatus = context.managedObjectDirectory?.summary?.overall_status || "unknown";
    const templateStatus = context.projectTemplates?.summary?.overall_status || "unknown";
    const resourceStatus = context.resourceCatalog?.summary?.overall_status || "unknown";
    hero.insertAdjacentHTML("afterend", `
      <section class="project-workspace-explainer" data-product-workbench>
        <div class="project-workspace-card ${acceptanceGateClass(projectStatus)}">
          <strong>客户项目目录</strong>
          <span>一套客户、一套现场、一套对象边界，避免方案商交付时串项目。</span>
          ${badge(projectStatus, acceptanceGateClass(projectStatus))}
        </div>
        <div class="project-workspace-card ${acceptanceGateClass(templateStatus)}">
          <strong>行业模板市场</strong>
          <span>沉淀工厂、园区、仓储、景区模板，新客户从模板复制而不是从零开发。</span>
          ${badge(templateStatus, acceptanceGateClass(templateStatus))}
        </div>
        <div class="project-workspace-card ${acceptanceGateClass(objectStatus)}">
          <strong>对象目录</strong>
          <span>把车辆、烟火、垃圾桶、游客、设备等对象绑定到模型、传感器、技能和验收用例。</span>
          ${badge(objectStatus, acceptanceGateClass(objectStatus))}
        </div>
        <div class="project-workspace-card ${acceptanceGateClass(resourceStatus)}">
          <strong>交付资源</strong>
          <span>检查视觉模型、传感器协议、技能包、验收脚本是否是可追溯资源。</span>
          ${badge(resourceStatus, acceptanceGateClass(resourceStatus))}
        </div>
        <div class="project-workspace-card ${acceptanceGateClass(readinessStatus)}">
          <strong>交付准入</strong>
          <span>导出和导入交付包前，先判断是否可交付、需复核或必须阻断。</span>
          ${badge(readinessStatus, acceptanceGateClass(readinessStatus))}
        </div>
      </section>
    `);
  }
  setText("#project-section-readiness h2", "客户交付总览");
  setText("#project-section-projects h2", "客户项目目录");
  setText("#project-section-package h2", "项目交付包导入导出");
  setText("#project-section-events h2", "事件归属检查");
  setText("#project-section-resources h2", "交付资源目录");
  setText("#project-section-acceptance h2", "验收证据目录");
  setText("#project-section-projects .project-lifecycle-grid h3", "项目生命周期");
  setText("#project-section-projects .project-lifecycle-grid .field-form:nth-child(2) h3", "对象下线");
  setText("#project-section-projects .field-form.compact-form:nth-of-type(2) h3", "项目基础信息");
  setText("[data-customer-project-filters] h3", "项目筛选");
  setText("[data-customer-project-template-filters] h3", "模板筛选");
  setText("[data-managed-object-directory] h3", "对象目录");
  setText("[data-object-change-log] h3", "对象变更记录");
  setText("[data-managed-object-editor] .object-editor-section:nth-child(1) strong", "基础对象");
  setText("[data-managed-object-editor] .object-editor-section:nth-child(2) strong", "识别范围");
  setText("[data-managed-object-editor] .object-editor-section:nth-child(3) strong", "客户范围保护");
  setText("[data-managed-object-editor] .object-editor-section:nth-child(4) strong", "运行绑定");
  setText("[data-managed-object-editor] .object-editor-section:nth-child(5) strong", "验收证据");
  setButtonText("[data-project-filter-apply]", "应用项目筛选");
  setButtonText("[data-project-filter-clear]", "清空项目筛选");
  setButtonText("[data-template-filter-apply]", "应用模板筛选");
  setButtonText("[data-template-filter-clear]", "清空模板筛选");
  setButtonText("[data-project-lifecycle-export]", "导出交付包");
  setButtonText("[data-project-execution-bindings]", "执行接入计划");
  setButtonText("[data-project-lifecycle-onsite-load]", "查看现场证据");
  setButtonText("[data-project-lifecycle-archive]", "归档项目");
  setButtonText("[data-object-delete]", "下线对象");
  setButtonText("[data-project-edit-load]", "加载项目");
  setButtonText("[data-project-edit-save]", "保存信息");
  setButtonText("[data-object-upsert]", "保存对象");
  setPlaceholder("#project-filter-customer", "customer_id / 客户");
  setPlaceholder("#project-filter-project", "project_id / 项目");
  setPlaceholder("#project-filter-site", "site_id / 现场");
  setPlaceholder("#project-filter-industry", "industry / 行业");
  setPlaceholder("#object-delete-reason", "下线原因，例如：客户现场已移除该对象");
  setPlaceholder("#project-edit-customer-name", "客户名称");
  setPlaceholder("#project-edit-industry", "行业");
  setPlaceholder("#project-edit-project-name", "项目名称");
  setPlaceholder("#project-edit-site-name", "现场名称");
  setPlaceholder("#project-edit-object-scope-note", "对象范围说明");
  setPlaceholder("#object-project-id", "项目 ID 或现场 ID");
  setPlaceholder("#object-id", "对象 ID，例如 line_1_motor");
  setPlaceholder("#object-display-name", "对象名称");
  setPlaceholder("#object-category", "对象类别");
  setPlaceholder("#object-labels", "识别标签，逗号分隔");
  setPlaceholder("#object-scenarios", "场景 ID，逗号分隔");
  setPlaceholder("#object-zone-types", "区域类型，逗号分隔");
  setPlaceholder("#object-device-sources", "设备来源，例如 camera, sensor, robot");
  setPlaceholder("#object-responder-group", "通知组，例如 security");
  setPlaceholder("#object-evidence-required", "必需证据，例如 photo, location");
}

function renderProjectPackageImportPanel() {
  return `
    <section id="project-section-package" class="card">
      <div class="section-title-row">
        <div>
          <h2>项目交付包导入</h2>
          <p>交付包、提案包、验收包是三种不同交付物。这里分开校验，避免把客户签收材料和项目导入数据混在一个文本框里。</p>
        </div>
        ${badge("先预检后写入", "ok")}
      </div>
      <div class="field-form">
        <div class="project-package-workbench">
          <label>
            <strong>项目交付包</strong>
            <span>用于导入、差异预览和复制新客户项目。</span>
            <textarea id="project-import-json" placeholder='粘贴 export 返回的 package JSON，或完整 { "package": ... } 响应'></textarea>
          </label>
          <label>
            <strong>客户提案包</strong>
            <span>用于校验发给客户或销售的 proposal bundle。</span>
            <textarea id="project-proposal-json" placeholder='粘贴 { "proposal": ... } 或 proposal bundle JSON'></textarea>
          </label>
          <label>
            <strong>验收证据包</strong>
            <span>用于校验客户签收前的验收材料。</span>
            <textarea id="project-dossier-json" placeholder='粘贴 { "dossier": ... } 或验收材料 JSON'></textarea>
          </label>
        </div>
        <label class="inline-check"><input id="project-import-overwrite" type="checkbox"> 允许覆盖已有项目</label>
        <div class="panel-actions">
          <button class="ghost-button" data-project-package-verify>验包</button>
          <button class="ghost-button" data-project-package-diff>预览差异</button>
          <button class="ghost-button" data-project-import="dry-run">导入演练</button>
          <button class="ghost-button" data-project-proposal-verify>验签提案包</button>
          <button class="ghost-button" data-project-dossier-verify>验签验收包</button>
          <button class="primary-button" data-project-import="apply">确认导入</button>
        </div>
      </div>
      <div id="project-import-result" class="project-import-result mini-list-empty">还没有导入预检结果。</div>
    </section>
  `;
}

function renderSolutionDeliveryReadiness(payload = {}) {
  const gates = Array.isArray(payload.gates) ? payload.gates : [];
  const summary = payload.summary || {};
  const status = payload.overall_status || "unknown";
  return `
    <section id="project-section-readiness" class="card solution-delivery-readiness">
      <div class="section-title-row">
        <div>
          <h2>客户交付总门禁</h2>
          <p>${esc(payload.customer_status || "汇总客户项目、模板市场、资源绑定和共享资源治理状态。")}</p>
        </div>
        ${badge(status, acceptanceGateClass(status))}
      </div>
      <div class="grid four">
        <div class="metric"><b>${esc(summary.project_count ?? 0)}</b><span>项目</span></div>
        <div class="metric"><b>${esc(summary.template_count ?? 0)}</b><span>模板</span></div>
        <div class="metric"><b>${esc(summary.resource_count ?? 0)}</b><span>资源</span></div>
        <div class="metric ${acceptanceGateClass(status)}"><b>${esc(summary.blocked_count ?? 0)}</b><span>阻塞门禁</span></div>
      </div>
      <div class="project-gate-grid">
        ${gates.map((gate) => `
          <div class="project-gate-card ${acceptanceGateClass(gate.status)}">
            <strong>${esc(gate.label || gate.gate_id || "gate")} ${badge(gate.status || "unknown", acceptanceGateClass(gate.status))}</strong>
            <span>${esc(gate.evidence || "")}</span>
            <small>${esc(gate.next_step || "")}</small>
          </div>
        `).join("") || `<div class="mini-list-empty">还没有交付门禁数据。</div>`}
      </div>
      <div class="project-reuse-assessment ${acceptanceGateClass(status)}">
        <strong>对外口径</strong>
        <p>${esc(payload.release_claim || "请先完成门禁检查，再形成客户承诺。")}</p>
        <span>${esc(payload.next_step || "")}</span>
      </div>
    </section>
  `;
}

function renderProductLaunchReadiness(payload = {}) {
  const gates = Array.isArray(payload.gates) ? payload.gates : [];
  const summary = payload.summary || {};
  const status = payload.overall_status || "unknown";
  const sources = Array.isArray(payload.evidence_sources) ? payload.evidence_sources : [];
  return `
    <section class="product-launch-readiness ${acceptanceGateClass(status)}" data-product-launch-readiness>
      <div class="product-launch-head">
        <div>
          <span>客户上线准入总览</span>
          <h2>${esc(payload.customer_status || "正在汇总客户上线证据。")}</h2>
          <p>${esc(payload.release_claim || "所有上线声明必须以身份、现场、交付包和客户项目证据为准。")}</p>
        </div>
        <div>
          ${badge(status, acceptanceGateClass(status))}
          <strong>${esc(payload.launch_stage || "unknown")}</strong>
          <small>${esc(payload.production_ready ? "可进入生产准入测试" : "不能声明无人值守生产上线")}</small>
        </div>
      </div>
      <div class="product-launch-metrics">
        <div><b>${esc(summary.ready_count ?? 0)}</b><span>通过门禁</span></div>
        <div><b>${esc(summary.manual_check_count ?? 0)}</b><span>需复核</span></div>
        <div><b>${esc(summary.blocked_count ?? 0)}</b><span>阻塞项</span></div>
        <div><b>${esc(summary.project_count ?? 0)}</b><span>客户项目</span></div>
      </div>
      <div class="product-launch-gates">
        ${gates.map((gate) => `
          <div class="${acceptanceGateClass(gate.status)}">
            <strong>${esc(gate.label || gate.gate_id || "gate")}</strong>
            <span>${esc(gate.customer_message || gate.evidence || "")}</span>
            <small>${esc(gate.next_step || "")}</small>
            ${badge(gate.status || "unknown", acceptanceGateClass(gate.status))}
          </div>
        `).join("") || `<div class="mini-list-empty">还没有上线准入证据。</div>`}
      </div>
      <div class="product-launch-next">
        <strong>下一步</strong>
        <p>${esc(payload.next_step || "补齐门禁证据后再安排客户验收。")}</p>
      </div>
      <div class="product-launch-sources">
        ${sources.map((item) => `<span>${esc(item.source_id || "-")} · ${esc(item.status || "-")}</span>`).join("")}
      </div>
    </section>
  `;
}

function renderProjectScopedEventPanel(payload = {}) {
  const projects = Array.isArray(payload.projects) ? payload.projects : [];
  const projectOptions = projects.map((project) => {
    const id = project.project_id || project.site_id || "";
    return `<option value="${esc(id)}">${esc(project.customer_name || project.customer_id || "客户")} / ${esc(project.project_name || id)}</option>`;
  }).join("");
  const objectOptions = projects.flatMap((project) => (
    Array.isArray(project.managed_objects) ? project.managed_objects : []
  )).map((item) => `<option value="${esc(item.object_id || "")}">${esc(item.display_name || item.object_id || "")}</option>`).join("");
  return `
    <section id="project-section-events" class="card">
      <div class="section-title-row">
        <div>
          <h2>事件归属检查</h2>
          <p>用客户项目和对象过滤现场事件，确认摄像头、传感器或手动事件没有串到别的客户项目。</p>
        </div>
        ${badge("scope check")}
      </div>
      <div class="field-form compact-form">
        <select id="project-event-project">
          <option value="">全部项目</option>
          ${projectOptions}
        </select>
        <select id="project-event-object">
          <option value="">全部对象</option>
          ${objectOptions}
        </select>
        <div class="panel-actions">
          <button class="ghost-button" data-project-events-refresh>查询事件</button>
        </div>
      </div>
      <div id="project-event-result" class="mini-list-empty">选择项目或对象后查询事件归属。</div>
    </section>
  `;
}

function renderAcceptanceRegistrySummary(payload = {}) {
  const summary = payload.summary || {};
  const references = Array.isArray(payload.references) ? payload.references : [];
  const consumers = Array.isArray(payload.consumers) ? payload.consumers : [];
  return `
    <section id="project-section-acceptance" class="card">
      <div class="section-title-row">
        <div>
          <h2>验收引用登记</h2>
          <p>统一管理客户项目和行业模板中使用的现场对象验收引用，避免验收材料缺失或引用失效。</p>
        </div>
        ${badge(summary.overall_status || "unknown", acceptanceGateClass(summary.overall_status))}
      </div>
      <div class="grid four">
        <div class="metric"><b>${esc(summary.reference_count ?? references.length)}</b><span>引用</span></div>
        <div class="metric ok"><b>${esc(summary.linked_count ?? 0)}</b><span>已关联</span></div>
        <div class="metric warn"><b>${esc(summary.manual_check_count ?? 0)}</b><span>待复核</span></div>
        <div class="metric err"><b>${esc(summary.blocked_count ?? 0)}</b><span>阻断</span></div>
      </div>
      <p class="muted-line">下一步：${esc(payload.next_step || "客户签收前复核阻断项和待人工确认的引用。")}</p>
      <div class="capability-list compact-list">
        ${references.slice(0, 8).map((item) => `
          <div class="row-item">
            <strong>${esc(item.reference || "acceptance reference")} ${badge(item.status || "unknown", acceptanceGateClass(item.status))}</strong>
            <span>${esc(item.consumer_count ?? 0)} 个对象 / 已关联 ${esc(item.linked_count ?? 0)} / 待复核 ${esc(item.manual_check_count ?? 0)} / 阻断 ${esc(item.blocked_count ?? 0)}</span>
          </div>
        `).join("") || `<div class="mini-list-empty">暂未发现验收引用。</div>`}
      </div>
      <div class="row-meta">
        <span>对象 ${esc(summary.object_count ?? 0)}</span>
        <span>项目 ${esc(summary.project_count ?? 0)}</span>
        <span>模板 ${esc(summary.template_count ?? 0)}</span>
        <span>使用方 ${esc(summary.consumer_count ?? consumers.length)}</span>
      </div>
    </section>
  `;
}

function deliveryResourceTypeLabel(resourceType = "") {
  return {
    vision_models: "Vision model",
    sensor_protocols: "Sensor protocol",
    skill_packages: "Skill package",
    acceptance_tests: "Acceptance test",
  }[resourceType] || resourceType || "Resource";
}

function renderDeliveryResourceTypeOptions(selected = "") {
  return DELIVERY_RESOURCE_TYPES.map((resourceType) => (
    `<option value="${esc(resourceType)}" ${resourceType === selected ? "selected" : ""}>${esc(deliveryResourceTypeLabel(resourceType))}</option>`
  )).join("");
}

function renderDeliveryResourceProjectOptions(projectPayload = {}) {
  const projects = Array.isArray(projectPayload.projects) ? projectPayload.projects : [];
  return projects.map((project) => {
    const projectId = project.project_id || project.site_id || "";
    if (!projectId) return "";
    const label = `${project.customer_name || project.customer_id || "客户"} / ${project.project_name || projectId}`;
    return `<option value="${esc(projectId)}">${esc(label)}</option>`;
  }).join("");
}

function renderDeliveryResourceRegistrationForm(projectPayload = {}) {
  const projectOptions = renderDeliveryResourceProjectOptions(projectPayload);
  return `
    <div class="delivery-resource-registry compact-form" data-delivery-resource-registry>
      <h3>交付资源登记</h3>
      <p>登记客户项目可绑定的真实资源，把识别模型、传感器、业务能力和验收项变成可审计的交付资产。</p>
      <div class="delivery-resource-form">
        <label>
          <span>Project scope</span>
          <select id="resource-project-id">
            <option value="">Shared across delivery workspace</option>
            ${projectOptions}
          </select>
        </label>
        <label>
          <span>Resource type</span>
          <select id="resource-type">${renderDeliveryResourceTypeOptions()}</select>
        </label>
        <label>
          <span>Resource ID</span>
          <input id="resource-id" placeholder="park_person_detector_v1">
        </label>
        <label>
          <span>Display name</span>
          <input id="resource-display-name" placeholder="Park person detector">
        </label>
        <label>
          <span>Version</span>
          <input id="resource-version" placeholder="v1.0.0">
        </label>
        <label>
          <span>Owner</span>
          <input id="resource-owner" placeholder="delivery.team">
        </label>
        <label>
          <span>Source</span>
          <input id="resource-source" placeholder="shared_registry">
        </label>
        <label class="delivery-resource-description">
          <span>Description</span>
          <textarea id="resource-description" placeholder="Where this resource is deployed, what it is allowed to support, and any customer-specific limits."></textarea>
        </label>
      </div>
      <div class="panel-actions">
        <button class="primary-button" data-resource-register>Register resource</button>
      </div>
      <div id="resource-register-result" class="mini-list-empty">No resource registration has been submitted yet.</div>
    </div>
  `;
}

function objectBindingInputId(resourceType = "") {
  return {
    vision_models: "object-vision-models",
    sensor_protocols: "object-sensor-protocols",
    skill_packages: "object-skill-packages",
    acceptance_tests: "object-acceptance-tests",
  }[resourceType] || "";
}

function renderObjectResourceBindingPicker(resources = []) {
  const bindable = (Array.isArray(resources) ? resources : [])
    .filter((item) => DELIVERY_RESOURCE_TYPES.includes(item.resource_type || ""))
    .filter((item) => item.status !== "unregistered")
    .sort((left, right) => (
      `${left.resource_type || ""}/${left.resource_id || ""}`
        .localeCompare(`${right.resource_type || ""}/${right.resource_id || ""}`)
    ));
  const options = bindable.map((item) => {
    const resourceType = item.resource_type || "";
    const resourceId = item.resource_id || "";
    const label = `${deliveryResourceTypeLabel(resourceType)} / ${item.display_name || resourceId} / ${resourceId}`;
    return `<option value="${esc(resourceType)}::${esc(resourceId)}">${esc(label)}</option>`;
  }).join("");
  return `
    <div class="object-resource-picker" data-object-resource-picker>
      <div>
        <strong>绑定交付资源</strong>
        <p>选择已登记的识别模型、传感器协议、能力包或验收项，系统会自动追加到对应的对象配置字段。</p>
      </div>
      <div class="object-resource-picker-row">
        <select id="object-resource-picker">
          ${options || `<option value="">暂无可绑定资源，请先登记交付资源</option>`}
        </select>
        <button class="ghost-button" type="button" data-object-resource-add>加入绑定</button>
      </div>
      <div id="object-resource-picker-result" class="small-note">请先登记资源，再绑定到现场对象。</div>
    </div>
  `;
}

function renderDeliveryResourceActionPlan(summary = {}, unregistered = []) {
  const status = summary.overall_status || (unregistered.length ? "manual_check" : "ready");
  if (unregistered.length) {
    return `
      <div class="delivery-resource-action-plan warn" data-delivery-resource-action-plan>
        <strong>资源绑定行动计划 ${badge("需要登记", "warn")}</strong>
        <p>这些绑定仍只是字符串引用。导出可复用客户交付包或客户签收前，需要先登记为真实资源。</p>
        <div class="mini-list">
          ${unregistered.slice(0, 6).map((item) => `
            <div class="mini-row">
              <b>${esc(deliveryResourceTypeLabel(item.resource_type))} / ${esc(item.resource_id || "-")}</b>
              <span>${esc(item.consumer_count ?? 0)} 个绑定，来源 ${esc(item.source || "未登记")}</span>
            </div>
          `).join("")}
        </div>
      </div>
    `;
  }
  return `
    <div class="delivery-resource-action-plan ${acceptanceGateClass(status)}" data-delivery-resource-action-plan>
      <strong>资源绑定行动计划 ${badge(status === "ready" ? "可导出交付包" : status, acceptanceGateClass(status))}</strong>
      <p>当前可见现场对象的资源绑定都能解析到已登记资源，可以继续导出交付包、绑定现场证据并进入验收复核。</p>
    </div>
  `;
}

function renderDeliveryResourceGovernancePanel(resources = []) {
  const sharedResources = resources.filter((item) => item.source === "shared_registry");
  return `
    <div class="delivery-resource-governance" data-delivery-resource-governance>
      <div class="section-title-row">
        <div>
          <h3>资源治理</h3>
          <p>查看共享资源历史，停用不安全资源，或在执行回滚前先做预检。</p>
        </div>
        ${badge(`${sharedResources.length} 个共享资源`, sharedResources.length ? "ok" : "warn")}
      </div>
      <div class="resource-governance-grid">
        <div class="project-import-card">
          <strong>变更历史</strong>
          <p>查看登记表修订、操作人、原因、校验值和修订编号。</p>
          <button class="ghost-button" type="button" data-resource-history>加载历史</button>
        </div>
        <div class="project-import-card">
          <strong>回滚申请</strong>
          <p>请先做预检。正式回滚会创建二次复核请求。</p>
          <input id="resource-rollback-id" placeholder="修订编号">
          <div class="panel-actions">
            <button class="ghost-button" type="button" data-resource-rollback="dry-run">预检</button>
            <button class="danger-button" type="button" data-resource-rollback="apply">申请回滚</button>
          </div>
        </div>
        <div class="project-import-card">
          <strong>审批队列</strong>
          <p>按时限、逾期状态和影响证据复核待处理的停用与回滚申请。</p>
          <div class="panel-actions">
            <button class="ghost-button" type="button" data-resource-governance-requests="all">加载申请</button>
            <button class="ghost-button" type="button" data-resource-governance-requests="overdue">只看逾期</button>
            <button class="ghost-button" type="button" data-resource-governance-escalate-overdue>升级逾期</button>
          </div>
        </div>
      </div>
      <div id="resource-governance-result" class="mini-list-empty">还没有执行资源治理操作。</div>
    </div>
  `;
}

function renderProjectResourceCatalogSummary(payload = {}, projectPayload = {}) {
  const summary = payload.summary || {};
  const resources = Array.isArray(payload.resources) ? payload.resources : [];
  currentDeliveryResourceItems = resources;
  const used = resources.filter((item) => Number(item.consumer_count || 0) > 0);
  const unregistered = resources.filter((item) => item.status === "unregistered");
  return `
    <section id="project-section-resources" class="card">
      <div class="section-title-row">
        <div>
          <h2>交付资源目录</h2>
          <p>核对每个客户对象绑定的视觉模型、传感器协议、技能包和验收用例，避免交付时只填了字符串但没有真实资源。</p>
        </div>
        ${badge(summary.overall_status || "unknown", acceptanceGateClass(summary.overall_status))}
      </div>
      <div class="grid four">
        <div class="metric"><b>${esc(summary.resource_count ?? resources.length)}</b><span>资源</span></div>
        <div class="metric ok"><b>${esc(summary.used_resource_count ?? used.length)}</b><span>已使用</span></div>
        <div class="metric warn"><b>${esc(summary.unregistered_resource_count ?? unregistered.length)}</b><span>待登记</span></div>
        <div class="metric"><b>${esc(summary.consumer_count ?? 0)}</b><span>绑定点</span></div>
      </div>
      ${renderDeliveryResourceRegistrationForm(projectPayload)}
      ${renderDeliveryResourceActionPlan(summary, unregistered)}
      ${renderDeliveryResourceGovernancePanel(resources)}
      <div class="capability-list compact-list">
        ${used.slice(0, 8).map((item) => `
          <div class="row-item">
            <strong>
              ${esc(item.display_name || item.resource_id || "resource")}
              ${badge(item.status || "registered", item.status === "unregistered" ? "warn" : acceptanceGateClass(item.status))}
              ${item.publish_status ? badge(item.publish_status, item.publish_status === "published" ? "ok" : item.publish_status === "disabled" || item.publish_status === "blocked" ? "err" : "warn") : ""}
            </strong>
            <span>${esc(item.resource_type || "-")} / ${esc(item.resource_id || "-")} / ${esc(item.consumer_count ?? 0)} 个绑定，来源 ${esc(item.source || "-")}</span>
            ${item.source === "shared_registry" ? `<button class="ghost-button compact-danger" type="button" data-resource-disable="${esc(item.resource_type || "")}::${esc(item.resource_id || "")}">申请停用</button>` : ""}
          </div>
        `).join("") || `<div class="mini-list-empty">还没有客户对象绑定交付资源。</div>`}
      </div>
      <p class="muted-line">下一步：${esc(payload.next_step || "导出客户验收包前，请先登记缺失资源。")}</p>
    </section>
  `;
}

function wireProjectConsoleControls() {
  wireCustomerProjectControls();
  wireCustomerProjectFilterControls();
  wireCustomerProjectTemplateFilterControls();
  document.querySelectorAll("[data-template-release]").forEach((button) => {
    button.addEventListener("click", () => updateTemplateRelease(
      button.dataset.templateRelease || "",
      button.dataset.releaseStatus || "pilot",
    ));
  });
  document.querySelectorAll("[data-template-history]").forEach((button) => {
    button.addEventListener("click", () => loadTemplateReleaseHistory(button.dataset.templateHistory || ""));
  });
  document.querySelectorAll("[data-template-release-request]").forEach((button) => {
    button.addEventListener("click", () => createTemplateReleaseRequest(
      button.dataset.templateReleaseRequest || "",
      button.dataset.releaseStatus || "published",
    ));
  });
  document.querySelectorAll("[data-template-release-requests]").forEach((button) => {
    button.addEventListener("click", () => loadTemplateReleaseRequests(button.dataset.templateReleaseRequests || ""));
  });
  const exportReleaseNotesButton = document.querySelector("[data-template-release-notes-export]");
  if (exportReleaseNotesButton) {
    exportReleaseNotesButton.addEventListener("click", exportTemplateReleaseNotesBundle);
  }
  wireTemplateReleaseReviewControls(document);
  document.querySelectorAll("[data-project-import]").forEach((button) => {
    button.addEventListener("click", () => importCustomerProjectPackage(button.dataset.projectImport === "dry-run"));
  });
  const packageVerifyButton = document.querySelector("[data-project-package-verify]");
  if (packageVerifyButton) packageVerifyButton.addEventListener("click", verifyCustomerProjectPackage);
  const packageDiffButton = document.querySelector("[data-project-package-diff]");
  if (packageDiffButton) packageDiffButton.addEventListener("click", diffCustomerProjectPackage);
  const proposalVerifyButton = document.querySelector("[data-project-proposal-verify]");
  if (proposalVerifyButton) proposalVerifyButton.addEventListener("click", verifyCustomerProjectProposalBundle);
  const dossierVerifyButton = document.querySelector("[data-project-dossier-verify]");
  if (dossierVerifyButton) dossierVerifyButton.addEventListener("click", verifyCustomerProjectAcceptanceDossier);
  const refreshButton = document.querySelector("[data-project-events-refresh]");
  if (refreshButton) refreshButton.addEventListener("click", refreshProjectScopedEvents);
  const resourceRegisterButton = document.querySelector("[data-resource-register]");
  if (resourceRegisterButton) resourceRegisterButton.addEventListener("click", registerDeliveryResourceFromForm);
  const resourceHistoryButton = document.querySelector("[data-resource-history]");
  if (resourceHistoryButton) resourceHistoryButton.addEventListener("click", loadDeliveryResourceHistory);
  document.querySelectorAll("[data-resource-governance-requests]").forEach((button) => {
    if (button.dataset.requestsWired === "true") return;
    button.dataset.requestsWired = "true";
    button.addEventListener("click", () => loadDeliveryResourceGovernanceRequests(button.dataset.resourceGovernanceRequests || "all"));
  });
  const resourceEscalateButton = document.querySelector("[data-resource-governance-escalate-overdue]");
  if (resourceEscalateButton) resourceEscalateButton.addEventListener("click", escalateOverdueDeliveryResourceGovernanceRequests);
  document.querySelectorAll("[data-resource-disable]").forEach((button) => {
    button.addEventListener("click", () => requestDeliveryResourceDisable(button.dataset.resourceDisable || ""));
  });
  document.querySelectorAll("[data-resource-rollback]").forEach((button) => {
    button.addEventListener("click", () => rollbackDeliveryResourceRegistry(button.dataset.resourceRollback || "dry-run"));
  });
  wireResourceGovernanceReviewControls(document);
}

function wireCustomerProjectFilterControls() {
  const applyButton = document.querySelector("[data-project-filter-apply]");
  if (applyButton) applyButton.addEventListener("click", applyCustomerProjectFilters);
  const clearButton = document.querySelector("[data-project-filter-clear]");
  if (clearButton) clearButton.addEventListener("click", clearCustomerProjectFilters);
}

function wireCustomerProjectTemplateFilterControls() {
  const applyButton = document.querySelector("[data-template-filter-apply]");
  if (applyButton) applyButton.addEventListener("click", applyCustomerProjectTemplateFilters);
  const clearButton = document.querySelector("[data-template-filter-clear]");
  if (clearButton) clearButton.addEventListener("click", clearCustomerProjectTemplateFilters);
}

async function applyCustomerProjectFilters() {
  const values = {
    tenant_id: document.getElementById("project-filter-tenant")?.value || "",
    delivery_namespace: document.getElementById("project-filter-namespace")?.value || "",
    customer_id: document.getElementById("project-filter-customer")?.value || "",
    project_id: document.getElementById("project-filter-project")?.value || "",
    site_id: document.getElementById("project-filter-site")?.value || "",
    industry: document.getElementById("project-filter-industry")?.value || "",
    gate_status: document.getElementById("project-filter-gate")?.value || "",
    deployment_stage: document.getElementById("project-filter-stage")?.value || "",
  };
  Object.entries(values).forEach(([key, value]) => {
    const storageKey = `askme.customer_project_filter.${key}`;
    if (String(value || "").trim()) {
      localStorage.setItem(storageKey, String(value || "").trim());
    } else {
      localStorage.removeItem(storageKey);
    }
  });
  await renderProjects();
}

async function clearCustomerProjectFilters() {
  CUSTOMER_PROJECT_FILTER_KEYS.forEach((key) => {
    localStorage.removeItem(`askme.customer_project_filter.${key}`);
  });
  await renderProjects();
}

async function applyCustomerProjectTemplateFilters() {
  const values = {
    tenant_id: document.getElementById("template-filter-tenant")?.value || "",
    delivery_namespace: document.getElementById("template-filter-namespace")?.value || "",
    industry: document.getElementById("template-filter-industry")?.value || "",
    publish_status: document.getElementById("template-filter-publish-status")?.value || "",
    product_status: document.getElementById("template-filter-product-status")?.value || "",
    template_id: document.getElementById("template-filter-template-id")?.value || "",
    release_channel: document.getElementById("template-filter-release-channel")?.value || "",
    owner: document.getElementById("template-filter-owner")?.value || "",
  };
  Object.entries(values).forEach(([key, value]) => {
    const storageKey = `askme.customer_project_template_filter.${key}`;
    if (String(value || "").trim()) {
      localStorage.setItem(storageKey, String(value || "").trim());
    } else {
      localStorage.removeItem(storageKey);
    }
  });
  await renderProjects();
}

async function clearCustomerProjectTemplateFilters() {
  CUSTOMER_PROJECT_TEMPLATE_FILTER_KEYS.forEach((key) => {
    localStorage.removeItem(`askme.customer_project_template_filter.${key}`);
  });
  await renderProjects();
}

function templateReleaseResultEl(templateId) {
  return document.getElementById(`template-release-result-${safeDomId(templateId)}`);
}

function templateReleaseGovernanceResultEl() {
  return document.getElementById("template-governance-result");
}

async function updateTemplateRelease(templateId, publishStatus) {
  if (!templateId) return;
  const resultEl = templateReleaseResultEl(templateId);
  const defaultReason = `Set ${templateId} to ${publishStatus}`;
  const reason = window.prompt("请输入发布治理原因", defaultReason);
  if (reason === null) return;
  const body = {
    operator_id: operatorId(),
    reason: reason || defaultReason,
    release: {
      publish_status: publishStatus,
      release_channel: publishStatus === "published" ? "stable" : publishStatus,
    },
  };
  const response = await postJson(
    `${ENDPOINTS.fieldCustomerProjectTemplates}/${encodeURIComponent(templateId)}/release`,
    body,
  );
  if (resultEl) resultEl.innerHTML = renderTemplateReleaseResult(response.payload, response.ok);
  if (response.ok) {
    await renderProjects();
    const refreshedEl = templateReleaseResultEl(templateId);
    if (refreshedEl) refreshedEl.innerHTML = renderTemplateReleaseResult(response.payload, response.ok);
  }
}

async function loadTemplateReleaseHistory(templateId) {
  if (!templateId) return;
  const resultEl = templateReleaseResultEl(templateId);
  const payload = await getJson(
    `${ENDPOINTS.fieldCustomerProjectTemplates}/${encodeURIComponent(templateId)}/history?limit=8`,
    { found: false, revisions: [], reason: "request_failed" },
  );
  if (resultEl) resultEl.innerHTML = renderTemplateReleaseHistory(payload);
}

async function createTemplateReleaseRequest(templateId, publishStatus) {
  if (!templateId) return;
  const resultEl = templateReleaseResultEl(templateId);
  const defaultReason = `Request ${templateId} ${publishStatus} release`;
  const reason = window.prompt("请输入模板发布申请原因", defaultReason);
  if (reason === null) return;
  const body = {
    operator_id: operatorId(),
    reason: reason || defaultReason,
    release: {
      publish_status: publishStatus,
      release_channel: publishStatus === "published" ? "stable" : publishStatus,
    },
  };
  const response = await postJson(
    `${ENDPOINTS.fieldCustomerProjectTemplates}/${encodeURIComponent(templateId)}/release-requests`,
    body,
  );
  if (resultEl) resultEl.innerHTML = renderTemplateReleaseRequestResult(response.payload, response.ok);
}

async function loadTemplateReleaseRequests(templateId = "") {
  const resultEl = templateReleaseResultEl(templateId);
  const params = new URLSearchParams({ limit: "10" });
  if (templateId) params.set("template_id", templateId);
  const payload = await getJson(
    `${ENDPOINTS.fieldCustomerProjectTemplateReleaseRequests}?${params.toString()}`,
    { requests: [], summary: {}, request_count: 0 },
  );
  if (resultEl) {
    resultEl.innerHTML = renderTemplateReleaseRequests(payload);
    wireTemplateReleaseReviewControls(resultEl, templateId);
  }
}

function wireTemplateReleaseReviewControls(root = document, fallbackTemplateId = "") {
  root.querySelectorAll("[data-template-request-review]").forEach((button) => {
    if (button.dataset.reviewWired === "true") return;
    button.dataset.reviewWired = "true";
    button.addEventListener("click", () => reviewTemplateReleaseRequest(
      button.dataset.templateRequestReview || "",
      button.dataset.reviewDecision || "approve",
      button.dataset.templateId || fallbackTemplateId,
    ));
  });
}

async function exportTemplateReleaseNotesBundle() {
  const resultEl = document.getElementById("template-release-notes-export-result");
  const customerName = window.prompt("客户方案包使用的客户名称", "客户");
  if (customerName === null) return;
  const projectName = window.prompt("Project name for proposal bundle", "AskMe Robot Deployment");
  if (projectName === null) return;
  const response = await postJson(ENDPOINTS.fieldCustomerProjectTemplateReleaseNotesExport, {
    operator_id: operatorId(),
    customer_context: {
      customer_name: customerName || "客户",
      project_name: projectName || "机器人现场交付项目",
    },
  });
  if (response.ok && response.payload?.bundle) {
    downloadTemplateReleaseNotesBundle(response.payload.bundle);
  }
  if (resultEl) resultEl.innerHTML = renderTemplateReleaseNotesBundleResult(response.payload, response.ok);
}

function downloadTemplateReleaseNotesBundle(bundle = {}) {
  const files = bundle.files || {};
  const jsonFilename = files.json_filename || "askme-template-release-notes.json";
  const htmlFilename = files.html_filename || "askme-template-release-notes.html";
  const jsonBundle = { ...bundle };
  delete jsonBundle.html;
  downloadTextFile(
    jsonFilename,
    JSON.stringify(jsonBundle, null, 2),
    "application/json;charset=utf-8",
  );
  downloadTextFile(
    htmlFilename,
    bundle.html || "",
    "text/html;charset=utf-8",
  );
}

async function reviewTemplateReleaseRequest(requestId, decision, templateId = "") {
  if (!requestId) return;
  const reason = window.prompt("模板发布复核原因", `${decision} ${requestId}`);
  if (reason === null) return;
  const governanceResult = templateReleaseGovernanceResultEl();
  if (governanceResult) governanceResult.textContent = "正在提交模板发布复核...";
  const response = await postJson(
    `${ENDPOINTS.fieldCustomerProjectTemplateReleaseRequests}/${encodeURIComponent(requestId)}/review`,
    {
      operator_id: operatorId(),
      decision,
      reason,
    },
  );
  const resultEl = templateReleaseResultEl(templateId || response.payload?.request?.template_id || "");
  if (resultEl) resultEl.innerHTML = renderTemplateReleaseReviewResult(response.payload, response.ok);
  if (governanceResult) governanceResult.innerHTML = renderTemplateReleaseReviewResult(response.payload, response.ok);
  if (response.ok) {
    await renderProjects();
    const refreshedEl = templateReleaseResultEl(templateId || response.payload?.request?.template_id || "");
    if (refreshedEl) refreshedEl.innerHTML = renderTemplateReleaseReviewResult(response.payload, response.ok);
    const refreshedGovernanceResult = templateReleaseGovernanceResultEl();
    if (refreshedGovernanceResult) {
      refreshedGovernanceResult.innerHTML = renderTemplateReleaseReviewResult(response.payload, response.ok);
    }
  }
}

function renderTemplateReleaseResult(payload = {}, ok = false) {
  const templatePackage = payload.template_package || {};
  const revision = payload.revision || {};
  return `
    <div class="project-import-card ${ok ? "ok" : "warn"}">
      <strong>模板发布 ${badge(ok ? "已接受" : "已拒绝", ok ? "ok" : "err")}</strong>
      <p>${esc(payload.reason || payload.next_step || templatePackage.customer_status || "模板发布治理结果。")}</p>
      <div class="row-meta">
        <span>模板 ${esc(payload.template_id || templatePackage.template_id || "-")}</span>
        <span>状态 ${esc(templatePackage.publish_status || "-")}</span>
        <span>产品 ${esc(templatePackage.product_status || "-")}</span>
        <span>版本 ${esc(templatePackage.version || "-")}</span>
        <span>修订 ${esc(revision.revision_id || "-")}</span>
      </div>
    </div>
  `;
}

function renderTemplateReleaseRequestResult(payload = {}, ok = false) {
  const request = payload.request || {};
  const proposed = request.proposed_template_package || payload.template_package || {};
  return `
    <div class="project-import-card ${ok ? "ok" : "warn"}">
      <strong>模板发布申请 ${badge(ok ? "已创建" : "已拒绝", ok ? "ok" : "err")}</strong>
      <p>${esc(payload.reason || payload.next_step || "需要第二位产品负责人复核。")}</p>
      <div class="row-meta">
        <span>申请 ${esc(request.request_id || "-")}</span>
        <span>模板 ${esc(request.template_id || proposed.template_id || "-")}</span>
        <span>状态 ${esc(request.status || "-")}</span>
        <span>目标状态 ${esc(proposed.publish_status || "-")}</span>
      </div>
    </div>
  `;
}

function renderTemplateReleaseNotesBundleResult(payload = {}, ok = false) {
  const bundle = payload.bundle || {};
  const manifest = bundle.manifest || {};
  const files = bundle.files || {};
  const proposal = bundle.proposal_insert || {};
  return `
    <div class="project-import-card ${ok ? "ok" : "warn"}">
      <strong>客户方案包 ${badge(ok ? "已生成" : "生成失败", ok ? "ok" : "err")}</strong>
      <p>${esc(payload.reason || payload.next_step || bundle.delivery_boundary || "方案包生成结果。")}</p>
      ${ok ? `<p class="muted-line">浏览器已下载 JSON 与可打印 HTML。</p>` : ""}
      <div class="row-meta">
        <span>说明 ${esc(manifest.release_note_count ?? bundle.release_note_count ?? 0)}</span>
        <span>模板 ${esc(manifest.template_count ?? bundle.summary?.template_count ?? 0)}</span>
        <span>json ${esc(files.json_filename || "-")}</span>
        <span>html ${esc(files.html_filename || "-")}</span>
      </div>
      ${proposal.section_title ? `<p class="muted-line">${esc(proposal.section_title)} / ${esc(proposal.customer_message || "")}</p>` : ""}
      <div class="mono">${esc(JSON.stringify({ manifest, customer_context: bundle.customer_context || {} }, null, 2))}</div>
    </div>
  `;
}

function renderTemplateReleaseRequests(payload = {}) {
  const requests = Array.isArray(payload.requests) ? payload.requests : [];
  return `
    <div class="project-import-card">
      <strong>模板发布申请</strong>
      <div class="row-meta">
        <span>待复核 ${esc(payload.summary?.pending_count ?? 0)}</span>
        <span>已通过 ${esc(payload.summary?.approved_count ?? 0)}</span>
        <span>已拒绝 ${esc(payload.summary?.rejected_count ?? 0)}</span>
      </div>
      <div class="mini-list">
        ${requests.map((item) => `
          <div class="mini-row">
            <b>${esc(item.template_id || "-")} ${badge(item.status || "unknown", acceptanceGateClass(item.status))}</b>
            <span>${esc(item.requested_by || "system")} / ${esc(new Date(Number(item.requested_at || 0) * 1000).toLocaleString())}</span>
            <span>${esc(item.release?.publish_status || "-")} / ${esc(item.proposed_template_package?.version || "-")}</span>
            ${item.status === "pending" ? `
              <span>
                <button class="ghost-button" data-template-request-review="${esc(item.request_id || "")}" data-review-decision="approve">通过</button>
                <button class="ghost-button" data-template-request-review="${esc(item.request_id || "")}" data-review-decision="reject">拒绝</button>
              </span>
            ` : ""}
          </div>
        `).join("") || `<div class="mini-list-empty">暂无发布申请。</div>`}
      </div>
    </div>
  `;
}

function renderTemplateReleaseReviewResult(payload = {}, ok = false) {
  const request = payload.request || {};
  return `
    <div class="project-import-card ${ok ? "ok" : "warn"}">
      <strong>模板发布复核 ${badge(ok ? request.status || "已接受" : "已拒绝", ok ? "ok" : "err")}</strong>
      <p>${esc(payload.reason || payload.next_step || "发布复核结果。")}</p>
      <div class="row-meta">
        <span>申请 ${esc(request.request_id || "-")}</span>
        <span>模板 ${esc(request.template_id || "-")}</span>
        <span>reviewer ${esc(request.reviewed_by || "-")}</span>
        <span>status ${esc(request.status || "-")}</span>
      </div>
    </div>
  `;
}

function renderTemplateReleaseGovernance(payload = {}) {
  const requests = Array.isArray(payload.requests) ? payload.requests : [];
  const summary = payload.summary || {};
  const pending = requests.filter((item) => item.status === "pending");
  const reviewed = requests.filter((item) => item.status !== "pending");
  return `
    <section id="project-section-template-governance" class="card">
      <div class="section-title-row">
        <div>
          <h2>模板发布治理</h2>
          <p>产品负责人先发起发布请求，第二个产品负责人复核后，模板才会进入客户可见的发布说明。这里用于防止销售材料直接引用未审批模板。</p>
        </div>
        ${badge(`${summary.pending_count ?? pending.length} 待审批`, Number(summary.pending_count || pending.length) ? "warn" : "ok")}
      </div>
      <div class="grid four">
        <div class="metric warn"><b>${esc(summary.pending_count ?? pending.length)}</b><span>待审批</span></div>
        <div class="metric ok"><b>${esc(summary.approved_count ?? 0)}</b><span>已批准</span></div>
        <div class="metric err"><b>${esc(summary.rejected_count ?? 0)}</b><span>已拒绝</span></div>
        <div class="metric"><b>${esc(payload.request_count ?? requests.length)}</b><span>请求总数</span></div>
      </div>
      <div class="template-governance-board">
        <div>
          <strong>待复核发布</strong>
          <div class="mini-list">
            ${pending.slice(0, 8).map(renderTemplateReleaseGovernanceRequest).join("") || `<div class="mini-list-empty">当前没有待复核发布请求。</div>`}
          </div>
        </div>
        <div>
          <strong>最近处理记录</strong>
          <div class="mini-list">
            ${reviewed.slice(0, 8).map(renderTemplateReleaseGovernanceRequest).join("") || `<div class="mini-list-empty">还没有已处理发布请求。</div>`}
          </div>
        </div>
      </div>
      <div id="template-governance-result" class="small-note"></div>
    </section>
  `;
}

function renderTemplateReleaseGovernanceRequest(item = {}) {
  const proposed = item.proposed_template_package || item.applied_template_package || {};
  const templateId = item.template_id || proposed.template_id || "";
  const status = item.status || "unknown";
  return `
    <div class="mini-row template-governance-request">
      <b>${esc(templateId || "-")} ${badge(status, acceptanceGateClass(status))}</b>
      <span>request ${esc(String(item.request_id || "").slice(0, 22))}</span>
      <span>${esc(item.requested_by || "system")} -> ${esc(proposed.publish_status || item.release?.publish_status || "-")} / v${esc(proposed.version || "-")}</span>
      <span>${esc(item.reason || item.review_reason || "-")}</span>
      ${status === "pending" ? `
        <span class="panel-actions inline-actions">
          <button class="ghost-button" data-template-request-review="${esc(item.request_id || "")}" data-review-decision="approve" data-template-id="${esc(templateId)}">批准</button>
          <button class="ghost-button" data-template-request-review="${esc(item.request_id || "")}" data-review-decision="reject" data-template-id="${esc(templateId)}">拒绝</button>
        </span>
      ` : `
        <span>复核人 ${esc(item.reviewed_by || "-")}</span>
      `}
    </div>
  `;
}

function renderTemplateReleaseHistory(payload = {}) {
  const revisions = Array.isArray(payload.revisions) ? payload.revisions : [];
  if (!payload.found) {
    return `<div class="mini-list-empty">模板发布历史不可用：${esc(payload.reason || "not found")}</div>`;
  }
  return `
    <div class="project-import-card">
      <strong>模板发布历史</strong>
      <div class="mini-list">
        ${revisions.map((item) => `
          <div class="mini-row">
            <b>${esc(item.action || item.revision_id || "release")}</b>
            <span>${esc(item.operator_id || "system")} / ${esc(new Date(Number(item.created_at || 0) * 1000).toLocaleString())}</span>
            <span>${esc(item.template_release?.publish_status || "-")} / ${esc(item.template_release?.version || "-")}</span>
          </div>
        `).join("") || `<div class="mini-list-empty">暂无发布修订记录。</div>`}
      </div>
    </div>
  `;
}

function projectPackageFromTextarea() {
  const raw = document.getElementById("project-import-json")?.value || "";
  if (!raw.trim()) throw new Error("请先粘贴客户项目包 JSON");
  const parsed = JSON.parse(raw);
  return parsed.package && typeof parsed.package === "object" ? parsed.package : parsed;
}

function projectProposalFromTextarea() {
  const raw = document.getElementById("project-proposal-json")?.value
    || document.getElementById("project-import-json")?.value
    || "";
  if (!raw.trim()) throw new Error("请先粘贴客户提案包 JSON");
  const parsed = JSON.parse(raw);
  return parsed.proposal && typeof parsed.proposal === "object" ? parsed.proposal : parsed;
}

function projectDossierFromTextarea() {
  const raw = document.getElementById("project-dossier-json")?.value
    || document.getElementById("project-import-json")?.value
    || "";
  if (!raw.trim()) throw new Error("请先粘贴客户验收证据包 JSON");
  const parsed = JSON.parse(raw);
  return parsed.dossier && typeof parsed.dossier === "object" ? parsed.dossier : parsed;
}

async function verifyCustomerProjectProposalBundle() {
  const resultEl = document.getElementById("project-import-result");
  try {
    const proposalPayload = projectProposalFromTextarea();
    const response = await postJson(ENDPOINTS.fieldCustomerProjectProposalBundleVerify, {
      operator_id: operatorId(),
      proposal: proposalPayload,
    });
    if (resultEl) resultEl.innerHTML = renderProjectProposalVerifyResult(response.payload, response.ok);
  } catch (error) {
    if (resultEl) {
      resultEl.innerHTML = `<div class="mini-list-empty">提案包验签失败：${esc(error.message)}</div>`;
    }
  }
}

async function verifyCustomerProjectAcceptanceDossier() {
  const resultEl = document.getElementById("project-import-result");
  try {
    const dossierPayload = projectDossierFromTextarea();
    const response = await postJson(ENDPOINTS.fieldCustomerProjectAcceptanceDossierVerify, {
      operator_id: operatorId(),
      dossier: dossierPayload,
    });
    if (resultEl) resultEl.innerHTML = renderProjectDossierVerifyResult(response.payload, response.ok);
  } catch (error) {
    if (resultEl) {
      resultEl.innerHTML = `<div class="mini-list-empty">验收证据包验签失败：${esc(error.message)}</div>`;
    }
  }
}

async function verifyCustomerProjectPackage() {
  const resultEl = document.getElementById("project-import-result");
  try {
    const packagePayload = projectPackageFromTextarea();
    const response = await postJson(ENDPOINTS.fieldCustomerProjectPackageVerify, {
      operator_id: operatorId(),
      package: packagePayload,
    });
    if (resultEl) resultEl.innerHTML = renderProjectPackageVerifyResult(response.payload, response.ok);
  } catch (error) {
    if (resultEl) {
      resultEl.innerHTML = `<div class="mini-list-empty">项目交付包验包失败：${esc(error.message)}</div>`;
    }
  }
}

async function diffCustomerProjectPackage() {
  const resultEl = document.getElementById("project-import-result");
  try {
    const packagePayload = projectPackageFromTextarea();
    const response = await postJson(ENDPOINTS.fieldCustomerProjectPackageDiff, {
      operator_id: operatorId(),
      package: packagePayload,
    });
    if (resultEl) resultEl.innerHTML = renderProjectImportResult(response.payload, response.ok, true, "差异预览");
  } catch (error) {
    if (resultEl) {
      resultEl.innerHTML = `<div class="mini-list-empty">项目交付包差异预览失败：${esc(error.message)}</div>`;
    }
  }
}

async function importCustomerProjectPackage(dryRun = true) {
  const resultEl = document.getElementById("project-import-result");
  try {
    const packagePayload = projectPackageFromTextarea();
    const body = {
      operator_id: operatorId(),
      dry_run: dryRun,
      overwrite: Boolean(document.getElementById("project-import-overwrite")?.checked),
      package: packagePayload,
    };
    const response = await postJson(ENDPOINTS.fieldCustomerProjectImport, body);
    if (resultEl) resultEl.innerHTML = renderProjectImportResult(response.payload, response.ok, dryRun);
    if (response.ok && !dryRun) await renderProjects();
  } catch (error) {
    if (resultEl) {
      resultEl.innerHTML = `<div class="mini-list-empty">项目交付包解析失败：${esc(error.message)}</div>`;
    }
  }
}

function renderProjectImportResult(payload = {}, ok = false, dryRun = true, modeLabel = "") {
  const verification = payload.verification || {};
  const diff = payload.diff || {};
  const failures = Array.isArray(verification.failures) ? verification.failures : [];
  const fieldChanges = Array.isArray(diff.field_changes) ? diff.field_changes : [];
  const legacyFields = diff.fields && typeof diff.fields === "object" ? diff.fields : {};
  const diffDetails = fieldChanges.length ? fieldChanges : legacyFields;
  const hasDiffDetails = fieldChanges.length || Object.keys(legacyFields).length;
  const incomingBinding = diff.incoming_binding_readiness_summary || {};
  const currentBinding = diff.current_binding_readiness_summary || {};
  const incomingDeliveryGate = payload.delivery_gate || diff.incoming_delivery_gate || verification.delivery_gate || {};
  const currentDeliveryGate = diff.current_delivery_gate || {};
  const title = modeLabel || (dryRun ? "导入演练" : "导入结果");
  const defaultMessage = ok
    ? "交付包结构、完整性和项目范围校验通过。"
    : "交付包未通过校验，请先处理范围、完整性或准入问题。";
  return `
    <div class="project-import-card ${ok ? "ok" : "warn"}">
      <strong>${esc(title)} ${badge(ok ? "通过" : "失败", ok ? "ok" : "err")}</strong>
      <p>${esc(payload.reason || payload.next_step || defaultMessage)}</p>
      ${renderProjectPackageScopeEvidence(payload, verification, diff)}
      <div class="row-meta">
        <span>完整性：${esc(verification.valid === false ? "未通过" : "通过")}</span>
        <span>变化类型：${esc(projectChangeTypeLabel(diff.change_type || "-"))}</span>
        <span>新包资源：${esc(deliveryStatusLabel(incomingBinding.overall_status || "-"))}</span>
        ${currentBinding.overall_status ? `<span>当前资源：${esc(deliveryStatusLabel(currentBinding.overall_status))}</span>` : ""}
        <span>是否会写入：${esc(payload.would_write === true ? "会" : payload.would_write === false ? "不会" : "-")}</span>
        <span>保存路径：${esc(payload.path || payload.profile_path || "-")}</span>
      </div>
      ${renderProjectPackageDeliveryGate(incomingDeliveryGate, "新交付包准入")}
      ${currentDeliveryGate.delivery_gate_status ? renderProjectPackageDeliveryGate(currentDeliveryGate, "当前项目准入") : ""}
      ${renderProjectBindingReadiness(incomingBinding, "新交付包资源绑定")}
      ${currentBinding.overall_status ? renderProjectBindingReadiness(currentBinding, "当前项目资源绑定") : ""}
      ${renderProjectImplementationHandoff(payload.implementation_handoff, dryRun ? "导入演练后的实施步骤" : "导入后的实施步骤")}
      ${renderProjectReuseAssessment(diff.incoming_reuse_assessment, "新交付包复用性")}
      ${diff.current_reuse_assessment ? renderProjectReuseAssessment(diff.current_reuse_assessment, "当前项目复用性") : ""}
      ${renderProjectCollisionCandidates(diff.collision_candidates)}
      ${failures.length ? `<div class="skill-validation">${failures.map((item) => `<span class="err">${esc(item)}</span>`).join("")}</div>` : ""}
      ${hasDiffDetails ? `<pre>${esc(JSON.stringify(diffDetails, null, 2))}</pre>` : ""}
    </div>
  `;
}

function renderProjectImplementationHandoff(handoff = {}, title = "项目实施步骤") {
  if (!handoff || typeof handoff !== "object" || !handoff.handoff_schema) return "";
  const summary = handoff.summary || {};
  const steps = Array.isArray(handoff.next_steps) ? handoff.next_steps : [];
  const todos = Array.isArray(handoff.object_binding_todo) ? handoff.object_binding_todo : [];
  const blockedObjects = todos.filter((item) => (
    Array.isArray(item.missing_binding_labels) && item.missing_binding_labels.length
  ));
  return `
    <div class="project-create-result-card ${blockedObjects.length ? "warn" : "ok"}" data-project-implementation-handoff>
      <div class="project-delivery-head">
        <strong>${esc(title)} ${badge(blockedObjects.length ? "待补齐" : "可继续验收", blockedObjects.length ? "warn" : "ok")}</strong>
        <span>${esc(handoff.customer_status || "请按步骤完成项目实施交接。")}</span>
      </div>
      <div class="project-scope-evidence">
        <div><b>客户/项目</b><span>${esc([handoff.customer_name, handoff.project_name || handoff.project_id].filter(Boolean).join(" / ") || "-")}</span></div>
        <div><b>现场</b><span>${esc(handoff.site_name || handoff.site_id || "-")}</span></div>
        <div><b>对象状态</b><span>${esc(summary.object_ready_count || 0)} 个已就绪，${esc(summary.object_needs_binding_count || 0)} 个待补齐</span></div>
      </div>
      <div class="project-create-next-steps">
        ${steps.slice(0, 4).map((step) => `
          <div>
            <b>${esc(step.label || step.step_id || "实施步骤")}</b>
            <span>${esc(step.customer_next_step || step.status || "")}</span>
          </div>
        `).join("")}
      </div>
      ${blockedObjects.length ? `
        <div class="mini-list">
          ${blockedObjects.slice(0, 6).map((item) => `
            <div class="mini-row">
              <b>${esc(item.display_name || item.object_id || "现场对象")}</b>
              <span>待补齐：${esc((item.missing_binding_labels || []).join("、") || "-")}</span>
            </div>
          `).join("")}
        </div>
      ` : ""}
    </div>
  `;
}

function renderManagedObjectWriteResult(payload = {}, ok = false, actionLabel = "对象已保存") {
  if (!ok || !payload.accepted) {
    return `
      <div class="project-import-card warn" data-managed-object-write-result>
        <strong>对象变更未生效 ${badge("blocked", "err")}</strong>
        <p>${esc(payload.reason || payload.error || "请检查项目、对象编号、验收项和操作权限。")}</p>
      </div>
    `;
  }
  const change = payload.object_change || {};
  const managedObject = payload.managed_object || payload.deleted_object || {};
  const handoff = payload.implementation_handoff || {};
  const displayName = managedObject.display_name || change.after?.display_name || change.before?.display_name || payload.object_id || "-";
  return `
    <div class="project-import-card ok" data-managed-object-write-result>
      <strong>${esc(actionLabel)} ${badge(payload.object_id || "object", "ok")}</strong>
      <p>${esc(payload.next_step || handoff.customer_status || "请继续根据对象配置完成资源绑定和现场验收。")}</p>
      <div class="row-meta">
        <span>现场对象：${esc(displayName)}</span>
        <span>动作：${esc(change.action || "-")}</span>
        <span>操作人：${esc(change.operator_id || operatorId())}</span>
        <span>保存路径：${esc(payload.profile_path || "-")}</span>
      </div>
      ${renderProjectImplementationHandoff(handoff, "对象变更后的实施步骤")}
    </div>
  `;
}

function renderProjectPackageVerifyResult(payload = {}, ok = false) {
  const verification = payload.verification || {};
  const failures = Array.isArray(verification.failures) ? verification.failures : [];
  const manifest = verification.manifest || {};
  const deliveryGate = verification.delivery_gate || {};
  return `
    <div class="project-import-card ${ok ? "ok" : "warn"}">
      <strong>交付包验包 ${badge(ok ? "通过" : "失败", ok ? "ok" : "err")}</strong>
      <p>${esc(payload.reason || (ok ? "交付包完整性和客户项目范围校验通过。" : "交付包校验失败，请先修正后再预览或导入。"))}</p>
      ${renderProjectPackageScopeEvidence(payload, verification, {})}
      <div class="row-meta">
        <span>完整性：${esc(verification.valid === false ? "未通过" : "通过")}</span>
        <span>项目：${esc(manifest.project_id || "-")}</span>
        <span>现场：${esc(manifest.site_id || "-")}</span>
        <span>交付包版本：${esc(manifest.package_schema || manifest.schema_version || "-")}</span>
      </div>
      ${renderProjectPackageDeliveryGate(deliveryGate, "交付包准入")}
      ${failures.length ? `<div class="skill-validation">${failures.map((item) => `<span class="err">${esc(item)}</span>`).join("")}</div>` : ""}
    </div>
  `;
}

function renderProjectProposalVerifyResult(payload = {}, ok = false) {
  const verification = payload.verification || {};
  const scope = payload.proposal_scope || verification.proposal_scope || {};
  const manifest = verification.manifest || {};
  const errors = Array.isArray(verification.errors) ? verification.errors : [];
  return `
    <div class="project-import-card ${ok ? "ok" : "warn"}">
      <strong>客户方案包校验 ${badge(ok ? "通过" : "失败", ok ? "ok" : "err")}</strong>
      <p>${esc(payload.reason || (ok ? "项目包、验收材料和模板发布记录一致。" : "客户方案包完整性校验失败。"))}</p>
      <div class="row-meta">
        <span>范围：${esc(renderProjectScopeLabel(scope))}</span>
        <span>项目：${esc(manifest.project_id || "-")}</span>
        <span>package: ${esc(String(verification.package_sha256 || manifest.package_sha256 || "").slice(0, 16))}</span>
        <span>dossier: ${esc(String(verification.dossier_sha256 || manifest.dossier_sha256 || "").slice(0, 16))}</span>
        <span>release: ${esc(String(verification.release_notes_sha256 || manifest.release_notes_sha256 || "").slice(0, 16))}</span>
      </div>
      ${errors.length ? `<div class="skill-validation">${errors.map((item) => `<span class="err">${esc(item)}</span>`).join("")}</div>` : ""}
      <div class="mono">${esc(JSON.stringify({ valid: verification.valid, manifest, scope }, null, 2))}</div>
    </div>
  `;
  return `
    <div class="project-import-card ${ok ? "ok" : "warn"}">
      <strong>提案包验签 ${badge(ok ? "通过" : "失败", ok ? "ok" : "err")}</strong>
      <p>${esc(payload.reason || (ok ? "客户项目、验收证据包和模板发布记录未发现篡改。" : "提案包未通过完整性校验。"))}</p>
      <div class="row-meta">
        <span>scope: ${esc(renderProjectScopeLabel(scope))}</span>
        <span>project: ${esc(manifest.project_id || "-")}</span>
        <span>package: ${esc(String(verification.package_sha256 || manifest.package_sha256 || "").slice(0, 16))}</span>
        <span>dossier: ${esc(String(verification.dossier_sha256 || manifest.dossier_sha256 || "").slice(0, 16))}</span>
        <span>release: ${esc(String(verification.release_notes_sha256 || manifest.release_notes_sha256 || "").slice(0, 16))}</span>
      </div>
      ${errors.length ? `<div class="skill-validation">${errors.map((item) => `<span class="err">${esc(item)}</span>`).join("")}</div>` : ""}
      <div class="mono">${esc(JSON.stringify({ valid: verification.valid, manifest, scope }, null, 2))}</div>
    </div>
  `;
}

function renderProjectDossierVerifyResult(payload = {}, ok = false) {
  const verification = payload.verification || {};
  const scope = payload.dossier_scope || verification.dossier_scope || {};
  const manifest = verification.manifest || {};
  const errors = Array.isArray(verification.errors) ? verification.errors : [];
  return `
    <div class="project-import-card ${ok ? "ok" : "warn"}">
      <strong>验收材料校验 ${badge(ok ? "通过" : "失败", ok ? "ok" : "err")}</strong>
      <p>${esc(payload.reason || (ok ? "验收材料清单、证据目录和完整性校验一致。" : "验收材料完整性校验失败。"))}</p>
      <div class="row-meta">
        <span>范围：${esc(renderProjectScopeLabel(scope))}</span>
        <span>项目：${esc(manifest.project_id || "-")}</span>
        <span>status: ${esc(manifest.overall_status || "-")}</span>
        <span>onsite: ${esc(manifest.onsite_evidence_status || "-")}</span>
        <span>payload: ${esc(String(verification.payload_sha256 || manifest.payload_sha256 || "").slice(0, 16))}</span>
      </div>
      ${errors.length ? `<div class="skill-validation">${errors.map((item) => `<span class="err">${esc(item)}</span>`).join("")}</div>` : ""}
      <div class="mono">${esc(JSON.stringify({ valid: verification.valid, manifest, scope }, null, 2))}</div>
    </div>
  `;
  return `
    <div class="project-import-card ${ok ? "ok" : "warn"}">
      <strong>验收包验签 ${badge(ok ? "通过" : "失败", ok ? "ok" : "err")}</strong>
      <p>${esc(payload.reason || (ok ? "客户验收包 manifest、证据清单和哈希一致。" : "客户验收包未通过完整性校验。"))}</p>
      <div class="row-meta">
        <span>scope: ${esc(renderProjectScopeLabel(scope))}</span>
        <span>project: ${esc(manifest.project_id || "-")}</span>
        <span>status: ${esc(manifest.overall_status || "-")}</span>
        <span>onsite: ${esc(manifest.onsite_evidence_status || "-")}</span>
        <span>payload: ${esc(String(verification.payload_sha256 || manifest.payload_sha256 || "").slice(0, 16))}</span>
      </div>
      ${errors.length ? `<div class="skill-validation">${errors.map((item) => `<span class="err">${esc(item)}</span>`).join("")}</div>` : ""}
      <div class="mono">${esc(JSON.stringify({ valid: verification.valid, manifest, scope }, null, 2))}</div>
    </div>
  `;
}

async function refreshProjectScopedEvents() {
  const resultEl = document.getElementById("project-event-result");
  const projectId = document.getElementById("project-event-project")?.value || "";
  const objectId = document.getElementById("project-event-object")?.value || "";
  const params = new URLSearchParams({ limit: "20" });
  if (projectId) params.set("project_id", projectId);
  if (objectId) params.set("managed_object_id", objectId);
  const payload = await getJson(`${ENDPOINTS.fieldEvents}?${params.toString()}`, { events: [], summary: {} });
  if (resultEl) resultEl.innerHTML = renderProjectScopedEvents(payload);
}

function renderProjectScopedEvents(payload = {}) {
  const events = Array.isArray(payload.events) ? payload.events : [];
  const summary = payload.summary || {};
  if (!events.length) {
    return `<div class="mini-list-empty">当前筛选范围没有事件。接入摄像头、传感器或手动触发后，这里会显示项目和对象归属。</div>`;
  }
  return `
    <div class="project-event-summary">
      <div><b>${esc(payload.filtered_total ?? events.length)}</b><span>事件</span></div>
      <div><b>${esc(Object.keys(summary.by_project || {}).length)}</b><span>项目分布</span></div>
      <div><b>${esc(Object.keys(summary.by_managed_object || {}).length)}</b><span>对象分布</span></div>
    </div>
    <div class="mini-list">
      ${events.slice(0, 12).map((event) => `
        <div class="mini-row">
          <b>${esc(event.title || event.scenario_id || event.event_id || "-")}</b>
          <span>${esc(event.customer_id || "-")} / ${esc(event.project_id || "-")} / ${esc(event.managed_object_id || "-")}</span>
          <span>${esc(event.location_name || event.site_name || "-")} · ${esc(event.status || "-")}</span>
        </div>
      `).join("")}
    </div>
  `;
}

function renderTemplateReleaseNotes(payload = {}) {
  const notes = Array.isArray(payload.notes) ? payload.notes : [];
  const summary = payload.summary || {};
  return `
    <section id="project-section-template-releases" class="card">
      <div class="section-title-row">
        <div>
          <h2>模板发布说明</h2>
          <p>面向客户的模板包只有在产品负责人审批后才会出现在这里。交付时仍需绑定现场数据、设备、凭证和现场验收证据。</p>
        </div>
        ${badge(`${summary.approved_release_count ?? notes.length} 个已审批发布`, Number(summary.approved_release_count || notes.length) ? "ok" : "warn")}
      </div>
      <div class="grid four">
        <div class="metric"><b>${esc(summary.approved_release_count ?? notes.length)}</b><span>已审批发布</span></div>
        <div class="metric"><b>${esc(summary.template_count ?? 0)}</b><span>模板</span></div>
        <div class="metric warn"><b>${esc(summary.manual_check_count ?? 0)}</b><span>待复核</span></div>
        <div class="metric ok"><b>${esc(summary.ready_count ?? 0)}</b><span>就绪</span></div>
      </div>
      <div class="capability-list compact-list">
        ${notes.slice(0, 8).map((item) => `
          <div class="row-item">
            <strong>${esc(item.template_id || "-")} v${esc(item.version || "-")} ${badge(item.product_status || "unknown", acceptanceGateClass(item.product_status))}</strong>
            <span>${esc(item.release_channel || "-")} / 审批人 ${esc(item.approved_by || "-")} / ${esc(item.customer_status || item.customer_claim || "")}</span>
          </div>
        `).join("") || `<div class="mini-list-empty">暂无已审批的客户可见模板发布。</div>`}
      </div>
      <div class="panel-actions">
        <button class="ghost-button" data-template-release-notes-export>导出客户方案包</button>
      </div>
      <div id="template-release-notes-export-result" class="small-note"></div>
    </section>
  `;
}

function renderIndustryTemplateCatalog(payload = {}) {
  const templates = Array.isArray(payload.templates) ? payload.templates : [];
  const summary = payload.summary || {};
  currentCustomerProjectTemplateItems = templates;
  const selectedTemplate = selectedProjectTemplateForCreate(templates);
  return `
    <section id="project-section-templates" class="card">
      <div class="section-title-row">
        <div>
          <h2>行业模板市场</h2>
          <p>交付团队可以从厂区、园区、仓储或景区模板开始，再替换客户范围、现场地图、设备和凭证。</p>
        </div>
        ${badge(`${summary.valid_count ?? 0}/${summary.template_count ?? 0} 可用`, Number(summary.valid_count || 0) ? "ok" : "warn")}
      </div>
      <div class="grid four">
        <div class="metric"><b>${esc(summary.template_count ?? 0)}</b><span>模板</span></div>
        <div class="metric"><b>${esc(summary.industry_count ?? 0)}</b><span>行业</span></div>
        <div class="metric"><b>${esc(summary.delivery_namespace_count ?? 0)}</b><span>交付空间</span></div>
        <div class="metric"><b>${esc(summary.managed_object_type_count ?? 0)}</b><span>默认对象</span></div>
      </div>
      ${renderCustomerProjectTemplateFilterControls(payload)}
      <div class="template-grid">
        ${templates.map(renderIndustryTemplateItem).join("") || `<div class="mini-list-empty">还没有行业模板。</div>`}
      </div>
      <div class="field-form compact-form">
        <h3>从模板创建项目</h3>
        <select id="project-template-id">
          ${templates.map((item) => `<option value="${esc(item.template_id || "")}">${esc(item.display_name || item.template_id)} / ${esc(item.industry || "-")}</option>`).join("")}
        </select>
        <div id="project-template-create-readiness">
          ${renderProjectTemplateCreateReadiness(selectedTemplate)}
        </div>
        <input id="project-tenant-id" placeholder="客户空间，例如 fanmu-group" value="default">
        <input id="project-delivery-namespace" placeholder="交付空间，例如 pilot" value="default">
        <input id="project-customer-id" placeholder="客户编号，例如 fanmu">
        <input id="project-customer-name" placeholder="客户名称，例如梵木创艺园">
        <input id="project-industry" placeholder="行业，例如园区/工厂/仓储">
        <input id="project-id" placeholder="项目编号，例如 fanmu-phase-1">
        <input id="project-name" placeholder="项目名称">
        <input id="project-site-id" placeholder="现场编号">
        <input id="project-site-name" placeholder="现场名称">
        <div class="panel-actions">
          <button class="primary-button" data-project-create>创建客户项目</button>
        </div>
        <div id="project-create-result" class="small-note"></div>
      </div>
    </section>
  `;
}

function selectedProjectTemplateForCreate(templates = currentCustomerProjectTemplateItems) {
  const selectedId = document.getElementById("project-template-id")?.value
    || localStorage.getItem("askme.project_create.template_id")
    || "";
  return templates.find((item) => item.template_id === selectedId) || templates[0] || {};
}

function renderProjectTemplateCreateReadiness(template = {}) {
  if (!template || !template.template_id) {
    return `<div class="project-create-readiness warn"><strong>创建准入</strong><p>请先选择一个行业模板。</p></div>`;
  }
  const summary = template.managed_objects_summary || {};
  const delivery = template.delivery_summary || {};
  const templatePackage = template.template_package || {};
  const objects = Array.isArray(delivery.default_objects)
    ? delivery.default_objects
    : Array.isArray(summary.objects)
      ? summary.objects
      : [];
  const prerequisites = Array.isArray(template.customer_prerequisites)
    ? template.customer_prerequisites
    : [];
  const criteria = Array.isArray(template.scenario_acceptance_criteria)
    ? template.scenario_acceptance_criteria
    : [];
  const outOfScope = Array.isArray(template.out_of_scope) ? template.out_of_scope : [];
  const applicability = template.applicability_scope || delivery.applicability_scope || {};
  const status = templatePackage.product_status || delivery.acceptance_status || "manual_check";
  return `
    <div class="project-create-readiness ${acceptanceGateClass(status)}" data-project-template-create-readiness>
      <div class="project-delivery-head">
        <strong>创建准入 ${badge(status, acceptanceGateClass(status))}</strong>
        <span>${esc(template.customer_status || templatePackage.customer_status || delivery.delivery_boundary || "先确认客户范围、现场对象、设备和验收边界。")}</span>
      </div>
      <div class="grid four">
        <div class="metric"><b>${esc(objects.length)}</b><span>首批对象</span></div>
        <div class="metric"><b>${esc(prerequisites.length)}</b><span>客户配合项</span></div>
        <div class="metric"><b>${esc(criteria.length)}</b><span>验收条件</span></div>
        <div class="metric"><b>${esc(outOfScope.length)}</b><span>暂不承诺</span></div>
      </div>
      <div class="project-create-readiness-grid">
        <div>
          <b>适用范围</b>
          <span>${esc([
            ...(Array.isArray(applicability.industries) ? applicability.industries : []),
            ...(Array.isArray(applicability.site_types) ? applicability.site_types : []),
          ].slice(0, 6).join(", ") || template.industry || "未声明")}</span>
        </div>
        <div>
          <b>首批对象</b>
          <span>${esc(objects.slice(0, 5).map((item) => item.display_name || item.object_id || item.category).filter(Boolean).join(", ") || "未声明")}</span>
        </div>
        <div>
          <b>客户配合</b>
          <span>${esc(prerequisites.slice(0, 3).map((item) => item.label || item.prerequisite_id || item.next_step).filter(Boolean).join("；") || "创建后补齐现场信息")}</span>
        </div>
        <div>
          <b>验收条件</b>
          <span>${esc(criteria.slice(0, 3).map((item) => item.scenario_id || item.label || item.criteria_id).filter(Boolean).join(", ") || "创建后绑定验收用例")}</span>
        </div>
        <div>
          <b>边界说明</b>
          <span>${esc(outOfScope.slice(0, 3).join("；") || "不承诺无人值守生产上线，需现场验收。")}</span>
        </div>
      </div>
    </div>
  `;
}

function renderIndustryTemplateItem(item = {}) {
  const summary = item.managed_objects_summary || {};
  const delivery = item.delivery_summary || {};
  const templatePackage = item.template_package || {};
  const templateId = item.template_id || templatePackage.template_id || "";
  const resultId = `template-release-result-${safeDomId(templateId)}`;
  const objects = Array.isArray(summary.objects) ? summary.objects : [];
  const defaultObjects = Array.isArray(delivery.default_objects) ? delivery.default_objects : objects;
  const scenarioIds = Array.isArray(delivery.scenario_ids) ? delivery.scenario_ids : Array.isArray(summary.scenario_ids) ? summary.scenario_ids : [];
  const skillPackages = Array.isArray(delivery.skill_packages) ? delivery.skill_packages : [];
  const deviceSources = Array.isArray(delivery.device_sources) ? delivery.device_sources : [];
  const acceptanceTests = Array.isArray(delivery.acceptance_tests) ? delivery.acceptance_tests : [];
  const applicability = item.applicability_scope || delivery.applicability_scope || {};
  const prerequisites = Array.isArray(item.customer_prerequisites) ? item.customer_prerequisites : [];
  const criteria = Array.isArray(item.scenario_acceptance_criteria) ? item.scenario_acceptance_criteria : [];
  const outOfScope = Array.isArray(item.out_of_scope) ? item.out_of_scope : [];
  return `
    <div class="template-market-card">
      <div class="template-market-head">
        <div>
          <strong>${esc(item.display_name || item.template_id || "模板")}</strong>
          <p>${esc(delivery.customer_fit || item.customer_claim || "可复用的客户项目起点。")}</p>
        </div>
        <div class="capability-badges">
          ${badge(item.status || "unknown", item.status === "passed" ? "ok" : "err")}
          ${badge(delivery.acceptance_status || "unknown", acceptanceGateClass(delivery.acceptance_status))}
          ${badge(templatePackage.product_status || "template", acceptanceGateClass(templatePackage.product_status))}
        </div>
      </div>
      <div class="template-market-meta">
        <span>${esc(item.tenant_id || "default")}</span>
        <span>${esc(item.delivery_namespace || "default")}</span>
        <span>${esc(item.industry || "-")}</span>
        <span>${esc(templatePackage.package_id || `v${item.template_version || delivery.template_version || "0.0.0"}`)}</span>
        <span>${esc(templatePackage.publish_status || item.publish_status || "draft")}</span>
        <span>${esc(templatePackage.upgrade_policy || "manual_review")}</span>
        <span>对象 ${esc(delivery.default_object_count ?? summary.object_type_count ?? objects.length)}</span>
        <span>场景 ${esc(scenarioIds.length)}</span>
        <span>能力 ${esc(skillPackages.length)}</span>
        <span>设备 ${esc(deviceSources.join(", ") || "-")}</span>
      </div>
      ${renderTemplatePackageReadiness(templatePackage)}
      ${renderTemplateApplicabilityScope(applicability)}
      ${renderTemplateObjectPreview(defaultObjects)}
      <div class="template-capability-strip">
        <div><b>场景范围</b><span>${esc(scenarioIds.slice(0, 5).join(", ") || "未声明")}</span></div>
        <div><b>能力包</b><span>${esc(skillPackages.slice(0, 5).join(", ") || "未绑定")}</span></div>
        <div><b>验收项</b><span>${esc(acceptanceTests.length ? `已关联 ${acceptanceTests.length} 项` : "未关联")}</span></div>
      </div>
      ${renderTemplateCustomerPrerequisites(prerequisites)}
      ${renderTemplateScenarioAcceptanceCriteria(criteria)}
      ${renderTemplateOutOfScope(outOfScope)}
      ${renderTemplateDeliveryChecklist(item.delivery_checklist)}
      <div class="panel-actions">
        <button class="ghost-button" data-template-select="${esc(templateId)}">使用此模板</button>
        <button class="ghost-button" data-template-history="${esc(templateId)}">发布历史</button>
        <button class="ghost-button" data-template-release-requests="${esc(templateId)}">发布申请</button>
        <button class="ghost-button" data-template-release="${esc(templateId)}" data-release-status="pilot">标记试点</button>
        <button class="primary-button" data-template-release-request="${esc(templateId)}" data-release-status="published">申请发布</button>
        <button class="danger-button" data-template-release="${esc(templateId)}" data-release-status="deprecated">废弃</button>
      </div>
      <div id="${esc(resultId)}" class="small-note template-release-result"></div>
      <p class="small-note">${esc(item.next_step || delivery.delivery_boundary || "先创建客户项目，再绑定现场设备和验收凭证。")}</p>
    </div>
  `;
}

function renderTemplateApplicabilityScope(scope = {}) {
  if (!scope || !scope.scope_type) return "";
  const industries = Array.isArray(scope.industries) ? scope.industries : [];
  const siteTypes = Array.isArray(scope.site_types) ? scope.site_types : [];
  const objectTypes = Array.isArray(scope.managed_object_types) ? scope.managed_object_types : [];
  return `
    <div class="template-capability-strip">
      <div><b>适用行业</b><span>${esc(industries.join(", ") || "未声明")}</span></div>
      <div><b>现场类型</b><span>${esc(siteTypes.join(", ") || "未声明")}</span></div>
      <div><b>对象类型</b><span>${esc(objectTypes.join(", ") || "未声明")}</span></div>
    </div>
  `;
}

function renderTemplatePackageReadiness(payload = {}) {
  if (!payload || !payload.product_status) return "";
  const blockers = Array.isArray(payload.blockers) ? payload.blockers : [];
  const manualChecks = Array.isArray(payload.manual_checks) ? payload.manual_checks : [];
  const deps = payload.dependencies || {};
  return `
    <div class="project-reuse-assessment ${acceptanceGateClass(payload.product_status)}">
      <strong>模板交付包 ${badge(payload.product_status, acceptanceGateClass(payload.product_status))}</strong>
      <p>${esc(payload.customer_status || payload.next_step || "")}</p>
      <div class="row-meta">
        <span>版本 ${esc(payload.version || "0.0.0")}</span>
        <span>阶段 ${esc(payload.release_channel || "-")}</span>
        <span>objects ${esc(deps.managed_object_count ?? 0)}</span>
        <span>acceptance ${esc(deps.acceptance_test_count ?? 0)}</span>
      </div>
      ${(blockers.length || manualChecks.length) ? `<div class="skill-validation">${[...blockers, ...manualChecks].slice(0, 5).map((item) => `<span class="${blockers.includes(item) ? "err" : "warn"}">${esc(item)}</span>`).join("")}</div>` : ""}
    </div>
  `;
}

function renderTemplateObjectPreview(objects = []) {
  const visible = Array.isArray(objects) ? objects.slice(0, 6) : [];
  if (!visible.length) return `<div class="mini-list-empty">暂未声明默认现场对象。</div>`;
  return `
    <div class="template-object-preview">
      ${visible.map((item) => `
        <span>
          <b>${esc(item.display_name || item.object_id || "object")}</b>
          <small>${esc(item.category || "uncategorized")}</small>
        </span>
      `).join("")}
    </div>
  `;
}

function renderTemplateCustomerPrerequisites(items = []) {
  const rows = Array.isArray(items) ? items.slice(0, 4) : [];
  if (!rows.length) return "";
  return `
    <div class="template-delivery-checklist">
      ${rows.map((item) => `
        <div class="template-check-item ${acceptanceGateClass(item.status)}">
          <b>${esc(item.label || item.prerequisite_id || "prerequisite")} ${badge(item.status || "manual_check", acceptanceGateClass(item.status))}</b>
          <span>${esc(item.owner || "")}</span>
          <small>${esc(item.next_step || "")}</small>
        </div>
      `).join("")}
    </div>
  `;
}

function renderTemplateScenarioAcceptanceCriteria(items = []) {
  const rows = Array.isArray(items) ? items.slice(0, 4) : [];
  if (!rows.length) return "";
  return `
    <div class="template-capability-strip">
      ${rows.map((item) => `
        <div>
          <b>${esc(item.scenario_id || "scenario")}</b>
          <span>${esc((item.required_evidence || []).slice(0, 4).join(", ") || "evidence not declared")}</span>
        </div>
      `).join("")}
    </div>
  `;
}

function renderTemplateOutOfScope(items = []) {
  const rows = Array.isArray(items) ? items.slice(0, 3) : [];
  if (!rows.length) return "";
  return `
    <div class="small-note">
      <strong>Delivery boundary:</strong> ${rows.map((item) => esc(item)).join(" | ")}
    </div>
  `;
}

function renderTemplateDeliveryChecklist(items = []) {
  const rows = Array.isArray(items) ? items : [];
  if (!rows.length) return `<div class="mini-list-empty">No delivery checklist available.</div>`;
  return `
    <div class="template-delivery-checklist">
      ${rows.map((item) => `
        <div class="template-check-item ${acceptanceGateClass(item.status)}">
          <b>${esc(item.label || item.step_id || "step")} ${badge(item.status || "unknown", acceptanceGateClass(item.status))}</b>
          <span>${esc(item.evidence || "")}</span>
          <small>${esc(item.next_step || "")}</small>
        </div>
      `).join("")}
    </div>
  `;
}

function renderSiteProfileCatalog(payload = {}) {
  const sites = Array.isArray(payload.sites) ? payload.sites : [];
  const summary = payload.summary || {};
  return `
    <section id="project-section-sites" class="card">
      <div class="section-title-row">
        <div>
          <h2>多现场交付</h2>
          <p>每个工厂、园区或区域都应有独立的现场档案：地图、问询点、响应组、设备登记、阈值和凭证就绪状态。</p>
        </div>
        ${badge(`${summary.configured_count ?? 0}/${summary.site_count ?? 0} configured`, Number(summary.blocked_count || 0) ? "warn" : "ok")}
      </div>
      <div class="grid four">
        <div class="metric"><b>${esc(summary.site_count ?? 0)}</b><span>现场</span></div>
        <div class="metric"><b>${esc(summary.configured_count ?? 0)}</b><span>有效档案</span></div>
        <div class="metric ${Number(summary.production_ready_count || 0) ? "ok" : "warn"}"><b>${esc(summary.production_ready_count ?? 0)}</b><span>可进入生产准入</span></div>
        <div class="metric ${Number(summary.env_missing_count || 0) ? "warn" : "ok"}"><b>${esc(summary.env_missing_count ?? 0)}</b><span>缺少环境配置</span></div>
      </div>
      <p class="muted-line">下一步：${esc(payload.next_step || "复制到更多现场前，先创建并验证现场档案。")}</p>
      <div class="capability-list">
        ${sites.slice(0, 12).map(renderSiteProfileCatalogItem).join("") || `<div class="mini-list-empty">deploy/site-profiles 下还没有现场档案。</div>`}
      </div>
    </section>
  `;
}

function renderCustomerProjectCatalog(payload = {}, resourcePayload = {}, directoryPayload = {}) {
  const projects = Array.isArray(payload.projects) ? payload.projects : [];
  const customers = Array.isArray(payload.customers) ? payload.customers : [];
  const summary = payload.summary || {};
  const acceptanceGate = payload.delivery_acceptance_gate || {};
  const resources = Array.isArray(resourcePayload.resources) ? resourcePayload.resources : currentDeliveryResourceItems;
  currentCustomerProjectItems = projects;
  return `
    <section id="project-section-projects" class="card">
      <div class="section-title-row">
        <div>
          <h2>客户项目目录</h2>
          <p>我们是方案商，同一套产品要服务不同客户、现场对象、场景、设备和响应团队；这里用于确认每个客户项目的交付边界。</p>
        </div>
        ${badge(`${summary.customer_count ?? customers.length} 个客户`, Number(summary.blocked_count || 0) ? "warn" : "ok")}
      </div>
      <div class="grid four">
        <div class="metric"><b>${esc(summary.customer_count ?? customers.length)}</b><span>客户</span></div>
        <div class="metric"><b>${esc(summary.project_count ?? projects.length)}</b><span>项目</span></div>
        <div class="metric"><b>${esc(summary.tenant_count ?? 0)}</b><span>客户空间</span></div>
        <div class="metric"><b>${esc(summary.delivery_namespace_count ?? 0)}</b><span>交付空间</span></div>
        <div class="metric"><b>${esc(summary.industry_count ?? 0)}</b><span>行业</span></div>
        <div class="metric"><b>${esc(summary.managed_object_type_count ?? 0)}</b><span>现场对象类型</span></div>
      </div>
      <p class="muted-line">对象类别：${esc((summary.managed_object_categories || []).join(", ") || "暂未配置")}</p>
      ${renderCustomerProjectFilterControls(payload)}
      ${renderDeliveryAcceptanceGate(acceptanceGate)}
      <div class="capability-list">
        ${projects.slice(0, 12).map(renderCustomerProjectCatalogItem).join("") || `<div class="mini-list-empty">No customer project profiles found.</div>`}
      </div>
      <div class="project-lifecycle-grid">
        <div class="field-form compact-form">
          <h3>项目生命周期</h3>
          <select id="project-lifecycle-id">
            ${renderProjectOptions(projects)}
          </select>
          <input id="project-rollback-revision" placeholder="revision_id for rollback">
          <input id="project-rollback-reason" placeholder="rollback reason, visible in delivery history">
          <div class="panel-actions">
            <button class="ghost-button" data-project-lifecycle-history>查看修订记录</button>
            <button class="ghost-button" data-project-lifecycle-rollback-dry>回滚预检</button>
            <button class="danger-button" data-project-lifecycle-rollback>执行回滚</button>
            <button class="ghost-button" data-project-lifecycle-export>导出交付包</button>
            <button class="ghost-button" data-project-lifecycle-proposal>客户方案包</button>
            <button class="ghost-button" data-project-execution-bindings>执行接入计划</button>
            <button class="ghost-button" data-project-lifecycle-onsite-load>查看现场证据</button>
            <button class="danger-button" data-project-lifecycle-archive>归档项目</button>
          </div>
          <div class="onsite-evidence-editor">
            <h3>现场验收证据</h3>
            <select id="project-onsite-evidence-type">
              <option value="device_ingest">设备上报</option>
              <option value="voice_playback">语音播报</option>
              <option value="notification_delivery">外部通知</option>
              <option value="runtime_roundtrip">任务回调</option>
              <option value="customer_review">客户复核</option>
            </select>
            <select id="project-onsite-evidence-status">
              <option value="passed">通过</option>
              <option value="manual_check">待复核</option>
              <option value="failed">失败</option>
            </select>
            <input id="project-onsite-evidence-path" placeholder="证据文件路径，例如 artifacts/field_operations/smoke/photo.json">
            <input id="project-onsite-evidence-summary" placeholder="现场证据说明，例如 西门服务点真实播报已录屏">
            <div class="panel-actions">
              <button class="primary-button" data-project-lifecycle-onsite-evidence>登记现场证据</button>
            </div>
          </div>
          <div class="onsite-evidence-editor">
            <h3>验收闭环</h3>
            <select id="project-acceptance-review-decision">
              <option value="accepted">同意提交验收</option>
              <option value="needs_fix">需要整改</option>
              <option value="rejected">拒绝验收</option>
              <option value="waived">豁免复核</option>
            </select>
            <input id="project-acceptance-review-reason" placeholder="复核意见，例如 现场证据齐备，同意提交客户签收">
            <input id="project-acceptance-evidence-refs" placeholder="证据引用，逗号分隔，例如 onsite:receipt-001, report:acceptance">
            <div class="acceptance-evidence-picker">
              <select id="project-acceptance-evidence-picker">
                <option value="">先点击“查看现场证据”，再选择要引用的证据</option>
              </select>
              <button class="ghost-button" data-acceptance-evidence-add type="button">加入引用</button>
            </div>
            <label class="checkbox-row">
              <input id="project-acceptance-risk-ack" type="checkbox">
              <span>我确认这只是客户项目验收结论，不代表无人值守生产上线</span>
            </label>
            <div class="panel-actions">
              <button class="ghost-button" data-project-lifecycle-closure>查看验收闭环</button>
              <button class="primary-button" data-project-lifecycle-review>提交复核结论</button>
            </div>
          </div>
          <div class="onsite-evidence-editor">
            <h3>客户签收</h3>
            <select id="project-customer-signoff-decision">
              <option value="accepted">客户同意验收</option>
              <option value="needs_fix">客户要求整改</option>
              <option value="rejected">客户拒收</option>
            </select>
            <input id="project-customer-signatory-name" placeholder="客户签收人姓名，例如 王经理">
            <input id="project-customer-signatory-role" placeholder="签收人职务，例如 园区运营负责人">
            <input id="project-customer-signoff-organization" placeholder="客户组织，例如 成都梵木创艺园">
            <input id="project-customer-signoff-reason" placeholder="签收意见，例如 首期试点验收通过">
            <input id="project-customer-signoff-evidence-refs" placeholder="签收证据引用，逗号分隔，例如 dossier:latest, onsite:receipt-001">
            <input id="project-customer-signoff-credential-ref" placeholder="签收凭证引用，例如 signed-dossier:20260515-001">
            <input id="project-customer-signoff-credential-sha256" placeholder="签收凭证 SHA-256，客户同意验收时必填">
            <label class="checkbox-row">
              <input id="project-customer-signoff-risk-ack" type="checkbox">
              <span>客户已确认：本次签收代表试点/项目验收，不代表无人值守生产上线</span>
            </label>
            <div class="panel-actions">
              <button class="ghost-button" data-project-customer-signoff-load>查看客户签收</button>
              <button class="primary-button" data-project-customer-signoff-submit>登记客户签收</button>
            </div>
          </div>
          <div id="project-lifecycle-result" class="small-note"></div>
        </div>
        <div class="field-form compact-form">
          <h3>对象下线</h3>
          <select id="object-delete-pair">
            ${renderProjectObjectPairOptions(projects)}
          </select>
          <input id="object-delete-reason" placeholder="下线原因，例如：客户现场已移除该对象">
          <div id="object-delete-impact" class="object-offline-impact">选择对象后，先查看下线影响再移除。</div>
          <div class="panel-actions">
            <button class="danger-button" data-object-delete>下线对象</button>
          </div>
          <div id="object-delete-result" class="small-note"></div>
        </div>
      </div>
      <div class="field-form compact-form">
        <h3>项目基础信息</h3>
        <select id="project-edit-id">
          ${renderProjectOptions(projects)}
        </select>
        <div id="project-edit-scope" class="small-note">请先加载项目，再编辑客户可见信息。</div>
        <input id="project-edit-customer-name" placeholder="客户名称">
        <input id="project-edit-industry" placeholder="industry">
        <input id="project-edit-project-name" placeholder="项目名称">
        <input id="project-edit-site-name" placeholder="现场名称">
        <input id="project-edit-object-scope-note" placeholder="对象范围说明">
        <div class="panel-actions">
          <button class="ghost-button" data-project-edit-load>加载项目</button>
          <button class="primary-button" data-project-edit-save>保存信息</button>
        </div>
        <div id="project-edit-result" class="small-note"></div>
      </div>
      <div id="project-section-objects">
        ${renderManagedObjectDirectory(projects, directoryPayload)}
      </div>
      ${renderObjectChangeLog(projects)}
      <div class="field-form compact-form">
        <h3>对象快速维护</h3>
        <div class="managed-object-editor" data-managed-object-editor>
          <div class="object-editor-section">
            <strong>基础对象</strong>
            <input id="object-project-id" placeholder="project_id or site_id">
            <input id="object-id" placeholder="object_id, for example: line_1_motor">
            <input id="object-display-name" placeholder="display_name">
            <input id="object-category" placeholder="category">
          </div>
          <div class="object-editor-section">
            <strong>识别范围</strong>
            <input id="object-labels" placeholder="object_labels, comma separated">
            <input id="object-scenarios" placeholder="scenario_ids, comma separated">
            <input id="object-zone-types" placeholder="zone_types, comma separated">
            <input id="object-device-sources" placeholder="device_sources, comma separated">
            <input id="object-responder-group" placeholder="responder_group, for example: security">
            <input id="object-evidence-required" placeholder="evidence_required, for example: photo, location">
          </div>
          <div class="object-editor-section">
            <strong>客户范围保护</strong>
            <input id="object-tenant-ids" placeholder="客户空间，选填">
            <input id="object-delivery-namespaces" placeholder="交付空间，选填">
            <input id="object-customer-ids" placeholder="客户 ID，选填">
            <input id="object-project-ids" placeholder="项目 ID，选填">
            <input id="object-site-ids" placeholder="现场 ID，选填">
            <p class="small-note">当模板或能力包复用到多个客户时，填写范围保护，避免别的客户现场事件误绑定到本项目。</p>
          </div>
          <div class="object-editor-section">
            <strong>能力配置</strong>
            <input id="object-vision-models" placeholder="识别能力，逗号分隔">
            <input id="object-sensor-protocols" placeholder="设备接入方式，逗号分隔">
            <input id="object-skill-packages" placeholder="业务能力，逗号分隔">
          </div>
          <div class="object-editor-section">
            <strong>验收证据</strong>
            <input id="object-acceptance-tests" placeholder="验收项，逗号分隔">
          </div>
        </div>
        ${renderObjectResourceBindingPicker(resources)}
        <div class="panel-actions">
          <button class="primary-button" data-object-upsert>保存对象</button>
        </div>
        <div id="object-upsert-result" class="small-note"></div>
      </div>
    </section>
  `;
}

function projectIdentifier(project = {}) {
  return project.project_id || project.site_id || project.customer_id || "";
}

function renderProjectScopeLabel(scope = {}) {
  const tenant = scope.tenant_id || "default";
  const namespace = scope.delivery_namespace || "default";
  const customer = scope.customer_id || "-";
  const project = scope.project_id || scope.site_id || "-";
  return `${tenant} / ${namespace} / ${customer} / ${project}`;
}

function renderProjectScopeCustomerLine(scope = {}) {
  const items = [
    ["客户空间", scope.tenant_id || "default"],
    ["交付空间", scope.delivery_namespace || "default"],
    ["客户", scope.customer_id || "-"],
    ["项目", scope.project_id || scope.site_id || "-"],
    ["现场", scope.site_id || "-"],
  ];
  return items.map(([label, value]) => `${label}: ${value}`).join(" / ");
}

function renderOperatorProjectScopeLine(scope = {}) {
  if (!scope || !Object.keys(scope).length) return "当前账号可查看全部授权项目";
  const items = [
    ["客户空间", scope.tenant_ids],
    ["交付空间", scope.delivery_namespaces],
    ["客户", scope.customer_ids],
    ["项目", scope.project_ids],
    ["现场", scope.site_ids],
  ].filter(([, values]) => Array.isArray(values) && values.length);
  if (!items.length) return "当前账号可查看全部授权项目";
  return items.map(([label, values]) => `${label}: ${values.join(", ")}`).join(" / ");
}

function renderProjectPackageScopeEvidence(payload = {}, verification = {}, diff = {}) {
  const packageScope = payload.package_scope
    || verification.package_scope
    || diff.incoming_delivery_scope
    || {};
  const currentScope = diff.current_delivery_scope || {};
  const operatorScope = payload.operator_project_scope || {};
  return `
    <div class="project-scope-evidence">
      <div>
        <strong>交付包归属</strong>
        <span>${esc(renderProjectScopeCustomerLine(packageScope))}</span>
      </div>
      ${currentScope.customer_id ? `
        <div>
          <strong>当前项目归属</strong>
          <span>${esc(renderProjectScopeCustomerLine(currentScope))}</span>
        </div>
      ` : ""}
      <div>
        <strong>当前账号可操作范围</strong>
        <span>${esc(renderOperatorProjectScopeLine(operatorScope))}</span>
      </div>
    </div>
  `;
}

function deliveryStatusLabel(value = "") {
  const labels = {
    ready: "可继续",
    manual_check: "需人工复核",
    blocked: "已阻断",
    warning: "需注意",
    missing: "缺失",
    "-": "-",
  };
  return labels[value] || value || "-";
}

function projectChangeTypeLabel(value = "") {
  const labels = {
    create: "新建项目",
    replace: "覆盖项目",
    unchanged: "无变化",
    blocked: "已阻断",
    "-": "-",
  };
  return labels[value] || value || "-";
}

function renderProjectScopeBadge(scope = {}) {
  return badge(`范围 ${renderProjectScopeLabel(scope)}`, "warn");
}

function renderProjectCollisionCandidates(candidates = []) {
  if (!Array.isArray(candidates) || !candidates.length) return "";
  return `
    <div class="project-collision-warning">
      <strong>交付包冲突项 ${badge("导入前复核", "warn")}</strong>
      <p>同名客户项目已存在于其他交付空间。本次导入不会覆盖这些项目，但交付负责人需要确认是否为同一客户的不同环境。</p>
      <div class="mini-list">
        ${candidates.slice(0, 5).map((item) => {
          const scope = item.delivery_scope || {};
          return `
            <div class="mini-row">
              <b>${esc(renderProjectScopeLabel(scope))}</b>
              <span>${esc(item.profile_path || "-")}</span>
            </div>
          `;
        }).join("")}
      </div>
    </div>
  `;
}

function renderProjectOptions(projects = []) {
  if (!projects.length) return `<option value="">No customer project</option>`;
  return projects.map((project) => {
    const id = projectIdentifier(project);
    const label = `${project.customer_name || project.customer_id || "客户"} / ${project.project_name || id} / ${project.delivery_namespace || "default"}`;
    return `<option value="${esc(id)}">${esc(label)}</option>`;
  }).join("");
}

function renderProjectObjectPairOptions(projects = []) {
  const options = projects.flatMap((project) => {
    const projectId = projectIdentifier(project);
    const objects = Array.isArray(project.managed_objects) ? project.managed_objects : [];
    return objects.map((item) => {
      const objectId = item.object_id || "";
      const label = `${project.project_name || projectId} / ${item.display_name || objectId}`;
      return `<option value="${esc(`${projectId}::${objectId}`)}">${esc(label)}</option>`;
    });
  });
  return options.join("") || `<option value="">暂无现场对象</option>`;
}

function objectBindingList(item = {}, key) {
  const bindings = item.bindings && typeof item.bindings === "object" ? item.bindings : {};
  const values = Array.isArray(bindings[key]) ? bindings[key] : [];
  return values.map((value) => String(value || "").trim()).filter(Boolean);
}

function managedObjectRows(projects = []) {
  return projects.flatMap((project) => {
    const projectId = projectIdentifier(project);
    const objects = Array.isArray(project.managed_objects) ? project.managed_objects : [];
    return objects.map((item) => ({ project, projectId, item }));
  });
}

function managedObjectDirectoryKey(projectId = "", objectId = "") {
  return `${String(projectId || "").trim()}::${String(objectId || "").trim()}`;
}

function managedObjectDeliveryStatus(item = {}) {
  const resourceStatus = item.resource_binding_status || {};
  const acceptanceStatus = item.acceptance_status || {};
  const resource = String(resourceStatus.overall_status || "manual_check");
  const acceptance = String(acceptanceStatus.status || "manual_check");
  if ([resource, acceptance].some((value) => ["blocked", "failed", "file_missing", "outside_project"].includes(value))) return "blocked";
  if (resource === "ready" && acceptance === "ready") return "ready";
  return "manual_check";
}

function managedObjectDirectorySummary(rows = []) {
  const summary = {
    object_count: rows.length,
    ready_count: 0,
    manual_check_count: 0,
    blocked_count: 0,
    customer_visible_count: 0,
    acceptance_test_count: 0,
    scoped_object_count: 0,
  };
  rows.forEach(({ item }) => {
    const status = managedObjectDeliveryStatus(item);
    if (status === "ready") summary.ready_count += 1;
    else if (status === "blocked") summary.blocked_count += 1;
    else summary.manual_check_count += 1;
    if (item.customer_visible !== false) summary.customer_visible_count += 1;
    summary.acceptance_test_count += objectBindingList(item, "acceptance_tests").length;
    if (["tenant_ids", "delivery_namespaces", "customer_ids", "project_ids", "site_ids"].some((key) => Array.isArray(item[key]) && item[key].length)) {
      summary.scoped_object_count += 1;
    }
  });
  summary.overall_status = summary.blocked_count ? "blocked" : summary.manual_check_count ? "manual_check" : rows.length ? "ready" : "manual_check";
  return summary;
}

function renderManagedObjectDirectory(projects = [], directoryPayload = {}) {
  const rows = managedObjectRows(projects);
  const directoryObjects = Array.isArray(directoryPayload.objects) ? directoryPayload.objects : [];
  const directoryByKey = new Map(directoryObjects.map((item) => [
    managedObjectDirectoryKey(item.project_id, item.object_id),
    item,
  ]));
  const displayRows = rows.map((row) => {
    const objectId = row.item?.object_id || "";
    const directoryItem = directoryByKey.get(managedObjectDirectoryKey(row.projectId, objectId)) || {};
    return {
      ...row,
      item: {
        ...row.item,
        action_plan: directoryItem.action_plan || row.item.action_plan || [],
        action_count: directoryItem.action_count ?? row.item.action_count ?? 0,
        blocked_action_count: directoryItem.blocked_action_count ?? row.item.blocked_action_count ?? 0,
        manual_check_action_count: directoryItem.manual_check_action_count ?? row.item.manual_check_action_count ?? 0,
        delivery_status: directoryItem.delivery_status || row.item.delivery_status,
      },
    };
  });
  const directorySummary = directoryPayload && typeof directoryPayload.summary === "object" ? directoryPayload.summary : {};
  const summary = Object.keys(directorySummary).length ? directorySummary : managedObjectDirectorySummary(rows);
  const apiObjectCount = Array.isArray(directoryPayload.objects) ? directoryPayload.objects.length : rows.length;
  return `
    <div class="managed-object-directory" data-managed-object-directory>
      <div class="section-title-row">
        <div>
          <h3>对象目录</h3>
          <p>客户可见的现场对象，以及对应的识别能力、设备接入方式、业务能力和验收项。当前可见 ${esc(apiObjectCount)} 个对象。</p>
        </div>
        ${badge(`${rows.length} 个对象`, rows.length ? "ok" : "warn")}
      </div>
      ${renderManagedObjectDirectorySummary(summary)}
      <div class="panel-actions compact">
        <button class="ghost-button" data-managed-object-export="all">导出对象目录</button>
        <button class="ghost-button" data-managed-object-export="deliverable">导出可交付对象清单</button>
      </div>
      <div class="managed-object-grid">
        ${displayRows.slice(0, 24).map(renderManagedObjectCard).join("") || `<div class="mini-list-empty">当前客户项目下还没有对象。</div>`}
      </div>
    </div>
  `;
}

function renderManagedObjectDirectorySummary(summary = {}) {
  return `
    <div class="managed-object-summary" data-managed-object-summary>
      <div class="metric ${acceptanceGateClass(summary.overall_status)}"><b>${esc(summary.object_count ?? 0)}</b><span>\u5ba2\u6237\u5bf9\u8c61</span></div>
      <div class="metric ok"><b>${esc(summary.ready_count ?? 0)}</b><span>\u53ef\u4ea4\u4ed8</span></div>
      <div class="metric warn"><b>${esc(summary.manual_check_count ?? 0)}</b><span>\u9700\u590d\u6838</span></div>
      <div class="metric err"><b>${esc(summary.blocked_count ?? 0)}</b><span>\u963b\u65ad</span></div>
      <div class="metric"><b>${esc(summary.acceptance_test_count ?? 0)}</b><span>\u9a8c\u6536\u7528\u4f8b</span></div>
      <div class="metric"><b>${esc(summary.scoped_object_count ?? 0)}</b><span>\u79df\u6237\u7ed1\u5b9a</span></div>
      <div class="metric ${Number(summary.action_count || 0) ? "warn" : "ok"}"><b>${esc(summary.action_count ?? 0)}</b><span>\u4fee\u590d\u52a8\u4f5c</span></div>
    </div>
  `;
}

function renderManagedObjectCard(row = {}) {
  const project = row.project || {};
  const item = row.item || {};
  const status = item.acceptance_status || {};
  const projectId = row.projectId || projectIdentifier(project);
  const objectId = item.object_id || "";
  const resourceStatus = item.resource_binding_status || {};
  const bindingRows = [
    ["识别能力", objectBindingList(item, "vision_models")],
    ["设备接入", objectBindingList(item, "sensor_protocols")],
    ["业务能力", objectBindingList(item, "skill_packages")],
    ["验收项", objectBindingList(item, "acceptance_tests")],
  ];
  return `
    <div class="managed-object-card">
      <div class="managed-object-head">
        <div>
          <strong>${esc(item.display_name || objectId || "现场对象")}</strong>
          <span>${esc(project.customer_name || project.customer_id || "客户")} / ${esc(project.project_name || projectId || "-")}</span>
        </div>
        ${badge(status.status || "unknown", acceptanceGateClass(status.status))}
      </div>
      <div class="row-meta">
        <span>对象 ${esc(objectId || "-")}</span>
        <span>类别 ${esc(item.category || "-")}</span>
        <span>响应组 ${esc(item.responder_group || "-")}</span>
        <span>资源 ${esc(resourceStatus.overall_status || "unknown")}</span>
      </div>
      <div class="managed-object-bindings">
        ${bindingRows.map(([label, values]) => `
          <div>
            <b>${esc(label)}</b>
            <span>${esc(values.join(", ") || "未配置")}</span>
          </div>
        `).join("")}
      </div>
      <div class="row-meta">
        ${badge(`资源 ${resourceStatus.overall_status || "unknown"}`, acceptanceGateClass(resourceStatus.overall_status))}
        ${badge(`交付 ${managedObjectDeliveryStatus(item)}`, acceptanceGateClass(managedObjectDeliveryStatus(item)))}
        <span>已关联 ${esc(resourceStatus.linked_count ?? 0)}</span>
        <span>需复核 ${esc(resourceStatus.manual_check_count ?? 0)}</span>
        <span>阻断 ${esc(resourceStatus.blocked_count ?? 0)}</span>
      </div>
      ${renderManagedObjectCheckDetails(item)}
      ${renderManagedObjectActionPlan(item)}
      <div class="panel-actions compact">
        <button class="ghost-button" data-object-load="${esc(`${projectId}::${objectId}`)}">加载到编辑区</button>
      </div>
    </div>
  `;
}

function renderManagedObjectActionPlan(item = {}) {
  const actions = Array.isArray(item.action_plan) ? item.action_plan : [];
  if (!actions.length) {
    return `<div class="managed-object-actions ready"><b>\u4fee\u590d\u6e05\u5355</b><span>\u5f53\u524d\u5bf9\u8c61\u6ca1\u6709\u7ed1\u5b9a\u963b\u65ad\u9879\u3002</span></div>`;
  }
  return `
    <div class="managed-object-actions" data-managed-object-action-plan>
      <b>\u4fee\u590d\u6e05\u5355 ${badge(`${actions.length} 项`, "warn")}</b>
      ${actions.slice(0, 4).map((action) => `
        <span class="${acceptanceGateClass(action.severity || action.status)}">
          ${esc(action.action_label || action.action || "待处理配置")} / ${esc(action.owner_label || action.owner || "交付负责人")}
          <small>${esc(action.reason_label || "配置需复核")}：${esc(action.customer_next_step || action.next_step || action.message || "交付前需要复核该能力配置。")}</small>
        </span>
      `).join("")}
    </div>
  `;
}

function renderManagedObjectCheckDetails(item = {}) {
  const resourceStatus = item.resource_binding_status || {};
  const acceptanceStatus = item.acceptance_status || {};
  const resourceChecks = Array.isArray(resourceStatus.checks) ? resourceStatus.checks : [];
  const acceptanceChecks = Array.isArray(acceptanceStatus.acceptance_checks) ? acceptanceStatus.acceptance_checks : [];
  const checks = [
    ...resourceChecks.slice(0, 4).map((check) => ({
      label: `${check.resource_type || "resource"}:${check.resource_id || "-"}`,
      status: check.status || "unknown",
      message: check.message || check.action_hint || check.next_step || "",
    })),
    ...acceptanceChecks.slice(0, 4).map((check) => ({
      label: `${check.node || check.path || "acceptance"}:${check.matched || check.status || ""}`,
      status: check.status || "unknown",
      message: check.next_step || check.message || check.path || "",
    })),
  ];
  if (!checks.length) return "";
  return `
    <div class="managed-object-checks">
      ${checks.slice(0, 6).map((check) => `
        <span class="${acceptanceGateClass(check.status)}">
          <b>${esc(check.label)}</b>
          <small>${esc(check.status || "unknown")} / ${esc(check.message || "review required")}</small>
        </span>
      `).join("")}
    </div>
  `;
}

function renderObjectChangeLog(projects = []) {
  const changes = projects.flatMap((project) => {
    const projectId = projectIdentifier(project);
    const entries = Array.isArray(project.object_change_log) ? project.object_change_log : [];
    return entries.map((entry) => ({ project, projectId, entry }));
  }).sort((a, b) => Number(b.entry.timestamp || 0) - Number(a.entry.timestamp || 0));
  return `
    <div class="object-change-log" data-object-change-log>
      <div class="section-title-row">
        <div>
          <h3>对象变更记录</h3>
          <p>记录对象新增、更新、下线及原因，方便交付复盘和客户验收追溯。</p>
        </div>
        ${badge(`${changes.length} 条变更`, changes.length ? "ok" : "warn")}
      </div>
      <div class="mini-list">
        ${changes.slice(0, 10).map(({ project, projectId, entry }) => `
          <div class="mini-row">
            <b>${esc(entry.action || "updated")} / ${esc(entry.object_id || "-")}</b>
            <span>${esc(project.customer_name || project.customer_id || "客户")} / ${esc(project.project_name || projectId || "-")}</span>
            <span>${esc(entry.operator_id || "系统")} / ${esc(entry.reason || "未填写原因")}</span>
          </div>
        `).join("") || `<div class="mini-list-empty">当前客户项目还没有对象变更记录。</div>`}
      </div>
    </div>
  `;
}

function findManagedObject(projectId, objectId) {
  const project = currentCustomerProjectItems.find((item) => projectIdentifier(item) === projectId);
  if (!project) return null;
  const objects = Array.isArray(project.managed_objects) ? project.managed_objects : [];
  const object = objects.find((item) => String(item.object_id || "") === objectId);
  return object ? { project, object } : null;
}

function renderManagedObjectOfflineImpact(pair = "") {
  const [projectId, objectId] = String(pair || "").split("::");
  const found = findManagedObject(projectId, objectId);
  if (!found) return "选择对象后，先查看下线影响再移除。";
  const object = found.object || {};
  const project = found.project || {};
  const scenarioCount = Array.isArray(object.scenario_ids) ? object.scenario_ids.length : 0;
  const modelCount = objectBindingList(object, "vision_models").length;
  const skillCount = objectBindingList(object, "skill_packages").length;
  const testCount = objectBindingList(object, "acceptance_tests").length;
  return [
    `项目：${project.customer_name || project.customer_id || projectId}`,
    `对象：${object.display_name || object.object_id || objectId}`,
    `影响：${scenarioCount} 个场景、${modelCount} 项识别能力、${skillCount} 项业务能力、${testCount} 个验收项将不再归属这个客户项目。`,
    "必须填写客户可见的下线原因。",
  ].join(" ");
}

function updateManagedObjectDeleteImpact() {
  const target = document.getElementById("object-delete-impact");
  if (!target) return;
  target.textContent = renderManagedObjectOfflineImpact(
    document.getElementById("object-delete-pair")?.value || "",
  );
}

function acceptanceGateClass(status) {
  const text = String(status || "").toLowerCase();
  if (["ready", "deliverable", "ready_for_acceptance", "ready_for_customer_signoff", "accepted_by_customer", "linked", "passed", "configured", "accepted"].includes(text)) return "ok";
  if (["manual_check", "manual_check_required", "needs_fix", "node_unresolved", "read_error", "not_run"].includes(text)) return "warn";
  if (["blocked", "file_missing", "invalid_reference", "outside_project", "failed"].includes(text)) return "err";
  return "";
}

function renderProjectAcceptanceEvidence(objects = []) {
  const visible = objects.slice(0, 6);
  if (!visible.length) return "";
  return `
    <div class="project-acceptance-evidence" data-acceptance-gates>
      ${visible.map((item) => {
        const status = item.acceptance_status || {};
        const checks = Array.isArray(status.acceptance_checks) ? status.acceptance_checks : [];
        const checkBadges = checks.slice(0, 3).map((check) => (
          badge(`${check.node || check.path || "test"} ${check.status || "unknown"}`, acceptanceGateClass(check.status))
        )).join("");
        return `
          <div class="row-meta">
            <span>${esc(item.display_name || item.object_id || "现场对象")}</span>
            ${badge(status.status || "unknown", acceptanceGateClass(status.status))}
            ${checkBadges || badge("no acceptance_checks", "err")}
          </div>
        `;
      }).join("")}
    </div>
  `;
}

async function refreshProjectSurface() {
  if (currentPage().key === "projects") {
    await renderProjects();
    return;
  }
  await renderDelivery();
}

const CUSTOMER_PROJECT_FILTER_KEYS = [
  "tenant_id",
  "delivery_namespace",
  "customer_id",
  "project_id",
  "site_id",
  "industry",
  "gate_status",
  "deployment_stage",
];

function customerProjectFilterState() {
  return Object.fromEntries(CUSTOMER_PROJECT_FILTER_KEYS.map((key) => [
    key,
    localStorage.getItem(`askme.customer_project_filter.${key}`) || "",
  ]));
}

function customerProjectFilterQuery() {
  const params = new URLSearchParams({ check_env: "true" });
  const filters = customerProjectFilterState();
  Object.entries(filters).forEach(([key, value]) => {
    if (String(value || "").trim()) params.set(key, String(value || "").trim());
  });
  return params.toString();
}

function renderCustomerProjectFilterControls(payload = {}) {
  const filters = customerProjectFilterState();
  const activeFilters = payload.filters || {};
  const gateStatus = filters.gate_status || activeFilters.gate_status || "";
  const deploymentStage = filters.deployment_stage || activeFilters.deployment_stage || "";
  return `
    <div class="field-form compact-form" data-customer-project-filters>
      <h3>项目筛选</h3>
      <div class="grid four">
        <input id="project-filter-tenant" placeholder="客户空间" value="${esc(filters.tenant_id || activeFilters.tenant_id || "")}">
        <input id="project-filter-namespace" placeholder="交付空间" value="${esc(filters.delivery_namespace || activeFilters.delivery_namespace || "")}">
        <input id="project-filter-customer" placeholder="customer_id / 客户" value="${esc(filters.customer_id || activeFilters.customer_id || "")}">
        <input id="project-filter-project" placeholder="project_id / 项目" value="${esc(filters.project_id || activeFilters.project_id || "")}">
        <input id="project-filter-site" placeholder="site_id / 现场" value="${esc(filters.site_id || activeFilters.site_id || "")}">
        <input id="project-filter-industry" placeholder="industry / 行业" value="${esc(filters.industry || activeFilters.industry || "")}">
        <select id="project-filter-gate">
          <option value="">所有验收状态</option>
          <option value="ready" ${gateStatus === "ready" ? "selected" : ""}>ready</option>
          <option value="manual_check" ${gateStatus === "manual_check" ? "selected" : ""}>manual_check</option>
          <option value="blocked" ${gateStatus === "blocked" ? "selected" : ""}>blocked</option>
        </select>
        <select id="project-filter-stage">
          <option value="">所有交付阶段</option>
          <option value="production_ready" ${deploymentStage === "production_ready" ? "selected" : ""}>production_ready</option>
          <option value="pilot_ready" ${deploymentStage === "pilot_ready" ? "selected" : ""}>pilot_ready</option>
          <option value="needs_review" ${deploymentStage === "needs_review" ? "selected" : ""}>needs_review</option>
          <option value="blocked" ${deploymentStage === "blocked" ? "selected" : ""}>blocked</option>
        </select>
      </div>
      <div class="panel-actions compact">
        <button class="ghost-button" type="button" data-project-filter-apply>应用筛选</button>
        <button class="ghost-button" type="button" data-project-filter-clear>清空筛选</button>
      </div>
    </div>
  `;
}

const CUSTOMER_PROJECT_TEMPLATE_FILTER_KEYS = [
  "tenant_id",
  "delivery_namespace",
  "industry",
  "publish_status",
  "product_status",
  "template_id",
  "release_channel",
  "owner",
];

function customerProjectTemplateFilterState() {
  return Object.fromEntries(CUSTOMER_PROJECT_TEMPLATE_FILTER_KEYS.map((key) => [
    key,
    localStorage.getItem(`askme.customer_project_template_filter.${key}`) || "",
  ]));
}

function customerProjectTemplateFilterQuery() {
  const params = new URLSearchParams();
  const filters = customerProjectTemplateFilterState();
  Object.entries(filters).forEach(([key, value]) => {
    if (String(value || "").trim()) params.set(key, String(value || "").trim());
  });
  return params.toString();
}

function renderCustomerProjectTemplateFilterControls(payload = {}) {
  const filters = customerProjectTemplateFilterState();
  const activeFilters = payload.filters || {};
  const publishStatus = filters.publish_status || activeFilters.publish_status || "";
  const productStatus = filters.product_status || activeFilters.product_status || "";
  return `
    <div class="field-form compact-form" data-customer-project-template-filters>
      <h3>模板筛选</h3>
      <div class="grid four">
        <input id="template-filter-tenant" placeholder="客户空间" value="${esc(filters.tenant_id || activeFilters.tenant_id || "")}">
        <input id="template-filter-namespace" placeholder="交付空间" value="${esc(filters.delivery_namespace || activeFilters.delivery_namespace || "")}">
        <input id="template-filter-industry" placeholder="industry / 行业" value="${esc(filters.industry || activeFilters.industry || "")}">
        <input id="template-filter-template-id" placeholder="template_id / 模板" value="${esc(filters.template_id || activeFilters.template_id || "")}">
        <input id="template-filter-release-channel" placeholder="发布通道" value="${esc(filters.release_channel || activeFilters.release_channel || "")}">
        <input id="template-filter-owner" placeholder="负责人" value="${esc(filters.owner || activeFilters.owner || "")}">
        <select id="template-filter-publish-status">
          <option value="">所有发布状态</option>
          <option value="draft" ${publishStatus === "draft" ? "selected" : ""}>draft</option>
          <option value="pilot" ${publishStatus === "pilot" ? "selected" : ""}>pilot</option>
          <option value="published" ${publishStatus === "published" ? "selected" : ""}>published</option>
          <option value="deprecated" ${publishStatus === "deprecated" ? "selected" : ""}>deprecated</option>
          <option value="blocked" ${publishStatus === "blocked" ? "selected" : ""}>blocked</option>
        </select>
        <select id="template-filter-product-status">
          <option value="">所有产品状态</option>
          <option value="ready" ${productStatus === "ready" ? "selected" : ""}>ready</option>
          <option value="manual_check" ${productStatus === "manual_check" ? "selected" : ""}>manual_check</option>
          <option value="blocked" ${productStatus === "blocked" ? "selected" : ""}>blocked</option>
        </select>
      </div>
      <div class="panel-actions compact">
        <button class="ghost-button" type="button" data-template-filter-apply>应用模板筛选</button>
        <button class="ghost-button" type="button" data-template-filter-clear>清空模板筛选</button>
      </div>
    </div>
  `;
}

function renderDeliveryAcceptanceGate(gate = {}) {
  if (!gate || !gate.gate_type) return "";
  const blocked = Array.isArray(gate.blocked_projects) ? gate.blocked_projects : [];
  const manual = Array.isArray(gate.manual_check_projects) ? gate.manual_check_projects : [];
  return `
    <div class="project-delivery-workflow" data-delivery-acceptance-gate>
      <div class="project-delivery-head">
        <strong>产品验收准入 ${badge(gate.overall_status || "unknown", acceptanceGateClass(gate.overall_status))}</strong>
        <span>${esc(gate.customer_status || "客户签收前请复核项目就绪状态。")}</span>
      </div>
      <div class="row-meta">
        <span>项目 ${esc(gate.project_count ?? 0)}</span>
        <span>就绪 ${esc(gate.ready_count ?? 0)}</span>
        <span>待复核 ${esc(gate.manual_check_count ?? 0)}</span>
        <span>阻断 ${esc(gate.blocked_count ?? 0)}</span>
      </div>
      <p class="muted-line">${esc(gate.next_step || "客户签收前复核项目准入项。")}</p>
      ${blocked.length || manual.length ? `
        <div class="mini-list">
          ${blocked.concat(manual).slice(0, 6).map((item) => `
            <div class="mini-row">
              <b>${esc(item.customer_id || "-")} / ${esc(item.project_id || "-")}</b>
              <span>${esc(item.tenant_id || "default")} / ${esc(item.delivery_namespace || "default")} / ${esc(item.site_id || "-")}</span>
              <small>${esc(item.next_step || "")}</small>
            </div>
          `).join("")}
        </div>
      ` : ""}
    </div>
  `;
}

function renderCustomerProjectCatalogItem(project = {}) {
  const summary = project.managed_objects_summary || {};
  const acceptance = summary.acceptance_summary || {};
  const productGate = project.product_acceptance_gate || {};
  const objects = Array.isArray(project.managed_objects) ? project.managed_objects : [];
  const objectNames = objects.slice(0, 4).map((item) => item.display_name || item.object_id).join(", ");
  return `
    <div class="capability-item">
      <div>
        <strong>${esc(project.customer_name || "未分配客户")} / ${esc(project.project_name || project.project_id || "-")}</strong>
        <p>${esc(objectNames || "Managed objects are not configured for this customer project.")}</p>
        <div class="row-meta">
          <span>客户空间 ${esc(project.tenant_id || "default")}</span>
          <span>交付空间 ${esc(project.delivery_namespace || "default")}</span>
          <span>${esc(project.industry || "unspecified")}</span>
          <span>site ${esc(project.site_id || "-")}</span>
          <span>objects ${esc(summary.object_type_count ?? objects.length)}</span>
          <span>acceptance ${esc(acceptance.ready_object_count ?? 0)}/${esc(acceptance.object_count ?? objects.length)}</span>
          <span>scenarios ${esc((summary.scenario_ids || []).length)}</span>
          <span>${esc(project.delivery_model || "solution_project")}</span>
        </div>
        ${renderProjectAcceptanceEvidence(objects)}
        ${renderProjectProductAcceptanceGate(productGate)}
        ${renderProjectDeliveryWorkflow(project.delivery_workflow)}
      </div>
      <div class="capability-badges">
        ${renderProjectScopeBadge(project)}
        ${badge(`gate ${productGate.overall_status || "unknown"}`, acceptanceGateClass(productGate.overall_status))}
        ${badge(`acceptance ${acceptance.overall_status || "unknown"}`, acceptance.overall_status === "ready" ? "ok" : acceptance.overall_status === "manual_check" ? "warn" : "err")}
        ${badge(project.deployment_stage || "unknown", project.deployment_stage === "production_ready" ? "ok" : project.status === "passed" ? "warn" : "err")}
        ${badge(project.status || "unknown", project.status === "passed" ? "ok" : "err")}
        <button class="ghost-button" data-project-acceptance-report="${esc(project.project_id || project.site_id || "")}">验收报告</button>
        <button class="ghost-button" data-project-acceptance-dossier="${esc(project.project_id || project.site_id || "")}">验收材料</button>
        <button class="ghost-button" data-project-proposal="${esc(project.project_id || project.site_id || "")}">客户方案</button>
        <button class="ghost-button" data-project-export="${esc(project.project_id || project.site_id || "")}">导出</button>
      </div>
    </div>
  `;
}

function renderProjectProductAcceptanceGate(gate = {}) {
  const gates = Array.isArray(gate.gates) ? gate.gates : [];
  if (!gates.length) return "";
  return `
    <div class="project-delivery-workflow compact" data-project-product-acceptance-gate>
      <div class="project-delivery-head">
        <strong>产品准入项 ${badge(gate.overall_status || "unknown", acceptanceGateClass(gate.overall_status))}</strong>
        <span>${esc(gate.next_step || "复核产品验收准入项。")}</span>
      </div>
      <div class="project-delivery-steps">
        ${gates.slice(0, 7).map((item) => `
          <div class="project-delivery-step ${acceptanceGateClass(item.status)}">
            <b>${esc(item.label || item.gate_id || "准入项")} ${badge(item.status || "unknown", acceptanceGateClass(item.status))}</b>
            <span>${esc(item.evidence || "")}</span>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderProjectDeliveryWorkflow(workflow = {}) {
  const steps = Array.isArray(workflow.steps) ? workflow.steps : [];
  if (!steps.length) return "";
  return `
    <div class="project-delivery-workflow" data-project-delivery-workflow>
      <div class="project-delivery-head">
        <strong>交付流程 ${badge(workflow.overall_status || "unknown", acceptanceGateClass(workflow.overall_status))}</strong>
        <span>${esc(workflow.customer_status || "请在客户交接前复核交付步骤。")}</span>
      </div>
      <div class="project-delivery-steps">
        ${steps.map((step) => `
          <div class="project-delivery-step ${acceptanceGateClass(step.status)}">
            <b>${esc(step.label || step.step_id || "step")} ${badge(step.status || "unknown", acceptanceGateClass(step.status))}</b>
            <span>${esc(step.evidence || "")}</span>
            <small>${esc(step.next_step || "")}</small>
          </div>
        `).join("")}
      </div>
      <p class="small-note">${esc(workflow.next_step || workflow.release_claim || "")}</p>
    </div>
  `;
}

function wireCustomerProjectControls() {
  wireProjectCreateResultControls(document);
  const createButton = document.querySelector("[data-project-create]");
  if (createButton) createButton.addEventListener("click", createCustomerProjectFromTemplate);
  const templateCreateSelect = document.getElementById("project-template-id");
  if (templateCreateSelect) {
    templateCreateSelect.addEventListener("change", () => {
      localStorage.setItem("askme.project_create.template_id", templateCreateSelect.value || "");
      renderProjectTemplateCreateReadinessIntoPanel(templateCreateSelect.value || "");
    });
  }
  const projectEditLoadButton = document.querySelector("[data-project-edit-load]");
  if (projectEditLoadButton) projectEditLoadButton.addEventListener("click", loadProjectProfileForEdit);
  const projectEditSaveButton = document.querySelector("[data-project-edit-save]");
  if (projectEditSaveButton) projectEditSaveButton.addEventListener("click", saveProjectProfileMetadata);
  const objectDeletePair = document.getElementById("object-delete-pair");
  if (objectDeletePair) objectDeletePair.addEventListener("change", updateManagedObjectDeleteImpact);
  document.querySelectorAll("[data-template-select]").forEach((button) => {
    button.addEventListener("click", () => selectTemplateForCreate(button.dataset.templateSelect || ""));
  });
  document.querySelectorAll("[data-object-load]").forEach((button) => {
    button.addEventListener("click", () => loadManagedObjectIntoEditor(button.dataset.objectLoad || ""));
  });
  document.querySelectorAll("[data-managed-object-export]").forEach((button) => {
    button.addEventListener("click", () => exportManagedObjectDirectory(button.dataset.managedObjectExport || "all"));
  });
  const objectButton = document.querySelector("[data-object-upsert]");
  if (objectButton) objectButton.addEventListener("click", upsertManagedObjectFromForm);
  const objectResourceAddButton = document.querySelector("[data-object-resource-add]");
  if (objectResourceAddButton) objectResourceAddButton.addEventListener("click", addSelectedObjectResourceBinding);
  const lifecycleExportButton = document.querySelector("[data-project-lifecycle-export]");
  if (lifecycleExportButton) lifecycleExportButton.addEventListener("click", exportSelectedCustomerProject);
  const lifecycleProposalButton = document.querySelector("[data-project-lifecycle-proposal]");
  if (lifecycleProposalButton) lifecycleProposalButton.addEventListener("click", exportSelectedCustomerProjectProposalBundle);
  const executionBindingsButton = document.querySelector("[data-project-execution-bindings]");
  if (executionBindingsButton) executionBindingsButton.addEventListener("click", loadSelectedCustomerProjectExecutionBindings);
  const lifecycleOnsiteLoadButton = document.querySelector("[data-project-lifecycle-onsite-load]");
  if (lifecycleOnsiteLoadButton) lifecycleOnsiteLoadButton.addEventListener("click", loadSelectedCustomerProjectOnsiteEvidence);
  const lifecycleOnsiteEvidenceButton = document.querySelector("[data-project-lifecycle-onsite-evidence]");
  if (lifecycleOnsiteEvidenceButton) lifecycleOnsiteEvidenceButton.addEventListener("click", registerSelectedCustomerProjectOnsiteEvidence);
  const lifecycleClosureButton = document.querySelector("[data-project-lifecycle-closure]");
  if (lifecycleClosureButton) lifecycleClosureButton.addEventListener("click", loadSelectedCustomerProjectAcceptanceClosure);
  const lifecycleReviewButton = document.querySelector("[data-project-lifecycle-review]");
  if (lifecycleReviewButton) lifecycleReviewButton.addEventListener("click", registerSelectedCustomerProjectAcceptanceReview);
  const customerSignoffLoadButton = document.querySelector("[data-project-customer-signoff-load]");
  if (customerSignoffLoadButton) customerSignoffLoadButton.addEventListener("click", loadSelectedCustomerProjectCustomerSignoff);
  const customerSignoffSubmitButton = document.querySelector("[data-project-customer-signoff-submit]");
  if (customerSignoffSubmitButton) customerSignoffSubmitButton.addEventListener("click", registerSelectedCustomerProjectCustomerSignoff);
  const acceptanceEvidenceAddButton = document.querySelector("[data-acceptance-evidence-add]");
  if (acceptanceEvidenceAddButton) {
    acceptanceEvidenceAddButton.addEventListener("click", addSelectedAcceptanceEvidenceRef);
  }
  const lifecycleHistoryButton = document.querySelector("[data-project-lifecycle-history]");
  if (lifecycleHistoryButton) lifecycleHistoryButton.addEventListener("click", loadSelectedCustomerProjectHistory);
  const lifecycleRollbackDryButton = document.querySelector("[data-project-lifecycle-rollback-dry]");
  if (lifecycleRollbackDryButton) lifecycleRollbackDryButton.addEventListener("click", () => rollbackSelectedCustomerProject(true));
  const lifecycleRollbackButton = document.querySelector("[data-project-lifecycle-rollback]");
  if (lifecycleRollbackButton) lifecycleRollbackButton.addEventListener("click", () => rollbackSelectedCustomerProject(false));
  const lifecycleArchiveButton = document.querySelector("[data-project-lifecycle-archive]");
  if (lifecycleArchiveButton) lifecycleArchiveButton.addEventListener("click", archiveSelectedCustomerProject);
  const objectDeleteButton = document.querySelector("[data-object-delete]");
  if (objectDeleteButton) objectDeleteButton.addEventListener("click", deleteManagedObjectFromForm);
  document.querySelectorAll("[data-project-acceptance-report]").forEach((button) => {
    button.addEventListener("click", () => loadCustomerProjectAcceptanceReport(button.dataset.projectAcceptanceReport || ""));
  });
  document.querySelectorAll("[data-project-acceptance-dossier]").forEach((button) => {
    button.addEventListener("click", () => exportCustomerProjectAcceptanceDossier(button.dataset.projectAcceptanceDossier || ""));
  });
  document.querySelectorAll("[data-project-proposal]").forEach((button) => {
    button.addEventListener("click", () => exportCustomerProjectProposalBundle(button.dataset.projectProposal || ""));
  });
  document.querySelectorAll("[data-project-export]").forEach((button) => {
    button.addEventListener("click", () => exportCustomerProject(button.dataset.projectExport || ""));
  });
  updateManagedObjectDeleteImpact();
}

function selectTemplateForCreate(templateId) {
  const select = document.getElementById("project-template-id");
  const resultEl = document.getElementById("project-create-result");
  if (select && templateId) select.value = templateId;
  if (templateId) localStorage.setItem("askme.project_create.template_id", templateId);
  renderProjectTemplateCreateReadinessIntoPanel(templateId);
  if (resultEl) {
    resultEl.textContent = templateId
      ? `已选择模板：${templateId}。请填写客户范围后创建项目。`
      : "未选择模板。";
  }
  select?.scrollIntoView({ block: "center", behavior: "smooth" });
}

function renderProjectTemplateCreateReadinessIntoPanel(templateId = "") {
  const target = document.getElementById("project-template-create-readiness");
  if (!target) return;
  const template = currentCustomerProjectTemplateItems.find((item) => item.template_id === templateId)
    || selectedProjectTemplateForCreate(currentCustomerProjectTemplateItems);
  target.innerHTML = renderProjectTemplateCreateReadiness(template);
}

function createdCustomerProjectProfile(payload = {}) {
  return payload.profile && typeof payload.profile === "object" ? payload.profile : {};
}

function createdCustomerProjectSummary(payload = {}) {
  const profile = createdCustomerProjectProfile(payload);
  const handoff = payload.implementation_handoff || {};
  const handoffSummary = handoff.summary || {};
  const customer = profile.customer || payload.customer || {};
  const site = profile.site || payload.site || {};
  const managedObjects = profile.managed_objects && typeof profile.managed_objects === "object"
    ? Object.values(profile.managed_objects)
    : [];
  return {
    projectId: handoff.project_id || customer.project_id || profile.project_id || payload.project_id || document.getElementById("project-id")?.value || "",
    projectName: handoff.project_name || customer.project_name || profile.project_name || payload.project_name || document.getElementById("project-name")?.value || "",
    customerName: handoff.customer_name || customer.customer_name || payload.customer_name || document.getElementById("project-customer-name")?.value || "",
    siteName: handoff.site_name || site.name || payload.site_name || document.getElementById("project-site-name")?.value || "",
    profilePath: handoff.profile_path || payload.profile_path || "",
    objectCount: managedObjects.length || Number(handoffSummary.object_count || 0),
    objectNeedsBindingCount: Number(handoffSummary.object_needs_binding_count || 0),
    customerStatus: handoff.customer_status || "",
  };
}

function renderCustomerProjectCreateResult(payload = {}, ok = false, template = {}) {
  const summary = createdCustomerProjectSummary(payload);
  if (!ok) {
    return `
      <div class="project-create-result-card err" data-project-create-result-card>
        <strong>客户项目创建失败</strong>
        <p>${esc(payload.message || payload.reason || payload.error || "请检查模板、客户编号、项目编号和现场编号。")}</p>
      </div>
    `;
  }
  const handoff = payload.implementation_handoff || {};
  const nextSteps = Array.isArray(handoff.next_steps) && handoff.next_steps.length
    ? handoff.next_steps.map((step) => [step.label || step.step_id || "实施步骤", step.customer_next_step || step.status || "待处理"])
    : [
        ["加载项目", "确认客户、现场、行业模板和项目基础信息。"],
        ["补齐对象绑定", "为首批对象绑定识别模型、传感器协议、能力包和验收项。"],
        ["登记现场证据", "补齐设备、语音、通知、客户复核和现场照片等交付证据。"],
        ["生成交付包", "先验包和预览差异，再导出可复用客户项目包。"],
      ];
  return `
    <div class="project-create-result-card ok" data-project-create-result-card>
      <div class="project-delivery-head">
        <strong>客户项目已创建 ${badge("下一步实施", "ok")}</strong>
        <span>${esc(summary.customerStatus || payload.next_step || "请继续补齐对象资源绑定和现场验收证据。")}</span>
      </div>
      <div class="project-scope-evidence">
        <div><b>行业模板</b><span>${esc(template.display_name || template.template_id || "未声明")}</span></div>
        <div><b>客户/项目</b><span>${esc([summary.customerName, summary.projectName || summary.projectId].filter(Boolean).join(" / ") || "-")}</span></div>
        <div><b>现场</b><span>${esc(summary.siteName || "-")}</span></div>
        <div><b>首批对象</b><span>${esc(summary.objectCount || 0)} 个，${esc(summary.objectNeedsBindingCount || 0)} 个待补齐</span></div>
      </div>
      <div class="project-create-next-steps">
        ${nextSteps.map(([title, text]) => `
          <div>
            <b>${esc(title)}</b>
            <span>${esc(text)}</span>
          </div>
        `).join("")}
      </div>
      <div class="panel-actions">
        ${summary.projectId ? `<button class="ghost-button" type="button" data-created-project-load="${esc(summary.projectId)}">加载新项目</button>` : ""}
        ${summary.projectId ? `<button class="primary-button" type="button" data-created-object-guide="${esc(summary.projectId)}">填写对象绑定</button>` : ""}
      </div>
      <p class="small-note">保存位置：${esc(summary.profilePath || "未返回路径")}</p>
    </div>
  `;
}

async function loadCreatedCustomerProject(projectId = "") {
  const input = document.getElementById("project-edit-id");
  if (input) input.value = projectId;
  await loadProjectProfileForEdit();
}

function guideCreatedProjectObjects(projectId = "") {
  const projectInput = document.getElementById("object-project-id");
  if (projectInput) projectInput.value = projectId;
  const resultEl = document.getElementById("object-upsert-result");
  if (resultEl) {
    resultEl.textContent = `已定位到项目 ${projectId || "-"}。请选择首批现场对象，补齐识别模型、传感器协议、能力包和验收项后保存。`;
  }
  document.getElementById("project-section-objects")?.scrollIntoView({ block: "start", behavior: "smooth" });
}

function wireProjectCreateResultControls(root = document) {
  root.querySelectorAll("[data-created-project-load]").forEach((button) => {
    button.addEventListener("click", () => {
      void loadCreatedCustomerProject(button.dataset.createdProjectLoad || "");
    });
  });
  root.querySelectorAll("[data-created-object-guide]").forEach((button) => {
    button.addEventListener("click", () => guideCreatedProjectObjects(button.dataset.createdObjectGuide || ""));
  });
}

function commaList(value) {
  return String(value || "").split(",").map((item) => item.trim()).filter(Boolean);
}

function onsiteReceiptEvidenceRef(receipt = {}) {
  const receiptId = String(receipt.receipt_id || "").trim();
  if (receiptId) return `onsite:${receiptId}`;
  const path = String(receipt.path || "").trim();
  if (path) return `path:${path}`;
  const external = String(receipt.external_reference || "").trim();
  return external ? `external:${external}` : "";
}

function onsiteReceiptEvidenceLabel(receipt = {}) {
  const type = receipt.evidence_type || "evidence";
  const status = receipt.status || "manual_check";
  const label = receipt.label || receipt.summary || receipt.path || receipt.receipt_id || "现场证据";
  return `${type} / ${status} / ${label}`;
}

function setAcceptanceEvidenceOptions(payload = {}) {
  const select = document.getElementById("project-acceptance-evidence-picker");
  if (!select) return;
  const evidence = payload.onsite_acceptance_evidence || {};
  const receipts = Array.isArray(evidence.receipts)
    ? evidence.receipts
    : Array.isArray(payload.receipts)
      ? payload.receipts
      : [];
  const options = receipts
    .map((receipt) => ({ ref: onsiteReceiptEvidenceRef(receipt), label: onsiteReceiptEvidenceLabel(receipt) }))
    .filter((item) => item.ref);
  select.innerHTML = options.length
    ? options.map((item) => `<option value="${esc(item.ref)}">${esc(item.label)}</option>`).join("")
    : `<option value="">暂无可引用证据，请先登记或读取现场证据</option>`;
}

function addSelectedAcceptanceEvidenceRef() {
  const select = document.getElementById("project-acceptance-evidence-picker");
  const input = document.getElementById("project-acceptance-evidence-refs");
  const ref = select?.value || "";
  if (!input || !ref) return;
  const refs = commaList(input.value);
  if (!refs.includes(ref)) refs.push(ref);
  input.value = refs.join(", ");
}

function addSelectedObjectResourceBinding() {
  const select = document.getElementById("object-resource-picker");
  const resultEl = document.getElementById("object-resource-picker-result");
  const value = select?.value || "";
  if (!value || !value.includes("::")) {
    if (resultEl) resultEl.textContent = "请先选择一个已登记资源。";
    return;
  }
  const [resourceType, resourceId] = value.split("::");
  const inputId = objectBindingInputId(resourceType);
  const input = inputId ? document.getElementById(inputId) : null;
  if (!input || !resourceId) {
    if (resultEl) resultEl.textContent = `无法识别资源类型：${resourceType || "unknown"}`;
    return;
  }
  const values = commaList(input.value);
  if (!values.includes(resourceId)) values.push(resourceId);
  input.value = values.join(", ");
  if (resultEl) {
    resultEl.textContent = `已把 ${resourceId} 加入${deliveryResourceTypeLabel(resourceType)}配置。请保存现场对象以生效。`;
  }
}

function managedObjectExportRows(mode = "all") {
  const rows = managedObjectRows(currentCustomerProjectItems);
  return rows
    .map(({ project, projectId, item }) => ({
      tenant_id: project.tenant_id || project.customer?.tenant_id || "",
      delivery_namespace: project.delivery_namespace || project.customer?.delivery_namespace || "",
      customer_id: project.customer_id || "",
      customer_name: project.customer_name || "",
      project_id: projectId,
      project_name: project.project_name || "",
      site_id: project.site_id || "",
      object_id: item.object_id || "",
      display_name: item.display_name || "",
      category: item.category || "",
      delivery_status: managedObjectDeliveryStatus(item),
      resource_status: item.resource_binding_status?.overall_status || "",
      acceptance_status: item.acceptance_status?.status || "",
      responder_group: item.responder_group || "",
      scenario_ids: (item.scenario_ids || []).join("|"),
      device_sources: (item.device_sources || []).join("|"),
      vision_models: objectBindingList(item, "vision_models").join("|"),
      sensor_protocols: objectBindingList(item, "sensor_protocols").join("|"),
      skill_packages: objectBindingList(item, "skill_packages").join("|"),
      acceptance_tests: objectBindingList(item, "acceptance_tests").join("|"),
      scope_guards: ["tenant_ids", "delivery_namespaces", "customer_ids", "project_ids", "site_ids"]
        .map((key) => `${key}:${Array.isArray(item[key]) && item[key].length ? item[key].join("|") : "*"}`)
        .join(";"),
    }))
    .filter((row) => mode !== "deliverable" || row.delivery_status === "ready");
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function managedObjectRowsToCsv(rows = []) {
  const headers = [
    "tenant_id",
    "delivery_namespace",
    "customer_id",
    "customer_name",
    "project_id",
    "project_name",
    "site_id",
    "object_id",
    "display_name",
    "category",
    "delivery_status",
    "resource_status",
    "acceptance_status",
    "responder_group",
    "scenario_ids",
    "device_sources",
    "vision_models",
    "sensor_protocols",
    "skill_packages",
    "acceptance_tests",
    "scope_guards",
  ];
  return [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => csvCell(row[header])).join(",")),
  ].join("\n");
}

function exportManagedObjectDirectory(mode = "all") {
  const rows = managedObjectExportRows(mode);
  const suffix = mode === "deliverable" ? "deliverable-objects" : "all-objects";
  if (mode === "deliverable") {
    downloadTextFile(`askme-managed-object-directory-${suffix}.csv`, managedObjectRowsToCsv(rows), "text/csv;charset=utf-8");
    return;
  }
  downloadTextFile(
    `askme-managed-object-directory-${suffix}.json`,
    JSON.stringify({
      exported_at: new Date().toISOString(),
      object_count: rows.length,
      summary: managedObjectDirectorySummary(managedObjectRows(currentCustomerProjectItems)),
      objects: rows,
    }, null, 2),
    "application/json;charset=utf-8",
  );
}

async function createCustomerProjectFromTemplate() {
  const template = selectedProjectTemplateForCreate(currentCustomerProjectTemplateItems);
  const body = {
    template_id: document.getElementById("project-template-id")?.value || "",
    customer: {
      tenant_id: document.getElementById("project-tenant-id")?.value || "default",
      delivery_namespace: document.getElementById("project-delivery-namespace")?.value || "default",
      customer_id: document.getElementById("project-customer-id")?.value || "",
      customer_name: document.getElementById("project-customer-name")?.value || "",
      industry: document.getElementById("project-industry")?.value || "",
      project_id: document.getElementById("project-id")?.value || "",
      project_name: document.getElementById("project-name")?.value || "",
      delivery_model: "solution_project",
    },
    site: {
      site_id: document.getElementById("project-site-id")?.value || "",
      name: document.getElementById("project-site-name")?.value || "",
    },
  };
  const response = await postJson(`${ENDPOINTS.fieldCustomerProjects}/from-template`, body);
  if (response.ok) await refreshProjectSurface();
  const latestResultEl = document.getElementById("project-create-result");
  if (latestResultEl) {
    latestResultEl.innerHTML = renderCustomerProjectCreateResult(response.payload, response.ok, template);
    wireProjectCreateResultControls(latestResultEl);
  }
}

function setProjectEditInput(id, value) {
  const element = document.getElementById(id);
  if (element) element.value = value || "";
}

function setObjectEditInput(id, value) {
  const element = document.getElementById(id);
  if (!element) return;
  if (Array.isArray(value)) {
    element.value = value.join(", ");
    return;
  }
  element.value = value || "";
}

function loadManagedObjectIntoEditor(pair) {
  const [projectId, objectId] = String(pair || "").split("::");
  const found = findManagedObject(projectId, objectId);
  const resultEl = document.getElementById("object-upsert-result");
  if (!found) {
    if (resultEl) resultEl.textContent = "Managed object was not found in the visible customer project catalog.";
    return;
  }
  const object = found.object || {};
  setObjectEditInput("object-project-id", projectId);
  setObjectEditInput("object-id", object.object_id || objectId);
  setObjectEditInput("object-display-name", object.display_name);
  setObjectEditInput("object-category", object.category);
  setObjectEditInput("object-labels", object.object_labels);
  setObjectEditInput("object-scenarios", object.scenario_ids);
  setObjectEditInput("object-zone-types", object.zone_types);
  setObjectEditInput("object-device-sources", object.device_sources);
  setObjectEditInput("object-tenant-ids", object.tenant_ids);
  setObjectEditInput("object-delivery-namespaces", object.delivery_namespaces);
  setObjectEditInput("object-customer-ids", object.customer_ids);
  setObjectEditInput("object-project-ids", object.project_ids);
  setObjectEditInput("object-site-ids", object.site_ids);
  setObjectEditInput("object-responder-group", object.responder_group);
  setObjectEditInput("object-evidence-required", object.evidence_required);
  setObjectEditInput("object-vision-models", objectBindingList(object, "vision_models"));
  setObjectEditInput("object-sensor-protocols", objectBindingList(object, "sensor_protocols"));
  setObjectEditInput("object-skill-packages", objectBindingList(object, "skill_packages"));
  setObjectEditInput("object-acceptance-tests", objectBindingList(object, "acceptance_tests"));
  const deletePair = document.getElementById("object-delete-pair");
  if (deletePair) deletePair.value = `${projectId}::${object.object_id || objectId}`;
  updateManagedObjectDeleteImpact();
  if (resultEl) resultEl.textContent = `Loaded object: ${object.display_name || object.object_id || objectId}`;
}

async function loadProjectProfileForEdit() {
  const resultEl = document.getElementById("project-edit-result");
  const scopeEl = document.getElementById("project-edit-scope");
  const identifier = document.getElementById("project-edit-id")?.value || "";
  if (!identifier) {
    if (resultEl) resultEl.textContent = "请先选择客户项目。";
    return;
  }
  const payload = await getJson(`${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}?check_env=true`, {});
  if (!payload?.found || !payload.profile) {
    currentProjectEditProfile = null;
    if (resultEl) resultEl.textContent = `Load failed: ${payload?.reason || "profile_not_found"}`;
    return;
  }
  currentProjectEditProfile = payload.profile;
  const customer = payload.customer || payload.profile.customer || {};
  const site = payload.site || payload.profile.site || {};
  setProjectEditInput("project-edit-customer-name", customer.customer_name);
  setProjectEditInput("project-edit-industry", customer.industry);
  setProjectEditInput("project-edit-project-name", customer.project_name);
  setProjectEditInput("project-edit-site-name", site.site_name || site.name);
  setProjectEditInput("project-edit-object-scope-note", customer.object_scope_note);
  if (scopeEl) scopeEl.textContent = `Editing scope: ${renderProjectScopeLabel({ ...customer, site_id: site.site_id })}`;
  if (resultEl) {
    resultEl.innerHTML = `
      <div class="project-import-card ok" data-project-detail-handoff>
        <strong>项目已加载 ${badge(payload.implementation_handoff?.status || "ready", "ok")}</strong>
        <p>保存位置：${esc(payload.profile_path || identifier)}</p>
        ${renderProjectImplementationHandoff(payload.implementation_handoff, "项目详情加载后的实施步骤")}
      </div>
    `;
  }
}

async function saveProjectProfileMetadata() {
  const resultEl = document.getElementById("project-edit-result");
  const identifier = document.getElementById("project-edit-id")?.value || "";
  if (!identifier) {
    if (resultEl) resultEl.textContent = "请先选择客户项目。";
    return;
  }
  let profile = currentProjectEditProfile;
  if (!profile) {
    const payload = await getJson(`${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}?check_env=true`, {});
    profile = payload?.profile;
  }
  if (!profile) {
    if (resultEl) resultEl.textContent = "Load the project profile before saving.";
    return;
  }
  const nextProfile = JSON.parse(JSON.stringify(profile));
  nextProfile.customer = nextProfile.customer || {};
  nextProfile.site = nextProfile.site || {};
  nextProfile.customer.customer_name = document.getElementById("project-edit-customer-name")?.value || nextProfile.customer.customer_name || "";
  nextProfile.customer.industry = document.getElementById("project-edit-industry")?.value || nextProfile.customer.industry || "";
  nextProfile.customer.project_name = document.getElementById("project-edit-project-name")?.value || nextProfile.customer.project_name || "";
  nextProfile.customer.object_scope_note = document.getElementById("project-edit-object-scope-note")?.value || "";
  nextProfile.site.name = document.getElementById("project-edit-site-name")?.value || nextProfile.site.name || "";
  const response = await postJson(ENDPOINTS.fieldCustomerProjects, {
    operator_id: operatorId(),
    overwrite: true,
    profile: nextProfile,
  });
  if (!response.ok) {
    if (resultEl) resultEl.textContent = `Save failed: ${response.payload.reason || response.payload.error || "unknown"}`;
    return;
  }
  currentProjectEditProfile = nextProfile;
  await refreshProjectSurface();
  const latestResultEl = document.getElementById("project-edit-result");
  if (latestResultEl) {
    latestResultEl.innerHTML = `
      <div class="project-import-card ok">
        <strong>项目信息已保存 ${badge("下一步实施", "ok")}</strong>
        <p>保存位置：${esc(response.payload.profile_path || identifier)}</p>
        ${renderProjectImplementationHandoff(response.payload.implementation_handoff, "保存后的实施步骤")}
      </div>
    `;
  }
}

function deliveryResourceFormValue(id) {
  return (document.getElementById(id)?.value || "").trim();
}

async function loadCustomerProjectProfile(identifier, resultEl = null) {
  if (!identifier) return null;
  const payload = await getJson(`${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}?check_env=true`, {});
  if (!payload?.found || !payload.profile) {
    if (resultEl) resultEl.textContent = `Load failed: ${payload?.reason || "profile_not_found"}`;
    return null;
  }
  return payload.profile;
}

async function registerDeliveryResourceFromForm() {
  const resultEl = document.getElementById("resource-register-result");
  const identifier = deliveryResourceFormValue("resource-project-id");
  const resourceType = deliveryResourceFormValue("resource-type");
  const resourceId = deliveryResourceFormValue("resource-id");
  if (!resourceType || !resourceId) {
    if (resultEl) resultEl.textContent = "Select a resource type and resource ID before registering.";
    return;
  }
  if (!DELIVERY_RESOURCE_TYPES.includes(resourceType)) {
    if (resultEl) resultEl.textContent = `Unsupported resource type: ${resourceType}`;
    return;
  }
  if (resultEl) resultEl.textContent = "Registering delivery resource...";
  const response = await postJson(ENDPOINTS.fieldDeliveryResourceRegistry, {
    operator_id: operatorId(),
    overwrite: true,
    reason: `register delivery resource ${resourceType}/${resourceId}`,
    resource: {
      resource_type: resourceType,
      resource_id: resourceId,
      display_name: deliveryResourceFormValue("resource-display-name") || resourceId,
      version: deliveryResourceFormValue("resource-version"),
      owner: deliveryResourceFormValue("resource-owner") || operatorId(),
      source: deliveryResourceFormValue("resource-source") || "shared_registry",
      description: deliveryResourceFormValue("resource-description"),
      project_id: identifier,
    },
  });
  if (!response.ok) {
    if (resultEl) resultEl.textContent = `Register failed: ${response.payload.reason || response.payload.error || "unknown"}`;
    return;
  }
  if (resultEl) {
    resultEl.innerHTML = `
      <div class="project-import-card ok">
        <strong>交付资源已登记 ${badge(resourceType, "ok")}</strong>
        <p>${esc(resourceId)} 已进入共享交付资源登记表，可用于现场对象能力配置。</p>
      </div>
    `;
  }
  await refreshProjectSurface();
  const refreshedResult = document.getElementById("resource-register-result");
  if (refreshedResult) {
    refreshedResult.innerHTML = `
      <div class="project-import-card ok">
        <strong>交付资源已登记 ${badge(resourceType, "ok")}</strong>
        <p>${esc(resourceId)} 已进入共享交付资源登记表，可用于现场对象能力配置。</p>
      </div>
    `;
  }
}

function resourceGovernanceResultEl() {
  return document.getElementById("resource-governance-result");
}

function renderDeliveryResourceHistory(payload = {}) {
  const revisions = Array.isArray(payload.revisions) ? payload.revisions : [];
  return `
    <div class="project-import-card ${revisions.length ? "ok" : "warn"}">
      <strong>资源目录历史 ${badge(`${revisions.length} revisions`, revisions.length ? "ok" : "warn")}</strong>
      <div class="mini-list">
        ${revisions.slice(0, 12).map((item) => `
          <div class="mini-row">
            <b>${esc(item.revision_id || "-")}</b>
            <span>${esc(item.created_at || "-")} / ${esc(item.operator_id || "-")} / ${esc(item.reason || "no reason")}</span>
            <small>${esc(item.registry_sha256 || "")}</small>
          </div>
        `).join("") || `<div class="mini-list-empty">还没有可回滚的资源目录修订。</div>`}
      </div>
    </div>
  `;
}

async function loadDeliveryResourceHistory() {
  const resultEl = resourceGovernanceResultEl();
  if (resultEl) resultEl.textContent = "正在读取资源目录历史...";
  const payload = await getJson(`${ENDPOINTS.fieldDeliveryResourceRegistryHistory}?limit=12`, { revisions: [] });
  if (payload.error || payload.reason) {
    if (resultEl) resultEl.textContent = `读取失败：${payload.reason || payload.error}`;
    return;
  }
  if (resultEl) resultEl.innerHTML = renderDeliveryResourceHistory(payload);
}

function renderDeliveryResourceGovernanceRequestResult(payload = {}, ok = false) {
  const request = payload.request || {};
  const operation = request.operation || {};
  const impact = request.preview?.impact || payload.preview?.impact || {};
  return `
    <div class="project-import-card ${ok ? "ok" : "warn"}">
      <strong>资源治理申请 ${badge(ok ? request.status || "已创建" : "已拒绝", ok ? "ok" : "err")}</strong>
      <p>${esc(payload.reason || payload.next_step || "需要第二位交付负责人复核此申请。")}</p>
      <div class="row-meta">
        <span>申请 ${esc(request.request_id || "-")}</span>
        <span>动作 ${esc(request.action || operation.action || "-")}</span>
        <span>对象 ${esc(operation.resource_type || "registry")}/${esc(operation.resource_id || operation.revision_id || "-")}</span>
        <span>申请人 ${esc(request.requested_by || "-")}</span>
      </div>
      ${renderDeliveryResourceGovernanceSla(request)}
      ${renderDeliveryResourceGovernanceEscalation(request)}
      ${renderDeliveryResourceGovernanceImpact(impact)}
    </div>
  `;
}

function resourceGovernanceSlaClass(state = "") {
  if (state === "overdue") return "err";
  if (state === "due_soon") return "warn";
  if (state === "closed") return "ok";
  return "ok";
}

function formatResourceGovernanceSeconds(value) {
  const seconds = Number(value || 0);
  if (!Number.isFinite(seconds)) return "-";
  const absolute = Math.abs(seconds);
  if (absolute >= 86400) return `${Math.round(absolute / 86400)}d`;
  if (absolute >= 3600) return `${Math.round(absolute / 3600)}h`;
  if (absolute >= 60) return `${Math.round(absolute / 60)}m`;
  return `${Math.round(absolute)}s`;
}

function formatResourceGovernanceTime(value) {
  const timestamp = Number(value || 0);
  if (!Number.isFinite(timestamp) || timestamp <= 0) return "-";
  return new Date(timestamp * 1000).toLocaleString();
}

function renderDeliveryResourceGovernanceSla(request = {}) {
  const sla = request.review_sla || {};
  const state = sla.state || "unknown";
  if (!request.due_at && state === "unknown") {
    return `<p class="muted-line">尚未分配复核时限。</p>`;
  }
  const dueAt = sla.due_at || request.due_at;
  const detail = state === "overdue"
    ? `overdue ${formatResourceGovernanceSeconds(sla.overdue_s)}`
    : state === "closed"
      ? `closed in ${formatResourceGovernanceSeconds(sla.age_s)}`
      : `remaining ${formatResourceGovernanceSeconds(sla.remaining_s)}`;
  return `
    <div class="resource-sla ${resourceGovernanceSlaClass(state)}">
      <strong>复核时限 ${badge(state, resourceGovernanceSlaClass(state))}</strong>
      <div class="row-meta">
        <span>截止 ${esc(formatResourceGovernanceTime(dueAt))}</span>
        <span>${esc(detail)}</span>
        <span>目标 ${esc(formatResourceGovernanceSeconds(sla.target_s || request.sla_target_s))}</span>
        ${sla.escalation_required ? `<span>升级 ${esc(sla.escalation_policy || request.escalation_policy || "delivery_owner_review_overdue")}</span>` : ""}
      </div>
      <p>${esc(sla.message || "Review this request before applying customer-facing resource changes.")}</p>
    </div>
  `;
}

function renderDeliveryResourceGovernanceEscalation(request = {}) {
  const last = request.last_escalation || {};
  const notification = last.notification || {};
  const delivery = Array.isArray(last.delivery_report) ? last.delivery_report : [];
  const sentChannels = Array.isArray(notification.sent_channels)
    ? notification.sent_channels.join(", ")
    : "";
  const count = Number(request.escalation_count || 0);
  if (!count && !last.escalation_id) return "";
  return `
    <div class="resource-escalation">
      <strong>Escalation ${badge(`${count || 1} record(s)`, "warn")}</strong>
      <div class="row-meta">
        <span>${esc(last.status || "queued")}</span>
        <span>${esc(last.delivery_group || "delivery_owners")}</span>
        <span>${esc(notification.delivery_mode || notification.channel || "local_queue")}</span>
        ${sentChannels ? `<span>sent ${esc(sentChannels)}</span>` : ""}
        <span>${esc(formatResourceGovernanceTime(last.escalated_at))}</span>
        <span>${esc(last.escalated_by || "system")}</span>
      </div>
      <p>${esc(last.notification?.message || "Overdue request has been escalated to the delivery owner queue.")}</p>
      ${delivery.length ? renderFieldDelivery(delivery, {}) : ""}
    </div>
  `;
}

function renderDeliveryResourceGovernanceImpact(impact = {}) {
  if (!impact || !impact.impact_type) {
    return `<p class="muted-line">资源影响分析尚未生成。</p>`;
  }
  const consumers = Array.isArray(impact.affected_consumers) ? impact.affected_consumers : [];
  const projects = Array.isArray(impact.affected_projects) ? impact.affected_projects : [];
  const templates = Array.isArray(impact.affected_templates) ? impact.affected_templates : [];
  return `
    <div class="resource-impact">
      <strong>资源治理影响</strong>
      <div class="row-meta">
        <span>分析 ${esc(impact.analysis_status || "-")}</span>
        <span>项目 ${esc(impact.affected_customer_project_count ?? projects.length ?? 0)}</span>
        <span>对象 ${esc(impact.affected_object_count ?? 0)}</span>
        <span>模板 ${esc(impact.affected_template_count ?? templates.length ?? 0)}</span>
        <span>使用方 ${esc(impact.affected_consumer_count ?? consumers.length ?? 0)}</span>
      </div>
      <p>${esc(impact.message || "审批前请复核受影响的客户项目、对象和模板。")}</p>
      ${consumers.length ? `
        <div class="mini-list">
          ${consumers.slice(0, 5).map((item) => `
            <div class="mini-row">
              <b>${esc(item.scope_type || "-")} / ${esc(item.project_id || item.template_id || "-")}</b>
              <span>${esc(item.object_id || "-")} / ${esc(item.display_name || "-")} / ${esc(item.status || "-")}</span>
              <small>${esc(item.profile_path || "")}</small>
            </div>
          `).join("")}
        </div>
      ` : ""}
    </div>
  `;
}

function renderDeliveryResourceGovernanceRequests(payload = {}) {
  const requests = Array.isArray(payload.requests) ? payload.requests : [];
  return `
    <div class="project-import-card">
      <strong>资源治理申请</strong>
      <div class="row-meta">
        <span>待复核 ${esc(payload.summary?.pending_count ?? 0)}</span>
        <span>处理中 ${esc(payload.summary?.active_count ?? 0)}</span>
        <span>即将逾期 ${esc(payload.summary?.due_soon_count ?? 0)}</span>
        <span>已逾期 ${esc(payload.summary?.overdue_count ?? 0)}</span>
        <span>已通过 ${esc(payload.summary?.approved_count ?? 0)}</span>
        <span>已拒绝 ${esc(payload.summary?.rejected_count ?? 0)}</span>
        ${payload.overdue_only ? `<span>仅看逾期</span>` : ""}
      </div>
      <div class="mini-list">
        ${requests.map((item) => {
          const operation = item.operation || {};
          const impact = item.preview?.impact || {};
          const sla = item.review_sla || {};
          const lastEscalation = item.last_escalation || {};
          const target = item.action === "rollback_registry"
            ? `revision ${operation.revision_id || "-"}`
            : `${operation.resource_type || "-"}/${operation.resource_id || "-"}`;
          return `
            <div class="mini-row">
              <b>
                ${esc(item.action || "-")}
                ${badge(item.status || "unknown", acceptanceGateClass(item.status))}
                ${badge(sla.state || "no_sla", resourceGovernanceSlaClass(sla.state || ""))}
              </b>
              <small>${esc(item.request_id || "-")}</small>
              <span>${esc(target)} / ${esc(item.requested_by || "system")} / ${esc(new Date(Number(item.requested_at || 0) * 1000).toLocaleString())}</span>
              <span>复核截止 ${esc(formatResourceGovernanceTime(sla.due_at || item.due_at))} / ${sla.state === "overdue" ? "已逾期" : "剩余"} ${esc(formatResourceGovernanceSeconds(sla.state === "overdue" ? sla.overdue_s : sla.remaining_s))}</span>
              ${Number(item.escalation_count || 0) ? `<span>已升级 ${esc(item.escalation_count)} 次 / ${esc(lastEscalation.status || "queued")}</span>` : ""}
              <span>影响：${esc(impact.affected_customer_project_count ?? 0)} 个项目、${esc(impact.affected_template_count ?? 0)} 个模板、${esc(impact.affected_consumer_count ?? 0)} 个使用方</span>
              <span>${esc(item.reason || "未填写原因")}</span>
              ${item.status === "pending" ? `
                <span>
                  <button class="ghost-button" data-resource-governance-review="${esc(item.request_id || "")}" data-review-decision="approve">通过</button>
                  <button class="ghost-button" data-resource-governance-review="${esc(item.request_id || "")}" data-review-decision="reject">拒绝</button>
                </span>
              ` : ""}
            </div>
          `;
        }).join("") || `<div class="mini-list-empty">No resource governance requests yet.</div>`}
      </div>
    </div>
  `;
}

async function loadDeliveryResourceGovernanceRequests(mode = "all") {
  const resultEl = resourceGovernanceResultEl();
  const overdueOnly = mode === "overdue";
  if (resultEl) resultEl.textContent = overdueOnly
    ? "Loading overdue resource governance requests..."
    : "Loading resource governance requests...";
  const payload = await getJson(
    `${ENDPOINTS.fieldDeliveryResourceGovernanceRequests}?limit=12${overdueOnly ? "&overdue_only=true" : ""}`,
    { requests: [], summary: {}, request_count: 0 },
  );
  if (resultEl) {
    resultEl.innerHTML = renderDeliveryResourceGovernanceRequests(payload);
    wireResourceGovernanceReviewControls(resultEl);
  }
}

async function escalateOverdueDeliveryResourceGovernanceRequests() {
  const resultEl = resourceGovernanceResultEl();
  const reason = window.prompt(
    "请输入升级原因",
    "升级逾期的交付资源治理申请",
  );
  if (reason === null) return;
  if (resultEl) resultEl.textContent = "正在升级逾期资源治理申请...";
  const response = await postJson(
    `${ENDPOINTS.fieldDeliveryResourceGovernanceRequests}/escalate-overdue`,
    {
      operator_id: operatorId(),
      reason: reason || "dashboard overdue escalation",
      limit: 50,
    },
  );
  if (!response.ok) {
    if (resultEl) resultEl.innerHTML = renderDeliveryResourceGovernanceEscalationResult(response.payload, false);
    return;
  }
  if (resultEl) resultEl.innerHTML = renderDeliveryResourceGovernanceEscalationResult(response.payload, true);
  wireResourceGovernanceReviewControls(resultEl || document);
}

function renderDeliveryResourceGovernanceEscalationResult(payload = {}, ok = false) {
  const escalations = Array.isArray(payload.escalations) ? payload.escalations : [];
  const skipped = Array.isArray(payload.skipped) ? payload.skipped : [];
  return `
    <div class="project-import-card ${ok ? "ok" : "warn"}">
      <strong>Overdue escalation ${badge(ok ? "submitted" : "failed", ok ? "ok" : "err")}</strong>
      <p>${esc(payload.reason || payload.next_step || "Escalation result returned.")}</p>
      <div class="row-meta">
        <span>checked ${esc(payload.checked_count ?? 0)}</span>
        <span>escalated ${esc(payload.escalated_count ?? escalations.length)}</span>
        <span>skipped ${esc(payload.skipped_count ?? skipped.length)}</span>
      </div>
      ${escalations.length ? `
        <div class="mini-list">
          ${escalations.slice(0, 5).map((item) => `
            <div class="mini-row">
              <b>${esc(item.request_id || "-")} ${badge(item.status || "queued", item.status === "sent" ? "ok" : "warn")}</b>
              <span>${esc(item.target || "-")} / overdue ${esc(formatResourceGovernanceSeconds(item.overdue_s))} / ${esc(item.notification?.delivery_mode || item.notification?.channel || "local_queue")}</span>
              <small>${esc(item.notification?.message || "Queued for delivery owner review.")}</small>
              ${Array.isArray(item.delivery_report) && item.delivery_report.length ? renderFieldDelivery(item.delivery_report, {}) : ""}
            </div>
          `).join("")}
        </div>
      ` : ""}
      ${skipped.length ? `<p class="muted-line">Skipped: ${esc(skipped.map((item) => item.reason || item.request_id || "-").join(", "))}</p>` : ""}
    </div>
    ${renderDeliveryResourceGovernanceRequests(payload)}
  `;
}

function wireResourceGovernanceReviewControls(root = document) {
  root.querySelectorAll("[data-resource-governance-review]").forEach((button) => {
    if (button.dataset.reviewWired === "true") return;
    button.dataset.reviewWired = "true";
    button.addEventListener("click", () => reviewDeliveryResourceGovernanceRequest(
      button.dataset.resourceGovernanceReview || "",
      button.dataset.reviewDecision || "approve",
    ));
  });
}

async function reviewDeliveryResourceGovernanceRequest(requestId, decision) {
  if (!requestId) return;
  const reason = window.prompt("资源治理复核原因", `${decision} ${requestId}`);
  if (reason === null) return;
  const resultEl = resourceGovernanceResultEl();
  if (resultEl) resultEl.textContent = "Submitting resource governance review...";
  const response = await postJson(
    `${ENDPOINTS.fieldDeliveryResourceGovernanceRequests}/${encodeURIComponent(requestId)}/review`,
    {
      operator_id: operatorId(),
      decision,
      reason,
    },
  );
  if (!response.ok) {
    if (resultEl) resultEl.innerHTML = renderDeliveryResourceGovernanceRequestResult(response.payload, false);
    return;
  }
  await refreshProjectSurface();
  const refreshed = resourceGovernanceResultEl();
  if (refreshed) refreshed.innerHTML = renderDeliveryResourceGovernanceRequestResult(response.payload, true);
}

async function disableDeliveryResource(key) {
  const parts = String(key || "").split("::");
  const resourceType = parts[0] || "";
  const resourceId = parts.slice(1).join("::");
  const resultEl = resourceGovernanceResultEl();
  if (!resourceType || !resourceId) {
    if (resultEl) resultEl.textContent = "无法停用：资源类型或资源 ID 缺失。";
    return;
  }
  const reason = window.prompt("停用原因", `停用 ${resourceType}/${resourceId}`);
  if (reason === null) return;
  if (resultEl) resultEl.textContent = `正在停用 ${resourceType}/${resourceId}...`;
  const response = await postJson(
    `${ENDPOINTS.fieldDeliveryResourceRegistry}/${encodeURIComponent(resourceType)}/${encodeURIComponent(resourceId)}/disable`,
    { operator_id: operatorId(), reason: reason || "disabled from dashboard" },
  );
  if (!response.ok) {
    if (resultEl) resultEl.textContent = `停用失败：${response.payload.reason || response.payload.error || "unknown"}`;
    return;
  }
  await refreshProjectSurface();
  const refreshed = resourceGovernanceResultEl();
  if (refreshed) {
    refreshed.innerHTML = `
      <div class="project-import-card ok">
        <strong>资源已停用 ${badge(resourceType, "ok")}</strong>
        <p>${esc(resourceId)} 后续不会再通过客户对象 readiness 检查。</p>
      </div>
    `;
  }
}

async function requestDeliveryResourceDisable(key) {
  const parts = String(key || "").split("::");
  const resourceType = parts[0] || "";
  const resourceId = parts.slice(1).join("::");
  const resultEl = resourceGovernanceResultEl();
  if (!resourceType || !resourceId) {
    if (resultEl) resultEl.textContent = "Cannot request disable: resource type or resource ID is missing.";
    return;
  }
  const reason = window.prompt("Disable request reason", `Disable ${resourceType}/${resourceId}`);
  if (reason === null) return;
  if (resultEl) resultEl.textContent = `Creating disable request for ${resourceType}/${resourceId}...`;
  const response = await postJson(
    ENDPOINTS.fieldDeliveryResourceGovernanceRequests,
    {
      operator_id: operatorId(),
      action: "disable_resource",
      operation: { resource_type: resourceType, resource_id: resourceId },
      reason: reason || "disable requested from dashboard",
    },
  );
  if (resultEl) {
    resultEl.innerHTML = renderDeliveryResourceGovernanceRequestResult(response.payload, response.ok);
  }
}

async function rollbackDeliveryResourceRegistry(mode = "dry-run") {
  const revisionId = deliveryResourceFormValue("resource-rollback-id");
  const apply = mode === "apply";
  const resultEl = resourceGovernanceResultEl();
  if (apply && revisionId) {
    const reason = window.prompt("回滚申请原因", `回滚资源登记表到 ${revisionId}`);
    if (reason === null) return;
    if (resultEl) resultEl.textContent = "Creating registry rollback request...";
    const response = await postJson(
      ENDPOINTS.fieldDeliveryResourceGovernanceRequests,
      {
        operator_id: operatorId(),
        action: "rollback_registry",
        operation: { revision_id: revisionId },
        reason: reason || "rollback requested from dashboard",
      },
    );
    if (resultEl) {
      resultEl.innerHTML = renderDeliveryResourceGovernanceRequestResult(response.payload, response.ok);
    }
    return;
  }
  if (!revisionId) {
    if (resultEl) resultEl.textContent = "请输入 revision id 后再回滚。";
    return;
  }
  if (apply && !window.confirm("回滚共享资源目录会影响多个客户项目，确定要继续吗？")) return;
  if (resultEl) resultEl.textContent = apply ? "正在执行资源目录回滚..." : "正在预演资源目录回滚...";
  const response = await postJson(ENDPOINTS.fieldDeliveryResourceRegistryRollback, {
    operator_id: operatorId(),
    revision_id: revisionId,
    reason: apply ? "apply dashboard registry rollback" : "dry-run dashboard registry rollback",
    dry_run: !apply,
  });
  if (!response.ok) {
    if (resultEl) resultEl.textContent = `回滚失败：${response.payload.reason || response.payload.error || "unknown"}`;
    return;
  }
  if (apply) await refreshProjectSurface();
  const refreshed = resourceGovernanceResultEl();
  if (refreshed) {
    const payload = response.payload || {};
    refreshed.innerHTML = `
      <div class="project-import-card ${apply ? "ok" : "warn"}">
        <strong>${apply ? "资源目录已回滚" : "资源目录回滚预演"} ${badge(payload.revision_id || revisionId, apply ? "ok" : "warn")}</strong>
        <p>目标资源数：${esc(payload.target_summary?.resource_count ?? "-")}；${apply ? "已经写入共享目录。" : "尚未写入，可确认后再执行 Apply rollback。"}</p>
      </div>
    `;
  }
}

async function upsertManagedObjectFromForm() {
  const resultEl = document.getElementById("object-upsert-result");
  const identifier = document.getElementById("object-project-id")?.value || "";
  const objectId = document.getElementById("object-id")?.value || "";
  const body = {
    managed_object: {
      display_name: document.getElementById("object-display-name")?.value || objectId,
      category: document.getElementById("object-category")?.value || "custom",
      object_labels: commaList(document.getElementById("object-labels")?.value),
      scenario_ids: commaList(document.getElementById("object-scenarios")?.value),
      zone_types: commaList(document.getElementById("object-zone-types")?.value),
      device_sources: commaList(document.getElementById("object-device-sources")?.value),
      tenant_ids: commaList(document.getElementById("object-tenant-ids")?.value),
      delivery_namespaces: commaList(document.getElementById("object-delivery-namespaces")?.value),
      customer_ids: commaList(document.getElementById("object-customer-ids")?.value),
      project_ids: commaList(document.getElementById("object-project-ids")?.value),
      site_ids: commaList(document.getElementById("object-site-ids")?.value),
      responder_group: document.getElementById("object-responder-group")?.value || "operations",
      evidence_required: commaList(document.getElementById("object-evidence-required")?.value),
      bindings: {
        vision_models: commaList(document.getElementById("object-vision-models")?.value),
        sensor_protocols: commaList(document.getElementById("object-sensor-protocols")?.value),
        skill_packages: commaList(document.getElementById("object-skill-packages")?.value),
        acceptance_tests: commaList(document.getElementById("object-acceptance-tests")?.value),
      },
      customer_visible: true,
    },
    reason: "save from managed object editor",
  };
  const response = await postJson(`${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/managed-objects/${encodeURIComponent(objectId)}`, body);
  if (resultEl) resultEl.innerHTML = renderManagedObjectWriteResult(response.payload, response.ok, "对象已保存");
  if (response.ok) await refreshProjectSurface();
}

async function exportSelectedCustomerProject() {
  const identifier = document.getElementById("project-lifecycle-id")?.value || "";
  await exportCustomerProject(identifier);
}

async function exportCustomerProject(identifier) {
  if (!identifier) return;
  const payload = await getJson(`${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/export`, {});
  const target = document.getElementById("project-lifecycle-result") || document.getElementById("object-upsert-result") || document.getElementById("project-create-result");
  if (target) target.innerHTML = renderProjectExportResult(payload);
  const importBox = document.getElementById("project-import-json");
  if (payload.accepted && payload.package && importBox && !importBox.value.trim()) {
    importBox.value = JSON.stringify(payload.package, null, 2);
  }
}

async function loadCustomerProjectAcceptanceReport(identifier) {
  if (!identifier) return;
  const target = document.getElementById("project-lifecycle-result") || document.getElementById("object-upsert-result") || document.getElementById("project-create-result");
  if (target) target.textContent = "正在生成客户项目验收报告...";
  const payload = await getJson(`${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/acceptance-report`, { found: false, reason: "request_failed" });
  setAcceptanceEvidenceOptions(payload);
  if (target) target.innerHTML = renderCustomerProjectAcceptanceReport(payload);
}

async function loadSelectedCustomerProjectExecutionBindings() {
  const identifier = document.getElementById("project-lifecycle-id")?.value || "";
  if (!identifier) return;
  const target = document.getElementById("project-lifecycle-result") || document.getElementById("object-upsert-result") || document.getElementById("project-create-result");
  if (target) target.textContent = "正在生成执行接入计划...";
  const payload = await getJson(
    `${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/${ENDPOINTS.fieldCustomerProjectExecutionBindingsSuffix}`,
    { found: false, reason: "request_failed" },
  );
  if (target) target.innerHTML = renderCustomerProjectExecutionBindings(payload);
  if (target) wireObjectExecutionRehearsalButtons();
}

function wireObjectExecutionRehearsalButtons() {
  document.querySelectorAll("[data-object-rehearsal]").forEach((button) => {
    button.addEventListener("click", async () => {
      const mode = button.dataset.objectRehearsal || "dry_run";
      const identifier = button.dataset.projectId || "";
      const objectId = button.dataset.objectId || "";
      if (!identifier || !objectId) return;
      let confirmShadowPost = false;
      if (mode === "shadow_post") {
        confirmShadowPost = window.confirm("Shadow post may create a lab field event. Continue only during a rehearsal window.");
        if (!confirmShadowPost) return;
      }
      await rehearseCustomerProjectObject(identifier, objectId, mode, confirmShadowPost);
    });
  });
}

async function rehearseCustomerProjectObject(identifier, objectId, mode = "dry_run", confirmShadowPost = false) {
  const target = document.getElementById("project-execution-rehearsal-result");
  if (target) target.textContent = "正在执行对象接入演练...";
  const response = await postJson(
    `${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/${ENDPOINTS.fieldCustomerProjectExecutionBindingsSuffix}/${encodeURIComponent(objectId)}/rehearsal`,
    {
      operator_id: operatorId(),
      mode,
      confirm_shadow_post: confirmShadowPost,
      register_onsite_evidence: mode === "shadow_post" && confirmShadowPost,
    },
  );
  if (target) target.innerHTML = renderObjectExecutionRehearsalResult(response.payload || {});
}

async function exportCustomerProjectAcceptanceDossier(identifier) {
  if (!identifier) return;
  const target = document.getElementById("project-lifecycle-result") || document.getElementById("object-upsert-result") || document.getElementById("project-create-result");
  if (target) target.textContent = "正在生成客户验收证据包...";
  const payload = await getJson(`${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/acceptance-dossier`, { accepted: false, reason: "request_failed" });
  if (payload.accepted && payload.dossier) {
    const dossierBox = document.getElementById("project-dossier-json");
    if (dossierBox) dossierBox.value = JSON.stringify(payload.dossier, null, 2);
  }
  if (target) target.innerHTML = renderCustomerProjectAcceptanceDossier(payload);
}

async function loadSelectedCustomerProjectOnsiteEvidence() {
  const identifier = document.getElementById("project-lifecycle-id")?.value || "";
  await loadCustomerProjectOnsiteEvidence(identifier);
}

async function loadCustomerProjectOnsiteEvidence(identifier) {
  if (!identifier) return;
  const target = document.getElementById("project-lifecycle-result") || document.getElementById("object-upsert-result") || document.getElementById("project-create-result");
  if (target) target.textContent = "正在读取现场验收证据...";
  const payload = await getJson(
    `${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/${ENDPOINTS.fieldCustomerProjectOnsiteEvidenceSuffix}?include_readiness_auto=true`,
    { found: false, reason: "request_failed" },
  );
  setAcceptanceEvidenceOptions(payload);
  if (target) target.innerHTML = renderCustomerProjectOnsiteEvidence(payload);
}

async function registerSelectedCustomerProjectOnsiteEvidence() {
  const resultEl = document.getElementById("project-lifecycle-result");
  const identifier = document.getElementById("project-lifecycle-id")?.value || "";
  if (!identifier) {
    if (resultEl) resultEl.textContent = "登记现场证据前请先选择客户项目。";
    return;
  }
  const body = {
    operator_id: operatorId(),
    reason: "Register onsite acceptance evidence from dashboard.",
    evidence: {
      evidence_type: document.getElementById("project-onsite-evidence-type")?.value || "customer_review",
      status: document.getElementById("project-onsite-evidence-status")?.value || "manual_check",
      path: document.getElementById("project-onsite-evidence-path")?.value || "",
      summary: document.getElementById("project-onsite-evidence-summary")?.value || "",
      source: "dashboard",
    },
  };
  const response = await postJson(
    `${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/${ENDPOINTS.fieldCustomerProjectOnsiteEvidenceSuffix}`,
    body,
  );
  setAcceptanceEvidenceOptions(response.payload || {});
  if (resultEl) resultEl.innerHTML = renderCustomerProjectOnsiteEvidence(response.payload, response.ok);
  if (response.ok) await refreshProjectSurface();
}

async function loadSelectedCustomerProjectAcceptanceClosure() {
  const identifier = document.getElementById("project-lifecycle-id")?.value || "";
  await loadCustomerProjectAcceptanceClosure(identifier);
}

async function loadCustomerProjectAcceptanceClosure(identifier) {
  if (!identifier) return;
  const target = document.getElementById("project-lifecycle-result") || document.getElementById("object-upsert-result") || document.getElementById("project-create-result");
  if (target) target.textContent = "正在读取验收闭环...";
  const payload = await getJson(
    `${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/${ENDPOINTS.fieldCustomerProjectAcceptanceClosureSuffix}`,
    { found: false, reason: "request_failed" },
  );
  if (target) target.innerHTML = renderCustomerProjectAcceptanceClosure(payload);
}

async function registerSelectedCustomerProjectAcceptanceReview() {
  const resultEl = document.getElementById("project-lifecycle-result");
  const identifier = document.getElementById("project-lifecycle-id")?.value || "";
  if (!identifier) {
    if (resultEl) resultEl.textContent = "提交复核前请先选择客户项目。";
    return;
  }
  const response = await postJson(
    `${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/${ENDPOINTS.fieldCustomerProjectAcceptanceReviewSuffix}`,
    {
      operator_id: operatorId(),
      reason: document.getElementById("project-acceptance-review-reason")?.value || "",
      review: {
        decision: document.getElementById("project-acceptance-review-decision")?.value || "needs_fix",
        reason: document.getElementById("project-acceptance-review-reason")?.value || "",
        risk_acknowledgement: document.getElementById("project-acceptance-risk-ack")?.checked === true,
        evidence_refs: commaList(document.getElementById("project-acceptance-evidence-refs")?.value),
      },
    },
  );
  if (resultEl) resultEl.innerHTML = renderCustomerProjectAcceptanceClosure(response.payload.closure || response.payload, response.ok);
  if (response.ok) await refreshProjectSurface();
}

async function loadSelectedCustomerProjectCustomerSignoff() {
  const identifier = document.getElementById("project-lifecycle-id")?.value || "";
  await loadCustomerProjectCustomerSignoff(identifier);
}

async function loadCustomerProjectCustomerSignoff(identifier) {
  if (!identifier) return;
  const target = document.getElementById("project-lifecycle-result")
    || document.getElementById("object-upsert-result")
    || document.getElementById("project-create-result");
  if (target) target.textContent = "正在读取客户签收记录...";
  const payload = await getJson(
    `${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/${ENDPOINTS.fieldCustomerProjectCustomerSignoffSuffix}`,
    { found: false, reason: "request_failed" },
  );
  if (target) target.innerHTML = renderCustomerProjectCustomerSignoff(payload);
}

async function registerSelectedCustomerProjectCustomerSignoff() {
  const resultEl = document.getElementById("project-lifecycle-result");
  const identifier = document.getElementById("project-lifecycle-id")?.value || "";
  if (!identifier) {
    if (resultEl) resultEl.textContent = "请先选择客户项目，再登记客户签收。";
    return;
  }
  const response = await postJson(
    `${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/${ENDPOINTS.fieldCustomerProjectCustomerSignoffSuffix}`,
    {
      operator_id: operatorId(),
      reason: document.getElementById("project-customer-signoff-reason")?.value || "",
      signoff: {
        decision: document.getElementById("project-customer-signoff-decision")?.value || "needs_fix",
        signatory_name: document.getElementById("project-customer-signatory-name")?.value || "",
        signatory_role: document.getElementById("project-customer-signatory-role")?.value || "",
        organization: document.getElementById("project-customer-signoff-organization")?.value || "",
        reason: document.getElementById("project-customer-signoff-reason")?.value || "",
        risk_acknowledgement: document.getElementById("project-customer-signoff-risk-ack")?.checked === true,
        credential_ref: document.getElementById("project-customer-signoff-credential-ref")?.value || "",
        credential_sha256: document.getElementById("project-customer-signoff-credential-sha256")?.value || "",
        evidence_refs: commaList(document.getElementById("project-customer-signoff-evidence-refs")?.value),
      },
    },
  );
  if (resultEl) resultEl.innerHTML = renderCustomerProjectAcceptanceClosure(response.payload.closure || response.payload, response.ok);
  if (response.ok) await refreshProjectSurface();
}

async function exportSelectedCustomerProjectProposalBundle() {
  const identifier = document.getElementById("project-lifecycle-id")?.value || "";
  await exportCustomerProjectProposalBundle(identifier);
}

async function exportCustomerProjectProposalBundle(identifier) {
  if (!identifier) return;
  const target = document.getElementById("project-lifecycle-result") || document.getElementById("object-upsert-result") || document.getElementById("project-create-result");
  if (target) target.textContent = "正在生成客户方案包...";
  const payload = await getJson(
    `${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/${ENDPOINTS.fieldCustomerProjectProposalBundleSuffix}`,
    { accepted: false, reason: "request_failed" },
  );
  if (payload.accepted && payload.proposal) {
    downloadCustomerProjectProposalBundle(payload.proposal);
    const proposalBox = document.getElementById("project-proposal-json");
    if (proposalBox) proposalBox.value = JSON.stringify(payload.proposal, null, 2);
  }
  if (target) target.innerHTML = renderCustomerProjectProposalBundle(payload);
}

function downloadCustomerProjectProposalBundle(proposal = {}) {
  const customer = proposal.customer || {};
  const projectId = customer.project_id || proposal.manifest?.project_id || "project";
  const customerId = customer.customer_id || proposal.manifest?.customer_id || "customer";
  const filenameBase = `${slugForDownload(customerId)}-${slugForDownload(projectId)}-proposal-bundle`;
  const jsonPayload = { ...proposal };
  delete jsonPayload.html;
  downloadTextFile(
    `${filenameBase}.json`,
    JSON.stringify(jsonPayload, null, 2),
    "application/json;charset=utf-8",
  );
  downloadTextFile(
    `${filenameBase}.html`,
    proposal.html || "",
    "text/html;charset=utf-8",
  );
}

function slugForDownload(value) {
  return String(value || "item").toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^[-_.]+|[-_.]+$/g, "") || "item";
}

function onsiteReceiptSourceLabel(receipt = {}) {
  const source = String(receipt.source || "");
  if (source === "field_readiness_auto_backfill" || receipt.auto_backfill) return "系统自动采信";
  if (source) return `人工登记 / ${source}`;
  return "人工登记";
}

function renderOnsiteReceiptMeta(receipt = {}) {
  const sha = String(receipt.sha256 || "").slice(0, 16) || "-";
  const ref = receipt.external_reference || receipt.path || "-";
  return `
    <div class="row-meta onsite-receipt-meta">
      <span>${esc(onsiteReceiptSourceLabel(receipt))}</span>
      <span>${esc(receipt.evidence_type || "evidence")}</span>
      <span>sha ${esc(sha)}</span>
      <span>${esc(ref)}</span>
    </div>
    ${renderEvidenceBoundaryTags(receipt)}
  `;
}

function evidenceBoundaryLabels(item = {}) {
  const tier = String(item.evidence_tier || "").trim();
  const labels = [];
  if (tier === "lab_rehearsal") labels.push("\u5b9e\u9a8c\u5ba4\u6f14\u7ec3");
  if (tier === "acceptance_candidate") labels.push("\u9a8c\u6536\u5019\u9009\u8bc1\u636e");
  if (tier === "site_acceptance") labels.push("\u73b0\u573a\u9a8c\u6536\u8bc1\u636e");
  if (item.production_eligible === false || item.production_claim_allowed === false) {
    labels.push("\u4e0d\u53ef\u4f5c\u4e3a\u751f\u4ea7\u4e0a\u7ebf\u8bc1\u636e");
  } else if (item.production_eligible === true || item.production_claim_allowed === true) {
    labels.push("\u53ef\u8fdb\u5165\u751f\u4ea7\u4e0a\u7ebf\u8bc4\u4f30");
  }
  return labels;
}

function renderEvidenceBoundaryTags(item = {}) {
  const labels = evidenceBoundaryLabels(item);
  if (!labels.length) return "";
  return `
    <div class="row-meta evidence-boundary-tags">
      ${labels.map((label) => `<span>${esc(label)}</span>`).join("")}
    </div>
  `;
}

function renderSiteAcceptanceChecklist(checklist = {}) {
  const items = Array.isArray(checklist.items) ? checklist.items : [];
  if (!items.length) return "";
  return `
    <div class="project-delivery-workflow">
      <div class="project-delivery-head">
        <strong>客户现场验收清单 ${badge(checklist.overall_status || "manual_check", acceptanceGateClass(checklist.overall_status))}</strong>
        <span>就绪 ${esc(checklist.ready_count ?? 0)} / 待人工复核 ${esc(checklist.manual_check_count ?? 0)} / 阻断 ${esc(checklist.blocked_count ?? 0)}</span>
      </div>
      <div class="project-delivery-steps">
        ${items.map((item) => `
          <div class="project-delivery-step ${acceptanceGateClass(item.status)}">
            <b>${esc(item.label || item.item_id || "验收项")}</b>
            <span>${esc(item.owner || "-")} / ${esc(item.status || "manual_check")}</span>
            <small>${esc(item.next_step || item.evidence || "-")}</small>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderCustomerProjectAcceptanceReport(payload = {}) {
  if (!payload.found) {
    return `<div class="project-import-card warn"><strong>验收报告不可用 ${badge("blocked", "err")}</strong><p>${esc(payload.reason || payload.error || "unknown")}</p></div>`;
  }
  const gates = Array.isArray(payload.gates) ? payload.gates : [];
  const summary = payload.acceptance_summary || {};
  const readiness = payload.field_readiness || {};
  const onsiteEvidence = payload.onsite_acceptance_evidence || {};
  const onsiteSummary = onsiteEvidence.summary || {};
  const onsiteReceipts = Array.isArray(onsiteEvidence.receipts) ? onsiteEvidence.receipts : [];
  const evidenceReports = Array.isArray(readiness.evidence_reports) ? readiness.evidence_reports : [];
  return `
    <div class="project-import-card ${payload.overall_status === "blocked" ? "warn" : "ok"}">
      <strong>验收报告 ${badge(payload.overall_status || "unknown", acceptanceGateClass(payload.overall_status))}</strong>
      <p>${esc(payload.customer_status || "-")}</p>
      <div class="row-meta">
        <span>就绪对象 ${esc(summary.ready_object_count ?? 0)}/${esc(summary.object_count ?? 0)}</span>
        <span>待人工复核 ${esc(summary.manual_check_object_count ?? 0)}</span>
        <span>阻断 ${esc(summary.blocked_object_count ?? 0)}</span>
        <span>现场就绪 ${esc(readiness.status || "-")}</span>
        <span>现场证据 ${esc(onsiteSummary.passed_required_count ?? 0)}/${esc(onsiteSummary.required_count ?? 4)}</span>
      </div>
      ${renderProjectDeliveryWorkflow(payload.delivery_workflow)}
      ${renderSiteAcceptanceChecklist(payload.site_acceptance_checklist)}
      ${renderCustomerProjectOnsiteEvidence({ found: true, onsite_acceptance_evidence: onsiteEvidence }, true)}
      <div class="capability-list compact-list">
        ${gates.map((gate) => `
          <div class="row-item">
            <strong>${esc(gate.label || gate.gate_id || "gate")} ${badge(gate.status || "unknown", acceptanceGateClass(gate.status))}</strong>
            <p>${esc(gate.evidence || "-")}</p>
            <span>${esc(gate.next_step || "-")}</span>
          </div>
        `).join("")}
      </div>
      ${evidenceReports.length ? `
        <div class="capability-list compact-list">
          ${evidenceReports.map((report) => `
            <div class="row-item">
              <strong>${esc(report.path || "evidence")} ${badge(report.status || "unknown", report.passed ? "ok" : "warn")}</strong>
              <span>local_server ${esc(report.local_server ?? "-")} / live_tts ${esc(report.live_tts ?? "-")} / external ${esc(report.external_services ?? "-")}</span>
            </div>
          `).join("")}
        </div>
      ` : ""}
      ${onsiteReceipts.length ? `
        <div class="capability-list compact-list">
          ${onsiteReceipts.slice(0, 6).map((receipt) => `
            <div class="row-item">
              <strong>${esc(receipt.label || receipt.evidence_type || "onsite evidence")} ${badge(receipt.status || "manual_check", acceptanceGateClass(receipt.status))}</strong>
              <span>${esc(receipt.summary || receipt.path || "-")}</span>
              ${renderOnsiteReceiptMeta(receipt)}
            </div>
          `).join("")}
        </div>
      ` : ""}
      <p class="muted-line">${esc(payload.release_claim || "")}</p>
    </div>
  `;
}

function renderCustomerProjectOnsiteEvidence(payload = {}, compact = false) {
  if (!payload.found && !payload.accepted) {
    return `<div class="project-import-card warn"><strong>现场证据不可用 ${badge("blocked", "err")}</strong><p>${esc(payload.reason || payload.error || "unknown")}</p></div>`;
  }
  const evidence = payload.onsite_acceptance_evidence || {};
  const summary = evidence.summary || payload.summary || {};
  const receipts = Array.isArray(evidence.receipts) ? evidence.receipts : Array.isArray(payload.receipts) ? payload.receipts : [];
  return `
    <div class="project-import-card ${summary.overall_status === "ready" ? "ok" : summary.overall_status === "blocked" ? "warn" : "manual_check"}">
      <strong>${compact ? "现场证据" : "现场验收证据"} ${badge(summary.overall_status || "manual_check", acceptanceGateClass(summary.overall_status))}</strong>
      <p>${esc(summary.customer_status || "客户验收前需要绑定真实现场证据。")}</p>
      <div class="row-meta">
        <span>必需证据 ${esc(summary.passed_required_count ?? 0)}/${esc(summary.required_count ?? 4)}</span>
        <span>凭证 ${esc(summary.receipt_count ?? receipts.length)}</span>
        <span>失败 ${esc(summary.failed_count ?? 0)}</span>
        <span>缺失 ${esc((summary.missing_required_types || []).join(", ") || "-")}</span>
      </div>
      ${compact ? "" : `
        <div class="capability-list compact-list">
          ${receipts.slice(0, 8).map((receipt) => `
            <div class="row-item">
              <strong>${esc(receipt.label || receipt.evidence_type || "receipt")} ${badge(receipt.status || "manual_check", acceptanceGateClass(receipt.status))}</strong>
              <p>${esc(receipt.summary || "-")}</p>
              ${renderOnsiteReceiptMeta(receipt)}
            </div>
          `).join("") || `<div class="mini-list-empty">暂无已登记的现场验收证据。</div>`}
        </div>
      `}
      <p class="muted-line">${esc(summary.next_step || "请绑定设备上报、语音播报、外部通知和执行回传证据。")}</p>
    </div>
  `;
  return `
    <div class="project-import-card ${summary.overall_status === "ready" ? "ok" : summary.overall_status === "blocked" ? "warn" : "manual_check"}">
      <strong>${compact ? "现场证据" : "现场验收证据"} ${badge(summary.overall_status || "manual_check", acceptanceGateClass(summary.overall_status))}</strong>
      <p>${esc(summary.customer_status || "需要补齐真实现场证据后才能提交验收。")}</p>
      <div class="row-meta">
        <span>必需证据 ${esc(summary.passed_required_count ?? 0)}/${esc(summary.required_count ?? 4)}</span>
        <span>凭证 ${esc(summary.receipt_count ?? receipts.length)}</span>
        <span>失败 ${esc(summary.failed_count ?? 0)}</span>
        <span>missing ${esc((summary.missing_required_types || []).join(", ") || "-")}</span>
      </div>
      ${compact ? "" : `
        <div class="capability-list compact-list">
          ${receipts.slice(0, 8).map((receipt) => `
            <div class="row-item">
              <strong>${esc(receipt.label || receipt.evidence_type || "receipt")} ${badge(receipt.status || "manual_check", acceptanceGateClass(receipt.status))}</strong>
              <p>${esc(receipt.summary || "-")}</p>
              ${renderOnsiteReceiptMeta(receipt)}
            </div>
          `).join("") || `<div class="mini-list-empty">还没有登记现场验收证据。</div>`}
        </div>
      `}
      <p class="muted-line">${esc(summary.next_step || "补齐设备上报、语音播报、外部通知和任务回调证据。")}</p>
    </div>
  `;
}

function renderCustomerProjectCustomerSignoff(payload = {}, compact = false) {
  const latest = payload.latest || {};
  const signoffs = Array.isArray(payload.signoffs) ? payload.signoffs : [];
  const decision = latest.decision || (signoffs.length ? signoffs[0].decision : "");
  const statusClass = acceptanceGateClass(decision || (payload.base_ready_for_signoff ? "ready_for_customer_signoff" : "manual_check"));
  if (!payload.found && !payload.signoff_count && !payload.base_ready_for_signoff && !compact) {
    return `<div class="project-import-card warn"><strong>客户签收不可用 ${badge("blocked", "err")}</strong><p>${esc(payload.reason || payload.error || "unknown")}</p></div>`;
  }
  return `
    <div class="project-import-card ${statusClass || "manual_check"}">
      <strong>客户签收 ${badge(decision || "未提交", statusClass)}</strong>
      <p>${esc(latest.reason || (payload.base_ready_for_signoff ? "内部交付门禁已就绪，等待客户签收。" : "客户签收前需要先完成内部交付门禁。"))}</p>
      <div class="row-meta">
        <span>签收人 ${esc(latest.signatory_name || "-")}</span>
        <span>职务 ${esc(latest.signatory_role || "-")}</span>
        <span>组织 ${esc(latest.organization || "-")}</span>
        <span>记录 ${esc(payload.signoff_count ?? signoffs.length ?? 0)}</span>
        <span>风险确认 ${esc(latest.risk_acknowledgement ? "已确认" : "未确认")}</span>
        <span>凭证 ${esc(latest.credential_ref || "-")}</span>
        <span>凭证哈希 ${esc((latest.credential_sha256 || "").slice(0, 12) || "-")}</span>
        <span>记录校验 ${esc(latest.integrity_valid === false ? "失败" : latest.signoff_payload_sha256 ? "通过" : "未生成")}</span>
      </div>
      ${compact ? "" : `
        <div class="capability-list compact-list">
          ${signoffs.slice(0, 8).map((item) => `
            <div class="row-item">
              <strong>${esc(item.signatory_name || "客户")} ${badge(item.decision || "unknown", acceptanceGateClass(item.decision))}</strong>
              <p>${esc(item.reason || "-")}</p>
              <span>${esc(item.signatory_role || "-")} / ${esc(item.organization || "-")} / ${esc(item.credential_ref || "-")} / ${esc(item.signoff_id || "-")}</span>
            </div>
          `).join("") || `<div class="mini-list-empty">还没有客户签收记录。</div>`}
        </div>
      `}
    </div>
  `;
}

function renderCustomerProjectExecutionBindings(payload = {}) {
  if (!payload.found) {
    return `<div class="project-import-card warn"><strong>执行接入计划不可用 ${badge("blocked", "err")}</strong><p>${esc(payload.reason || payload.error || "unknown")}</p></div>`;
  }
  const summary = payload.summary || {};
  const plans = Array.isArray(payload.plans) ? payload.plans : [];
  const projectScope = payload.project_scope || {};
  const projectIdentifier = projectScope.project_id || payload.customer?.project_id || payload.site?.site_id || "";
  return `
    <div class="project-import-card ${acceptanceGateClass(summary.overall_status)}">
      <strong>执行接入计划 ${badge(summary.overall_status || "unknown", acceptanceGateClass(summary.overall_status))}</strong>
      <p>${esc(payload.customer_claim || "对象绑定已生成接入计划。")}</p>
      <div class="row-meta">
        <span>对象 ${esc(summary.object_count || 0)}</span>
        <span>可接入 ${esc(summary.ready_object_count || 0)}</span>
        <span>需复核 ${esc(summary.manual_check_object_count || 0)}</span>
        <span>阻断 ${esc(summary.blocked_object_count || 0)}</span>
      </div>
      <div class="capability-list compact-list">
        ${plans.map((plan) => `
          <div class="row-item">
            <strong>${esc(plan.display_name || plan.object_id)} ${badge(plan.overall_status || "unknown", acceptanceGateClass(plan.overall_status))}</strong>
            <div class="row-actions">
              <button class="primary-button mini-button" data-object-rehearsal="dry_run" data-project-id="${esc(projectIdentifier)}" data-object-id="${esc(plan.object_id || "")}">接入演练</button>
              <button class="ghost-button mini-button" data-object-rehearsal="shadow_post" data-project-id="${esc(projectIdentifier)}" data-object-id="${esc(plan.object_id || "")}">实验室投递</button>
            </div>
            ${renderExecutionAdapterContracts(plan)}
            <p>${esc(plan.customer_status || "-")}</p>
            ${renderExecutionScopeConstraints(plan.scope_constraints)}
            <span>输入 ${esc((plan.required_sources || []).join(", ") || "-")} / 技能 ${esc(((plan.skill_routes || []).map((item) => `${item.capability || item.resource_id || "-"} · ${item.safety_level || "unknown"} · ${item.approval_policy || "-"}`)).join(", ") || "-")}</span>
            <span>工具 ${esc(((plan.skill_routes || []).map((item) => item.tool || "-")).join(", ") || "-")} / 输出 ${esc(((plan.skill_routes || []).map((item) => item.output_contract || "-")).join(", ") || "-")}</span>
            <span>入口 ${esc(plan.ingest_contract?.endpoint || "-")} / 回传 ${esc(plan.runtime_contract?.callback_endpoint || "-")}</span>
            <span>边界 ${esc(((plan.skill_routes || []).map((item) => item.hardware_boundary || item.safety_boundary || "-")).join("；") || "-")}</span>
            ${(plan.blockers || []).length ? `<p class="error-text">阻断：${esc((plan.blockers || []).join("；"))}</p>` : ""}
            ${(plan.manual_checks || []).length ? `<p class="muted-line">复核：${esc((plan.manual_checks || []).join("；"))}</p>` : ""}
          </div>
        `).join("") || `<div class="mini-list-empty">还没有执行接入计划。</div>`}
      </div>
      <div id="project-execution-rehearsal-result" class="capability-list compact-list"></div>
      <p class="muted-line">${esc(payload.next_step || "接入真实设备数据，并记录执行服务回传。")}</p>
    </div>
  `;
}

function renderExecutionScopeConstraints(scope = {}) {
  if (!scope || typeof scope !== "object") return "";
  const parts = [
    ["客户空间", scope.tenant_ids],
    ["交付空间", scope.delivery_namespaces],
    ["客户", scope.customer_ids],
    ["项目", scope.project_ids],
    ["现场", scope.site_ids],
  ].map(([label, values]) => `${label}:${Array.isArray(values) && values.length ? values.join("|") : "*"}`);
  return `<span>范围约束 ${esc(parts.join(" / "))}</span>`;
}

function renderExecutionAdapterContracts(plan = {}) {
  const adapters = Array.isArray(plan.input_adapters) ? plan.input_adapters : [];
  const sourcePlans = Array.isArray(plan.source_plans) ? plan.source_plans : [];
  const adapterRows = adapters.map((adapter) => {
    const contract = adapter.adapter_contract || {};
    const bridgeName = contract.bridge || "field-ingest-bridge";
    return `
      <span>设备接入 ${esc(adapter.protocol_id || "-")} / ${esc(adapter.adapter || "-")} / ${esc(adapter.status || "-")}</span>
      <span>接入桥 ${esc(bridgeName)} / 签名 ${esc(contract.device_signature_required ? "需要" : "不需要")} / 密钥 ${esc((contract.device_secret_envs || []).join(", ") || "-")}</span>
      <span>Dry-run ${esc(contract.dry_run_command || "-")}</span>
      <span>Live ${esc(contract.live_command || "-")}</span>
    `;
  }).join("");
  const deviceRows = sourcePlans.map((source) => {
    const devices = Array.isArray(source.devices) ? source.devices : [];
    const names = devices.map((device) => `${device.device_id || device.name || "-"}:${device.zone_id || "-"}`).join(", ");
    return `<span>设备 ${esc(source.source || "-")} ${esc(source.status || "-")} / ${esc(source.device_count || 0)} / ${esc(names || "-")}</span>`;
  }).join("");
  const bridge = plan.bridge_contract || {};
  const bridgeRow = bridge.bridge ? `<span>接入合同 ${esc(bridge.bridge)} / ${esc(bridge.ingest_endpoint || "-")} / 签收需现场上报 ${esc(bridge.live_post_required_for_customer_signoff ? "是" : "否")}</span>` : "";
  return `${adapterRows}${deviceRows}${bridgeRow}`;
}

function renderObjectExecutionRehearsalResult(payload = {}) {
  const rehearsal = payload.rehearsal || {};
  const normalized = payload.normalized || {};
  const plan = payload.plan || {};
  const ingest = payload.ingest_result || {};
  const onsite = payload.onsite_evidence_registration || {};
  const statusClass = payload.accepted ? "ok" : payload.status === "manual_check" ? "warn" : "err";
  return `
    <div class="row-item">
      <strong>对象接入演练 ${badge(payload.status || "unknown", statusClass)}</strong>
      <p>${esc(payload.customer_status || rehearsal.customer_status || "实验室演示证据，不能作为生产上线验收依据。")}</p>
      ${renderEvidenceBoundaryTags({ ...rehearsal, ...payload })}
      <span>对象 ${esc(payload.object_id || plan.object_id || "-")} / 模式 ${esc(rehearsal.mode || "-")} / 能力 ${esc(((plan.skill_routes || []).map((item) => item.capability || item.resource_id || "-")).join(", ") || "-")}</span>
      <span>归一化 ${esc(normalized.source || "-")} / ${esc(normalized.scenario_id || "-")} / ${esc(normalized.zone_id || "-")} / ${esc(normalized.managed_object_id || "-")}</span>
      <span>生产声明 ${esc(payload.production_claim_allowed ? "allowed" : "not allowed")} / ${esc(payload.reason || ingest.reason || "adapter rehearsal completed")}</span>
      ${onsite.requested ? `
        <div class="row-meta">
          <span>\u9a8c\u6536\u8bc1\u636e\u767b\u8bb0 ${esc(onsite.registered ? "\u5df2\u767b\u8bb0" : "\u672a\u767b\u8bb0")}</span>
          <span>${esc(onsite.status || onsite.reason || "manual_check")}</span>
          <span>${esc(onsite.customer_status || "\u4ec5\u4f5c\u4e3a\u9a8c\u6536\u5019\u9009\uff0c\u4e0d\u4ee3\u8868\u751f\u4ea7\u4e0a\u7ebf")}</span>
        </div>
        ${renderEvidenceBoundaryTags(onsite)}
      ` : ""}
      <div class="mono">${esc(JSON.stringify({
        normalized,
        event_id: payload.event_id || ingest.event_id || "",
        onsite_evidence_registration: onsite,
        release_claim: payload.release_claim || rehearsal.release_claim || "",
      }, null, 2))}</div>
    </div>
  `;
}

function renderCustomerProjectAcceptanceClosure(payload = {}) {
  if (!payload.found && !payload.accepted) {
    return `<div class="project-import-card warn"><strong>验收闭环不可用 ${badge("blocked", "err")}</strong><p>${esc(payload.reason || payload.error || "unknown")}</p></div>`;
  }
  const gates = Array.isArray(payload.gates) ? payload.gates : [];
  const timeline = Array.isArray(payload.evidence_timeline) ? payload.evidence_timeline : [];
  const manual = payload.manual_review || {};
  const latest = manual.latest || {};
  const signoff = payload.customer_signoff || {};
  const latestSignoff = signoff.latest || {};
  const checklist = payload.site_acceptance_checklist || {};
  return `
    <div class="project-import-card ${acceptanceGateClass(payload.overall_status) || "manual_check"}">
      <strong>验收闭环 ${badge(payload.overall_status || "manual_check", acceptanceGateClass(payload.overall_status))}</strong>
      <p>${esc(payload.customer_claim || "-")}</p>
      <div class="row-meta">
        <span>内部复核 ${esc(latest.decision || "未提交")}</span>
        <span>复核人 ${esc(latest.operator_id || "-")}</span>
        <span>客户签收 ${esc(latestSignoff.decision || "未提交")}</span>
        <span>签收人 ${esc(latestSignoff.signatory_name || "-")}</span>
        <span>时间线 ${esc(timeline.length)}</span>
      </div>
      ${renderSiteAcceptanceChecklist(checklist)}
      ${renderCustomerProjectCustomerSignoff(signoff, true)}
      <div class="capability-list compact-list">
        ${gates.map((gate) => `
          <div class="row-item">
            <strong>${esc(gate.label || gate.gate_id || "gate")} ${badge(gate.status || "unknown", acceptanceGateClass(gate.status))}</strong>
            <p>${esc(gate.evidence || "-")}</p>
            <span>${esc(gate.next_step || "-")}</span>
          </div>
        `).join("")}
      </div>
      <div class="capability-list compact-list">
        ${timeline.slice(0, 8).map((item) => `
          <div class="row-item">
            <strong>${esc(item.label || item.type || "evidence")} ${badge(item.status || "unknown", acceptanceGateClass(item.status))}</strong>
            <span>${esc(item.summary || item.ref || "-")}</span>
          </div>
        `).join("") || `<div class="mini-list-empty">还没有验收闭环记录。</div>`}
      </div>
      <p class="muted-line">${esc(payload.next_step || "")}</p>
    </div>
  `;
}

function renderCustomerProjectAcceptanceDossier(payload = {}) {
  if (!payload.accepted) {
    return `<div class="project-import-card warn"><strong>验收材料不可用 ${badge("blocked", "err")}</strong><p>${esc(payload.reason || payload.error || "unknown")}</p></div>`;
  }
  const dossier = payload.dossier || {};
  const manifest = dossier.manifest || {};
  const workflow = dossier.delivery_workflow || {};
  const evidence = Array.isArray(dossier.evidence_inventory) ? dossier.evidence_inventory : [];
  return `
    <div class="project-import-card ${dossier.overall_status === "blocked" ? "warn" : "ok"}">
      <strong>客户验收材料 ${badge(dossier.overall_status || "unknown", acceptanceGateClass(dossier.overall_status))}</strong>
      <p>JSON: ${esc(payload.dossier_path || "-")}</p>
      <p>Printable HTML: ${esc(payload.html_path || "-")}</p>
      <div class="row-meta">
        <span>项目：${esc(manifest.project_id || "-")}</span>
        <span>现场就绪：${esc(manifest.field_readiness_status || "-")}</span>
        <span>现场证据：${esc(manifest.onsite_evidence_status || "-")}</span>
        <span>证据数：${esc(manifest.evidence_count ?? evidence.length)}</span>
        <span>sha256: ${esc(String(manifest.payload_sha256 || "").slice(0, 16))}</span>
        <span>签名：${esc(manifest.signature_alg || "unsigned")}</span>
        <span>HTML：${esc(payload.html_path ? "已生成" : "缺失")}</span>
      </div>
      ${renderProjectDeliveryWorkflow(workflow)}
      <div class="capability-list compact-list">
        ${evidence.slice(0, 8).map((item) => `
          <div class="row-item">
            <strong>${esc(item.path || "证据")} ${badge(item.exists ? "已校验" : "缺失", item.exists ? "ok" : "err")}</strong>
            <span>${esc(String(item.sha256 || "").slice(0, 16))} / ${esc(item.size_bytes ?? 0)} bytes</span>
          </div>
        `).join("")}
      </div>
      <p class="muted-line">${esc(dossier.handoff_boundary || dossier.release_claim || "")}</p>
    </div>
  `;
}

function renderCustomerProjectProposalBundle(payload = {}) {
  if (!payload.accepted) {
    return `<div class="project-import-card warn"><strong>客户方案包不可用 ${badge("blocked", "err")}</strong><p>${esc(payload.reason || payload.error || "unknown")}</p></div>`;
  }
  const proposal = payload.proposal || {};
  const manifest = proposal.manifest || {};
  const projectPackage = proposal.customer_project_package || {};
  const dossier = proposal.acceptance_dossier || {};
  const releaseBundle = proposal.approved_template_release_bundle || {};
  const releaseSummary = releaseBundle.summary || {};
  const proposalInsert = proposal.proposal_insert || {};
  return `
    <div class="project-import-card ok">
      <strong>客户方案包 ${badge("已生成", "ok")}</strong>
      <p>JSON: ${esc(payload.proposal_path || "-")}</p>
      <p>Printable HTML: ${esc(payload.html_path || "-")}</p>
      <div class="row-meta">
        <span>项目：${esc(manifest.project_id || "-")}</span>
        <span>交付包：${esc(String(manifest.package_sha256 || "").slice(0, 16))}</span>
        <span>验收材料：${esc(String(manifest.dossier_sha256 || "").slice(0, 16))}</span>
        <span>已审批模板：${esc(releaseSummary.approved_release_count ?? 0)}</span>
      </div>
      <p class="muted-line">${esc(proposalInsert.section_title || "")}</p>
      <p class="muted-line">${esc(proposal.delivery_boundary || "")}</p>
      <div class="mono">${esc(JSON.stringify({
        manifest,
        package_manifest: projectPackage.manifest || {},
        dossier_manifest: dossier.manifest || {},
      }, null, 2))}</div>
    </div>
  `;
}

function renderProjectExportResult(payload = {}) {
  if (!payload.accepted) {
    return `<div class="project-import-card warn"><strong>交付包导出失败 ${badge("失败", "err")}</strong><p>${esc(payload.reason || payload.error || "unknown")}</p></div>`;
  }
  const manifest = payload.package?.manifest || {};
  const reuse = payload.package?.reuse_assessment || {};
  const bindingReadiness = payload.package?.binding_readiness_summary || {};
  const deliveryGate = payload.package?.package_delivery_gate || {};
  const acceptanceStatus = manifest.acceptance_overall_status || payload.package?.acceptance_summary?.overall_status || "unknown";
  const bindingStatus = manifest.resource_binding_overall_status || bindingReadiness.overall_status || "unknown";
  return `
    <div class="project-import-card ok">
      <strong>客户项目交付包已生成 ${badge("完整性通过", "ok")}</strong>
      <p>${esc(payload.package_path || "-")}</p>
      <div class="row-meta">
        <span>客户空间：${esc(manifest.tenant_id || "default")}</span>
        <span>交付空间：${esc(manifest.delivery_namespace || "default")}</span>
        <span>客户：${esc(manifest.customer_id || "-")}</span>
        <span>项目：${esc(manifest.project_id || "-")}</span>
        <span>对象：${esc(manifest.managed_object_count ?? "-")}</span>
        ${badge(`验收 ${acceptanceStatus}`, acceptanceGateClass(acceptanceStatus))}
        ${badge(`资源 ${bindingStatus}`, acceptanceGateClass(bindingStatus))}
        <span>未登记资源：${esc(manifest.resource_binding_unregistered_resource_count ?? bindingReadiness.unregistered_resource_count ?? 0)}</span>
        <span>已登记资源：${esc(manifest.delivery_resource_count ?? "-")}</span>
        <span>sha256: ${esc(String(manifest.payload_sha256 || "").slice(0, 16))}</span>
      </div>
      ${renderProjectPackageDeliveryGate(deliveryGate, "交付包准入")}
      ${renderProjectBindingReadiness(bindingReadiness, "交付资源绑定")}
      ${renderProjectReuseAssessment(reuse, "复用就绪度")}
    </div>
  `;
}

function renderProjectBindingReadiness(payload = {}, title = "Resource bindings") {
  if (!payload || typeof payload !== "object" || !payload.overall_status) return "";
  const status = payload.overall_status || "unknown";
  const unregistered = Array.isArray(payload.unregistered_resources) ? payload.unregistered_resources : [];
  return `
    <div class="project-reuse-assessment ${acceptanceGateClass(status)}">
      <strong>${esc(title)} ${badge(status, acceptanceGateClass(status))}</strong>
      <p>${esc(payload.object_count ?? 0)} 个对象，${esc(payload.unregistered_resource_count ?? 0)} 个资源未登记。</p>
      <div class="row-meta">
        <span>就绪 ${esc(payload.ready_object_count ?? 0)}</span>
        <span>待复核 ${esc(payload.manual_check_object_count ?? 0)}</span>
        <span>阻断 ${esc(payload.blocked_object_count ?? 0)}</span>
      </div>
      ${unregistered.length ? `<div class="skill-validation">${unregistered.slice(0, 8).map((item) => `<span class="warn">${esc(item.object_id || "-")} / ${esc(item.resource_type || "-")} / ${esc(item.resource_id || "-")}</span>`).join("")}</div>` : ""}
    </div>
  `;
}

function renderProjectPackageDeliveryGate(payload = {}, title = "交付包准入") {
  if (!payload || typeof payload !== "object" || !payload.delivery_gate_status) return "";
  const status = payload.delivery_gate_status || "unknown";
  const reasons = Array.isArray(payload.delivery_gate_reasons) ? payload.delivery_gate_reasons : [];
  return `
    <div class="project-reuse-assessment project-package-delivery-gate ${acceptanceGateClass(status)}" data-project-package-delivery-gate>
      <strong>${esc(title)} ${badge(status, acceptanceGateClass(status))}</strong>
      <p>${esc(payload.customer_status || payload.next_step || "")}</p>
      <div class="row-meta">
        <span>可客户交接：${esc(payload.customer_handoff_ready === true ? "是" : "否")}</span>
        <span>允许导入：${esc(payload.import_allowed === false ? "否" : "是")}</span>
        <span>动作 ${esc(payload.action_count ?? reasons.length)}</span>
        <span>阻断 ${esc(payload.blocked_action_count ?? 0)}</span>
        <span>待复核 ${esc(payload.manual_check_action_count ?? 0)}</span>
      </div>
      ${reasons.length ? `
        <div class="skill-validation">
          ${reasons.slice(0, 8).map((item) => `
            <span class="${item.severity === "blocked" ? "err" : "warn"}">
              ${esc(item.object_id || item.source || "package")} / ${esc(item.reason_code || "-")} / ${esc(item.next_step || item.reason_label || "")}
            </span>
          `).join("")}
        </div>
      ` : ""}
    </div>
  `;
}

function renderProjectReuseAssessment(payload = {}, title = "复用就绪度") {
  if (!payload || typeof payload !== "object" || !payload.status) return "";
  const deps = payload.dependencies || {};
  const blockers = Array.isArray(payload.blockers) ? payload.blockers : [];
  const manualChecks = Array.isArray(payload.manual_checks) ? payload.manual_checks : [];
  const depText = [
    `${deps.device_count ?? 0} 台设备`,
    `${(deps.device_sources || []).length} 类数据源`,
    `${(deps.vision_models || []).length} 个视觉模型`,
    `${(deps.sensor_protocols || []).length} 个传感器协议`,
    `${(deps.skill_packages || []).length} 个能力包`,
    `${deps.missing_env_count ?? 0} 个缺失环境项`,
  ].join(" / ");
  return `
    <div class="project-reuse-assessment ${acceptanceGateClass(payload.status)}">
      <strong>${esc(title)} ${badge(payload.status, acceptanceGateClass(payload.status))}</strong>
      <p>${esc(payload.customer_status || payload.next_step || "")}</p>
      <div class="row-meta"><span>${esc(depText)}</span><span>阻断 ${esc(payload.blocker_count ?? blockers.length)}</span><span>人工复核 ${esc(payload.manual_check_count ?? manualChecks.length)}</span></div>
      ${(blockers.length || manualChecks.length) ? `<div class="skill-validation">${[...blockers, ...manualChecks].slice(0, 6).map((item) => `<span class="${blockers.includes(item) ? "err" : "warn"}">${esc(item)}</span>`).join("")}</div>` : ""}
    </div>
  `;
}

async function loadSelectedCustomerProjectHistory() {
  const resultEl = document.getElementById("project-lifecycle-result");
  const identifier = document.getElementById("project-lifecycle-id")?.value || "";
  if (!identifier) {
    if (resultEl) resultEl.textContent = "请先选择客户项目。";
    return;
  }
  const payload = await getJson(`${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/history?limit=12`, { found: false, reason: "request_failed" });
  if (resultEl) resultEl.innerHTML = renderProjectRevisionHistory(payload);
}

function renderProjectRevisionHistory(payload = {}) {
  if (!payload.found) {
    return `<div class="project-import-card warn"><strong>修订记录不可用 ${badge("阻断", "err")}</strong><p>${esc(payload.reason || payload.error || "unknown")}</p></div>`;
  }
  const revisions = Array.isArray(payload.revisions) ? payload.revisions : [];
  return `
    <div class="project-import-card ${revisions.length ? "ok" : "warn"}">
      <strong>项目修订记录 ${badge(`${revisions.length} 条`, revisions.length ? "ok" : "warn")}</strong>
      <p>${esc(payload.next_step || "")}</p>
      <div class="capability-list compact-list">
        ${revisions.map((item) => `
          <div class="row-item">
            <strong>${esc(item.action || "revision")} ${badge(String(item.revision_id || "").slice(0, 18), "warn")}</strong>
            <p>${esc(item.reason || "-")}</p>
            <div class="row-meta">
              <span>${esc(item.operator_id || "system")}</span>
              <span>${esc(new Date(Number(item.created_at || 0) * 1000).toLocaleString())}</span>
              <span>sha ${esc(String(item.profile_sha256 || "").slice(0, 16))}</span>
            </div>
          </div>
        `).join("") || `<div class="mini-list-empty">暂无修订记录。请先保存项目或变更现场对象。</div>`}
      </div>
    </div>
  `;
}

async function rollbackSelectedCustomerProject(dryRun = true) {
  const resultEl = document.getElementById("project-lifecycle-result");
  const identifier = document.getElementById("project-lifecycle-id")?.value || "";
  const revisionId = document.getElementById("project-rollback-revision")?.value || "";
  const reason = document.getElementById("project-rollback-reason")?.value || "";
  if (!identifier || !revisionId) {
    if (resultEl) resultEl.textContent = "Select a project and enter a revision_id before rollback.";
    return;
  }
  if (!dryRun && !window.confirm(`确认将客户项目 ${identifier} 回滚到修订 ${revisionId}？`)) return;
  const response = await postJson(`${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/rollback`, {
    operator_id: operatorId(),
    revision_id: revisionId,
    reason: reason || "从客户项目控制台发起回滚。",
    dry_run: dryRun,
  });
  if (resultEl) resultEl.innerHTML = renderProjectRollbackResult(response.payload, response.ok);
  if (response.ok && !dryRun) await refreshProjectSurface();
}

function renderProjectRollbackResult(payload = {}, ok = false) {
  if (!ok || !payload.accepted) {
    return `<div class="project-import-card warn"><strong>暂不能回滚 ${badge("阻断", "err")}</strong><p>${esc(payload.reason || payload.error || "unknown")}</p></div>`;
  }
  const revision = payload.revision || {};
  const changes = Array.isArray(payload.field_changes) ? payload.field_changes : [];
  return `
    <div class="project-import-card ${payload.dry_run ? "warn" : "ok"}">
      <strong>${payload.dry_run ? "回滚预检" : "已执行回滚"} ${badge(revision.revision_id || "revision", payload.dry_run ? "warn" : "ok")}</strong>
      <p>${esc(payload.next_step || "复核变更并重新执行验收检查。")}</p>
      <div class="row-meta">
        <span>变更 ${esc(changes.length)}</span>
        <span>当前 ${esc(String(payload.current_profile_sha256 || "").slice(0, 16))}</span>
        <span>目标 ${esc(String(payload.target_profile_sha256 || "").slice(0, 16))}</span>
      </div>
      <div class="capability-list compact-list">
        ${changes.slice(0, 8).map((item) => `
          <div class="row-item">
            <strong>${esc(item.path || item.field || "field")} ${badge(item.change || item.type || "changed", "warn")}</strong>
            <span>${esc(JSON.stringify(item).slice(0, 180))}</span>
          </div>
        `).join("") || `<div class="mini-list-empty">没有检测到字段变更。</div>`}
      </div>
    </div>
  `;
}

async function archiveSelectedCustomerProject() {
  const resultEl = document.getElementById("project-lifecycle-result");
  const identifier = document.getElementById("project-lifecycle-id")?.value || "";
  if (!identifier) {
    if (resultEl) resultEl.textContent = "请选择要归档的客户项目。";
    return;
  }
  if (!window.confirm(`确认归档客户项目 ${identifier}？归档后不会永久删除，但会从当前项目目录移走。`)) return;
  const response = await postJson(`${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/archive`, {
    operator_id: operatorId(),
    reason: "archive from customer project console",
  });
  if (resultEl) {
    resultEl.innerHTML = response.ok
      ? `<div class="project-import-card ok"><strong>项目已归档 ${badge("archived", "ok")}</strong><p>${esc(response.payload.archived_path || "-")}</p></div>`
      : `<div class="project-import-card warn"><strong>归档失败 ${badge("blocked", "err")}</strong><p>${esc(response.payload.reason || response.payload.error || "unknown")}</p></div>`;
  }
  if (response.ok) await refreshProjectSurface();
}

async function deleteManagedObjectFromForm() {
  const resultEl = document.getElementById("object-delete-result");
  const pair = document.getElementById("object-delete-pair")?.value || "";
  const [identifier, objectId] = pair.split("::");
  const reason = (document.getElementById("object-delete-reason")?.value || "").trim();
  if (!reason) {
    if (resultEl) resultEl.textContent = "移除现场对象前必须填写下线原因。";
    return;
  }
  if (!identifier || !objectId) {
    if (resultEl) resultEl.textContent = "请选择要下线的对象。";
    return;
  }
  if (!window.confirm(`确认删除对象 ${objectId}？后续事件将不能再归属到这个对象。`)) return;
  const response = await deleteJson(`${ENDPOINTS.fieldCustomerProjects}/${encodeURIComponent(identifier)}/managed-objects/${encodeURIComponent(objectId)}`, {
    operator_id: operatorId(),
    reason,
  });
  if (resultEl) {
    resultEl.innerHTML = renderManagedObjectWriteResult(response.payload, response.ok, "对象已下线");
  }
  if (response.ok) await refreshProjectSurface();
}

function renderSiteProfileCatalogItem(site = {}) {
  const summary = site.summary || {};
  const managedObjects = site.managed_objects_summary || {};
  return `
    <div class="capability-item">
      <div>
        <strong>${esc(site.site_name || site.site_id || "未命名现场")}</strong>
        <p>${esc(site.customer_status || "Site profile status unknown.")}</p>
        <div class="row-meta">
          <span>${esc(site.site_id || "-")}</span>
          <span>区域 ${esc(summary.zone_count ?? 0)}</span>
          <span>设备 ${esc(summary.device_count ?? 0)}</span>
          <span>问询点 ${esc(summary.help_point_count ?? 0)}</span>
          <span>对象 ${esc(managedObjects.object_type_count ?? 0)}</span>
          <span>缺少配置 ${esc(site.env_missing_count ?? 0)}</span>
        </div>
        <p class="small-note">下一步：${esc(site.next_step || "-")}</p>
      </div>
      <div class="capability-badges">
        ${badge(site.deployment_stage || "unknown", site.deployment_stage === "production_ready" ? "ok" : site.status === "passed" ? "warn" : "err")}
        ${badge(site.status || "unknown", site.status === "passed" ? "ok" : "err")}
      </div>
    </div>
  `;
}

function getAuditWindow() {
  return {
    since: localStorage.getItem("askme.audit_since") || "",
    until: localStorage.getItem("askme.audit_until") || "",
    project_id: localStorage.getItem("askme.audit_project_id") || "",
    managed_object_id: localStorage.getItem("askme.audit_managed_object_id") || "",
  };
}

function auditWindowToQuery(window = {}) {
  const parts = [];
  if (window.since) parts.push(`since=${encodeURIComponent(window.since)}`);
  if (window.until) parts.push(`until=${encodeURIComponent(window.until)}`);
  if (window.project_id) parts.push(`project_id=${encodeURIComponent(window.project_id)}`);
  if (window.managed_object_id) {
    parts.push(`managed_object_id=${encodeURIComponent(window.managed_object_id)}`);
  }
  return parts.length ? `&${parts.join("&")}` : "";
}

function auditRecordsForReview(payload = {}, reviews = {}) {
  const byId = new Map();
  [...(payload.review_queue || []), ...(payload.records || [])].forEach((record) => {
    if (record?.record_id) byId.set(record.record_id, record);
  });
  const history = auditReviewHistoryByRecord(reviews);
  byId.forEach((record, recordId) => {
    record.review_history = history[recordId] || [];
  });
  return [...byId.values()];
}

function auditReviewHistoryByRecord(payload = {}) {
  const result = {};
  (payload.records || []).forEach((item) => {
    const recordId = item?.record_id || "";
    if (!recordId) return;
    result[recordId] = result[recordId] || [];
    result[recordId].push(item);
  });
  return result;
}

function renderAuditWindowControls(window = {}, customerProjects = {}) {
  const projects = Array.isArray(customerProjects.projects) ? customerProjects.projects : [];
  const projectOptions = projects.map((project) => {
    const id = project.project_id || "";
    if (!id) return "";
    const label = `${project.customer_name || project.customer_id || "客户"} / ${project.project_name || id}`;
    return `<option value="${esc(id)}" ${window.project_id === id ? "selected" : ""}>${esc(label)}</option>`;
  }).join("");
  const objects = projects.flatMap((project) => (
    Array.isArray(project.managed_objects)
      ? project.managed_objects.map((item) => ({ ...item, project_id: project.project_id || "" }))
      : []
  ));
  const objectOptions = objects
    .filter((item) => !window.project_id || item.project_id === window.project_id)
    .map((item) => {
      const id = item.object_id || "";
      if (!id) return "";
      const label = `${item.display_name || id}${item.object_type ? ` / ${item.object_type}` : ""}`;
      return `<option value="${esc(id)}" ${window.managed_object_id === id ? "selected" : ""}>${esc(label)}</option>`;
    })
    .join("");
  return `
    <div class="audit-window">
      <div>
        <strong>审计范围</strong>
        <p>生成验收证据前，先按客户项目、现场对象和时间范围筛选。</p>
      </div>
      <div style="display:none">
        <strong>验收证据时间范围</strong>
        <p>用于按试点周期、客户验收日或事故复盘周期筛选审计，并按同一范围生成审计包。</p>
      </div>
      <select id="audit-project">
        <option value="">全部客户项目</option>
        ${projectOptions}
      </select>
      <select id="audit-object">
        <option value="">全部对象</option>
        ${objectOptions}
      </select>
      <input id="audit-since" placeholder="开始时间，例如 2026-05-14T09:00:00+08:00 或 1715658000" value="${esc(window.since || "")}">
      <input id="audit-until" placeholder="结束时间" value="${esc(window.until || "")}">
      <button class="ghost-button" data-audit-window="apply">应用</button>
      <button class="ghost-button" data-audit-window="clear">清空</button>
    </div>
  `;
}

function renderAuditProductSummary(payload = {}) {
  const product = payload.product_summary || {};
  const summary = payload.summary || {};
  const report = payload.customer_report || {};
  const dossier = payload.delivery_dossier || {};
  const integrity = product.integrity || {};
  const reviewQueue = Array.isArray(payload.review_queue) ? payload.review_queue : [];
  const status = product.status || "unknown";
  return `
    ${report.handoff_brief ? `
      <div class="audit-handoff-brief ${report.customer_ready ? "ok" : "warn"}">
        <strong>${esc(report.handoff_brief.claim || report.status_label || "审计交付状态")}</strong>
        <p>${esc(report.handoff_brief.customer_message || report.summary_sentence || "")}</p>
        <div class="row-meta">
          <span>负责人：${esc(report.handoff_brief.delivery_owner || "-")}</span>
          <span>下一步：${esc(report.handoff_brief.next_step || "-")}</span>
        </div>
      </div>
    ` : ""}
    ${renderAuditDeliveryDossier(dossier)}
    ${renderAuditAcceptanceChecklist(report.acceptance_checklist || [])}
    <div class="audit-summary-grid">
      <div><b>${esc(product.record_count ?? payload.filtered_total ?? 0)}</b><span>范围内记录</span></div>
      <div class="${Number(product.requires_review_count || 0) ? "warn" : "ok"}"><b>${esc(product.requires_review_count ?? summary.requires_review_count ?? 0)}</b><span>需要复核</span></div>
      <div class="${Number(product.high_or_critical_count || 0) ? "warn" : "ok"}"><b>${esc(product.high_or_critical_count ?? 0)}</b><span>高风险/关键</span></div>
      <div><b>${esc(integrity.signed_record_count ?? 0)}</b><span>签名记录</span></div>
      <div><b>${esc(integrity.hash_chained_record_count ?? 0)}</b><span>哈希链记录</span></div>
      <div class="${status === "needs_review" ? "warn" : "ok"}"><b>${esc(status)}</b><span>${esc(product.customer_status || "audit status")}</span></div>
    </div>
    ${reviewQueue.length ? renderAuditReviewQueue(reviewQueue) : ""}
  `;
}

function renderAuditDeliveryDossier(dossier = {}) {
  const safeDossier = dossier && typeof dossier === "object" ? dossier : {};
  const allowed = Array.isArray(safeDossier.allowed_uses) ? safeDossier.allowed_uses : [];
  const blocked = Array.isArray(safeDossier.blocked_uses) && safeDossier.blocked_uses.length
    ? safeDossier.blocked_uses
    : [
      "无人值守生产上线声明",
      "替代现场验收结论",
      "替代安全负责人或客户负责人签收",
    ];
  const mustFix = Array.isArray(safeDossier.must_fix) ? safeDossier.must_fix : [];
  const watchItems = Array.isArray(safeDossier.watch_items) ? safeDossier.watch_items : [];
  const scope = safeDossier.record_scope || {};
  return `
    <div class="audit-delivery-dossier ${safeDossier.decision === "ready" ? "ok" : "warn"}">
      <div class="section-title-row compact">
        <div>
          <h3>${esc(safeDossier.title || "客户交付审计材料")}</h3>
          <p>${esc(safeDossier.customer_claim || "The audit package explains the operation scope, evidence, and review status.")}</p>
        </div>
        ${badge(safeDossier.decision_label || safeDossier.decision || "pending", safeDossier.decision === "ready" ? "ok" : "warn")}
      </div>
      <div class="grid three">
        <div>
          <strong>允许用途</strong>
          ${renderAuditTextList(allowed, "暂未声明允许用途")}
        </div>
        <div>
          <strong>禁止声明</strong>
          ${renderAuditTextList(blocked, "暂未声明禁止项")}
        </div>
        <div>
          <strong>范围指标</strong>
          <p>${esc(scope.record_count ?? 0)} 条记录 / ${esc(scope.high_or_critical_count ?? 0)} 条高风险 / ${esc(scope.evidence_linked_count ?? 0)} 条证据关联</p>
          <p>负责人：${esc(safeDossier.handoff_owner || "-")}</p>
          <p>下一步：${esc(safeDossier.next_step || "-")}</p>
        </div>
      </div>
      ${(mustFix.length || watchItems.length) ? `
        <div class="row-meta">
          ${mustFix.length ? `<span>Blockers: ${esc(mustFix.join(" / "))}</span>` : ""}
          ${watchItems.length ? `<span>Watch items: ${esc(watchItems.join(" / "))}</span>` : ""}
        </div>
      ` : ""}
    </div>
  `;
}

function renderAuditTextList(items = [], emptyText = "暂无") {
  if (!items.length) return `<p>${esc(emptyText)}</p>`;
  return `
    <ul>
      ${items.slice(0, 6).map((item) => `<li>${esc(item)}</li>`).join("")}
    </ul>
  `;
}

function renderAuditAcceptanceChecklist(items = []) {
  if (!Array.isArray(items) || !items.length) return "";
  return `
    <div class="audit-checklist">
      ${items.map((item) => `
        <div class="audit-check-item ${item.status === "passed" ? "ok" : item.status === "blocked" ? "warn" : ""}">
          <b>${esc(item.label || item.id || "-")}</b>
          ${badge(item.status || "-", item.status === "passed" ? "ok" : item.status === "blocked" ? "warn" : "")}
          <span>${esc(item.detail || "")}</span>
          <small>${esc(item.next_step || "")}</small>
        </div>
      `).join("")}
    </div>
  `;
}

function renderAuditReviewIntegrity(payload = {}) {
  const integrity = payload.review_integrity || {};
  const valid = integrity.valid !== false;
  const failures = Array.isArray(integrity.failures) ? integrity.failures : [];
  return `
    <div class="audit-integrity-strip ${valid ? "ok" : "warn"}">
      <div>
        <strong>复核日志完整性 ${badge(valid ? "有效" : "异常", valid ? "ok" : "err")}</strong>
        <p>${valid ? "复核决定采用追加式哈希链记录，可作为交付证据。" : "复核日志存在篡改、缺行或哈希不一致，交付门禁会阻断。"}</p>
      </div>
      <div class="row-meta">
        <span>${esc(integrity.path || "artifacts/audit/reviews.jsonl")}</span>
        <span>${esc(integrity.checked_count ?? 0)} checks</span>
        <span>${esc(failures.length)} failures</span>
      </div>
    </div>
  `;
}

function renderAuditSourceHealth(payload = {}) {
  const sources = payload.source_health || {};
  const entries = Object.entries(sources);
  if (!entries.length) return "";
  const invalidTotal = entries.reduce((total, [, item]) => total + Number(item.invalid_record_count || 0), 0);
  const unreadable = entries.filter(([, item]) => item.exists && item.readable === false).length;
  return `
    <div class="audit-source-health ${invalidTotal || unreadable ? "warn" : "ok"}">
      <div class="section-title-row compact">
        <div>
          <h3>审计源健康</h3>
          <p>检查技能、现场、运行和复核日志是否可读，是否存在坏行，避免导出包遗漏关键记录。</p>
        </div>
        ${badge(`${invalidTotal} invalid / ${unreadable} unreadable`, invalidTotal || unreadable ? "warn" : "ok")}
      </div>
      <div class="audit-source-grid">
        ${entries.map(([name, item]) => {
          const exists = item.exists === true;
          const readable = item.readable !== false;
          const invalid = Number(item.invalid_record_count || 0);
          const cls = !exists ? "" : (!readable || invalid ? "warn" : "ok");
          const label = payload.customer_report?.sections?.source_health?.source_labels?.[name] || name;
          return `
            <div class="audit-source-item ${cls}">
              <strong>${esc(label)} ${badge(!exists ? "not configured" : readable ? "readable" : "unreadable", cls || "warn")}</strong>
              <span>${esc(item.valid_record_count ?? item.record_count ?? 0)} valid / ${esc(invalid)} invalid</span>
              <small>${esc(item.path || "-")}</small>
              ${item.error ? `<small>${esc(item.error)}</small>` : ""}
            </div>
          `;
        }).join("")}
      </div>
    </div>
  `;
}

function getLastAuditExportResult() {
  try {
    return JSON.parse(localStorage.getItem("askme.audit_last_export") || "{}");
  } catch {
    return {};
  }
}

function renderAuditExportHistory(payload = {}) {
  const exports = Array.isArray(payload.exports) ? payload.exports : [];
  if (exports.length) {
    return `
      <div class="audit-export-history">
        <div class="section-title-row compact">
          <div>
            <h3>审计包历史</h3>
            <p>服务器上最近生成的审计包，交付、测试和客户成功看到的是同一份历史。</p>
          </div>
          ${badge(`${exports.length}/${payload.total ?? exports.length}`, payload.invalid ? "warn" : "ok")}
        </div>
        ${payload.invalid ? `<div class="mini-list-empty">有 ${esc(payload.invalid)} 个 manifest 无法读取，请检查 ${esc(payload.output_dir || "")}</div>` : ""}
        <div class="audit-export-list">
          ${exports.map((item) => renderAuditExportResult({ export: item, created_at: item.created_at })).join("")}
        </div>
      </div>
    `;
  }
  return renderAuditExportResult(getLastAuditExportResult());
}

function renderAuditExportResult(result = {}) {
  const exportPayload = result.export || {};
  if (!exportPayload.export_id) return "";
  const evidence = exportPayload.evidence_summary || {};
  const delivery = result.delivery || {};
  const deliveryStatus = delivery
    ? (delivery.sent ? "delivered" : (delivery.reason || "not_delivered"))
    : "local_only";
  return `
    <div class="audit-export-result ${evidence.ready === false ? "warn" : "ok"}">
      <div>
        <strong>最近审计包 ${badge(evidence.ready === false ? "证据缺失" : "证据完整", evidence.ready === false ? "warn" : "ok")}</strong>
        <p>${esc(exportPayload.manifest_path || "-")}</p>
      </div>
      <div class="row-meta">
        <span>${esc(exportPayload.export_id || "-")}</span>
        <span>${esc(exportPayload.record_count ?? 0)} records</span>
        <span>${esc(evidence.ref_count ?? 0)} evidence</span>
        <span>${esc(evidence.local_missing_count ?? 0)} missing</span>
        <span>${esc(deliveryStatus)}</span>
        <span>${esc(result.created_at || "-")}</span>
      </div>
    </div>
  `;
}

function renderAuditReviewQueue(records = []) {
  return `
    <div class="sub-card audit-review-card">
      <div class="section-title-row compact">
        <div>
          <h3>待复核审计</h3>
          <p>这些记录代表被拒绝、被阻断、未授权、关键风险或需要人工确认的操作。</p>
        </div>
        ${badge(`${records.length} 条`, "warn")}
      </div>
      <div class="mini-list">
        ${records.slice(0, 6).map((record) => `
          <div class="mini-row">
            <b>${esc(record.customer_label || record.source || "-")} ${badge(record.severity || "-", auditSeverityClass(record.severity))}</b>
            <span>${esc(record.reason || record.message || record.record_id || "-")} / ${esc(record.operator_id || "system")} / ${esc(record.timestamp || "-")}</span>
            <div class="mini-actions">
              <button class="ghost-button" data-audit-review-open="accepted" data-record-id="${esc(record.record_id || "")}">已复核</button>
              <button class="ghost-button" data-audit-review-open="waived" data-record-id="${esc(record.record_id || "")}">豁免</button>
              <button class="ghost-button" data-audit-review-open="escalated" data-record-id="${esc(record.record_id || "")}">升级</button>
            </div>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderUnifiedAudit(payload = {}) {
  const records = Array.isArray(payload.records) ? payload.records : [];
  if (!records.length) return `<div class="mini-list-empty">暂无统一审计记录</div>`;
  return `
    <div class="table-list">
      ${records.map((record) => `
        <div class="row-item audit-row ${record.requires_review ? "requires-review" : ""}">
          <strong>${esc(record.display_title || record.customer_label || record.source || "-")} ${badge(record.outcome_label || record.outcome || "-", statusClass(record.outcome))} ${badge(record.severity_label || record.severity || "low", auditSeverityClass(record.severity))} ${record.requires_review ? badge("需复核", "warn") : ""}</strong>
          <p>${esc(record.customer_copy?.what_happened || record.message || record.reason || record.subject || "")}</p>
          <p>${esc(record.customer_copy?.next_step || record.recommended_action || "")}</p>
          <div class="row-meta">
            <span>${esc(record.record_id || "-")}</span>
            <span>${esc(record.operator_id || "unknown")}</span>
            <span>${esc(record.resource_type || "-")}: ${esc(record.resource_id || record.subject || "-")}</span>
            <span>${esc(record.timestamp || "-")}</span>
            <span>${esc(record.integrity?.signed ? "signed" : "unsigned")}</span>
            <span>${esc(record.integrity?.hash_chain ? "hash-chain" : "no-chain")}</span>
            ${(record.evidence_refs || []).length ? `<span>evidence ${(record.evidence_refs || []).length}</span>` : ""}
          </div>
          ${record.requires_review ? `<div class="row-actions"><button class="ghost-button" data-audit-review-open="accepted" data-record-id="${esc(record.record_id || "")}">标记已复核</button><button class="ghost-button" data-audit-review-open="waived" data-record-id="${esc(record.record_id || "")}">复核豁免</button><button class="ghost-button" data-audit-review-open="escalated" data-record-id="${esc(record.record_id || "")}">升级处理</button></div>` : ""}
        </div>
      `).join("")}
    </div>
  `;
}

function renderAuditReviewPanel(review = {}) {
  const record = review.record || {};
  const decision = review.decision || "accepted";
  const evidence = Array.isArray(record.evidence_refs) ? record.evidence_refs : [];
  const history = Array.isArray(record.review_history) ? record.review_history : [];
  const latest = record.review_decision || history[0] || null;
  return `
    <div class="audit-review-panel">
      <div class="section-title-row compact">
        <div>
          <h3>审计复核</h3>
          <p>确认这条记录是否已处理。复核会写入独立哈希链日志，不会修改原始审计记录。</p>
        </div>
        <button class="ghost-button" data-audit-review-close>关闭</button>
      </div>
      <div class="audit-review-body">
        <div class="audit-review-evidence">
          <strong>${esc(record.customer_label || record.source || "产品审计")} ${badge(record.severity || "low", auditSeverityClass(record.severity))}</strong>
          <p>${esc(record.message || record.reason || record.subject || "需要人工确认的审计记录")}</p>
          <div class="row-meta">
            <span>${esc(record.record_id || "-")}</span>
            <span>${esc(record.operator_id || "system")}</span>
            <span>${esc(record.resource_type || "-")}: ${esc(record.resource_id || record.subject || "-")}</span>
            <span>${esc(record.timestamp || "-")}</span>
          </div>
          ${latest ? `<div class="audit-review-latest">${badge(latest.decision || "reviewed", latest.clears_review ? "ok" : "warn")}<span>${esc(latest.reviewer_id || "-")} / ${esc(latest.note || "-")}</span></div>` : ""}
          ${renderAuditEvidenceRefs(evidence)}
          ${renderAuditReviewHistory(history)}
        </div>
        <div class="audit-review-form">
          <label>处理决定</label>
          <select id="audit-review-decision">
            ${[
              ["accepted", "已复核，处理有效"],
              ["waived", "豁免，不再阻断交付"],
              ["false_positive", "误报"],
              ["resolved", "问题已解决"],
              ["escalated", "升级继续处理"],
              ["rejected", "复核拒绝，继续阻断"],
            ].map(([value, label]) => `<option value="${value}" ${value === decision ? "selected" : ""}>${label}</option>`).join("")}
          </select>
          <label>复核说明</label>
          <textarea id="audit-review-note" placeholder="说明处理依据、责任人、客户沟通结果或后续动作">${esc(review.note || defaultAuditReviewNote(decision))}</textarea>
          <div class="panel-actions compact">
            <button data-audit-review-submit>提交复核</button>
            <button class="ghost-button" data-audit-review-close>取消</button>
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderAuditEvidenceRefs(refs = []) {
  if (!refs.length) return `<div class="mini-list-empty">暂无证据附件</div>`;
  return `
    <div class="audit-evidence-list">
      <h4>关联证据</h4>
      ${refs.slice(0, 6).map((item) => {
        const path = item.path || item.url || "";
        const href = auditEvidenceHref(path);
        const isLink = Boolean(href);
        return `
          <div class="audit-evidence-row">
            <b>${esc(item.label || item.type || "evidence")}</b>
            ${isLink ? `<a href="${esc(href)}" target="_blank" rel="noreferrer">${esc(path)}</a>` : `<span>${esc(path || "-")}</span>`}
            ${isImageEvidence(path) && href ? `<img class="audit-evidence-thumb" src="${esc(href)}" alt="${esc(item.label || "evidence")}">` : ""}
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function auditEvidenceHref(path = "") {
  const text = String(path || "").trim();
  if (!text) return "";
  if (/^https?:\/\//i.test(text) || text.startsWith("/api/")) return text;
  return `/api/field/evidence?path=${encodeURIComponent(text)}`;
}

function isImageEvidence(path = "") {
  return /\.(avif|bmp|gif|jpe?g|png|webp)$/i.test(String(path || "").split("?")[0]);
}

function renderAuditReviewHistory(history = []) {
  if (!history.length) return `<div class="mini-list-empty">暂无复核历史</div>`;
  return `
    <div class="audit-history-list">
      <h4>复核历史</h4>
      ${history.slice(0, 6).map((item) => `
        <div class="audit-history-row">
          <b>${esc(item.decision || "-")} ${badge(item.clears_review ? "解除阻断" : "继续关注", item.clears_review ? "ok" : "warn")}</b>
          <span>${esc(item.reviewer_id || "-")} / ${esc(item.created_at || "-")}</span>
          <p>${esc(item.note || "-")}</p>
        </div>
      `).join("")}
    </div>
  `;
}

function defaultAuditReviewNote(decision) {
  if (decision === "escalated") return "需要继续升级处理";
  if (decision === "waived") return "已确认不影响本次交付";
  if (decision === "false_positive") return "已确认该记录为误报";
  if (decision === "rejected") return "复核未通过，继续阻断交付";
  return "已人工复核";
}

function renderAuditRetryStatus(payload = {}) {
  const pending = Number(payload.pending || 0);
  const invalid = Number(payload.invalid || 0);
  const items = Array.isArray(payload.items) ? payload.items : [];
  const cls = pending || invalid ? "warn" : "ok";
  return `
    <div class="sub-card">
      <div class="section-title-row compact">
        <div>
          <h3>外部审计投递</h3>
          <p>SIEM/WORM webhook 临时失败后进入重试队列，保证验收证据不断链。</p>
        </div>
        ${badge(`${pending} 待投递 / ${invalid} 异常`, cls)}
      </div>
      ${
        items.length
          ? `<div class="table-list compact-list">
              ${items.map((item) => `
                <div class="row-item">
                  <strong>${esc(item.export_id || `line-${item.line || "-"}`)} ${badge(item.status || "pending", statusClass(item.status))}</strong>
                  <p>${esc(item.reason || item.error || item.webhook_url || "等待重试投递")}</p>
                  <div class="row-meta">
                    <span>${esc(item.queued_at || "-")}</span>
                    <span>${esc(item.retry_count ?? 0)} retries</span>
                    <span>${esc(item.record_count ?? 0)} records</span>
                  </div>
                </div>
              `).join("")}
            </div>`
          : `<div class="mini-list-empty">暂无失败投递，审计外送队列为空</div>`
      }
    </div>
  `;
}

function wireAuditExportControls() {
  document.querySelectorAll("[data-audit-export]").forEach((button) => {
    button.addEventListener("click", async () => {
      const mode = button.dataset.auditExport || "local";
      button.disabled = true;
      const response = await postJson(ENDPOINTS.auditExport, {
        operator_id: operatorId(),
        limit: 500,
        since: getAuditWindow().since,
        until: getAuditWindow().until,
        project_id: getAuditWindow().project_id,
        managed_object_id: getAuditWindow().managed_object_id,
        deliver: mode === "deliver",
      });
      button.disabled = false;
      if (!response.ok) {
        alert(response.payload?.error || response.payload?.reason || "审计包生成失败");
        return;
      }
      const exportPayload = response.payload?.export || {};
      localStorage.setItem("askme.audit_last_export", JSON.stringify({
        mode,
        created_at: new Date().toISOString(),
        export: exportPayload,
        delivery: response.payload?.delivery || null,
      }));
      const manifest = exportPayload.manifest_path || "";
      const evidence = exportPayload.evidence_summary || {};
      alert(manifest ? `审计包已生成：${manifest}` : "审计包已生成");
      await renderDelivery();
    });
  });
}

function wireAuditWindowControls() {
  document.querySelectorAll("[data-audit-window]").forEach((button) => {
    button.addEventListener("click", async () => {
      const action = button.dataset.auditWindow || "apply";
      if (action === "clear") {
        localStorage.removeItem("askme.audit_since");
        localStorage.removeItem("askme.audit_until");
        localStorage.removeItem("askme.audit_project_id");
        localStorage.removeItem("askme.audit_managed_object_id");
      } else {
        localStorage.setItem("askme.audit_since", document.getElementById("audit-since")?.value || "");
        localStorage.setItem("askme.audit_until", document.getElementById("audit-until")?.value || "");
        localStorage.setItem("askme.audit_project_id", document.getElementById("audit-project")?.value || "");
        localStorage.setItem("askme.audit_managed_object_id", document.getElementById("audit-object")?.value || "");
      }
      await renderDelivery();
    });
  });
}

function wireAuditReviewOpenControls() {
  document.querySelectorAll("[data-audit-review-open]").forEach((button) => {
    button.addEventListener("click", async () => {
      const recordId = button.dataset.recordId || "";
      const decision = button.dataset.auditReviewOpen || "accepted";
      if (!recordId) return;
      const record = auditRecordCache.find((item) => item.record_id === recordId) || { record_id: recordId };
      selectedAuditReview = { record_id: recordId, record, decision, note: defaultAuditReviewNote(decision) };
      const panel = document.getElementById("audit-review-panel");
      if (panel) panel.innerHTML = renderAuditReviewPanel(selectedAuditReview);
      wireAuditReviewPanelControls();
      panel?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  });
}

function wireAuditReviewPanelControls() {
  document.querySelectorAll("[data-audit-review-close]").forEach((button) => {
    button.addEventListener("click", () => {
      selectedAuditReview = null;
      const panel = document.getElementById("audit-review-panel");
      if (panel) panel.innerHTML = "";
    });
  });
  document.querySelectorAll("[data-audit-review-submit]").forEach((button) => {
    button.addEventListener("click", async () => {
      if (!selectedAuditReview?.record_id) return;
      const decision = document.getElementById("audit-review-decision")?.value || selectedAuditReview.decision || "accepted";
      const note = document.getElementById("audit-review-note")?.value || defaultAuditReviewNote(decision);
      button.disabled = true;
      const response = await postJson(ENDPOINTS.auditReviews, {
        operator_id: operatorId(),
        record_id: selectedAuditReview.record_id,
        decision,
        note,
      });
      button.disabled = false;
      if (!response.ok) {
        alert(response.payload?.error || response.payload?.reason || "审计复核提交失败");
        return;
      }
      selectedAuditReview = null;
      await renderDelivery();
    });
  });
}

function auditSeverityClass(value) {
  const text = String(value || "").toLowerCase();
  if (text === "critical" || text === "high") return "err";
  if (text === "medium") return "warn";
  return "ok";
}

function wireAuditRetryControls() {
  document.querySelectorAll("[data-audit-retry]").forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      const response = await postJson(ENDPOINTS.auditExportRetry, {
        operator_id: operatorId(),
        limit: 50,
      });
      button.disabled = false;
      if (!response.ok) {
        alert(response.payload?.error || response.payload?.reason || "审计投递重试失败");
        return;
      }
      const payload = response.payload || {};
      alert(`审计投递重试完成：已尝试 ${payload.attempted || 0} 条，成功 ${payload.sent || 0} 条，剩余 ${payload.remaining ?? 0} 条`);
      await renderDelivery();
    });
  });
}

async function render() {
  await refreshGlobalStatus();
  const page = currentPage();
  setHeader(page);
  renderNav(page);
  if (page.key === "overview") await renderOverview();
  if (page.key === "conversation") renderConversation();
  if (page.key === "projects") await renderProjects();
  if (page.key === "field") await renderField();
  if (page.key === "space") await renderSpace();
  if (page.key === "knowledge") renderKnowledge();
  if (page.key === "capabilities") await renderCapabilities();
  if (page.key === "voice") await renderVoice();
  if (page.key === "delivery" || page.key === "audit") await renderDelivery();
  document.querySelectorAll("[data-route]").forEach((button) => {
    button.addEventListener("click", () => routeTo(button.dataset.route));
  });
}

window.addEventListener("popstate", render);
setInterval(() => {
  refreshGlobalStatus();
  if (currentPage().key === "conversation") pollLive();
}, 5000);
render();
