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
  currentOperator: "/api/governance/current-operator",
  authorize: "/api/governance/authorize",
  auditEvents: "/api/audit/events",
  auditExport: "/api/audit/export",
  auditExportRetry: "/api/audit/export/retry",
  knowledgePreview: "/api/knowledge/preview",
  knowledgeImport: "/api/knowledge/import",
  knowledgeList: "/api/knowledge/list",
  knowledgeUpdate: "/api/knowledge/update",
  memorySearch: "/api/memory/search",
  capabilityCenter: "/api/capability-center",
  skillAudit: "/api/skill-audit",
  agentProfiles: "/api/agent-profiles",
  generatedSkills: "/api/skills/generated",
  skillPackages: "/api/skill-packages",
  skillGrowthBacklog: "/api/skill-growth/backlog",
};

const pages = [
  { path: "/dashboard", key: "overview", label: "总览", hint: "功能地图", title: "现场任务平台", kicker: "产品总览", desc: "给客户看的功能地图：语音入口、现场事件、知识库、音色和交付检查分开验收。" },
  { path: "/dashboard/conversation", key: "conversation", label: "对话", hint: "语音和文本", title: "语音和文本对话", kicker: "真实交互", desc: "用于输入任务、问路、知识问答和安全确认。回答会展示可引用证据和任务状态。" },
  { path: "/dashboard/field", key: "field", label: "现场事件", hint: "安防巡检", title: "现场事件处置", kicker: "园区场景", desc: "覆盖摔倒、卡住、陌生人拍照、违停、烟雾火灾、垃圾桶满溢、人群聚集、访客问路和带路。" },
  { path: "/dashboard/knowledge", key: "knowledge", label: "知识库", hint: "上传审批", title: "知识管理", kicker: "可审计回答", desc: "上传、预览、审批、检索和重建索引。过期、冲突或未审批知识不能直接进入回答。" },
  { path: "/dashboard/capabilities", key: "capabilities", label: "能力中心", hint: "技能增长", title: "机器人能力中心", kicker: "客户可见能力", desc: "按巡检、异常处置、安防、访客服务、空间认知和在线增长展示机器人当前能做什么、缺什么、哪些能力需要审批。" },
  { path: "/dashboard/voice", key: "voice", label: "语音音色", hint: "播报策略", title: "语音音色和实时链路", kicker: "声音系统", desc: "按巡检、访客、安防、紧急告警、夜间低扰等场景切换音色和提示音，并查看端到端延迟。" },
  { path: "/dashboard/delivery", key: "delivery", label: "交付检查", hint: "可验收", title: "交付检查", kicker: "上线门禁", desc: "把演示、试点、真实硬件和外部通知的缺口拆成清晰门禁，避免把实验室能力说成生产上线。" },
];

let health = {};
let governance = { operators: [] };
let operatorSession = null;
let liveBaseline = null;
let chatStarted = false;
let chatRenderedCount = 0;
let selectedFieldEventId = null;
let fieldActionResult = null;
let selectedGeneratedSkillPreview = null;
let selectedAgentProfilePreview = null;

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
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
    const response = await fetch(BASE + path);
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
  const [healthPayload, governancePayload, operatorPayload] = await Promise.all([
    getJson("/health", {}),
    getJson(ENDPOINTS.governance, governance),
    getJson(currentOperatorPath, operatorSession),
  ]);
  health = healthPayload || {};
  governance = governancePayload || governance || { operators: [] };
  operatorSession = operatorPayload || operatorSession;
  const ok = health.status === "ok";
  globalStatusDot.className = `status-dot ${ok ? "ok" : "err"}`;
  globalStatusText.textContent = ok ? "服务在线" : "服务异常";
}

async function renderOverview() {
  const [eventsPayload, scenariosPayload, readiness, notification] = await Promise.all([
    getJson("/api/field/events?limit=6&needs_attention=true", { events: [] }),
    getJson("/api/field/scenarios", { scenarios: [] }),
    getJson("/api/field/readiness", {}),
    getJson("/api/field/notification-preflight", {}),
  ]);
  const events = eventsPayload.events || eventsPayload.items || [];
  const scenarios = scenariosPayload.scenarios || scenariosPayload.items || [];
  app.innerHTML = `
    <section class="ops-hero">
      <div>
        <p class="page-kicker">客户验收视角</p>
        <h2>现场事件闭环看板</h2>
        <p>客户可直接验收：发生了什么、证据在哪里、通知给谁、谁负责、当前风险是什么。</p>
      </div>
      <div class="ops-summary">
        <div><b>${esc(events.length)}</b><span>待关注事件</span></div>
        <div><b>${esc(scenarios.length || 8)}</b><span>覆盖场景</span></div>
        <div><b>${esc((notification.blockers || []).length)}</b><span>通知阻塞</span></div>
        <div><b>${esc(readiness.status || "unknown")}</b><span>交付门禁</span></div>
      </div>
    </section>
    <section class="grid three">
      ${renderOperatorCard()}
      ${renderReadinessCard(readiness, notification)}
      <div class="card">
        <h2>客户现在能看什么</h2>
        <div class="metric"><b>事件</b><span>场景、地点、风险、状态</span></div>
        <div class="metric"><b>证据</b><span>照片/传感器/运行记录</span></div>
        <div class="metric"><b>处置</b><span>通知目标、负责人、下一步</span></div>
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
  return `
    <div class="card">
      <h2>交付门禁</h2>
      <div class="metric"><b>现场 readiness</b>${badge(readiness.status || "unknown", statusClass(readiness.status))}</div>
      <div class="metric"><b>通知链路</b>${badge(ready ? "可通知" : "需配置", ready ? "ok" : "warn")}</div>
      <div class="metric"><b>阻塞项</b><span>${esc(((readiness.blockers || []).length) + ((notification.blockers || []).length))}</span></div>
    </div>
  `;
}

function renderScenarioLanes(scenarios = []) {
  const fallback = [
    { title: "机器人卡住", notification_group: "保安群", priority: "P1", evidence: "位置+照片+运行状态" },
    { title: "夜间陌生人拍照", notification_group: "保安群", priority: "P1", evidence: "照片+地点+时间" },
    { title: "车辆违停", notification_group: "保安群", priority: "P2", evidence: "车牌/照片+区域" },
    { title: "烟雾火灾", notification_group: "保安群", priority: "P0", evidence: "烟感/温度+照片" },
    { title: "垃圾桶满溢", notification_group: "保洁群", priority: "P3", evidence: "定点照片" },
    { title: "访客问路", notification_group: "不通知", priority: "P4", evidence: "帮助点+意图" },
  ];
  const rows = scenarios.length ? scenarios : fallback;
  return rows.slice(0, 8).map((scenario) => {
    const name = scenario.title || scenario.name || scenario.label || scenario.scenario_id || "现场场景";
    const group = scenario.notification_group || scenario.notify_group || scenario.notification || "按规则";
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
          <span>地点 ${esc(event.location || "-")}</span>
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
  const [center, auditPayload, agentPayload, generatedPayload, packagePayload, growthPayload] = await Promise.all([
    getJson(ENDPOINTS.capabilityCenter, {}),
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
  const scenarioSummary = scenarioBlueprints.summary || {};
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
          <h2>场景能力蓝图</h2>
          <p>按客户场景展示需要哪些技能、传感器/数据依赖、通知归档和验收标准，避免只看散乱的技能清单。</p>
        </div>
        ${badge(`${scenarioSummary.ready_count ?? 0} ready / ${scenarioSummary.scenario_count ?? scenarioItems.length} scenarios`, (scenarioSummary.blocked_count ?? 0) ? "warn" : "ok")}
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
          <h2>Customer Skill Packages</h2>
          <p>Approved generated skills are enabled by customer/site package, not globally.</p>
        </div>
        ${badge(`${skillPackages.length} packages`)}
      </div>
      <div class="knowledge-form skill-package-form">
        <input id="skill-package-id" placeholder="Package ID, e.g. fanmu-phase-1">
        <input id="skill-package-name" placeholder="Package name, e.g. Fanmu pilot package">
        <input id="skill-package-site" placeholder="Site ID, e.g. fanmu-park">
        <input id="skill-package-customer" placeholder="Customer name">
        <select id="skill-package-enabled">
          <option value="true">Enabled</option>
          <option value="false">Disabled</option>
        </select>
        <select id="skill-package-channel">
          <option value="draft">Draft</option>
          <option value="pilot">Pilot</option>
          <option value="prod">Prod</option>
        </select>
        <input id="skill-package-rollout" type="number" min="0" max="100" value="100" placeholder="Rollout %">
        <textarea id="skill-package-description" placeholder="Package scope, acceptance boundary, rollout note"></textarea>
        <button id="skill-package-save" class="primary-button">Save Package</button>
      </div>
      <div class="capability-list">
        ${skillPackages.map(renderSkillPackage).join("") || `<div class="mini-list-empty">No skill packages.</div>`}
      </div>
    </section>
    <section class="capability-grid">
      ${groups.map(renderCapabilityGroup).join("") || `<div class="loading-card">能力中心暂无数据。</div>`}
    </section>
  `;
  wireGeneratedSkillReview();
}

function renderScenarioBlueprint(item = {}) {
  const skills = Array.isArray(item.required_skills) ? item.required_skills : [];
  const dependencies = Array.isArray(item.dependencies) ? item.dependencies : [];
  const evidence = Array.isArray(item.required_evidence) ? item.required_evidence : [];
  const acceptance = Array.isArray(item.acceptance_criteria) ? item.acceptance_criteria : [];
  const status = item.coverage_status || "blocked";
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
          `).join("") || `<span class="err">未配置 required skills</span>`}
        </div>
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
        ${badge(`skills ${esc(item.enabled_count ?? 0)}/${esc(item.required_skill_count ?? skills.length)}`)}
        ${badge(item.runtime_entry || "field_event_trigger")}
        <span class="small-note">${esc(item.next_action || "")}</span>
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
  const packageLabel = inDefaultPackage ? "Remove package" : "Assign package";
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
        <strong>${esc(item.display_name || item.package_id || "skill package")}</strong>
        <p>${esc(item.description || "Customer/site scoped ability package.")}</p>
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
          ${(skills.length ? skills : ["no assigned skills"]).map((name) => `
            <span class="${missing.includes(name) ? "warn" : ""}">${esc(name)}</span>
          `).join("")}
        </div>
      </div>
      <div class="capability-badges">
        ${badge(item.enabled ? "enabled" : "disabled", item.enabled ? "ok" : "warn")}
        ${badge(rollout > 0 ? `${rollout}% rollout` : "rollout paused", rollout > 0 ? "ok" : "warn")}
        <button class="ghost-button" data-package-release="pilot" data-package-id="${esc(item.package_id || "default-demo")}" data-package-rollout="25">Pilot 25%</button>
        <button class="ghost-button" data-package-release="prod" data-package-id="${esc(item.package_id || "default-demo")}" data-package-rollout="100">Prod 100%</button>
        <button class="ghost-button" data-package-rollback="${esc(item.package_id || "default-demo")}">Rollback</button>
        ${badge(item.customer_name || "customer scoped")}
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
        <pre>${esc(preview.raw_body || "Built-in profile has no project Markdown body.")}</pre>
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
        ${badge(validation.ok ? "validation ok" : "validation failed", validation.ok ? "ok" : "err")}
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
        alert(response.payload?.error || response.payload?.reason || "Agent profile save failed");
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
        alert(response.payload?.error || response.payload?.reason || "Skill package save failed");
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
        alert(response.payload?.error || response.payload?.reason || "Skill package release failed");
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
        alert(response.payload?.error || response.payload?.reason || "Skill package rollback failed");
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

async function renderDelivery() {
  const auditPath = `${ENDPOINTS.auditEvents}?actor_id=${encodeURIComponent(operatorId())}&limit=12`;
  const retryPath = `${ENDPOINTS.auditExportRetry}?actor_id=${encodeURIComponent(operatorId())}&limit=8`;
  const [readiness, devices, runtime, audit, retry] = await Promise.all([
    getJson("/api/field/readiness", {}),
    getJson("/api/field/devices", {}),
    getJson("/api/runtime/context", {}),
    getJson(auditPath, { records: [], summary: {} }),
    getJson(retryPath, { pending: 0, invalid: 0, items: [] }),
  ]);
  app.innerHTML = `
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
  wireAuditExportControls();
  wireAuditRetryControls();
}

function renderUnifiedAudit(payload = {}) {
  const records = Array.isArray(payload.records) ? payload.records : [];
  if (!records.length) return `<div class="mini-list-empty">暂无统一审计记录</div>`;
  return `
    <div class="table-list">
      ${records.map((record) => `
        <div class="row-item">
          <strong>${esc(record.source || "-")} / ${esc(record.action || "-")} ${badge(record.outcome || "-", statusClass(record.outcome))}</strong>
          <p>${esc(record.message || record.reason || record.subject || "")}</p>
          <div class="row-meta">
            <span>${esc(record.operator_id || "unknown")}</span>
            <span>${esc(record.subject || "-")}</span>
            <span>${esc(record.timestamp || "-")}</span>
          </div>
        </div>
      `).join("")}
    </div>
  `;
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
        deliver: mode === "deliver",
      });
      button.disabled = false;
      if (!response.ok) {
        alert(response.payload?.error || response.payload?.reason || "审计包生成失败");
        return;
      }
      const manifest = response.payload?.export?.manifest_path || "";
      alert(manifest ? `审计包已生成：${manifest}` : "审计包已生成");
      await renderDelivery();
    });
  });
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
  if (page.key === "field") await renderField();
  if (page.key === "knowledge") renderKnowledge();
  if (page.key === "capabilities") await renderCapabilities();
  if (page.key === "voice") await renderVoice();
  if (page.key === "delivery") await renderDelivery();
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
