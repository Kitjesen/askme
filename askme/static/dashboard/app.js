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
  knowledgePreview: "/api/knowledge/preview",
  knowledgeImport: "/api/knowledge/import",
  knowledgeList: "/api/knowledge/list",
  knowledgeUpdate: "/api/knowledge/update",
  memorySearch: "/api/memory/search",
};

const pages = [
  { path: "/dashboard", key: "overview", label: "总览", hint: "功能地图", title: "现场任务平台", kicker: "产品总览", desc: "给客户看的功能地图：语音入口、现场事件、知识库、音色和交付检查分开验收。" },
  { path: "/dashboard/conversation", key: "conversation", label: "对话", hint: "语音和文本", title: "语音和文本对话", kicker: "真实交互", desc: "用于输入任务、问路、知识问答和安全确认。回答会展示可引用证据和任务状态。" },
  { path: "/dashboard/field", key: "field", label: "现场事件", hint: "安防巡检", title: "现场事件处置", kicker: "园区场景", desc: "覆盖摔倒、卡住、陌生人拍照、违停、烟雾火灾、垃圾桶满溢、人群聚集、访客问路和带路。" },
  { path: "/dashboard/knowledge", key: "knowledge", label: "知识库", hint: "上传审批", title: "知识管理", kicker: "可审计回答", desc: "上传、预览、审批、检索和重建索引。过期、冲突或未审批知识不能直接进入回答。" },
  { path: "/dashboard/voice", key: "voice", label: "语音音色", hint: "播报策略", title: "语音音色和实时链路", kicker: "声音系统", desc: "按巡检、访客、安防、紧急告警、夜间低扰等场景切换音色和提示音，并查看端到端延迟。" },
  { path: "/dashboard/delivery", key: "delivery", label: "交付检查", hint: "可验收", title: "交付检查", kicker: "上线门禁", desc: "把演示、试点、真实硬件和外部通知的缺口拆成清晰门禁，避免把实验室能力说成生产上线。" },
];

let health = {};
let governance = { operators: [] };
let liveBaseline = null;
let chatStarted = false;
let chatRenderedCount = 0;

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
  const operators = Array.isArray(governance.operators) ? governance.operators : [];
  return operators.find((operator) => operator.operator_id === id) || { operator_id: id, roles: ["operator"], source: "localStorage" };
}

function operatorRolesText(operator = currentOperator()) {
  const roles = Array.isArray(operator.roles) ? operator.roles : [];
  return roles.length ? roles.join(" / ") : "-";
}

function renderOperatorCard() {
  const operators = Array.isArray(governance.operators) ? governance.operators : [];
  const active = currentOperator();
  const options = operators.length
    ? operators.map((operator) => `<option value="${esc(operator.operator_id)}" ${operator.operator_id === active.operator_id ? "selected" : ""}>${esc(operator.display_name || operator.operator_id)} (${esc(operatorRolesText(operator))})</option>`).join("")
    : `<option value="${esc(active.operator_id)}">${esc(active.operator_id)} (${esc(operatorRolesText(active))})</option>`;
  return `
    <div class="operator-card">
      <div>
        <strong>当前操作人</strong>
        <p>${esc(governance.mode || "demo_config")} / ${esc(governance.identity_provider || "local_config")}；生产环境需绑定企业账号体系。</p>
      </div>
      <select id="operator-select">${options}</select>
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
  const [healthPayload, governancePayload] = await Promise.all([
    getJson("/health", {}),
    getJson(ENDPOINTS.governance, governance),
  ]);
  health = healthPayload || {};
  governance = governancePayload || governance || { operators: [] };
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
  const [scenarios, events] = await Promise.all([
    getJson("/api/field/scenarios", { scenarios: [] }),
    getJson("/api/field/events?limit=20", { events: [] }),
  ]);
  const scenarioRows = scenarios.scenarios || scenarios.items || [];
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
        <h2>场景范围</h2>
        <p>这些不是框架名，而是客户要验收的真实场景：异常播报、夜间陌生人拍照、违停、烟雾火灾、垃圾桶满溢、突发巡检、人群聚集、路人问路和带路。</p>
      </div>
    </section>
    <section class="card">
      <h2>最近现场事件</h2>
      <div id="field-events" class="table-list">${renderFieldEvents(events)}</div>
    </section>
  `;
  document.getElementById("field-submit").addEventListener("click", submitFieldEvent);
}

function renderFieldEvents(payload) {
  const rows = payload.events || payload.items || [];
  if (!rows.length) return `<div class="row-item"><strong>暂无现场事件</strong><p>可以从上方选择场景创建一条演示事件。</p></div>`;
  return rows.map((event) => `
    <div class="row-item">
      <strong>${esc(event.title || event.scenario_id || event.event_type || "现场事件")}</strong>
      <p>${esc(event.narrative || event.summary || event.location || "-")}</p>
      <div class="row-meta">
        <span>状态 ${esc(event.status || "-")}</span>
        <span>地点 ${esc(event.location || "-")}</span>
        <span>通知 ${esc(event.notification_group || event.notification?.group || "-")}</span>
      </div>
    </div>
  `).join("");
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
  });
  const box = document.getElementById("field-events");
  box.innerHTML = response.ok ? renderFieldEvents({ events: [response.payload.event || response.payload] }) : `<div class="row-item"><strong>提交失败</strong><p>${esc(response.payload.error || response.payload.message || "未知错误")}</p></div>`;
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

async function renderDelivery() {
  const [readiness, devices, runtime] = await Promise.all([
    getJson("/api/field/readiness", {}),
    getJson("/api/field/devices", {}),
    getJson("/api/runtime/context", {}),
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
    <section class="card"><h2>原始门禁证据</h2><div class="mono">${esc(JSON.stringify(readiness, null, 2))}</div></section>
  `;
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
