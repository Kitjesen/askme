const BASE = location.origin;
const STORAGE_KEY = "askme.voice_lab.active_run_id";
const OPERATOR_KEY = "askme.operator_id";

const STEPS = [
  ["设备检查", "输入、输出与回采"],
  ["安静校准", "采集房间噪声底"],
  ["20 次静默", "观察误触发"],
  ["20 次覆盖", "主观打断感受"],
  ["20 次响应", "自然问答听感"],
  ["严格报告", "诊断与物理门禁"],
];

const SCENARIOS = {
  speaker_only: {
    title: "静默播放观察",
    kicker: "SPEAKER ONLY",
    instruction: "点击播放标准长句后保持安静，观察机器人是否被自己的回声误打断、误停播或异常重启监听。",
    question: "这次是否发生了误打断或误停播？",
    positive: ["没有误触发", "播放过程稳定"],
    negative: ["发生误触发", "出现误停播、抢话或异常"],
    field: "false_barge_in",
    positiveValue: false,
    negativeValue: true,
    triggerLabel: "播放标准长句",
  },
  human_overlap: {
    title: "覆盖说话观察",
    kicker: "BARGE-IN",
    instruction: "点击播放后，在机器人说话约 1 秒时清楚地说“停一下”。观察它是否停止，而不是等整句播完。",
    question: "你覆盖说话后，机器人是否停止播报？",
    positive: ["停止了", "主观感觉可以打断"],
    negative: ["没有停止", "继续播完或反应明显异常"],
    field: "detected",
    positiveValue: true,
    negativeValue: false,
    triggerLabel: "开始覆盖打断",
  },
  assistant_response: {
    title: "自然响应观察",
    kicker: "RESPONSE",
    instruction: "面对机器人自然说：“小算，请用一句话介绍你自己。” 等它回答后，记录首字是否完整、声音是否清楚。",
    question: "这次是否听到了可理解的完整回答？",
    positive: ["听到了", "回答可理解且能继续对话"],
    negative: ["没有听清", "无声、截字、断音或超时"],
    field: "heard",
    positiveValue: true,
    negativeValue: false,
    triggerLabel: "我已开始自然提问",
  },
};

const STATUS_TEXT = {
  needs_device_check: "待设备检查",
  needs_calibration: "待安静校准",
  running: "测试进行中",
  blocked: "当前被阻断",
  paused: "已暂停",
  ready_for_report: "可生成报告",
  completed: "诊断已完成",
  aborted: "已终止",
};

const BLOCK_REASON_TEXT = {
  missing_isolated_speaker_monitor: "缺少隔离扬声器监听通道：覆盖场景不能证明物理停播。",
  physical_first_sound_collector_not_connected: "物理首音自动采集器尚未接入。",
};

const state = {
  devices: null,
  run: null,
  selectedOutcome: null,
  selectedQuality: "clear",
  busy: false,
  trialExecutionKeys: new Map(),
  trialExecutionFailures: new Map(),
};

const workPanel = document.getElementById("work-panel");
const alertBox = document.getElementById("lab-alert");

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

function operatorId() {
  return localStorage.getItem(OPERATOR_KEY) || "dashboard.operator";
}

function idempotencyKey(action) {
  const random = crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `voice-lab:${action}:${random}`;
}

function trialExecutionKey(attemptId) {
  if (!state.trialExecutionKeys.has(attemptId)) {
    state.trialExecutionKeys.set(attemptId, idempotencyKey(`execute-${attemptId}`));
  }
  return state.trialExecutionKeys.get(attemptId);
}

async function api(path, { method = "GET", body = null, version = null, key = null } = {}) {
  const headers = { "X-Askme-Operator-Id": operatorId() };
  if (body !== null) headers["Content-Type"] = "application/json";
  if (key) headers["Idempotency-Key"] = key;
  if (version !== null) headers["If-Match"] = String(version);
  const response = await fetch(BASE + path, {
    method,
    headers,
    body: body === null ? undefined : JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `请求失败 (${response.status})`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function setConnection(text, kind = "neutral") {
  const element = document.getElementById("connection-state");
  element.textContent = text;
  element.className = `status-pill ${kind}`;
}

function showAlert(message, kind = "warn") {
  alertBox.hidden = !message;
  alertBox.textContent = message || "";
  alertBox.className = `lab-alert ${kind}`;
}

function showBusy(label) {
  state.busy = true;
  workPanel.innerHTML = `<div class="loading-card"><span class="spinner" aria-hidden="true"></span><p>${escapeHtml(label)}</p></div>`;
  syncDock();
}

function finishBusy() {
  state.busy = false;
  render();
}

async function loadDevices() {
  try {
    state.devices = await api("/api/voice/lab/devices");
    setConnection("设备 API 已连接", "ok");
  } catch (error) {
    state.devices = { status: "error", error: error.message, devices: [], recommendation: {} };
    setConnection("设备 API 不可用", "err");
  }
}

async function restoreRun() {
  const runId = localStorage.getItem(STORAGE_KEY);
  if (!runId) return;
  try {
    state.run = await api(`/api/voice/lab/runs/${encodeURIComponent(runId)}`);
    setConnection("已恢复上次进度", "ok");
  } catch (error) {
    localStorage.removeItem(STORAGE_KEY);
    if (error.status !== 404) showAlert(`恢复上次测试失败：${error.message}`);
  }
}

async function boot() {
  wirePersistentActions();
  await loadDevices();
  await restoreRun();
  render();
}

function wirePersistentActions() {
  document.getElementById("pause-button").addEventListener("click", () => mutate("pause", {}));
  document.getElementById("resume-button").addEventListener("click", () => mutate("resume", {}));
  document.getElementById("forget-button").addEventListener("click", () => {
    localStorage.removeItem(STORAGE_KEY);
    state.run = null;
    state.selectedOutcome = null;
    state.trialExecutionKeys.clear();
    state.trialExecutionFailures.clear();
    showAlert("仅从此浏览器移除了运行引用；服务器证据文件仍被保留。", "warn");
    render();
  });
}

function render() {
  renderSteps();
  syncDock();
  document.getElementById("evidence-json").textContent = state.run
    ? JSON.stringify(state.run, null, 2)
    : JSON.stringify(state.devices || {}, null, 2);
  if (state.busy) return;
  if (!state.run) renderSetup();
  else if (state.run.status === "needs_device_check" || state.run.status === "blocked") renderDeviceCheck();
  else if (state.run.status === "needs_calibration") renderCalibration();
  else if (state.run.status === "running") renderTrial();
  else if (state.run.status === "paused") renderPaused();
  else if (state.run.status === "ready_for_report") renderReportReady();
  else if (state.run.status === "completed") renderReport();
  else renderTerminal();
}

function currentStep() {
  if (!state.run || ["needs_device_check", "blocked"].includes(state.run.status)) return 0;
  if (state.run.status === "needs_calibration" || state.run.status === "paused") return 1;
  const action = state.run.next_action || {};
  if (action.scenario === "speaker_only") return 2;
  if (action.scenario === "human_overlap") return 3;
  if (action.scenario === "assistant_response") return 4;
  return 5;
}

function completedSteps() {
  const run = state.run;
  if (!run) return new Set();
  const done = new Set();
  if (run.device_check?.status === "ok") done.add(0);
  if (run.calibration?.status === "ok") done.add(1);
  if ((run.progress?.speaker_only || 0) >= 20) done.add(2);
  if ((run.progress?.human_overlap || 0) >= 20) done.add(3);
  if ((run.progress?.assistant_response || 0) >= 20) done.add(4);
  if (run.status === "completed") done.add(5);
  return done;
}

function renderSteps() {
  const active = currentStep();
  const done = completedSteps();
  document.getElementById("step-list").innerHTML = STEPS.map(([title, desc], index) => `
    <li class="step-item ${done.has(index) ? "done" : ""} ${active === index ? "active" : ""}" data-step="${index + 1}" ${active === index ? 'aria-current="step"' : ""}>
      <strong>${escapeHtml(title)}</strong><span>${escapeHtml(desc)}</span>
    </li>
  `).join("");
  const progress = state.run?.progress || { total: 0, required_total: 60, percent: 0 };
  document.getElementById("progress-label").textContent = `${progress.total || 0} / ${progress.required_total || 60}`;
  document.getElementById("progress-bar").style.width = `${Math.max(0, Math.min(100, progress.percent || 0))}%`;
}

function syncDock() {
  const run = state.run;
  document.getElementById("run-status").textContent = run ? (STATUS_TEXT[run.status] || run.status) : "未创建";
  document.getElementById("run-id").textContent = run?.run_id || "—";
  document.getElementById("run-input").textContent = run ? deviceName(run.device_binding?.input_device_id) : "—";
  document.getElementById("run-output").textContent = run ? deviceName(run.device_binding?.output_device_id) : "—";
  document.getElementById("run-room").textContent = run?.room || "—";
  const gate = run?.product_gate?.status;
  document.getElementById("evidence-grade").textContent = gate === "passed" ? "物理通过" : "诊断 / 待物理证据";
  const reasons = run?.product_gate_blocked_reasons || [];
  document.getElementById("gate-list").innerHTML = reasons.length
    ? reasons.map((reason) => `<div class="gate-row"><span class="gate-dot"></span><span>${escapeHtml(BLOCK_REASON_TEXT[reason] || reason)}</span></div>`).join("")
    : '<div class="gate-row"><span class="gate-dot" style="background:var(--green)"></span><span>当前采集能力具备物理验收条件。</span></div>';
  const pause = document.getElementById("pause-button");
  const resume = document.getElementById("resume-button");
  const forget = document.getElementById("forget-button");
  pause.hidden = !run || state.busy || !["needs_device_check", "needs_calibration", "running", "blocked"].includes(run.status);
  resume.hidden = !run || state.busy || run.status !== "paused";
  forget.hidden = !run;
}

function deviceName(id) {
  const match = (state.devices?.devices || []).find((device) => String(device.index) === String(id));
  return match ? `${match.name} (#${match.index})` : String(id ?? "—");
}

function deviceOptions(kind) {
  const recommendation = state.devices?.recommendation || {};
  const preferred = kind === "input" ? recommendation.input_device : recommendation.output_device;
  const rows = (state.devices?.devices || []).filter((device) => kind === "input" ? device.is_input : device.is_output);
  return rows.map((device) => `<option value="${escapeHtml(device.index)}" ${String(device.index) === String(preferred) ? "selected" : ""}>${escapeHtml(device.name)} · #${escapeHtml(device.index)} · ${escapeHtml(device.default_samplerate || "rate?")}Hz</option>`).join("");
}

function renderSetup() {
  const unavailable = state.devices?.status !== "ok";
  workPanel.innerHTML = `
    <p class="panel-kicker">STEP 1 · DEVICE BINDING</p>
    <h2 class="panel-title">先确认机器人实际使用的麦克风和音箱</h2>
    <p class="panel-copy">创建后会播放一段短测试音并从麦克风回采。请先把系统音量调到日常对话水平。页面不会启用 ROS2，也不会上传原始校准音频。</p>
    ${unavailable ? `<div class="callout">设备枚举不可用：${escapeHtml(state.devices?.error || "未知错误")}</div>` : ""}
    <form id="setup-form" class="form-grid">
      <div class="field full"><label for="lab-room">测试房间</label><input id="lab-room" name="room" value="办公室" maxlength="128" required><small>报告只对这个房间、摆位和音量有效。</small></div>
      <div class="field"><label for="lab-input">麦克风输入</label><select id="lab-input" name="input" required>${deviceOptions("input")}</select></div>
      <div class="field"><label for="lab-output">扬声器输出</label><select id="lab-output" name="output" required>${deviceOptions("output")}</select></div>
      <div class="field"><label for="lab-aec">当前回声控制</label><select id="lab-aec" name="aec"><option value="none">未证明 / none</option><option value="system">系统级（待证明）</option><option value="hardware">硬件级（待证明）</option><option value="native">本地 WebRTC AEC（待证明）</option></select></div>
      <div class="field"><label for="lab-operator">操作员</label><input id="lab-operator" value="${escapeHtml(operatorId())}" readonly></div>
      <div class="field full"><div class="callout">当前采集器能自动完成发声检查和麦克风噪声校准；尚不能自动证明覆盖说话时的物理停播，因此首轮报告会严格标记为诊断。</div></div>
      <div class="field full"><div class="action-row"><button class="button primary" type="submit" ${unavailable ? "disabled" : ""}>创建 20+20+20 测试</button><button id="refresh-devices" class="button secondary" type="button">重新读取设备</button></div></div>
    </form>
  `;
  document.getElementById("setup-form").addEventListener("submit", startRun);
  document.getElementById("refresh-devices").addEventListener("click", async () => {
    showBusy("正在重新读取 Windows 音频设备…");
    await loadDevices();
    finishBusy();
  });
}

async function startRun(event) {
  event.preventDefault();
  const inputId = document.getElementById("lab-input").value;
  const outputId = document.getElementById("lab-output").value;
  const input = (state.devices?.devices || []).find((device) => String(device.index) === inputId);
  const output = (state.devices?.devices || []).find((device) => String(device.index) === outputId);
  const hostapis = state.devices?.hostapis || [];
  const hostapi = hostapis[input?.hostapi] || hostapis[output?.hostapi] || {};
  showBusy("正在创建可恢复测试运行…");
  try {
    state.run = await api("/api/voice/lab/runs", {
      method: "POST",
      key: idempotencyKey("create"),
      body: {
        operator_id: operatorId(),
        room: document.getElementById("lab-room").value.trim(),
        device_binding: {
          input_device_id: Number.isNaN(Number(inputId)) ? inputId : Number(inputId),
          output_device_id: Number.isNaN(Number(outputId)) ? outputId : Number(outputId),
          audio_device: `${input?.name || inputId} + ${output?.name || outputId}`,
          audio_driver: hostapi.name || "PortAudio",
          input_sample_rate_hz: Math.round(input?.default_samplerate || 48000),
          output_sample_rate_hz: Math.round(output?.default_samplerate || 48000),
          aec_backend: document.getElementById("lab-aec").value,
        },
      },
    });
    state.trialExecutionKeys.clear();
    state.trialExecutionFailures.clear();
    localStorage.setItem(STORAGE_KEY, state.run.run_id);
    showAlert("测试已创建。下一步会播放短测试音，请确认周围人员已知情。", "ok");
  } catch (error) {
    showAlert(`创建失败：${error.message}`);
  } finally {
    finishBusy();
  }
}

function renderDeviceCheck() {
  const check = state.run.device_check || {};
  workPanel.innerHTML = `
    <p class="panel-kicker">STEP 1 · ACOUSTIC ROUTE CHECK</p>
    <h2 class="panel-title">播放测试音，确认音箱能被麦克风听见</h2>
    <p class="panel-copy">点击后会播放约 0.8 秒、880Hz 的短音，并记录约 2 秒回采。它只证明设备路由可用，不是产品级首音证据。</p>
    <div class="instruction-card"><div class="instruction-index">♪</div><div><h3>操作前检查</h3><p>保持设备在日常摆位；音量不要临时调得特别大；测试期间不要说话。</p></div></div>
    ${check.status === "degraded" || check.status === "error" ? `<div class="callout">上次检查未通过：${escapeHtml(check.blocking_reason || check.failure_reason || check.error || "未检测到稳定测试音")}</div>` : ""}
    <div class="action-row"><button id="device-check-button" class="button primary" type="button">我已准备好，播放测试音</button></div>
  `;
  document.getElementById("device-check-button").addEventListener("click", () => mutate("device-check", {}, "正在播放并回采测试音…"));
}

function renderCalibration() {
  workPanel.innerHTML = `
    <p class="panel-kicker">STEP 2 · NOISE FLOOR</p>
    <h2 class="panel-title">保持安静 2 秒，建立这个房间的噪声底</h2>
    <p class="panel-copy">校准直接从所选麦克风采集，不保存原始 PCM，只保留 RMS 分位数和检测阈值。有人说话、碰桌子或移动设备时请重做。</p>
    <div class="instruction-card"><div class="instruction-index">2s</div><div><h3>现在不要说话</h3><p>关闭临时音乐和提示音，保持机器人与人之后测试时相同的位置。</p></div></div>
    <div class="action-row"><button id="calibrate-button" class="button primary" type="button">开始安静校准</button></div>
  `;
  document.getElementById("calibrate-button").addEventListener("click", () => mutate("calibration", { duration_s: 2.0 }, "正在采集房间噪声…"));
}

function renderTrial() {
  const action = state.run.next_action || {};
  const scenario = SCENARIOS[action.scenario];
  if (!scenario) return renderReportReady();
  if (action.action === "trial") return renderTrialStart(action, scenario);
  const activeTrial = state.run.active_trial || {};
  const turnEvidence = activeTrial.turn_evidence || null;
  const executionFailure = state.trialExecutionFailures.get(activeTrial.attempt_id) || "";
  state.selectedOutcome = null;
  state.selectedQuality = "clear";
  workPanel.innerHTML = `
    <p class="panel-kicker">${escapeHtml(scenario.kicker)} · TRIAL ${escapeHtml(action.ordinal)} / 20</p>
    <h2 class="panel-title">${escapeHtml(scenario.title)}</h2>
    <p class="panel-copy">${escapeHtml(scenario.instruction)}</p>
    <div class="instruction-card"><div class="instruction-index">${escapeHtml(action.ordinal)}</div><div><h3>${escapeHtml(scenario.question)}</h3><p>先完成动作，再选择结果。人工选择只记录主观体验，不会产生毫秒级物理证据。</p></div></div>
    <div class="trial-meta"><span class="mini-pill">${escapeHtml(action.scenario)}</span><span class="mini-pill">服务端试次已锁定</span><span class="mini-pill">${turnEvidence ? "服务端证据已锁定" : "等待服务端执行"}</span></div>
    ${turnEvidence
      ? renderTurnEvidence(turnEvidence, activeTrial.product_gate_usable)
      : `<div class="evidence-pending"><strong>${executionFailure ? "服务端运行证据尚不可用" : "尚未采集服务端运行证据"}</strong><span>${escapeHtml(executionFailure || "点击下方按钮执行本次试验；如果证据服务暂时不可用，本次 active attempt 会保留以便重试。")}</span><span>仍可选择仅保存人工诊断，但这会结束当前 active attempt，之后不能再采集本试次的服务端证据。</span></div>`}
    <div class="action-row"><button id="trigger-trial" class="button secondary" type="button" ${turnEvidence ? "disabled" : ""}>${turnEvidence ? "服务端证据已采集" : executionFailure ? "重试服务端证据" : escapeHtml(scenario.triggerLabel)}</button></div>
    <div class="choice-group" role="group" aria-label="本次结果">
      <button class="choice-button" type="button" data-outcome="positive"><strong>${escapeHtml(scenario.positive[0])}</strong><span>${escapeHtml(scenario.positive[1])}</span></button>
      <button class="choice-button" type="button" data-outcome="negative"><strong>${escapeHtml(scenario.negative[0])}</strong><span>${escapeHtml(scenario.negative[1])}</span></button>
    </div>
    <div class="field full" style="margin-top:1.2rem"><label>声音质量</label><div class="quality-group" role="group" aria-label="声音质量"><button class="quality-button selected" type="button" data-quality="clear">完整清楚</button><button class="quality-button" type="button" data-quality="clipped">首字截断</button><button class="quality-button" type="button" data-quality="choppy">断断续续</button><button class="quality-button" type="button" data-quality="unintelligible">听不清</button></div></div>
    <div class="field full" style="margin-top:1rem"><label for="trial-notes">备注（可选）</label><textarea id="trial-notes" maxlength="500" placeholder="例如：停播后还有半个字尾音；第一字偏轻…"></textarea></div>
    <div class="action-row"><button id="submit-trial" class="button primary" type="button">${turnEvidence ? "保存本次并进入下一次" : "仅保存人工诊断并结束本试次"}</button></div>
  `;
  document.getElementById("trigger-trial").addEventListener("click", () => executeActiveTrial(action.scenario));
  document.querySelectorAll("[data-outcome]").forEach((button) => button.addEventListener("click", () => {
    state.selectedOutcome = button.dataset.outcome;
    document.querySelectorAll("[data-outcome]").forEach((item) => item.classList.toggle("selected", item === button));
  }));
  document.querySelectorAll("[data-quality]").forEach((button) => button.addEventListener("click", () => {
    state.selectedQuality = button.dataset.quality;
    document.querySelectorAll("[data-quality]").forEach((item) => item.classList.toggle("selected", item === button));
  }));
  document.getElementById("submit-trial").addEventListener("click", () => submitTrial(action, scenario, Boolean(turnEvidence)));
}

function booleanEvidence(value) {
  return value === true ? "是" : "否";
}

function numericEvidence(value, suffix = "") {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)}${suffix}` : "—";
}

function renderEvidenceFacts(facts) {
  return facts.map(([label, value]) => `
    <div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>
  `).join("");
}

function renderEvidenceTimeline(timeline) {
  const orderedTimeline = timeline.slice().sort((left, right) => Number(left.sequence || 0) - Number(right.sequence || 0));
  const rows = orderedTimeline.length
    ? orderedTimeline.map((item) => `
      <li>
        <span class="timeline-sequence">#${escapeHtml(item.sequence)}</span>
        <div><strong>${escapeHtml(item.event)}</strong><span>${escapeHtml(item.stage)}</span></div>
        <time>${escapeHtml(numericEvidence(item.offset_ms, " ms"))}</time>
      </li>
    `).join("")
    : '<li class="timeline-empty">服务端没有返回可展示的时间线里程碑。</li>';
  return `<div class="evidence-section"><h4>有序时间线</h4><ol class="evidence-timeline">${rows}</ol></div>`;
}

function renderFallbackEvidence(fallback) {
  const route = fallback.used
    ? `${fallback.from || "未知"} → ${fallback.to || "未知"}`
    : "未发生";
  return `
    <article class="evidence-card">
      <h4>Fallback</h4>
      <dl>${renderEvidenceFacts([
        ["是否发生", booleanEvidence(fallback.used)],
        ["路由", route],
        ["原因", fallback.reason || "—"],
      ])}</dl>
    </article>
  `;
}

function renderInterruptEvidence(interrupt) {
  return `
    <article class="evidence-card">
      <h4>Interrupt</h4>
      <dl>${renderEvidenceFacts([
        ["检测到", booleanEvidence(interrupt.detected)],
        ["已确认", booleanEvidence(interrupt.confirmed)],
        ["已驳回", booleanEvidence(interrupt.dismissed)],
        ["恢复播放", booleanEvidence(interrupt.playback_resumed)],
      ])}</dl>
    </article>
  `;
}

function renderAecEvidence(aec) {
  return `
    <article class="evidence-card aec-evidence">
      <div class="evidence-card-title"><h4>AEC 算法遥测</h4><span>${escapeHtml(aec.evidence_kind || "未知")}</span></div>
      <dl>${renderEvidenceFacts([
        ["后端", aec.backend || "—"],
        ["活跃", booleanEvidence(aec.active)],
        ["降级", booleanEvidence(aec.degraded)],
        ["ERL", numericEvidence(aec.erl_db, " dB")],
        ["ERLE", numericEvidence(aec.erle_db, " dB")],
        ["残余回声概率", numericEvidence(aec.residual_echo_likelihood)],
      ])}</dl>
      <p class="evidence-caveat">算法 AEC 遥测不等于物理门禁；它只能说明运行时算法状态，不能代替隔离麦克风或目标硬件上的残余音频测量。</p>
    </article>
  `;
}

function renderResidualAudioEvidence(residual) {
  if (!residual) {
    return '<div class="physical-evidence-missing">本次没有物理 residual_audio 证据；不会把算法遥测渲染成物理通过。</div>';
  }
  return `
    <article class="evidence-card physical-evidence">
      <div class="evidence-card-title"><h4>物理残余音频</h4><span>${escapeHtml(residual.evidence_kind || "未知")}</span></div>
      <dl>${renderEvidenceFacts([
        ["测量来源", residual.measurement_source || "—"],
        ["时钟域", residual.clock_domain || "—"],
        ["丢帧", residual.dropped_frames ?? "—"],
        ["尾音", numericEvidence(residual.tail_ms, " ms")],
      ])}</dl>
      <p class="evidence-caveat">这是物理测量元数据，但单个试次仍不代表整套产品门禁已经通过；最终状态以严格报告为准。</p>
    </article>
  `;
}

function renderTurnEvidence(evidence, productGateUsable = false) {
  const timeline = Array.isArray(evidence.timeline) ? evidence.timeline : [];
  const gateReviewable = productGateUsable === true;
  return `
    <section class="turn-evidence" aria-label="服务端运行证据">
      <div class="evidence-heading">
        <div><p class="panel-kicker">SERVER-OWNED TURN EVIDENCE</p><h3>本次运行事实</h3></div>
        <span class="evidence-gate ${gateReviewable ? "reviewable" : "blocked"}">${gateReviewable ? "可进入最终门禁评估" : "不可用于产品门禁"}</span>
      </div>
      <dl class="evidence-identity">${renderEvidenceFacts([
        ["来源", evidence.source || "未知"],
        ["关联 ID", evidence.correlation_id || "—"],
        ["采集时间", evidence.captured_at || "—"],
      ])}</dl>
      ${renderEvidenceTimeline(timeline)}
      <div class="evidence-grid">
        ${renderFallbackEvidence(evidence.fallback || {})}
        ${renderInterruptEvidence(evidence.interrupt || {})}
      </div>
      ${renderAecEvidence(evidence.aec_stats || {})}
      ${renderResidualAudioEvidence(evidence.residual_audio || null)}
    </section>
  `;
}

function renderTrialStart(action, scenario) {
  workPanel.innerHTML = `
    <p class="panel-kicker">${escapeHtml(scenario.kicker)} · TRIAL ${escapeHtml(action.ordinal)} / 20</p>
    <h2 class="panel-title">准备开始 ${escapeHtml(scenario.title)}</h2>
    <p class="panel-copy">${escapeHtml(scenario.instruction)}</p>
    <div class="instruction-card"><div class="instruction-index">${escapeHtml(action.ordinal)}</div><div><h3>先锁定本次试验</h3><p>开始后服务器会生成唯一 attempt_id；只有这次试验的结果可以提交，暂停会使它失效。</p></div></div>
    <div class="action-row"><button id="begin-trial" class="button primary" type="button">开始并锁定本次试验</button></div>
  `;
  document.getElementById("begin-trial").addEventListener("click", () => {
    mutate("trials/begin", {}, `正在开始 ${scenario.title} #${action.ordinal}…`);
  });
}

async function executeActiveTrial(scenario) {
  if (!state.run || state.busy) return;
  const attemptId = state.run.active_trial?.attempt_id;
  if (!attemptId) {
    showAlert("本次试验没有服务端 attempt_id，请刷新后重新开始。", "warn");
    return;
  }
  const executePath = `/api/voice/lab/runs/${encodeURIComponent(state.run.run_id)}/trials/${encodeURIComponent(attemptId)}/execute`;
  const instruction = scenario === "human_overlap"
    ? "服务端开始播报后，请在听到声音约 1 秒时说“停一下”。"
    : scenario === "assistant_response"
      ? "服务端开始采集后，请直接面对机器人说测试句并等待回答。"
      : "服务端开始播报后，请保持安静并观察。";
  const preservedRun = state.run;
  showAlert(instruction, "ok");
  showBusy("正在执行试验并采集服务端证据…");
  try {
    state.run = await api(executePath, {
      method: "POST",
      version: state.run.version,
      key: trialExecutionKey(attemptId),
    });
    state.trialExecutionFailures.delete(attemptId);
    localStorage.setItem(STORAGE_KEY, state.run.run_id);
    showAlert("服务端试验已完成，可信运行证据已锁定。请记录你的主观结果。", "ok");
  } catch (error) {
    if (error.status === 409) {
      try {
        state.run = await api(`/api/voice/lab/runs/${encodeURIComponent(state.run.run_id)}`);
        if (state.run.active_trial?.turn_evidence) state.trialExecutionFailures.delete(attemptId);
        showAlert(`执行状态已由其他页面更新，已自动刷新：${error.message}`, "warn");
      } catch {
        state.run = preservedRun;
        state.trialExecutionFailures.set(attemptId, `执行状态冲突且刷新失败：${error.message}`);
        showAlert(`执行状态冲突且刷新失败：${error.message}`, "warn");
      }
    } else if (error.status === 503) {
      state.run = preservedRun;
      state.trialExecutionFailures.set(attemptId, `服务端证据不可用（503）：${error.message}。本次 active attempt 已保留，可直接重试。`);
      showAlert(`服务端证据不可用（503）：${error.message}。本次 active attempt 已保留，可直接重试。`, "warn");
    } else {
      state.run = preservedRun;
      state.trialExecutionFailures.set(attemptId, `服务端证据采集失败：${error.message}。本次 active attempt 已保留，可重试。`);
      showAlert(`服务端证据采集失败：${error.message}。本次 active attempt 已保留，可重试。`, "warn");
    }
  } finally {
    finishBusy();
  }
}

async function submitTrial(action, scenario, hasServerEvidence) {
  if (!state.selectedOutcome) {
    showAlert("请先选择本次结果。", "warn");
    return;
  }
  const positive = state.selectedOutcome === "positive";
  const attemptId = state.run.active_trial?.attempt_id || action.attempt_id;
  if (!attemptId) {
    showAlert("本次试验没有服务端 attempt_id，请刷新后重新开始。", "warn");
    return;
  }
  if (!hasServerEvidence) {
    const continueWithoutEvidence = window.confirm(
      "服务端运行证据尚未采集。继续只会保存人工诊断，并结束这个 active attempt；之后不能重试该试次的服务端证据。是否继续？",
    );
    if (!continueWithoutEvidence) {
      showAlert("本次 active attempt 已保留，可继续重试服务端证据。", "warn");
      return;
    }
  }
  const body = {
    attempt_id: attemptId,
    scenario: action.scenario,
    ordinal: action.ordinal,
    quality: state.selectedQuality,
    notes: document.getElementById("trial-notes").value.trim(),
    [scenario.field]: positive ? scenario.positiveValue : scenario.negativeValue,
  };
  await mutate("trials", body, `正在保存 ${scenario.title} #${action.ordinal}…`);
}

function renderPaused() {
  workPanel.innerHTML = `
    <p class="panel-kicker">PAUSED · SAFE TO LEAVE</p><h2 class="panel-title">进度已保存</h2>
    <p class="panel-copy">已完成 ${escapeHtml(state.run.progress?.total || 0)} / 60。恢复时必须重新做设备检查和安静校准，已有试次不会丢失。</p>
    <div class="action-row"><button id="panel-resume" class="button primary" type="button">恢复并重新检查设备</button></div>`;
  document.getElementById("panel-resume").addEventListener("click", () => mutate("resume", {}));
}

function renderReportReady() {
  workPanel.innerHTML = `
    <p class="panel-kicker">STEP 6 · STRICT REPORT</p><h2 class="panel-title">60 次人工协作测试已完成</h2>
    <p class="panel-copy">现在生成硬件 schema v2 报告。当前人工标记会完整保留，但物理停播和物理首音样本仍为 0，因此报告会诚实地显示未通过产品门禁。</p>
    <div class="callout">“跑完”表示主观诊断完整，不等于物理验收通过。这正是 fail-closed 设计。</div>
    <div class="action-row"><button id="report-button" class="button primary" type="button">生成严格报告</button></div>`;
  document.getElementById("report-button").addEventListener("click", () => mutate("report", {}, "正在生成证据报告…"));
}

function renderReport() {
  const gate = state.run.product_gate || {};
  const report = gate.report || {};
  const failed = report.failed_checks || [];
  const passed = gate.status === "passed";
  workPanel.innerHTML = `
    <p class="panel-kicker">REPORT · ${passed ? "PHYSICAL PASS" : "DIAGNOSTIC COMPLETE"}</p>
    <h2 class="panel-title">${passed ? "产品级物理门禁已通过" : "诊断完成，但不能宣称产品级通过"}</h2>
    <p class="panel-copy">${escapeHtml(gate.reason || "报告已生成")}</p>
    <div class="instruction-card"><div class="instruction-index" style="background:${passed ? "var(--green)" : "var(--amber)"}">${passed ? "✓" : "!"}</div><div><h3>${escapeHtml(state.run.progress?.total || 0)} / 60 人工试次已保存</h3><p>报告 artifact：${escapeHtml(gate.artifact || "—")}</p></div></div>
    ${failed.length ? `<h3 style="margin-top:1.5rem">仍未满足的严格检查</h3><ul class="report-list">${failed.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}
    <div class="action-row"><button id="download-report" class="button primary" type="button">下载报告 JSON</button><a class="button secondary" href="/dashboard/voice">返回语音系统</a></div>`;
  document.getElementById("download-report").addEventListener("click", () => downloadJson(`${state.run.run_id}-hardware-report.json`, report));
}

function renderTerminal() {
  workPanel.innerHTML = `<p class="panel-kicker">RUN ${escapeHtml(state.run.status)}</p><h2 class="panel-title">本次运行已结束</h2><p class="panel-copy">服务器证据仍保留。可从浏览器移除引用后创建新测试。</p>`;
}

async function mutate(action, body = {}, busyLabel = "正在保存…") {
  if (!state.run || state.busy) return;
  const activeAttemptId = state.run.active_trial?.attempt_id;
  showBusy(busyLabel);
  try {
    state.run = await api(`/api/voice/lab/runs/${encodeURIComponent(state.run.run_id)}/${action}`, {
      method: "POST",
      body,
      version: state.run.version,
      key: idempotencyKey(`${action}-${state.run.version}`),
    });
    if (action === "trials" && activeAttemptId) {
      state.trialExecutionKeys.delete(activeAttemptId);
      state.trialExecutionFailures.delete(activeAttemptId);
    }
    localStorage.setItem(STORAGE_KEY, state.run.run_id);
    state.selectedOutcome = null;
    showAlert(action === "trials" ? "本次已保存。" : "操作完成。", "ok");
  } catch (error) {
    if (error.status === 409) {
      try {
        state.run = await api(`/api/voice/lab/runs/${encodeURIComponent(state.run.run_id)}`);
        showAlert(`状态已由其他页面更新，已自动刷新：${error.message}`, "warn");
      } catch {
        showAlert(`状态冲突且刷新失败：${error.message}`);
      }
    } else {
      showAlert(`操作失败：${error.message}`);
    }
  } finally {
    finishBusy();
  }
}

function downloadJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

boot();
