"""Browser-facing dashboard contracts exercised with Node's built-in VM."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "askme" / "static" / "dashboard" / "app.js"
NODE = shutil.which("node")


NODE_HARNESS = r"""
const fs = require("node:fs");
const vm = require("node:vm");
const { webcrypto } = require("node:crypto");

const appPath = process.argv[1];
const scenario = process.argv[2];
const source = fs.readFileSync(appPath, "utf8").replace(
  /\r?\nbootDashboard\(\);\s*$/,
  "\n",
);

function createElement(id = "") {
  return {
    id,
    value: "",
    innerHTML: "",
    textContent: "",
    className: "",
    dataset: {},
    disabled: false,
    children: [],
    style: {},
    classList: {
      add() {},
      remove() {},
      toggle() {},
      contains() { return false; },
    },
    addEventListener() {},
    appendChild(child) { this.children.push(child); },
    querySelector() { return null; },
    setAttribute() {},
    focus() {},
  };
}

function loadDashboard() {
  const elements = new Map();
  const requests = [];
  const storageWrites = [];
  const storage = new Map([["askme.operator_id", "operator-private"]]);
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, createElement(id));
    return elements.get(id);
  };
  const document = {
    body: createElement("body"),
    getElementById: element,
    createElement,
    querySelector() { return null; },
    querySelectorAll() { return []; },
    addEventListener() {},
  };
  const context = {
    Blob,
    URL,
    console,
    crypto: webcrypto,
    document,
    fetch: async (_url, options = {}) => {
      requests.push(JSON.parse(options.body || "{}"));
      return {
        ok: true,
        async json() { return { reply: "ok" }; },
      };
    },
    history: { pushState() {} },
    localStorage: {
      getItem(key) { return storage.has(key) ? storage.get(key) : null; },
      setItem(key, value) {
        storage.set(key, String(value));
        storageWrites.push([key, String(value)]);
      },
    },
    location: {
      origin: "http://dashboard.test",
      pathname: "/dashboard/conversation",
    },
    setInterval() { return 0; },
    clearInterval() {},
    setTimeout() { return 0; },
    clearTimeout() {},
  };
  context.window = context;
  context.globalThis = context;
  context.addEventListener = () => {};
  context.matchMedia = () => ({
    matches: false,
    addEventListener() {},
  });
  vm.createContext(context);
  vm.runInContext(
    `${source}\n;globalThis.__dashboardContract = { sendChat, renderVoiceModels };`,
    context,
    { filename: appPath },
  );
  return { context, elements, requests, storageWrites };
}

function modelCard(html, component) {
  const match = html.match(new RegExp(
    `<article class="voice-model-card"[^>]*data-model-card="${component}"[\\s\\S]*?</article>`,
  ));
  if (!match) throw new Error(`missing ${component} model card`);
  return match[0];
}

function currentModel(card) {
  const match = card.match(/<div class="voice-current-model">([\s\S]*?)<\/div>/);
  if (!match) throw new Error("missing current model block");
  return match[1];
}

async function runChatContract() {
  async function pageLifecycle(messages) {
    const dashboard = loadDashboard();
    for (const message of messages) {
      dashboard.context.document.getElementById("chat-input").value = message;
      await dashboard.context.__dashboardContract.sendChat();
    }
    return {
      requests: dashboard.requests,
      storageWrites: dashboard.storageWrites,
    };
  }
  async function overlappingSubmissions() {
    const dashboard = loadDashboard();
    const input = dashboard.context.document.getElementById("chat-input");
    const button = dashboard.context.document.getElementById("chat-send");
    let releaseFirst;
    let requestCount = 0;
    dashboard.context.fetch = async (_url, options = {}) => {
      dashboard.requests.push(JSON.parse(options.body || "{}"));
      requestCount += 1;
      if (requestCount === 1) {
        await new Promise((resolve) => { releaseFirst = resolve; });
      }
      return {
        ok: true,
        async json() { return { reply: "ok" }; },
      };
    };

    input.value = "first";
    const first = dashboard.context.__dashboardContract.sendChat();
    await Promise.resolve();
    input.value = "second";
    await dashboard.context.__dashboardContract.sendChat();
    const during = { input: input.disabled, button: button.disabled, value: input.value };
    releaseFirst();
    await first;
    const after = { input: input.disabled, button: button.disabled, value: input.value };
    await dashboard.context.__dashboardContract.sendChat();
    return { requests: dashboard.requests, during, after };
  }

  return {
    firstPage: await pageLifecycle(["first visitor question", "follow-up"]),
    secondPage: await pageLifecycle(["unrelated visitor"]),
    overlap: await overlappingSubmissions(),
  };
}

function runVoiceContract() {
  const dashboard = loadDashboard();
  const renderVoiceModels = dashboard.context.__dashboardContract.renderVoiceModels;
  const modernHtml = renderVoiceModels({
    runtime: {
      llm: { provider: "deepseek", model: "deepseek-chat" },
      asr: {
        provider: "local",
        local: { provider: "sherpa_onnx", available: true },
        cloud: {
          provider: "volcengine_seed_asr",
          model: "configured-cloud-asr",
          available: false,
        },
      },
      tts: {
        backend: "minimax",
        minimax: { model: "configured-cloud-tts", voice_id: "configured-voice" },
      },
      switches: {
        asr: {
          state: "pending",
          effective: { provider: "local", model: "local" },
          desired: { provider: "volcengine", model: "configured-cloud-asr" },
          pending: { provider: "volcengine", model: "configured-cloud-asr" },
          failed: null,
        },
        tts: {
          state: "failed",
          effective: {
            backend: "edge",
            model: "zh-CN-XiaoxiaoNeural",
            voice_id: "zh-CN-XiaoxiaoNeural",
          },
          desired: {
            backend: "minimax",
            model: "configured-cloud-tts",
            voice_id: "configured-voice",
          },
          pending: null,
          failed: { reason: "provider warmup failed" },
        },
      },
    },
    catalog: {
      llm: [],
      asr: [
        { provider: "local", models: ["local"], credential_ready: true },
        {
          provider: "volcengine",
          models: ["configured-cloud-asr"],
          credential_ready: true,
        },
      ],
      tts: [
        { backend: "edge", models: ["zh-CN-XiaoxiaoNeural"], credential_ready: true },
        { backend: "minimax", models: ["configured-cloud-tts"], credential_ready: true },
      ],
    },
  });
  const legacyHtml = renderVoiceModels({
    runtime: {
      llm: { provider: "deepseek", model: "deepseek-chat" },
      asr: {
        provider: "local",
        local: { provider: "sherpa_onnx", available: true },
        cloud: {
          provider: "volcengine_seed_asr",
          model: "legacy-configured-cloud",
          available: false,
        },
      },
      tts: {
        backend: "minimax",
        minimax: { model: "speech-2.8-turbo", voice_id: "legacy-voice" },
      },
    },
    catalog: { llm: [], asr: [], tts: [] },
  });
  const directLegacyHtml = renderVoiceModels({
    runtime: {
      llm: { provider: "deepseek", model: "deepseek-chat" },
      asr: {
        provider: "volcengine_seed_asr",
        model: "direct-configured-cloud",
        available: false,
        local: { provider: "sherpa_onnx", available: true },
      },
      tts: { backend: "local", model: "local" },
    },
    catalog: { llm: [], asr: [], tts: [] },
  });
  const modernAsr = modelCard(modernHtml, "asr");
  const modernTts = modelCard(modernHtml, "tts");
  const legacyAsr = modelCard(legacyHtml, "asr");
  const legacyTts = modelCard(legacyHtml, "tts");
  return {
    modernAsr,
    modernAsrCurrent: currentModel(modernAsr),
    modernTts,
    modernTtsCurrent: currentModel(modernTts),
    legacyAsr,
    legacyAsrCurrent: currentModel(legacyAsr),
    legacyTtsCurrent: currentModel(legacyTts),
    directLegacyAsrCurrent: currentModel(modelCard(directLegacyHtml, "asr")),
  };
}

Promise.resolve(
  scenario === "chat" ? runChatContract() : runVoiceContract(),
).then((result) => {
  process.stdout.write(JSON.stringify(result));
}).catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
"""


def _run_dashboard_contract(scenario: str) -> dict:
    if NODE is None:
        pytest.skip("Node.js is required for the dashboard browser contract")
    result = subprocess.run(
        [NODE, "-e", NODE_HARNESS, str(APP_JS), scenario],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_browser_chat_uses_one_private_thread_id_per_page_lifecycle() -> None:
    result = _run_dashboard_contract("chat")
    first_requests = result["firstPage"]["requests"]
    second_requests = result["secondPage"]["requests"]

    assert len(first_requests) == 2
    first_id = first_requests[0]["conversation_session_id"]
    assert first_requests[0]["conversation_thread_id"] == first_id
    assert first_requests[1]["conversation_session_id"] == first_id
    assert first_requests[1]["conversation_thread_id"] == first_id
    assert second_requests[0]["conversation_session_id"] != first_id
    assert first_id.startswith("web-")
    assert len(first_id) == 40
    assert "operator-private" not in first_id
    assert "visitor" not in first_id
    assert all(
        "conversation" not in key.lower() and "session" not in key.lower()
        for key, _value in result["firstPage"]["storageWrites"]
    )
    overlap = result["overlap"]
    assert [request["text"] for request in overlap["requests"]] == ["first", "second"]
    assert overlap["during"] == {"input": True, "button": True, "value": "second"}
    assert overlap["after"] == {"input": False, "button": False, "value": "second"}


def test_voice_model_cards_show_effective_runtime_and_switch_state() -> None:
    result = _run_dashboard_contract("voice")

    assert 'data-model-state="pending"' in result["modernAsr"]
    assert ">PENDING<" in result["modernAsr"]
    assert "<strong>local</strong>" in result["modernAsrCurrent"]
    assert "Sherpa-ONNX" in result["modernAsrCurrent"]
    assert "configured-cloud-asr" not in result["modernAsrCurrent"]

    assert 'data-model-state="failed"' in result["modernTts"]
    assert ">FAILED<" in result["modernTts"]
    assert "<strong>zh-CN-XiaoxiaoNeural</strong>" in result["modernTtsCurrent"]
    assert "Microsoft Edge TTS" in result["modernTtsCurrent"]
    assert "configured-cloud-tts" not in result["modernTtsCurrent"]
    assert "provider warmup failed" in result["modernTts"]

    assert 'data-model-state="active"' in result["legacyAsr"]
    assert "<strong>local</strong>" in result["legacyAsrCurrent"]
    assert "legacy-configured-cloud" not in result["legacyAsrCurrent"]
    assert "<strong>speech-2.8-turbo</strong>" in result["legacyTtsCurrent"]
    assert "<strong>local</strong>" in result["directLegacyAsrCurrent"]
    assert "direct-configured-cloud" not in result["directLegacyAsrCurrent"]
