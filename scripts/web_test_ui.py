#!/usr/bin/env python3
"""
Askme Web Test UI — 浏览器实时交互 + 日志流 + 技能测试面板
用法: python scripts/web_test_ui.py [--port 8765] [--host 0.0.0.0]
访问: http://<sunrise-ip>:8765
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ── WebSocket log handler ──────────────────────────────────────────────────

_loop: asyncio.AbstractEventLoop | None = None
_connections: list[WebSocket] = []


def _safe_broadcast(msg: dict) -> None:
    """Thread-safe fire-and-forget broadcast."""
    if _loop is None:
        return
    asyncio.run_coroutine_threadsafe(_broadcast(msg), _loop)


async def _broadcast(msg: dict) -> None:
    for ws in list(_connections):
        try:
            await ws.send_json(msg)
        except Exception:
            pass


class _WsLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _safe_broadcast({
                "type": "log",
                "level": record.levelname.lower(),
                "name": record.name.replace("askme.", ""),
                "msg": self.format(record),
            })
        except Exception:
            pass


_ws_handler = _WsLogHandler()
_ws_handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d [%(name)s] %(message)s", "%H:%M:%S"))
_ws_handler.setLevel(logging.DEBUG)


# ── Mock AudioAgent ────────────────────────────────────────────────────────

class WebAudio:
    """Captures TTS text; optionally delegates to real AudioAgent for speaker output.

    Set ``voice_enabled = True`` to also play audio on sunrise's speaker.
    """

    def __init__(self, real_audio: Any = None) -> None:
        self._real = real_audio
        self.voice_enabled: bool = False

    def speak(self, text: str) -> None:
        _safe_broadcast({"type": "speak", "text": text})
        if self.voice_enabled and self._real is not None:
            try:
                self._real.speak(text)
            except Exception:
                pass

    def start_playback(self) -> None:
        if self.voice_enabled and self._real is not None:
            try:
                self._real.start_playback()
            except Exception:
                pass

    def stop_playback(self) -> None:
        if self.voice_enabled and self._real is not None:
            try:
                self._real.stop_playback()
            except Exception:
                pass

    def wait_speaking_done(self) -> None:
        if self.voice_enabled and self._real is not None:
            try:
                self._real.wait_speaking_done()
            except Exception:
                pass

    def drain_buffers(self) -> None:
        if self.voice_enabled and self._real is not None:
            try:
                self._real.drain_buffers()
            except Exception:
                pass

    def is_active(self) -> bool:
        if self.voice_enabled and self._real is not None:
            try:
                return self._real.is_active()
            except Exception:
                pass
        return False

    @property
    def tts(self) -> "WebAudio":
        return self


# ── Pipeline factory ───────────────────────────────────────────────────────

def build_components() -> tuple[Any, Any, Any]:
    """Instantiate App (text mode) and replace audio with WebAudio mock."""
    import asyncio
    from askme.blueprints.text import text as text_blueprint
    from askme.config import get_config

    cfg = get_config()
    app_instance = asyncio.get_event_loop().run_until_complete(
        text_blueprint.build(cfg),
    )

    pipeline_mod = app_instance.modules.get("pipeline")
    pipeline = getattr(pipeline_mod, "brain_pipeline", None)

    skill_mod = app_instance.modules.get("skill")
    dispatcher = getattr(skill_mod, "skill_dispatcher", None)
    skill_manager = getattr(skill_mod, "skill_manager", None)

    text_mod = app_instance.modules.get("text")
    voice_mod = app_instance.modules.get("voice")
    audio = getattr(voice_mod, "audio", None) if voice_mod else None
    if audio is None:
        audio = getattr(text_mod, "_text_audio", None) if text_mod else None

    # Replace the real AudioAgent with our WebAudio mock so TTS goes to
    # WebSocket instead of a speaker, and mic calls are safely no-ops.
    web_audio = WebAudio(real_audio=audio)
    if pipeline is not None:
        pipeline._audio = web_audio  # noqa: SLF001
    if dispatcher is not None:
        dispatcher._audio = web_audio  # noqa: SLF001

    return pipeline, dispatcher, skill_manager


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(title="Askme Web Test UI")
_pipeline: Any = None
_dispatcher: Any = None
_skill_mgr: Any = None


@app.on_event("startup")
async def _startup() -> None:
    global _pipeline, _dispatcher, _skill_mgr, _loop
    _loop = asyncio.get_running_loop()

    root_logger = logging.getLogger("askme")
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(_ws_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _pipeline, _dispatcher, _skill_mgr = build_components()
    logging.getLogger("askme.web_ui").info("Web Test UI ready.")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _HTML


@app.get("/api/skills")
async def api_skills() -> list[dict]:
    if _skill_mgr is None:
        return []
    return [
        {"name": s.name, "description": s.description, "timeout": s.timeout,
         "confirm": getattr(s, "confirm_before_execute", False)}
        for s in _skill_mgr.get_enabled()
    ]


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _connections.append(ws)
    logger = logging.getLogger("askme.web_ui")
    logger.info("Browser connected.")
    try:
        while True:
            data = await ws.receive_json()
            mtype = data.get("type", "")
            text: str = data.get("text", "").strip()

            if mtype == "chat":
                if not text:
                    continue
                await _broadcast({"type": "user", "text": text})
                try:
                    result = await _dispatcher.handle_general(text, source="web")
                    await _broadcast({"type": "assistant", "text": result})
                except Exception as exc:
                    logger.error("handle_general error: %s", exc)
                    await _broadcast({"type": "error", "text": f"[Error] {type(exc).__name__}: {exc}"})

            elif mtype == "skill":
                skill_name: str = data.get("skill", "").strip()
                if not skill_name:
                    continue
                user_input = text or f"执行 {skill_name}"
                await _broadcast({"type": "skill_call", "skill": skill_name, "text": user_input})
                try:
                    result = await _dispatcher.dispatch(skill_name, user_input, source="web")
                    await _broadcast({"type": "skill_result", "skill": skill_name, "text": result})
                except Exception as exc:
                    logger.error("dispatch error [%s]: %s", skill_name, exc)
                    await _broadcast({"type": "error", "text": f"[Error] {skill_name}: {type(exc).__name__}: {exc}"})

            elif mtype == "set_voice":
                enabled: bool = bool(data.get("enabled", False))
                if _pipeline and hasattr(_pipeline, "_audio"):
                    _pipeline._audio.voice_enabled = enabled  # type: ignore[attr-defined]
                state = "开启" if enabled else "关闭"
                await _broadcast({"type": "system", "text": f"语音播放已{state}（sunrise 扬声器）。"})

            elif mtype == "clear_history":
                if _pipeline and hasattr(_pipeline, "_conversation"):
                    _pipeline._conversation.clear()
                await _broadcast({"type": "system", "text": "对话历史已清空。"})

            elif mtype == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Browser disconnected.")
    finally:
        if ws in _connections:
            _connections.remove(ws)


# ── HTML frontend ──────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Askme Web Test UI</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Consolas','Menlo',monospace;font-size:13px;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
header{background:#16213e;padding:10px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #0f3460}
header h1{font-size:15px;color:#4fc3f7;letter-spacing:1px}
#status{padding:3px 8px;border-radius:12px;font-size:11px;background:#c62828;color:#fff}
#status.ok{background:#2e7d32}
.main{flex:1;display:flex;overflow:hidden}
/* Left: conversation */
#conv-panel{flex:1;display:flex;flex-direction:column;border-right:1px solid #0f3460;min-width:300px}
#conv-title{padding:8px 12px;background:#16213e;font-size:11px;color:#90a4ae;border-bottom:1px solid #0f3460}
#conv-msgs{flex:1;overflow-y:auto;padding:10px;display:flex;flex-direction:column;gap:6px}
.msg{padding:8px 12px;border-radius:6px;max-width:90%;word-break:break-word;line-height:1.5}
.msg.user{background:#1565c0;align-self:flex-end;color:#e3f2fd}
.msg.assistant{background:#263238;align-self:flex-start;color:#e0e0e0;border-left:3px solid #4fc3f7}
.msg.speak{background:#1a237e;align-self:flex-start;color:#c5cae9;font-style:italic}
.msg.skill_call{background:#4a148c;align-self:flex-start;color:#e1bee7}
.msg.skill_result{background:#1b5e20;align-self:flex-start;color:#c8e6c9;border-left:3px solid #66bb6a}
.msg.error{background:#b71c1c;align-self:flex-start;color:#ffcdd2}
.msg.system{background:#37474f;align-self:center;color:#b0bec5;font-size:11px}
.msg .tag{font-size:10px;color:#78909c;margin-bottom:3px}
/* Input area */
#input-area{padding:10px;border-top:1px solid #0f3460;background:#16213e;display:flex;gap:8px}
#msg-input{flex:1;background:#263238;border:1px solid #455a64;color:#e0e0e0;padding:8px 10px;border-radius:4px;font-size:13px;font-family:inherit}
#msg-input:focus{outline:none;border-color:#4fc3f7}
button{padding:7px 14px;border:none;border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px}
#send-btn{background:#1565c0;color:#fff}
#clear-btn{background:#37474f;color:#ccc}
/* Right: logs + skills */
#right-panel{width:420px;display:flex;flex-direction:column;min-width:280px}
/* Skills */
#skills-section{padding:8px;background:#16213e;border-bottom:1px solid #0f3460}
#skills-title{font-size:11px;color:#90a4ae;margin-bottom:6px}
#skills-grid{display:flex;flex-wrap:wrap;gap:5px;max-height:100px;overflow-y:auto}
.skill-btn{padding:4px 8px;background:#0d47a1;color:#e3f2fd;border-radius:3px;font-size:11px;cursor:pointer;border:1px solid #1565c0}
.skill-btn:hover{background:#1565c0}
.skill-btn.confirm{border-color:#f57f17;background:#e65100}
/* Logs */
#log-title{padding:8px 12px;background:#16213e;font-size:11px;color:#90a4ae;border-bottom:1px solid #0f3460;display:flex;justify-content:space-between;align-items:center}
#log-filter{background:#263238;border:1px solid #455a64;color:#ccc;padding:2px 6px;border-radius:3px;font-size:11px}
#log-panel{flex:1;overflow-y:auto;padding:8px;font-size:11px;line-height:1.4}
.log{padding:2px 0;border-bottom:1px solid #1e2a38;white-space:pre-wrap;word-break:break-all}
.log.debug{color:#546e7a}
.log.info{color:#80cbc4}
.log.warning{color:#ffcc02}
.log.error{color:#ef5350}
.log.critical{color:#ff1744;font-weight:bold}
#log-autoscroll{accent-color:#4fc3f7}
</style>
</head>
<body>
<header>
  <h1>🤖 Askme Web Test UI</h1>
  <span id="status">连接中...</span>
  <span style="color:#78909c;font-size:11px" id="conn-info"></span>
</header>
<div class="main">
  <!-- Left: conversation -->
  <div id="conv-panel">
    <div id="conv-title">对话 · 文本交互</div>
    <div id="conv-msgs"></div>
    <div id="input-area">
      <input id="msg-input" placeholder="输入消息，或 /skill 技能名 参数" autocomplete="off">
      <button id="send-btn" onclick="sendChat()">发送</button>
      <button id="voice-btn" onclick="toggleVoice()" style="background:#37474f;color:#ccc">🔇 静音</button>
      <button id="clear-btn" onclick="clearHistory()">清空历史</button>
    </div>
  </div>
  <!-- Right: skills + logs -->
  <div id="right-panel">
    <div id="skills-section">
      <div id="skills-title">可用技能 (点击执行)</div>
      <div id="skills-grid">加载中...</div>
    </div>
    <div id="log-title">
      实时日志
      <div style="display:flex;gap:8px;align-items:center">
        <select id="log-filter" onchange="filterLogs()">
          <option value="all">全部</option>
          <option value="info">≥ INFO</option>
          <option value="warning">≥ WARNING</option>
        </select>
        <label style="font-size:11px;color:#90a4ae">
          <input type="checkbox" id="log-autoscroll" checked> 自动滚动
        </label>
        <button onclick="clearLogs()" style="font-size:10px;padding:2px 6px;background:#37474f;color:#ccc">清空</button>
      </div>
    </div>
    <div id="log-panel"></div>
  </div>
</div>

<script>
let ws;
let logFilter = 'all';
const convMsgs = document.getElementById('conv-msgs');
const logPanel = document.getElementById('log-panel');
const statusEl = document.getElementById('status');
const connInfo = document.getElementById('conn-info');

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    statusEl.textContent = '已连接';
    statusEl.className = 'ok';
    connInfo.textContent = location.host;
    loadSkills();
  };
  ws.onclose = () => {
    statusEl.textContent = '断开';
    statusEl.className = '';
    setTimeout(connect, 2000);
  };
  ws.onerror = () => {};
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handleMsg(msg);
  };
}

function handleMsg(msg) {
  switch (msg.type) {
    case 'user':
      addConvMsg('user', msg.text, '你');
      break;
    case 'assistant':
      addConvMsg('assistant', msg.text, 'Thunder');
      break;
    case 'speak':
      addConvMsg('speak', msg.text, '🔊 TTS');
      break;
    case 'skill_call':
      addConvMsg('skill_call', `[${msg.skill}] ${msg.text}`, '▶ 技能调用');
      break;
    case 'skill_result':
      addConvMsg('skill_result', msg.text, `✓ ${msg.skill}`);
      break;
    case 'error':
      addConvMsg('error', msg.text, '⚠ 错误');
      break;
    case 'system':
      addConvMsg('system', msg.text, '');
      break;
    case 'log':
      addLog(msg);
      break;
    case 'pong':
      break;
  }
}

function addConvMsg(type, text, tag) {
  const div = document.createElement('div');
  div.className = `msg ${type}`;
  if (tag) {
    const tagEl = document.createElement('div');
    tagEl.className = 'tag';
    tagEl.textContent = tag;
    div.appendChild(tagEl);
  }
  const body = document.createElement('div');
  body.textContent = text;
  div.appendChild(body);
  convMsgs.appendChild(div);
  convMsgs.scrollTop = convMsgs.scrollHeight;
}

function addLog(msg) {
  const div = document.createElement('div');
  div.className = `log ${msg.level}`;
  div.dataset.level = msg.level;
  div.textContent = msg.msg;
  div.style.display = shouldShowLog(msg.level) ? '' : 'none';
  logPanel.appendChild(div);
  // Keep max 500 log lines
  while (logPanel.children.length > 500) {
    logPanel.removeChild(logPanel.firstChild);
  }
  if (document.getElementById('log-autoscroll').checked) {
    logPanel.scrollTop = logPanel.scrollHeight;
  }
}

function shouldShowLog(level) {
  if (logFilter === 'all') return true;
  const order = ['debug', 'info', 'warning', 'error', 'critical'];
  return order.indexOf(level) >= order.indexOf(logFilter);
}

function filterLogs() {
  logFilter = document.getElementById('log-filter').value;
  for (const el of logPanel.querySelectorAll('.log')) {
    el.style.display = shouldShowLog(el.dataset.level) ? '' : 'none';
  }
}

function clearLogs() { logPanel.innerHTML = ''; }
function clearHistory() {
  ws.send(JSON.stringify({type: 'clear_history'}));
  convMsgs.innerHTML = '';
}

function sendChat() {
  const input = document.getElementById('msg-input');
  const text = input.value.trim();
  if (!text || !ws || ws.readyState !== 1) return;
  input.value = '';
  // Parse /skill commands
  const m = text.match(/^\/skill\s+(\S+)\s*(.*)/);
  if (m) {
    ws.send(JSON.stringify({type: 'skill', skill: m[1], text: m[2]}));
  } else {
    ws.send(JSON.stringify({type: 'chat', text}));
  }
}

async function loadSkills() {
  const grid = document.getElementById('skills-grid');
  try {
    const res = await fetch('/api/skills');
    const skills = await res.json();
    grid.innerHTML = '';
    if (!skills.length) { grid.textContent = '无可用技能'; return; }
    for (const s of skills) {
      const btn = document.createElement('button');
      btn.className = 'skill-btn' + (s.confirm ? ' confirm' : '');
      btn.title = s.description;
      btn.textContent = s.name;
      btn.onclick = () => {
        const input = document.getElementById('msg-input');
        const extra = input.value.trim();
        ws.send(JSON.stringify({type: 'skill', skill: s.name, text: extra}));
        input.value = '';
      };
      grid.appendChild(btn);
    }
  } catch(e) { grid.textContent = '加载失败'; }
}

let voiceEnabled = false;
function toggleVoice() {
  voiceEnabled = !voiceEnabled;
  const btn = document.getElementById('voice-btn');
  btn.textContent = voiceEnabled ? '🔊 语音' : '🔇 静音';
  btn.style.background = voiceEnabled ? '#1b5e20' : '#37474f';
  ws.send(JSON.stringify({type: 'set_voice', enabled: voiceEnabled}));
}

document.getElementById('msg-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') sendChat();
});

connect();
</script>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"\n  Askme Web Test UI")
    print(f"  http://{args.host}:{args.port}\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
    )
