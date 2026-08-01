# askme Operations

更新时间：2026-07-27

本文合并配置、S100P 验收、语音健康检查、交接信息和故障排查。

## 本地快速启动

LiteLLM 是默认 LLM 控制面。启动 Runtime 前，必须按 [LiteLLM 运行手册](LITELLM_GATEWAY.md) 启动 sidecar、等待 readiness，并在根环境中配置 AskMe scoped virtual key。Proxy、readiness 或 scoped key 缺失时会 fail-closed，不会自动回退到 MiniMax/DeepSeek 直连。

```powershell
cd <repo-root>
python -m askme.blueprints.presets.edge_robot
```

Dashboard 默认：

```text
http://127.0.0.1:8765/dashboard
```

健康接口：

```powershell
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/healthz
curl http://127.0.0.1:8765/metrics/prometheus
```

## MCP 接入

项目根目录提供 `.mcp.json`，当前真实入口是：

```json
{
  "mcpServers": {
    "askme": {
      "command": "python",
      "args": ["-m", "askme.mcp.server"],
      "cwd": "D:\\inovxio\\tools\\askme"
    }
  }
}
```

也可以直接运行：

```powershell
python -m askme.mcp.server
```

已验证的 stdio 能力：

- `initialize` 正常。
- `list_tools` 返回 15 个工具：memory、robot、skill、vision、chat、voice。
- `list_resources` 返回 10 个资源：health、perception、memory、robot、skills、config。
- `read_resource("askme://health")` 正常返回健康快照。
- `call_tool("robot_state")` 在未接机器人时返回结构化 `robot_not_connected`，不会误触发硬件。

注意：MCP 能接入 askme 工具层，但真实机器人动作仍受 `robot.enabled`、SafetyPreflight、runtime profile 和外部机器人服务约束。未启用机器人硬件时，robot 工具只应返回结构化不可用结果。

## 配置入口

| 文件 | 用途 |
| --- | --- |
| `config.yaml` | 非机密运行配置 |
| `.env` | API key、服务地址等机密配置，不提交 |
| `.env.example` | `.env` 模板 |
| `prompts/SOUL.md` | 语音人格、表达风格、边界 |
| `askme/config.py` | YAML + dotenv + `${VAR}` 解析 |

`config.yaml` 和 `.mcp.json` 保留在项目根目录是有意的：`askme/config.py`
默认从根目录加载 `config.yaml`，Codex/Claude Desktop 等 MCP 客户端也按项目
根目录发现 `.mcp.json`。提示词和规划文件不放根目录，分别放在 `prompts/`
和 `plans/`。

关键环境变量：

```text
LITELLM_BASE_URL=http://127.0.0.1:4000/v1
LITELLM_VIRTUAL_KEY=
ZEROCLAW_LITELLM_VIRTUAL_KEY=
VISION_LITELLM_VIRTUAL_KEY=
NO_PROXY=127.0.0.1,localhost,litellm
MINIMAX_API_KEY=  # 默认仅供 TTS
MINIMAX_GROUP_ID=
DOG_CONTROL_SERVICE_URL=http://localhost:5080
DOG_SAFETY_SERVICE_URL=http://localhost:5070
NAV_GATEWAY_URL=http://localhost:8088
```

真实硬件相关配置必须显式启用。默认 fake/sim/shadow 不应该触碰硬件。

## 常用验证

软件语音门：

```powershell
python -m askme runtime voice-health --json
```

S100P/Sunrise 音频门：

```powershell
python -m askme runtime sunrise-voice-readiness --json
```

Runtime handoff readiness：

```powershell
python scripts\eval\simulate_robot_interaction.py
python scripts\eval\evaluate_robot_scenarios.py
python scripts\eval\check_runtime_handoff_readiness.py --require-audit
```

RAG Trust：

```powershell
python scripts\eval\evaluate_rag_trust_scenarios.py --output artifacts\rag_trust\scenario-evaluation.json
```

Voice E2E：

```powershell
python scripts\eval\evaluate_voice_e2e_scenarios.py --output artifacts\voice_e2e\scenario-evaluation.json
```

核心测试：

```powershell
python -m pytest tests\test_interaction_gate.py tests\scenario_tests\test_voice_e2e_evaluation.py -q
python -m pytest tests\test_runtime_handoff.py tests\test_runtime_handoff_module.py tests\test_runtime_arbiter_client.py -q
python -m pytest tests\test_memory_catalog.py tests\test_runtime_modules.py tests\test_health.py -q
```

静态检查：

```powershell
python -m ruff check askme tests scripts
```

Dashboard JS 语法检查：

```powershell
node -e "const fs=require('fs');const s=fs.readFileSync('askme/static/dashboard.html','utf8');const m=s.match(/<script>([\s\S]*)<\/script>/);if(!m) throw new Error('script not found');new Function(m[1]);console.log('dashboard script ok');"
```

## 证据包

| 证据 | 路径 |
| --- | --- |
| RAG Trust | `artifacts/rag_trust/scenario-evaluation.json` |
| Voice E2E | `artifacts/voice_e2e/scenario-evaluation.json` |
| Runtime handoff simulation | `artifacts/runtime_handoff/` |
| S100P readiness bundle | 命令输出指定目录 |

Dashboard 运营诊断会读取：

- `rag_trust`
- `voice_e2e`
- `voice_pipeline_status`
- `runtime_handoff`
- latency buckets
- recent traces

## S100P 现场上线

固定入口：

| 项 | 约定 |
| --- | --- |
| systemd 源文件 | `deploy/askme.service` |
| systemd 入口 | `python -m askme.blueprints.presets.edge_robot` |
| 安装脚本 | `bash deploy/install.sh` |
| 健康端口 | `health_server.port=8765` |

现场前置条件：

- 当前 commit 已记录。
- `.env` 包含 LiteLLM base URL、AskMe scoped virtual key、TTS/ASR 与机器人服务 URL；master/provider key 只放在受控的 `docker/.env.litellm`。
- 如 Cloud ASR 是部署硬门，必须配置对应 API key 并启用。
- ASR/VAD/TTS 模型已在 `models/`，不要依赖现场临时下载。
- `sunrise` 用户在 `audio` 组内。
- MCP01 USB 声卡已接入。

安装：

```bash
cd /home/sunrise/data/inovxio/askme
bash deploy/install.sh
sudo systemctl start askme.service
sudo systemctl status askme.service --no-pager
journalctl -u askme.service -f
```

验收原则：

- 没有真机证据，不声明生产就绪。
- `degraded` 不等于失败，但必须解释并留证。
- 任何真实硬件动作前，确认 runtime profile、SafetyPreflight、operator 权限和审计日志。

## 故障排查

### 启动后没声音或麦克风不响应

1. 跑 `voice-health`。
2. 跑 `sunrise-voice-readiness`。
3. 查看 Dashboard 的 Voice Console：
   - Mic
   - VAD
   - ASR
   - TTS
   - Runtime
   - Safety
   - Interrupt
4. 检查 `voice.input_device`、`voice.output_device`。
5. Windows 开发机没有 S100P 声卡时，硬件门返回 degraded 是正常的。

### 回复慢

看 Dashboard Latency：

- `asr_final_ms`
- `llm_ttft_ms`
- `tts_first_audio_ms`
- `playback_done_ms`
- `barge_in_stop_ms`

先分清是 ASR、LLM、TTS 还是播放设备慢。

### 不知道什么时候能说话

看 Voice Console 的主状态：

- `正在听`：可以说话。
- `正在思考`：已收到输入，等待回复。
- `正在回复`：正在播报，可以打断。
- `请稍等`：处于 cooldown。
- `语音未就绪`：看 readiness 项。

### 问答没有依据

检查：

```powershell
python scripts\eval\evaluate_rag_trust_scenarios.py --output artifacts\rag_trust\scenario-evaluation.json
```

再看 Dashboard：

- bot 气泡的回答依据。
- Knowledge Trust 卡片。
- Knowledge Console 的知识状态。

### 路人闲聊触发回复

先跑：

```powershell
python scripts\eval\evaluate_voice_e2e_scenarios.py --output artifacts\voice_e2e\scenario-evaluation.json
```

重点看：

- `noise_bystander_casual_recorded_only`
- `multi_person_ambiguous_clarifies`
- `false_respond_rate`

如果现场仍误触发，需要补真实录音回放评测，而不是只调提示词。

### 中文乱码

仓库源码和文档按 UTF-8 存储。PowerShell 显示乱码通常是控制台 code page 问题，不一定是文件损坏。

```powershell
.\scripts\dev\enable_utf8_console.ps1
python scripts\dev\check_text_encoding.py
```

## 发布前检查

发布前至少保留：

- git commit。
- `voice-health` JSON。
- RAG Trust JSON。
- Voice E2E JSON。
- runtime handoff readiness JSON。
- Dashboard health snapshot。
- systemd 日志。
- 已知 degraded 项解释。

未完成真实 S100P/MCP01/现场噪声证据前，对外口径应保持为“软件链路通过”或“仿真/shadow 通过”；现场上线状态以 readiness 报告的客户状态和交付清单为准。
# Field Runtime Callback Contract

Field incidents that require robot motion must not update the archive by informal text logs. Runtime, shadow, lab, or robot processes must callback through:

```text
POST /api/field/events/{event_id}/runtime-delivery
```

Required production behavior:

- Set `ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET` on the askme service.
- Send `runtime_signature_timestamp` on every callback.
- Sign the JSON payload with HMAC-SHA256 over the unsigned payload and send it as `runtime_signature`.
- Use a controlled `status`: `submitted`, `queued`, `executing`, `paused`, `blocked`, `completed`, `failed`, `cancelled`, `rejected`, `shadowed`, or another status listed by the 422 error response.
- Send a stable `runtime_callback_id`. If omitted, askme derives one from the callback payload, but real runtime producers should provide their own id.

Archive guarantees:

- Bad signatures are rejected with `403`.
- Unknown statuses are rejected with `422` and do not mutate the FieldIncident.
- Duplicate `runtime_callback_id` values are idempotent and do not append duplicate receipts.
- Accepted callbacks write `runtime_delivery` plus `runtime_delivery_receipts` so customer audit can reconstruct the runtime side of the incident.

## Runtime Callback Helper

Real shadow, lab, or robot runtime processes can use the helper below to produce a signed callback without hand-writing the HMAC payload:

```powershell
python scripts\runtime\post_field_runtime_callback.py --event-id <event_id> --status executing --run-id <run_id> --secret $env:ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET
```

Use `--dry-run` to print the signed payload without posting.

When a shadow/lab runtime has a `RuntimeHandoffService.submit_plan_payload()` JSON result, it can post or preview the full status sequence:

```powershell
python scripts\runtime\post_field_runtime_callback.py --result-json runtime-result.json --secret $env:ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET --dry-run
```

When `/api/field/events` is called with a configured runtime handler, its response includes `runtime_handoff_result`. Use that JSON as the `--result-json` input for shadow/lab callback posting.

Acceptance smoke:

```powershell
python scripts\eval\smoke_field_runtime_roundtrip.py --secret $env:ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET --output artifacts\runtime_handoff\field-runtime-roundtrip-smoke.json
```

The smoke is successful only when it creates a FieldIncident, submits it to shadow runtime, posts signed runtime callbacks, and the archived workflow reaches the final runtime status with matching receipts.

To prove real HTTP routing without relying on an already-running service:

```powershell
python scripts\eval\smoke_field_runtime_roundtrip.py --start-local-server --secret $env:ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET --output artifacts\runtime_handoff\field-runtime-roundtrip-live-smoke.json
```

Deployment readiness consumes the same artifact through
`runtime_roundtrip_report_path`, defaulting to:

```text
artifacts/runtime_handoff/field-runtime-roundtrip-live-smoke.json
```

Readiness blocks when the roundtrip smoke is missing or failed. A passing
temporary local-server run is acceptable lab evidence, but production readiness
requires a run against an existing deployed askme service with trusted callback
receipts and a final runtime delivery status of `shadowed` or `completed`.
