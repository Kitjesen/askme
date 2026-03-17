# Sunrise 部署测试计划

## 目标
在 sunrise (192.168.66.190) 上用新代码启动 askme 语音服务，验证工业级改造的真实效果。

## 前置条件（已确认）
- ✅ 代码已同步 (`git push sunrise master`)
- ✅ 新模块 76 测试通过
- ✅ Node.js v20 + npm 11 可用
- ✅ 音频设备 MCP01 USB Audio (hw:1,0) 可用
- ✅ ASR/VAD/KWS 模型文件已部署
- ❌ TTS 本地模型缺失（需传 ~511MB）
- ❌ .env 文件缺失（需传）
- ⚠️ 磁盘 87%（5.7GB 剩余，足够）
- ⚠️ 旧 askme 进程 PID 390673 占音频设备

## 步骤

### Step 1: 停旧进程
```bash
ssh sunrise "kill 390673"  # 停旧 askme --legacy
```
验证：`ps aux | grep askme` 无进程

### Step 2: 传 .env 和 TTS 模型
```bash
scp .env sunrise:~/askme/.env
scp -r models/tts/vits-melo-tts-zh_en sunrise:~/askme/models/tts/
```
验证：`ls ~/askme/.env ~/askme/models/tts/vits-melo-tts-zh_en/model.onnx`

### Step 3: 跑全量测试
```bash
ssh sunrise "cd ~/askme && source .venv/bin/activate && python -m pytest tests/ -q --tb=short"
```
验证：全部通过（预期 491+，排除 1 个环境相关）

### Step 4: 启动新服务
```bash
ssh sunrise "cd ~/askme && tmux send-keys -t askme 'source .venv/bin/activate && python -m askme --legacy --voice' Enter"
```
验证：`curl http://192.168.66.190:8765/health` 返回 `{"status":"ok"}`

### Step 5: 真机验证清单
| 测试项 | 操作 | 预期 | pass/fail |
|--------|------|------|-----------|
| 健康端点 | `curl :8765/health` | status: ok, voice pipeline ok | |
| 追踪端点 | `curl :8765/trace` | 返回 summary + recent | |
| 唤醒+回答 | 说唤醒词 → "现在几点了" | chime → 回答 <3s | |
| 思考音 | 问复杂问题 | 1.2s 后听到 900Hz 思考音 | |
| 打断 | TTS 播报中说"够了" | TTS 立即停止 | |
| 连续对话 | 30s 内连续问 3 个问题 | 无需重复唤醒词 | |
| 确认流程 | 触发需确认技能 → 说"好的" | "好的"被接受（不被过滤） | |
| MIC peak 日志 | 查 log | 有 peak 数据，HPF 生效 | |

## 风险
- 磁盘 87% — TTS 模型 511MB 传完后剩 ~5.2GB，够用
- tmux 会话 "askme" 已存在 — 可能需要先 kill-session 再建
- MiniMax API 需要外网 — sunrise 无外网，TTS 必须用 local 后端

## 关键配置调整
sunrise 上 config.yaml 需要改：
- `tts.backend: local`（MiniMax 需外网，用本地 TTS）
- `voice.audio_filter.enabled: true`（启用 HPF）
- `voice.noise_gate_peak: auto`（自动校准）

## 验收标准
1. `/health` 返回 status: ok
2. `/trace` 显示 pipeline timing
3. 语音对话端到端可用（唤醒 → 回答 → 连续对话）
4. "够了" 停止 TTS（不触发 ESTOP）
5. 确认流程 "好的" 被正确接受
