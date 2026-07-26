# LiteLLM 网关接入与运行手册

> 状态：可靠网关和模型调用控制上下文已接入，默认仍保持现有 DeepSeek
> 直连。真实凭据 A/B、故障演练和真机声学验收完成前，不把 LiteLLM
> 宣称为已提速或默认主通道。

## 1. 产品边界

LiteLLM 以独立 Proxy sidecar 运行，不安装进机器人 Python 进程，也不替代 AskMe 的对话内核。

| 能力 | 唯一负责人 |
| --- | --- |
| 会话、轮次、记忆、工具调用、审批、急停 | AskMe |
| 模型别名、上游密钥、负载均衡、模型重试与 fallback | LiteLLM |
| ASR、TTS、AEC、全双工打断 | AskMe 语音链路 |
| 本地向量化与检索 | AskMe FastEmbed/ONNX |
| 视觉采集与 VLM 调用 | AskMe VisionBridge（尚未并入本次网关验收） |

这条边界避免两层重试造成尾延迟放大。使用 `provider: litellm` 时，AskMe 会自动清空本地模型 fallback、把 transport retry 固定为 0，并禁止已有 MiniMax 直连客户端绕过代理。

打断也贯通到了上游：用户插话或轮次取消后，AskMe 停止消费 token，并主动关闭 LiteLLM 流，避免后台继续生成和计费。

每次模型调用由 AskMe 创建 `LLMCallContext`。它把以下安全字段贯通到
LiteLLM：

- `traceparent`：跨 AskMe、Proxy 和上游的 W3C trace。
- `trace_id`、`turn_id`、匿名 `call_id`、`purpose`、`request_class`、
  `privacy_class`、产品模型别名和时限预算：进入 allowlist metadata。
- 同一匿名 `call_id` 同时放入 `x-litellm-call-id`，用于代理侧定位调用。

`session_id`、操作员身份、证据 ID、原始音频/图像和 provider key 不会进入
代理 metadata。适配器也不会透传调用方传入的任意 `extra_headers`，因此
`Authorization`、Cookie 或厂商 key 不会越过这条边界。Conversation Core
仍是会话和轮次的唯一事实来源；LiteLLM 只持有关联标识，不得回写对话历史。

Gateway 在内存中保留最近 100 条 message-free 调用诊断，可通过内部
`recent_call_diagnostics()` 查询 call/trace/turn、请求类别、请求/解析模型、
模式、结果、首语义状态和耗时。通用健康状态只暴露数量和最近结果，不暴露
高基数 ID；诊断记录不包含 prompt、回复、session、操作员或证据内容。

语音请求使用逐调用时限，而不是等待全局 `request_timeout: 10s`。当前
`voice_llm_latency_budget_ms: 1500` 是可回滚的起始策略，不是实测成绩。
Gateway 只把首个非空文本或 tool call 视为“首语义”，空 keep-alive 不会
解除时限；如果流正常结束却始终没有首语义，会产生明确的
`LLMNoSemanticResponse`，而不是被记成成功。首语义之前发生连接故障时，
fallback 只能使用本次调用的剩余预算。首语义之后禁止在 AskMe 侧切换模型。

## 2. 产品能力别名

业务代码不依赖具体厂商型号。`docker/litellm-config.yaml` 暴露稳定能力别名：

| 别名 | 用途 | 当前部署 |
| --- | --- | --- |
| `voice-fast` | 普通语音和文本首答 | DeepSeek V4 Flash |
| `voice-quality` | `voice-fast` 的质量 fallback | DeepSeek V4 Pro |
| `robot-action` | 工具/机器人动作语义 | DeepSeek V4 Flash；无默认跨模型 fallback |
| `memory-compact` | 后台会话压缩 | DeepSeek V4 Flash，可回退 `voice-quality` |
| `health-probe` | 独立短探针 | DeepSeek V4 Flash |

旧 `deepseek-v4-*` 名称暂时保留给迁移和运维脚本。新增火山、MiniMax
或其他候选模型时，只修改 LiteLLM 路由和验收配置，不修改机器人业务调用点。

## 3. 固定版本与供应链要求

- Sidecar 固定为带 Prisma 工具链的 `ghcr.io/berriai/litellm-database:v1.93.0`，并锁定
  2026-07-25 解析到的多架构清单 digest
  `sha256:72360d8bd5602faa49be5098a8ac3dd069d9fb74503d6bd014242d96dc753e43`，
  不是浮动 `latest`。
- v1.93.0 官方发布页提供了 Docker cosign 签名。上线前必须验证镜像签名。
- 不在机器人进程安装 LiteLLM SDK；这样可以独立升级/回滚代理，并缩小供应链事件影响面。
- 已知历史风险：PyPI `1.82.7`/`1.82.8` 曾发生供应链事件；SQL 注入公告影响 `>=1.81.16,<1.83.7`。本项目不得降级到这些范围。

推荐使用发布页给出的固定公钥提交验证：

```powershell
cosign verify --key https://raw.githubusercontent.com/BerriAI/litellm/0112e53046018d726492c814b3644b7d376029d0/cosign.pub ghcr.io/berriai/litellm-database@sha256:72360d8bd5602faa49be5098a8ac3dd069d9fb74503d6bd014242d96dc753e43
```

参考：

- [LiteLLM v1.93.0 release](https://github.com/BerriAI/litellm/releases/tag/v1.93.0)
- [LiteLLM config](https://docs.litellm.ai/docs/proxy/configs)
- [LiteLLM virtual keys](https://docs.litellm.ai/docs/proxy/virtual_keys)
- [GHSA-r75f-5x8p-qvmc](https://github.com/BerriAI/litellm/security/advisories/GHSA-r75f-5x8p-qvmc)
- [PyPI 1.82.7/1.82.8 incident](https://github.com/BerriAI/litellm/issues/24518)

## 4. 启动 sidecar

1. 复制环境模板：

```powershell
Copy-Item docker/litellm.env.example docker/.env.litellm
```

2. 填写 `docker/.env.litellm`。`LITELLM_MASTER_KEY` 只用于管理，必须以 `sk-` 开头；`LITELLM_SALT_KEY` 一旦用于加密数据库凭据就要稳定保存。数据库密码请使用 URL 安全字符。

3. 启动：

```powershell
docker compose --env-file docker/.env.litellm -f docker/docker-compose.litellm.yml up -d
docker compose --env-file docker/.env.litellm -f docker/docker-compose.litellm.yml ps
```

Proxy 只绑定 `127.0.0.1:4000`，不会直接暴露到局域网。PostgreSQL 也不映射主机端口。

4. 检查健康：

```powershell
Invoke-RestMethod http://127.0.0.1:4000/health/liveliness
```

容器按 API-only、read-only root filesystem 运行；迁移目录与 Prisma/XDG cache 分别映射到可写 tmpfs。首次启动必须等 `/health/readiness` 成功，不能只看进程存活。

部署配置还默认启用 `LITELLM_MODE=PRODUCTION`、JSON 日志、消息正文禁记和
调用方密钥信息脱敏。不要用 `--detailed_debug` 覆盖这些生产默认值。

## 5. 给机器人创建最小权限密钥

不要把 master key 或 DeepSeek/MiniMax provider key 放到 AskMe `.env`。使用 master key 调用 `/key/generate`，只授权机器人需要的模型：

```powershell
curl.exe -X POST http://127.0.0.1:4000/key/generate `
  -H "Authorization: Bearer <LITELLM_MASTER_KEY>" `
  -H "Content-Type: application/json" `
  -d '{"models":["voice-fast","voice-quality","robot-action","memory-compact","health-probe"],"user_id":"askme-robot","duration":"30d","max_parallel_requests":4,"rpm_limit":120,"metadata":{"service":"askme-robot"}}'
```

把返回的 `sk-...` 写入项目根目录 `.env`：

```dotenv
LITELLM_BASE_URL=http://127.0.0.1:4000/v1
LITELLM_VIRTUAL_KEY=sk-generated-virtual-key
```

该 key 只授权当前产品能力，并设置 30 天有效期、并发和 RPM 上限。到期前先生成新 key、灰度切换，再撤销旧 key；不要给默认 key 加未使用的厂商模型权限。

根 `.env` 中的 `MINIMAX_API_KEY` 只在 MiniMax TTS 或明确回退到直连 LLM 时保留；LiteLLM preset 会把 LLM 的 `minimax_api_key` 清空。若不使用 MiniMax TTS，也应从应用环境删除该 provider key。

## 6. 切换 AskMe

`config.yaml` 已提供 `brain.provider_presets.litellm`，运行中的控制面可以选择 `litellm`。要让重启后也默认使用代理，将 brain 的入口改成：

```yaml
brain:
  provider: litellm
  api_key: ${LITELLM_VIRTUAL_KEY}
  base_url: ${LITELLM_BASE_URL}
  model: voice-fast
  voice_model: voice-fast
  max_retries: 0
  fallback_models: []
```

模型 fallback 只配置在 `docker/litellm-config.yaml`。不要同时恢复 AskMe 的 `fallback_models`；适配器仍会在运行时强制单一 routing owner，但配置保持一致更利于审计。

如果 AskMe 也运行在容器内，`127.0.0.1` 指向 AskMe 容器本身。应把两个服务加入同一内部网络并使用 `http://litellm:4000/v1`，同时不要额外暴露 Proxy 公网端口。

## 7. 验收门禁

上线前至少完成以下检查：

- 功能：普通对话、tools、短回复、20 轮连续上下文均与直连一致；每个
  tool follow-up 必须拥有独立 `call_id`，但保持同一 `trace_id/turn_id`。
- 打断：TTS 播放中插话，旧轮不得继续出字；LiteLLM 日志中旧请求应结束。
- 路由：健康状态显示 `routing_owner=litellm`，且 `fallback_models=[]`。
- 安全：AskMe 日志和健康接口不出现 master/provider key、消息正文、session
  或原始证据；应用只持有 virtual key。
- 延迟冒烟：直连/Proxy 各 20 次；发布门各至少 50 次，条件允许时 100 次。
  冷态/热态、tools/无 tools 分组，并随机交错执行；保存原始样本和
  p50/p95/p99。Proxy 增量目标 p95 不高于 50 ms；未测量前不宣称提速。
- 追踪：空 chunk 不计入 semantic TTFT；可用 `trace_id + x-litellm-call-id`
  找到最终模型、fallback、错误和耗时，且这些 ID 不作为 Prometheus
  高基数标签。
- 故障：断开 `voice-fast` 后由 LiteLLM 切到 `voice-quality`；AskMe
  不做第二轮 fallback。还要覆盖 429、500、永久挂起、流中断、Proxy/DB
  重启、virtual key 撤销以及“插话与 fallback 同时发生”。
- 恢复：Proxy/数据库重启后虚拟密钥仍有效，机器人可以重新建立流。

执行代码回归：

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_litellm_provider.py tests/test_litellm_deployment.py tests/test_llm_gateway.py tests/test_llm_client.py tests/test_llm_retry.py -q
```

`/health/liveliness` 只说明进程活着，`/health/readiness` 用于流量准入；
会触发真实模型检查和费用的 `/health` 只能由独立 `health-probe` key
受控调用。

## 8. 视觉、向量化与语音的关系

- LiteLLM 只进入 LLM/VLM/embedding 的 API 管理面，不处理麦克风、ASR、TTS、AEC 或声学打断；因此不需要 ROS2。
- 当前记忆检索继续使用本地 FastEmbed/ONNX，启动后热态快、无云端网络延迟，也不会把记忆文本外发。除非远端 embedding 的准确率有实测收益，不应为了“统一”而牺牲实时性和隐私。
- 当前 `VisionBridge` 是独立客户端。本次不能把 `supports_vision` 当作已经完成的多模态验收。后续应先在 LiteLLM 配置一个经过验证的 vision model alias，再让 VisionBridge 使用 scoped virtual key/base URL，并做图像泄露与延迟验收。
- 火山端到端 S2S 是自由对话声学通道；机器人命令、工具、审批和急停仍走 AskMe 级联裁决。它与 LiteLLM 是并行能力，不互相替代。

## 9. 缓存边界

- 多轮语音、工具/动作、个性化记忆和视觉请求默认 `allow_cache=false`。
- 只有无身份、无工具、已审核的固定 FAQ 才能通过显式 allowlist 开启
  LLM 精确缓存。
- 语义缓存不得用于机器人动作；“相似”不等于安全语义相同。
- “好的”“请稍等”等低延迟反馈继续走本地 TTS phrase cache，不经过
  LLM 响应缓存。

## 10. 回滚

将 `brain.provider`、`api_key`、`base_url` 切回原 DeepSeek 直连配置即可；会话、记忆和轮次数据不受影响。停止 sidecar：

```powershell
docker compose --env-file docker/.env.litellm -f docker/docker-compose.litellm.yml down
```

不要加 `-v`，除非明确要删除 LiteLLM 的 PostgreSQL 数据和虚拟密钥。
