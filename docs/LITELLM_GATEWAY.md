# LiteLLM 网关接入与运行手册

> 状态：LiteLLM 已是 AskMe 的产品默认 LLM 控制面。ZeroClaw v0.1.7 只保留在
> 显式 `experimental-zeroclaw` profile 中，默认产品栈不启动；即使实验进程运行，它也尚未接通 AskMe MCP，不能视为集成可用。Proxy、readiness 或 scoped virtual key 缺失时会 fail-closed，不会自动回退直连。
> 真实凭据 A/B、故障演练和真机声学验收尚未完成，因此不宣称已经提速。

## 1. 产品边界

LiteLLM 以独立 Proxy sidecar 运行，不安装进机器人 Python 进程，也不替代 AskMe 的对话内核。

| 能力 | 唯一负责人 |
| --- | --- |
| 会话、轮次、记忆、工具调用、审批、急停 | AskMe |
| 模型别名、上游密钥、负载均衡、模型重试与 fallback | LiteLLM |
| ASR、TTS、AEC、全双工打断 | AskMe 语音链路 |
| 本地向量化与检索 | AskMe FastEmbed/ONNX |
| 视觉采集与 VLM 调用 | AskMe VisionBridge；云 VLM 默认关闭，待 `vision-scene` 别名验收后再启用 |

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

2. 填写 `docker/.env.litellm`。`LITELLM_MASTER_KEY` 只用于管理，必须以 `sk-` 开头；`LITELLM_SALT_KEY` 一旦用于加密数据库凭据就要稳定保存。master、salt、数据库密码必须分别生成，至少 24 个字符，不能包含 `replace`、`placeholder`、`example`、`generated`、`changeme` 等模板标记，不能使用低多样性或重复块。数据库密码只允许 URL unreserved 字符（`A-Z a-z 0-9 . _ ~ -`），因为当前 Compose 会把它插入 `DATABASE_URL`。
可选的 MiniMax LLM 上游使用 `LITELLM_MINIMAX_PROVIDER_API_KEY`；它与 `docker/.env` 中仅供 TTS 的 `MINIMAX_API_KEY` 必须分名、分文件管理。

3. 启动。独立 sidecar Compose 会先运行不联网的 `--control-plane-only`
门禁；PostgreSQL 和 LiteLLM 都等待 master/salt/DB 检查成功。此阶段尚未签发
AskMe virtual key，所以不会要求它：

```powershell
docker compose --env-file docker/.env.litellm -f docker/docker-compose.litellm.yml up -d --wait litellm
docker compose --env-file docker/.env.litellm -f docker/docker-compose.litellm.yml ps
```

Proxy 只绑定 `127.0.0.1:4000`，不会直接暴露到局域网。PostgreSQL 也不映射主机端口。

4. 检查健康：

```powershell
curl.exe --noproxy 127.0.0.1 http://127.0.0.1:4000/health/readiness
```

容器按 API-only、read-only root filesystem 运行；迁移目录与 Prisma/XDG cache 分别映射到可写 tmpfs。首次启动必须等 `/health/readiness` 成功，不能只看进程存活。开发机若设置了 `HTTP_PROXY`/`HTTPS_PROXY`，必须保留 `NO_PROXY=127.0.0.1,localhost,litellm` 或显式使用 `--noproxy 127.0.0.1`，否则本地探测可能被系统代理劫持。

部署配置还默认启用 `LITELLM_MODE=PRODUCTION`、JSON 日志、消息正文禁记和
调用方密钥信息脱敏。不要用 `--detailed_debug` 覆盖这些生产默认值。

## 5. 给机器人创建最小权限密钥

不要把 master key 或 DeepSeek/MiniMax provider key 放到 AskMe `.env`。使用 master key 调用 `/key/generate`，只授权机器人需要的模型：

```powershell
# AskMe：语音、工具、记忆与健康别名
curl.exe --noproxy 127.0.0.1 -X POST http://127.0.0.1:4000/key/generate `
  -H "Authorization: Bearer <LITELLM_MASTER_KEY>" `
  -H "Content-Type: application/json" `
  -d '{"models":["voice-fast","voice-quality","robot-action","memory-compact","health-probe"],"user_id":"askme-robot","duration":"30d","max_parallel_requests":4,"rpm_limit":120,"metadata":{"service":"askme-robot"}}'

# 可选实验 ZeroClaw：仅启用 experimental-zeroclaw 时单独签发，只允许 robot-action
curl.exe --noproxy 127.0.0.1 -X POST http://127.0.0.1:4000/key/generate `
  -H "Authorization: Bearer <LITELLM_MASTER_KEY>" `
  -H "Content-Type: application/json" `
  -d '{"models":["robot-action"],"user_id":"askme-zeroclaw","duration":"30d","max_parallel_requests":2,"rpm_limit":60,"metadata":{"service":"askme-zeroclaw"}}'
```

把 AskMe 返回的 `sk-...` 写入 `docker/.env`（本机原生 quickstart 也读取该文件）。只有计划运行实验 ZeroClaw 时才写入它的独立 key：

```dotenv
LITELLM_BASE_URL=http://127.0.0.1:4000/v1
LITELLM_VIRTUAL_KEY=<paste-the-issued-AskMe-sk-key-here>
ZEROCLAW_LITELLM_VIRTUAL_KEY=<paste-the-issued-ZeroClaw-sk-key-here>
NO_PROXY=127.0.0.1,localhost,litellm
```

默认 Compose 与 `local` quickstart 校验 master、AskMe virtual key、salt 与数据库密码均满足强度要求且两两不同；不要求 ZeroClaw key。启用 `experimental-zeroclaw`、`docker-zeroclaw` 或 `local-zeroclaw` 时，独立门禁再要求 ZeroClaw virtual key，并纳入全部角色隔离。错误只报告变量名与规则，不打印 secret。每把 virtual key 都应设置有效期、并发和 RPM 上限；到期前先生成新 key、灰度切换，再撤销旧 key。云 VLM 关闭时不应签发预留的视觉 key。

根 `.env` 中的 `MINIMAX_API_KEY` 只供 MiniMax TTS 使用；默认 LLM 路径不会读取它，也不会把它当作自动直连回退。若不使用 MiniMax TTS，也应从应用环境删除该 provider key。

## 6. 默认产品启动

`config.yaml` 与 `config.board.yaml` 已把 LiteLLM 设为默认控制面：

```yaml
brain:
  provider: litellm
  api_key: ${LITELLM_VIRTUAL_KEY}
  base_url: ${LITELLM_BASE_URL}
  model: voice-fast
  voice_model: voice-fast
  health_model: health-probe
  max_retries: 0
  fallback_models: []
```

模型 retry/fallback 只配置在 `docker/litellm-config.yaml`。应用层缺少
`LITELLM_BASE_URL` 或 `LITELLM_VIRTUAL_KEY` 时会在创建 transport 前失败；
不会创建 dummy client，也不会因为存在 MiniMax/DeepSeek 凭据而自动直连。

容器产品栈使用两份职责分离的环境文件，并严格分两阶段：

```powershell
docker compose --env-file docker/.env.litellm -f docker/docker-compose.litellm.yml up -d --wait litellm
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml -f docker/docker-compose.edge-linux.yml up -d
```

默认 Compose 与独立 sidecar 使用同一 `name: askme-litellm`、`askme-net`
和 PostgreSQL volume。先按第 4、5 节启动 sidecar 并签发 AskMe key，再运行默认
Compose；后者会复用控制面状态，并等待 LiteLLM readiness 与默认 key-policy
成功后才启动 AskMe。AskMe 容器 healthcheck 使用聚合 `/ready`；
`/healthz` 只承担 liveness。容器内 AskMe 固定使用
`http://litellm:4000/v1`，不接受远端 URL override；本机原生进程才使用
`http://127.0.0.1:4000/v1`。远端 Proxy 必须另建显式部署 profile，不能让本地 sidecar 与远端 Proxy 同时成为 routing owner。

ZeroClaw 不属于默认产品启动。只为调试模型路由时，签发独立
`robot-action` key 后显式运行：

```powershell
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml -f docker/docker-compose.edge-linux.yml --profile experimental-zeroclaw up -d
```

实验进程被限制到 `custom:http://litellm:4000/v1` 与
`robot-action`，并由独立 key-policy fail-closed；但 v0.1.7 schema
没有 MCP connector，当前容器并未接通 AskMe MCP。该进程启动不能作为工具、记忆或 handoff 集成证据。

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
会触发真实模型检查和费用的 `/health` 只能由拥有 `health-probe`
别名权限的 AskMe scoped key 受控调用。

## 8. 视觉、向量化与语音的关系

- LiteLLM 只进入 LLM/VLM/embedding 的 API 管理面，不处理麦克风、ASR、TTS、AEC 或声学打断；因此不需要 ROS2。
- 当前记忆检索继续使用本地 FastEmbed/ONNX，启动后热态快、无云端网络延迟，也不会把记忆文本外发。除非远端 embedding 的准确率有实测收益，不应为了“统一”而牺牲实时性和隐私。
- `config.yaml` 与 `config.board.yaml` 当前都将云端 `vlm_enabled` 设为 `false`。VisionBridge 已预留 `openai` backend、`vision-scene`、LiteLLM base URL 与独立 scoped key，但在代理中建立并验证该别名前必须保持 fail-closed；不能把 `supports_vision` 当作已经完成的多模态验收。
  VLM 启用后也只允许通过同一个 LiteLLM Proxy 控制面：`brain.provider=litellm`、`vision.vlm_model=vision-scene`、`vision.vlm_base_url` 必须与 `brain.base_url` 相同，并且视觉 key 必须和 AskMe 文本 key 分离。VisionBridge 复用中央 LiteLLM envelope，只发送 allowlist metadata、AskMe 生成的 `traceparent` 和匿名 `x-litellm-call-id`；prompt、图像内容、session/operator/evidence 标识、provider key 和任意调用方 header 不进入代理控制面字段。
- 火山端到端 S2S 是自由对话声学通道；机器人命令、工具、审批和急停仍走 AskMe 级联裁决。它与 LiteLLM 是并行能力，不互相替代。

## 9. 缓存边界

- 多轮语音、工具/动作、个性化记忆和视觉请求默认 `allow_cache=false`。
- 只有无身份、无工具、已审核的固定 FAQ 才能通过显式 allowlist 开启
  LLM 精确缓存。
- 语义缓存不得用于机器人动作；“相似”不等于安全语义相同。
- “好的”“请稍等”等低延迟反馈继续走本地 TTS phrase cache，不经过
  LLM 响应缓存。

## 10. 回滚

默认路径没有自动直连 fallback。回滚必须走变更审批：优先在
`docker/litellm-config.yaml` 内把稳定能力别名切回已验证上游；只有 Proxy
本身不可恢复时，才配置一个显式、审计过的 direct provider preset，并分别
迁移 AskMe 与 ZeroClaw。直接路径的 key、retry、fallback 和健康检查必须
重新明确唯一负责人，不能复用默认 LiteLLM 环境契约。

确认所有消费者已迁出且业务健康后，才停止 sidecar：

```powershell
docker compose --env-file docker/.env.litellm -f docker/docker-compose.litellm.yml down
```

不要加 `-v`，除非已经审批并确认要永久删除 LiteLLM 的 PostgreSQL 数据、virtual key 与审计状态。会话、记忆和轮次仍由 AskMe 持有，但模型路由审计数据不会随 direct 回滚自动迁移。
