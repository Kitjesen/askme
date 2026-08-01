# Askme 生产部署指南

> 本文档面向交付/运维人员，描述 Askme 边缘运行时在园区巡检机器狗上的生产部署流程。

---

## 目录

- [系统要求](#系统要求)
- [部署结构](#部署结构)
- [前置准备](#前置准备)
- [Docker 部署](#docker-部署)
- [环境变量清单](#环境变量清单)
- [健康检查验证](#健康检查验证)
- [日志查看](#日志查看)
- [监控接入](#监控接入)
- [故障排查](#故障排查)
- [备份恢复](#备份恢复)
- [安全注意事项](#安全注意事项)

---

## 系统要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 2 cores | 4 cores (x86_64 / ARM64) |
| 内存 | 2 GB | 4 GB |
| 磁盘 | 10 GB | 20 GB (SSD) |
| OS | Ubuntu 22.04+ / Debian 12+ | Ubuntu 22.04 LTS |
| Docker | 24.0+ | 25.0+ |
| Docker Compose | v2.24+ | v2.27+ |

---

## 部署结构

```
docker/
  docker-compose.yml          # 默认 AskMe + LiteLLM；ZeroClaw 仅显式实验 profile
  docker-compose.litellm.yml  # LiteLLM + PostgreSQL 两阶段引导
  docker-compose.prod.yml     # 生产覆盖
  litellm-config.yaml         # 模型别名、重试和 fallback 唯一策略
  litellm.env.example         # Proxy master/provider 密钥模板
  Dockerfile.askme            # Askme 服务镜像
  Dockerfile.zeroclaw         # ZeroClaw 网关镜像
  docker-entrypoint.sh        # 容器入口脚本
  .env.example                # AskMe scoped key 与可选实验凭据模板
  zeroclaw/config.toml        # 仅约束 ZeroClaw 模型路由，不提供 MCP connector

data/                         # 运行时数据（持久化）
  memory/                     # 记忆层数据
  sessions/                   # 会话记录
  tasks/                      # 任务历史

models/                       # ONNX 模型文件（只读挂载）
  sherpa-onnx/                # 语音识别模型
  punctuation/                # 标点恢复模型
```

---

## 前置准备

### 1. 安装 Docker

```bash
# Ubuntu / Debian
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $USER
# 登出后重新登录使组生效
```

### 2. 准备模型文件

模型文件通过 volume 挂载到容器内 `/app/models` 目录：

```bash
# 这些路径来自 config.yaml，入口会逐项校验实际模型文件
test -f models/asr/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30/encoder.int8.onnx
test -f models/vad/silero_vad.onnx
test -f models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt
```

ASR、VAD、KWS 是默认 `edge_robot` 的必需能力。TTS 是否要求本地
模型由实际配置决定：有效云端 TTS 可以不挂本地 TTS；配置选择
`fallback_backend: local` 时应同时准备对应模型。缺失的必需文件会在
启动前被逐项报告，进程以 78 退出，不会启动一个永久 503 的半成品。

### 3. 准备 Linux 音频设备

默认容器运行 `edge_robot`，因此产品部署必须提供真实输入和输出设备：

```bash
test -d /dev/snd
getent group audio
export ASKME_AUDIO_GID="$(getent group audio | cut -d: -f3)"
test -n "$ASKME_AUDIO_GID"
```

`docker-compose.edge-linux.yml` 把 `/dev/snd` 映射进容器，并把容器用户
加入宿主机 audio GID。入口随后使用 PortAudio 校验所选输入、输出通道与
格式权限。Docker Desktop/Windows 不提供这一 Linux 硬件契约，不能作为
默认 edge 语音产品验收环境。

### 4. 配置环境变量

```bash
cp docker/.env.example docker/.env
cp docker/litellm.env.example docker/.env.litellm
# docker/.env 只放应用 scoped key；provider/master 密钥只放 .env.litellm
vi docker/.env
vi docker/.env.litellm
chmod 600 docker/.env docker/.env.litellm
```

Windows 手动部署必须在写入 secret 后收紧 ACL；quickstart.bat 会执行同等检查并在失败时终止：

```powershell
icacls docker\.env /inheritance:r /grant:r "$env:USERDOMAIN\${env:USERNAME}:(M)" "*S-1-5-18:(F)" "*S-1-5-32-544:(F)"
if ($LASTEXITCODE -ne 0) { throw "Failed to secure docker/.env" }

icacls docker\.env.litellm /inheritance:r /grant:r "$env:USERDOMAIN\${env:USERNAME}:(M)" "*S-1-5-18:(F)" "*S-1-5-32-544:(F)"
if ($LASTEXITCODE -ne 0) { throw "Failed to secure docker/.env.litellm" }
```

---

## Docker 部署

### 标准部署

```bash
# 进入项目目录
cd /opt/askme

# 第一阶段：先校验 master/salt/DB，再启动控制面并等待 readiness
# （不能只看进程存活）
docker compose --env-file docker/.env.litellm -f docker/docker-compose.litellm.yml up -d --wait litellm
curl --noproxy 127.0.0.1 -fsS http://127.0.0.1:4000/health/readiness

# 按 docs/LITELLM_GATEWAY.md 创建 AskMe 最小权限 virtual key，
# 将其写入 LITELLM_VIRTUAL_KEY；默认产品启动不要求 ZeroClaw key。

# 第二阶段：密钥门禁通过后启动编排服务；同一 project/network/DB 复用控制面
docker compose --env-file docker/.env --env-file docker/.env.litellm \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.edge-linux.yml \
  up -d

# 查看启动日志
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml logs -f

# 确认聚合 readiness；任一必需组件 degraded 时返回 HTTP 503
curl -fsS http://localhost:8765/ready | python -m json.tool
```

### 实验 ZeroClaw 启动（非 MCP 集成）

默认 `docker compose up` 不启动 ZeroClaw。仅在调试其 LiteLLM
`robot-action` 路由时，先按运行手册签发独立 key，再显式启用实验 profile：

```bash
docker compose --env-file docker/.env \
  --env-file docker/.env.litellm \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.edge-linux.yml \
  --profile experimental-zeroclaw \
  up -d
```

> 当前 ZeroClaw v0.1.7 配置 schema 没有 MCP connector。启动 ZeroClaw 进程不等于 AskMe MCP 集成可用；该 profile 不得进入生产验收，等待受支持的连接方案和端到端证据。

### 生产部署（推荐）

```bash
# 使用生产覆盖配置
docker compose --env-file docker/.env \
  --env-file docker/.env.litellm \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.edge-linux.yml \
  -f docker/docker-compose.prod.yml \
  up -d
```

生产覆盖会移除 AskMe 的宿主端口；ZeroClaw 不属于默认生产栈。反向代理必须加入 `askme-litellm_askme-net`，并使用 `http://askme:8765` 作为 upstream；容器内 AskMe 由 Compose 设置 `ASKME_HEALTH_HOST=0.0.0.0`，同时强制要求 `ASKME_CONTROL_API_KEY`。

### 仅启动 Askme

```bash
docker compose --env-file docker/.env --env-file docker/.env.litellm \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.edge-linux.yml \
  up -d askme
```

### 重建镜像

```bash
# 拉取最新代码后重建
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml build --pull

# 滚动更新
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml -f docker/docker-compose.edge-linux.yml up -d --pull always
```

### 停止服务

```bash
# 停止并保留数据
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml down

# 危险：永久删除 LiteLLM 数据库、virtual key 与应用 volume；仅审批后执行
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml down -v
```

---

## 环境变量清单

### 必需

| 变量 | 说明 | 示例 |
|------|------|------|
| `LITELLM_VIRTUAL_KEY` | AskMe scoped virtual key；禁止使用 master key | `sk-xxxx` |
| `ASKME_CONTROL_API_KEY` | 容器 HTTP 控制面认证；远程绑定必填 | 强随机值 |

### LiteLLM 控制面（仅 `docker/.env.litellm`）

| 变量 | 说明 | 约束 |
|------|------|------|
| `LITELLM_MASTER_KEY` | 管理和签发 virtual key | `sk-` 开头、至少 24 字符；不得注入 AskMe/ZeroClaw |
| `LITELLM_SALT_KEY` | 数据库密钥加密盐 | 至少 24 字符；首次上线后稳定保存，不得轮换丢失 |
| `LITELLM_DATABASE_PASSWORD` | PostgreSQL 密码 | 至少 24 字符，仅使用 `A-Z a-z 0-9 . _ ~ -` |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | 当前上游 provider 凭据 | 只由 LiteLLM 持有 |
| `LITELLM_MINIMAX_PROVIDER_API_KEY` | 可选 MiniMax LLM 上游凭据 | 与 AskMe 的 TTS `MINIMAX_API_KEY` 分离 |

### 实验 ZeroClaw 凭据

`ZEROCLAW_LITELLM_VIRTUAL_KEY` 仅在显式启用
`experimental-zeroclaw` profile 时才必填；默认 AskMe + LiteLLM
产品栈不读取也不要求它。该 key 只能授权 `robot-action`，并且必须与
master key、AskMe scoped key、salt、数据库密码全部不同。模板词、低多样性和
重复块会被启动门禁拒绝；错误不会回显 secret。

### 条件启用的语音供应商

以下凭据只有启用对应在线语音能力时才必填，不能作为 LLM provider key 注入 AskMe。

| 变量 | 说明 | 示例 |
|------|------|------|
| `MINIMAX_API_KEY` | MiniMax TTS API Key；默认不用于 LLM | `mx-xxxx` |
| `DASHSCOPE_API_KEY` | 阿里云 ASR API Key | `sk-xxxx` |

### 语音配置

Docker profile 不消费独立的 TTS 音色、语速或情感环境变量，因此
`docker/.env.example` 不发布这些伪配置项。语音参数必须通过当前
`config.yaml` 的 `voice.tts` 配置或受支持的控制面更新，
并在供应商实际支持的取值范围内验收。

### 机器人服务（可选）

这些外部服务默认留空。启用时必须填写 AskMe 容器可解析且可路由的 DNS
名称或 URL；不能填写 `localhost`，因为容器内 loopback 指向 AskMe
容器自身。

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NAV_GATEWAY_URL` | 导航网关 | (空) |
| `DOG_CONTROL_SERVICE_URL` | 控制服务 | (空) |
| `DOG_SAFETY_SERVICE_URL` | 安全服务 | (空) |
| `ASKME_EDGE_SERVICE_URL` | 运行时桥接 | (空) |

### 运行时认证（可选）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RUNTIME_BEARER_TOKEN` | 运行时 Bearer Token | (空) |
| `RUNTIME_OPERATOR_ID` | 操作员 ID | `askme` |
| `NOVA_DOG_RUNTIME_API_KEY` | Nova Dog Runtime API Key | (空) |

### Compose 管理的网络变量

| 变量 | 说明 | 默认行为 |
|------|------|----------|
| `ASKME_HEALTH_HOST` | HTTP 监听地址 | 产品 Compose 固定注入 `0.0.0.0`；原生运行未设置时回退 loopback |
| `LITELLM_BASE_URL` | 原生进程访问 sidecar | `docker/.env` 固定为 `http://127.0.0.1:4000/v1`；产品容器固定使用 `http://litellm:4000/v1` |
| `NO_PROXY` | 本机探针和 Compose 服务名绕过系统代理 | 默认 `127.0.0.1,localhost,litellm`，无需手工设为必需项 |

产品 Compose 不接受远端 LiteLLM base URL 覆盖。远端网关必须使用独立、明确评审的部署 profile，不能与本地 sidecar 同时成为 routing owner。

---

## 健康检查验证

### 内置健康端点

服务启动后，通过以下端点验证：

```bash
# LiteLLM 流量准入；必须先于 AskMe readiness 成功
curl --noproxy 127.0.0.1 -fsS http://127.0.0.1:4000/health/readiness

# AskMe 聚合 readiness（Compose healthcheck 使用；degraded 返回 503）
curl -fsS http://localhost:8765/ready

# AskMe liveness（仅证明进程存活，不代表组件可服务）
curl -fsS http://localhost:8765/healthz

# AskMe 详细健康快照
curl -s http://localhost:8765/health | python -m json.tool
# 其中 LLM 应显示 routing_owner=litellm、fallback_models=[]、health_model=health-probe。
```

### 预期响应

`/ready` 只有在已注册的必需组件全部 healthy 时返回 HTTP 200；
任一组件 degraded/unhealthy 时返回 HTTP 503。`/healthz` 只用于
liveness，HTTP 200 不能作为业务 readiness 证据。

在 HTTP 服务启动前，容器入口先运行 fail-closed edge preflight。模型、
输入设备或输出设备不满足时，主进程不会启动；应先查看容器日志中的逐项
错误，而不是等待 `/ready`。

`/health` 返回详细运行时状态：

```json
{
  "status": "ok",
  "service": "askme",
  "version": "4.1.0",
  "snapshot_at": "2026-06-01T12:00:00.000Z",
  "uptime_seconds": 3600.0,
  "model_name": "...",
  "voice_pipeline_status": { "pipeline_ok": true }
}
```

### Docker 健康检查

```bash
# 查看容器健康状态
docker inspect --format='{{json .State.Health}}' askme

# 仅查看健康状态摘要
docker ps --filter name=askme --format "{{.Names}}\t{{.Status}}"
```

---

## 日志查看

### 实时日志

```bash
# 所有服务日志
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml logs -f

# 仅 Askme 日志
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml logs -f askme

# 最近 100 行
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml logs --tail=100 askme
```

### 日志持久化

Docker 使用 `json-file` 驱动，自动轮转：

- 每个日志文件最大 10MB
- 保留最近 3 个文件
- 生产环境保留 5 个文件并压缩

### 日志分析

```bash
# 错误过滤
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml logs askme 2>&1 | grep -i "error\|exception\|traceback"

# 启动耗时
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml logs askme 2>&1 | grep "started"
```

---

## 监控接入

### Prometheus 指标

Askme 在 `:8765/metrics` 暴露 Prometheus 格式指标：

```bash
curl -s http://localhost:8765/metrics/prometheus
```

支持的自定义指标包括：

- `askme_health_status` — 整体健康状态 (1=ok, 0=degraded)
- `askme_llm_latency_ms` — LLM 调用延迟
- `askme_conversation_count` — 会话总数

### 接入 Prometheus + Grafana

生产 Prometheus 容器需加入 `askme-litellm_askme-net`，并在 `scrape_configs` 中添加：

```yaml
scrape_configs:
  - job_name: 'askme'
    static_configs:
      - targets: ['askme:8765']
    metrics_path: '/metrics/prometheus'
```

### 接入 Loki

使用 Promtail 或 Grafana Alloy 采集 Docker 日志到 Loki：

```yaml
scrape_configs:
  - job_name: 'askme-logs'
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '^/?askme($|-.*)'
        action: keep
```

---

## 故障排查

### 容器无法启动

```bash
# 查看详细日志
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml logs askme

# 只验证 Compose；--quiet 不渲染包含 secret 的配置
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml config --quiet

# 运行不联网的一次性强度/占位/角色隔离门禁；只输出 OK/INVALID，不输出 key
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml run --rm --no-deps litellm-key-policy
```

### 健康检查失败

```bash
# 手动测试 readiness；失败时保留 HTTP 非零退出码
docker exec askme curl -fsS http://localhost:8765/ready

# 检查容器内实际监听配置（不会打印环境变量）
docker inspect --format='{{json .Config.Healthcheck}}' askme
```

### 内存不足

```bash
# 检查容器资源使用
docker stats askme

# 检查系统资源
free -h
df -h
```

### 磁盘空间不足

```bash
# 检查容器日志占用
du -sh /var/lib/docker/containers/*/*-json.log

# 清理旧容器和镜像
docker system prune -f

# 先审计占用；不要使用 --volumes，以免删除尚未备份的命名卷
docker system df
```

### 模型文件缺失

```bash
# 检查挂载的模型目录
docker exec askme ls -la /app/models/

# 确认宿主机模型路径
ls -la /opt/askme/models/
```

---

## 备份恢复

产品 Compose 使用命名卷，而不是仓库 `data/` 目录。至少需要纳入同一恢复点：

- `askme-litellm_askme_data`：AskMe 的 `/app/data`。
- `askme-litellm_litellm_db`：LiteLLM PostgreSQL、virtual-key 状态。
- `askme-litellm_zeroclaw_workspace`：仅在启用
  `experimental-zeroclaw` profile 后存在；当前不代表 MCP 已接通。

先停止写入并确认默认产品卷，再使用部署平台批准的 volume
snapshot/backup 机制：

```bash
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml down
docker volume inspect askme-litellm_askme_data askme-litellm_litellm_db
```

如果确实运行过实验 profile，再单独检查并备份其 workspace：

```bash
docker volume inspect askme-litellm_zeroclaw_workspace
```

本文不提供伪造的 `tar data/` 恢复步骤：它不会覆盖上述命名卷。上线前必须在隔离环境完成一次实际 snapshot restore 演练，并验证 AskMe 数据与 LiteLLM virtual key 同一恢复点可用，才可把备份门禁标记为通过。

环境文件备份本身包含 secret，只能生成 `0600` 副本并进入批准的 secrets backup：

```bash
install -m 600 docker/.env "docker/.env.backup.$(date +%Y%m%d)"
install -m 600 docker/.env.litellm "docker/.env.litellm.backup.$(date +%Y%m%d)"
```

不要备份 example 文件来替代 live 配置，也不要把 live env 或其副本提交到版本库。

---

## 安全注意事项

1. **最小权限** — Askme 容器以 uid 1001 (`askme` 用户) 运行；Askme 业务容器当前未启用 `read_only`，只有 LiteLLM 与一次性 key-policy 已显式只读，不能扩大宣称
2. **API 认证** — 生产环境必须设置 `ASKME_CONTROL_API_KEY`
3. **网络隔离** — 服务在 `askme-net` 内部通信；LiteLLM 仅绑定宿主机回环地址，PostgreSQL 不映射端口；生产反向代理必须加入同一网络，不能重新发布 AskMe 宿主端口
4. **密钥管理** — API Key 通过环境变量注入，不硬编码在镜像中；live env/backup 必须限制为当前操作员和系统管理员可读
5. **日志轮转** — 日志文件自动轮转，防止磁盘写满
6. **镜像签名** — 生产环境建议使用私有镜像仓库，验证镜像完整性

### 反向代理参考 (Nginx)

以下 Nginx 必须作为容器加入 `askme-litellm_askme-net`；客户端认证头应原样转发。

```nginx
server {
    listen 443 ssl;
    server_name askme.example.com;

    ssl_certificate /etc/ssl/certs/askme.crt;
    ssl_certificate_key /etc/ssl/private/askme.key;

    location / {
        proxy_pass http://askme:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

> 最后更新: 2026-07-27
> 维护: 穹沛科技 运维团队
