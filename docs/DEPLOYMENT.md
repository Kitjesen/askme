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
  docker-compose.yml          # 基础编排
  docker-compose.prod.yml     # 生产覆盖
  Dockerfile.askme            # Askme 服务镜像
  Dockerfile.zeroclaw         # ZeroClaw 网关镜像
  docker-entrypoint.sh        # 容器入口脚本
  .env.example                # 环境变量模板

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
# 确认 models/ 目录结构
ls -la models/
# 应包含 sherpa-onnx/ 等子目录
```

### 3. 配置环境变量

```bash
cp docker/.env.example docker/.env
# 编辑 docker/.env 填入实际值
vi docker/.env
```

---

## Docker 部署

### 标准部署

```bash
# 进入项目目录
cd /opt/askme

# 启动全部服务
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d

# 查看启动日志
docker compose -f docker/docker-compose.yml logs -f

# 确认服务就绪
curl -s http://localhost:8765/healthz | python -m json.tool
```

### 生产部署（推荐）

```bash
# 使用生产覆盖配置
docker compose --env-file docker/.env \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.prod.yml \
  up -d
```

### 仅启动 Askme

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d askme
```

### 重建镜像

```bash
# 拉取最新代码后重建
docker compose -f docker/docker-compose.yml build --pull

# 滚动更新
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d --pull always
```

### 停止服务

```bash
# 停止并保留数据
docker compose -f docker/docker-compose.yml down

# 停止并清除数据卷
docker compose -f docker/docker-compose.yml down -v
```

---

## 环境变量清单

### 必需

| 变量 | 说明 | 示例 |
|------|------|------|
| `LLM_API_KEY` | LLM 提供商 API Key | `sk-xxxx` |
| `LLM_BASE_URL` | LLM API 地址 | `https://api.example.com/v1` |
| `MINIMAX_API_KEY` | MiniMax TTS API Key | `mx-xxxx` |
| `DASHSCOPE_API_KEY` | 阿里云 ASR API Key | `sk-xxxx` |

### 语音配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TTS_VOICE_ID` | TTS 音色 ID | `male-qn-qingse` |
| `TTS_SPEED` | 语速 (0.5-2.0) | `1` |
| `TTS_EMOTION` | 情感 | `happy` |

### 机器人服务（可选）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NAV_GATEWAY_URL` | 导航网关 | `http://localhost:8088` |
| `DOG_CONTROL_SERVICE_URL` | 控制服务 | `http://localhost:5080` |
| `DOG_SAFETY_SERVICE_URL` | 安全服务 | `http://localhost:5070` |
| `ASKME_EDGE_SERVICE_URL` | 运行时桥接 | `http://nova_dog:5100` |
| `OTA_SERVER_URL` | OTA 服务器 | (空) |
| `ROBOT_SERIAL_PORT` | 机械臂串口 | (空) |

### 运行时认证（可选）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RUNTIME_BEARER_TOKEN` | 运行时 Bearer Token | (空) |
| `RUNTIME_OPERATOR_ID` | 操作员 ID | `askme` |
| `NOVA_DOG_RUNTIME_API_KEY` | Nova Dog Runtime API Key | (空) |

### 生产环境

| 变量 | 说明 | 建议 |
|------|------|------|
| `ASKME_CONTROL_API_KEY` | 控制 API 认证 | 生成强随机密钥 |

---

## 健康检查验证

### 内置健康端点

服务启动后，通过以下端点验证：

```bash
# 简洁健康检查（推荐用于 healthcheck）
curl -s http://localhost:8765/healthz

# 详细健康快照
curl -s http://localhost:8765/health | python -m json.tool
```

### 预期响应

`/healthz` 返回 HTTP 200 表示服务就绪：

```json
{"status": "ok"}
```

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
docker inspect --format='{{json .State.Health}}' askme-edge

# 仅查看健康状态摘要
docker ps --filter name=askme-edge --format "{{.Names}}\t{{.Status}}"
```

---

## 日志查看

### 实时日志

```bash
# 所有服务日志
docker compose -f docker/docker-compose.yml logs -f

# 仅 Askme 日志
docker compose -f docker/docker-compose.yml logs -f askme

# 最近 100 行
docker compose -f docker/docker-compose.yml logs --tail=100 askme
```

### 日志持久化

Docker 使用 `json-file` 驱动，自动轮转：

- 每个日志文件最大 10MB
- 保留最近 3 个文件
- 生产环境保留 5 个文件并压缩

### 日志分析

```bash
# 错误过滤
docker compose -f docker/docker-compose.yml logs askme 2>&1 | grep -i "error\|exception\|traceback"

# 启动耗时
docker compose -f docker/docker-compose.yml logs askme 2>&1 | grep "started"
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

在 Prometheus `scrape_configs` 中添加：

```yaml
scrape_configs:
  - job_name: 'askme'
    static_configs:
      - targets: ['localhost:8765']
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
        regex: 'askme-.*'
        action: keep
```

---

## 故障排查

### 容器无法启动

```bash
# 查看详细日志
docker compose -f docker/docker-compose.yml logs askme

# 检查配置
docker compose --env-file docker/.env -f docker/docker-compose.yml config

# 检查环境变量
docker compose --env-file docker/.env -f docker/docker-compose.yml run --rm askme env | grep -E "^(LLM_|ASKME_)"
```

### 健康检查失败

```bash
# 手动测试健康端点
docker exec askme-edge curl -s http://localhost:8765/healthz

# 检查端口绑定
docker exec askme-edge ss -tlnp

# 检查进程
docker exec askme-edge ps aux
```

### 内存不足

```bash
# 检查容器资源使用
docker stats askme-edge

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

# 清理所有未使用资源（谨慎）
docker system prune -a -f --volumes
```

### 模型文件缺失

```bash
# 检查挂载的模型目录
docker exec askme-edge ls -la /app/models/

# 确认宿主机模型路径
ls -la /opt/askme/docker/models/
```

---

## 备份恢复

### 备份运行时数据

```bash
# 停止服务（保持数据一致性）
docker compose -f docker/docker-compose.yml down

# 备份数据目录
tar -czf askme-backup-$(date +%Y%m%d).tar.gz data/

# 重新启动
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d
```

### 恢复数据

```bash
# 停止服务
docker compose -f docker/docker-compose.yml down

# 解压备份
tar -xzf askme-backup-20260601.tar.gz

# 重新启动
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d
```

### 配置备份

```bash
# 备份环境变量和配置文件
cp docker/.env docker/.env.backup.$(date +%Y%m%d)
cp docker/.env.example docker/.env.example.backup
```

---

## 安全注意事项

1. **非 root 运行** — Askme 容器以 uid 1001 (`askme` 用户) 运行，rootfs 只读
2. **API 认证** — 生产环境必须设置 `ASKME_CONTROL_API_KEY`
3. **网络隔离** — 服务仅在 `askme-network` 内部通信，端口不暴露到宿主机（通过反向代理暴露）
4. **密钥管理** — API Key 通过环境变量注入，不硬编码在镜像中
5. **日志轮转** — 日志文件自动轮转，防止磁盘写满
6. **镜像签名** — 生产环境建议使用私有镜像仓库，验证镜像完整性

### 反向代理参考 (Nginx)

```nginx
server {
    listen 443 ssl;
    server_name askme.example.com;

    ssl_certificate /etc/ssl/certs/askme.crt;
    ssl_certificate_key /etc/ssl/private/askme.key;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

> 最后更新: 2026-06-01
> 维护: 穹沛科技 运维团队
