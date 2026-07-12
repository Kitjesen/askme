# API Key 轮换指南

> 适用范围：Askme 生产环境部署
> 目标：在发现密钥可能泄露时安全、完整地轮换所有 API Key，确保服务不中断

---

## 1. 确认泄露范围

在轮换前，先确定哪些密钥需要轮换：

```bash
# 1a. 扫描 git 历史中所有硬编码的密钥
git log --all -p | grep -E "sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{35}|cr_[A-Za-z0-9]{20,}"

# 1b. 找出这些值出现在哪些 commit 中
git log --all -S "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" --format="%H %ai %s"

# 1c. 检查这些 commit 是否已被推送
git branch -r --contains <COMMIT_HASH>
```

### 已知泄露案例

本仓库曾有一个 DeepSeek API Key 被硬编码在 `examples/` 目录的脚本中，并在初始提交 `40737e19` 中推送到 GitHub。虽然文件已在 `2f7cea8a` 中删除，但密钥仍存在于 git 历史中。**必须轮换该密钥。**

**轮换清单（按优先级排序）：**

| 密钥 | 服务 | 轮换方式 | 是否已推送到远端 |
|------|------|---------|----------------|
| DeepSeek API Key (`sk-030f66...affc5ec`) | DeepSeek LLM | DeepSeek 控制台 | 是 (GitHub) |
| MiniMax API Key | MiniMax TTS/LLM | MiniMax 控制台 | 待确认 |
| DashScope API Key | Alibaba Cloud ASR | 阿里云控制台 | 待确认 |
| Telegram Bot Token | Telegram 接口 | @BotFather | 待确认 |
| RUNTIME_BEARER_TOKEN | Runtime 认证 | 重新生成 | 待确认 |

---

## 2. 轮换流程

### 2.1 生成新密钥

访问各平台控制台生成新密钥：

| 服务 | 控制台地址 |
|------|-----------|
| MiniMax | https://platform.minimax.chat/ |
| DeepSeek | https://platform.deepseek.com/api_keys |
| DashScope | https://dashscope.aliyun.com/ |
| OpenAI | https://platform.openai.com/api-keys |
| Telegram | https://t.me/BotFather |
| Anthropic | https://console.anthropic.com/ |

### 2.2 更新 `.env`

```bash
# 用新密钥替换旧密钥
# 注意：不要直接在 git 管理的文件中编辑带密钥的 .env
# 生产环境的 .env 应通过安全通道分发

# 编辑 .env（该文件已被 .gitignore 忽略，不会提交）
$EDITOR .env
```

### 2.3 重新部署服务

```bash
# 重启 Askme 进程使新密钥生效
# --- systemd ---
sudo systemctl restart askme

# --- Docker ---
docker-compose restart askme

# --- 手动启动 ---
# 先停掉旧进程，重启
pkill -f "python -m askme"
python -m askme
```

### 2.4 验证新密钥可用

```bash
# 验证 LLM 路由
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello", "model": "minimax"}'

# 验证 ASR
python scripts/demo/demo_asr.py --file test.wav

# 验证 TTS
python -c "from askme.voice.tts import speak; speak('测试语音')"

# 运行预提交密钥扫描（确认新密钥没有被硬编码）
python scripts/audit/check_secrets.py
```

---

## 3. 清理 Git 历史中的泄露密钥

如果一个密钥已经被推送到 Git 远程仓库，仅仅删除当前文件是不够的——密钥仍存在于历史中。有两种策略：

### 方案 A：轮换密钥（推荐，低风险）

不修改 git 历史，直接轮换密钥使旧密钥失效。这是最简单、最安全的方式。

### 方案 B：BFG Repo-Cleaner 重写历史（高风险，破坏性操作）

> 仅在绝对必要时使用。重写历史后所有协作者需要重新克隆。

```bash
# 1. 下载 BFG
# https://rtyley.github.io/bfg-repo-cleaner/

# 2. 用 BFG 替换特定密钥
java -jar bfg.jar --replace-text passwords.txt askme.git

# 3. 清理并强制推送
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push --force origin master
```

**重写历史的风险：**
- 所有协作者必须重新克隆仓库
- 如果有基于旧历史的 PR/issue，关联会断裂
- 如果有 CI/CD 流水线引用旧 commit，会出错
- 如果有其他分支未合并，需要逐个处理

---

## 4. 预防措施

### 4.1 启用 pre-commit 密钥扫描

```bash
# 在 .git/hooks/pre-commit 中添加：
#!/bin/sh
python scripts/audit/check_secrets.py --all-files
if [ $? -ne 0 ]; then
    echo "ERROR: 检测到可能的密钥泄露，提交已阻止。"
    exit 1
fi
```

### 4.2 定期全量扫描

```bash
# 扫描所有 git 跟踪的文件
python scripts/audit/check_secrets.py --all-files --json

# 扫描整个代码库目录
python scripts/audit/check_secrets.py --path /path/to/askme --json
```

### 4.3 对照清单

每次添加新的外部 API 集成时，检查：

- [ ] API Key 只出现在 `.env`（已 gitignored）中
- [ ] `.env.example` 中对应 key 的值为空或 `your-key-here`
- [ ] 代码中的配置项通过 `os.getenv()` 读取，不设置默认值
- [ ] 没有任何测试文件、示例脚本、文档中包含真实密钥
- [ ] CI 中的密钥通过 GitHub Secrets / 环境变量注入

---

## 5. 检查 .env.example 是否泄露

`.env.example` 不应包含真实密钥：

```bash
# 检查当前版本
cat .env.example

# 检查 git 历史中 .env.example 的所有版本
git log --all -p -- .env.example | grep -E "sk-|AIza|ghp_|cr_"
```

如果 `.env.example` 的历史版本中包含真实密钥：
1. 立即按本指南第 3 节轮换密钥
2. 考虑使用 BFG 重写 `.env.example` 的历史
3. 用空值或 `your-key-here` 替换所有 `.env.example` 中的真实密钥

---

## 6. 快速参考命令

```bash
# 查找 git 历史中所有 sk- 开头的密钥
git log --all -p | grep -E "sk-[A-Za-z0-9]{20,}"

# 查找当前代码中硬编码的 api_key 赋值
git grep -n "api_key.*=.*\"[A-Za-z0-9_-]\{20,\}" -- "*.py" "*.yaml"

# 运行预提交扫描
python scripts/audit/check_secrets.py

# 验证 .env 是否被 git 跟踪（应该不跟踪）
git ls-files .env
```
