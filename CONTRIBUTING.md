# 贡献指南 / Contributing Guide

欢迎为 **Askme** 项目做出贡献！Askme 是穹沛科技（inovxio）面向园区的机器人现场任务与智能交互平台。

---

## 开发环境设置 / Development Setup

```bash
# 克隆仓库
git clone https://github.com/inovxio/askme.git
cd askme

# 创建虚拟环境（需要 Python 3.11+）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装开发依赖（包含 dev 和 embed 可选组）
pip install -e ".[dev,embed]"
```

### 依赖分组 / Dependency Groups

| 分组 | 用途 |
|------|------|
| `dev` | 测试、lint、类型检查（pytest, ruff, mypy） |
| `robot` | 机器人硬件控制（pyserial, onnxruntime） |
| `memory` | Mem0 记忆后端与本地 fastembed/ONNX 向量后端 |
| `robotmem` | RobotMem 记忆后端（CJK 支持） |
| `mempalace` | MemPalace 记忆后端 |
| `vision` | 视觉感知（qp-perception） |
| `embed` | 文本嵌入（fastembed/ONNX） |

### FastEmbed 离线模型就绪 / Offline Model Readiness

生产运行严格使用 `local_files_only=True`，不会在启动或查询时下载模型。
请预先把 Hugging Face 仓库
`Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q` 放入 FastEmbed 缓存。
默认缓存是系统临时目录下的 `fastembed_cache`（Linux 通常为
`/tmp/fastembed_cache`）；可用 `FASTEMBED_CACHE_PATH` 显式指定持久化目录。
对应 snapshot 必须包含：

- `config.json`
- `model_optimized.onnx`
- `special_tokens_map.json`
- `tokenizer.json`
- `tokenizer_config.json`

在 memory 健康信息中，`vector_model_status.reason=model_artifacts_missing`
且 `selected_backend_dependency.runtime_ready=false` 表示 Python 依赖可能已安装，
但本地模型仍不完整；`vector_model_status.reason=local_model_ready`、
`selected_backend_dependency.runtime_ready=true` 且
`selected_backend_ready=true` 才表示向量检索可运行。

---

## 代码风格 / Code Style

本项目使用 [ruff](https://docs.astral.sh/ruff/) 进行代码检查和格式化：

```bash
# 检查代码
ruff check .

# 自动格式化
ruff format .

# 类型检查（可选）
mypy askme/
```

配置规则见 `pyproject.toml` 中的 `[tool.ruff]` 和 `[tool.mypy]` 部分。

---

## 运行测试 / Running Tests

```bash
# 运行所有测试（跳过慢速测试）
pytest

# 运行全部测试（包含慢速）
pytest --run-slow

# 运行指定测试文件
pytest tests/test_voice_loop.py

# 带覆盖率
pytest --cov=askme
```

测试标记说明：
- `slow`：较慢的集成测试，默认排除
- `scenario`：客户场景 / 运行态场景评估测试
- `e2e`：端到端工作流测试
- `benchmark`：基准 / 回归阈值测试

使用 `pytest -m <marker>` 运行指定标记的测试。

---

## 提交规范 / Commit Convention

本项目采用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <简短描述>

<可选的详细描述>

<可选的 footer>
```

### 类型 / Types

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `docs` | 文档变更 |
| `style` | 代码风格调整（不影响功能） |
| `refactor` | 重构（既不修 bug 也不加功能） |
| `test` | 测试相关 |
| `chore` | 构建、CI、依赖等杂项 |
| `perf` | 性能优化 |
| `ci` | CI 配置变更 |

### 示例 / Examples

```
feat(voice): 添加流式 ASR 支持
fix(memory): 修复向量检索中的路径编码问题
docs: 更新 README 中的 API 示例
test(runtime): 添加模块热加载测试
```

---

## PR 流程 / Pull Request Process

1. **确保测试通过**：提交前运行 `pytest`，确保所有测试通过
2. **确保代码风格合规**：运行 `ruff check .` 和 `ruff format .` 无报错
3. **保持小粒度**：一个 PR 解决一个问题。大型变更请拆分为多个小 PR
4. **描述清晰**：PR 标题和描述清楚说明变更内容和动机
5. **更新文档**：如果变更涉及接口或行为变化，同步更新相关文档
6. **Code Review**：至少一位维护者 review 后方可合并

---

## 多 Agent 协作 / Multi-Agent Collaboration

Askme 项目支持多个 AI agent 并行协作开发。详细的协作流程和分工指南请参阅：

**[docs/MULTI_AGENT_WORKFLOW.md](docs/MULTI_AGENT_WORKFLOW.md)**

核心原则：
- 每个 agent 拥有明确的模块边界
- 主 agent 负责任务分解和最终集成
- Worker agent 只操作分配给它的模块范围
- 主 agent 负责最终测试验证

---

## 项目结构 / Project Structure

```
askme/
├── askme/              # 主源码
│   ├── api/            # API 路由和中间件
│   ├── mcp/            # MCP 协议实现
│   ├── memory/         # 记忆系统（L0-L6）
│   ├── runtime/        # 运行态模块
│   ├── voice_gateway/  # 语音网关
│   ├── providers/      # 外部服务适配器
│   ├── ports/          # 端口抽象层
│   └── tools/          # 内置工具
├── docs/               # 文档
├── tests/              # 测试
├── data/               # 运行时数据
├── scripts/            # 工具脚本
└── prompts/            # LLM 提示词模板
```

---

## 获取帮助 / Getting Help

- 提交 [Issue](https://github.com/inovxio/askme/issues)
- 联系维护者：森哥（Kitjesen）

---

再次感谢你的贡献！ / Thank you for contributing!
