# 安全策略 / Security Policy

## 支持的版本 / Supported Versions

当前仅 4.x 系列接收安全更新。

| 版本 / Version | 支持状态 / Supported          |
| -------------- | ----------------------------- |
| 4.x            | :white_check_mark: 是 / Yes   |
| < 4.0          | :x: 否 / No                   |

---

## 报告安全漏洞 / Reporting a Vulnerability

如果你发现了 Askme 的安全漏洞，**请勿公开提交 Issue**。请直接发送邮件至：

**security@inovxio.com**

### 响应时间 / Response Timeline

我们承诺：

1. **48 小时内**确认收到你的报告
2. **7 个工作日内**给出初步评估和修复计划
3. **修复完成后**通知你，并在新版本发布时致谢

### 期望流程 / What to Expect

1. 你发送漏洞详情至 security@inovxio.com
2. 我们确认收到并开始评估
3. 我们与您协商合理的披露时间
4. 修复完成并发布安全更新后，我们会公开致谢（若你同意）

---

## 安全设计原则 / Security Design Principles

Askme 作为面向园区的机器人现场任务平台，安全性是核心设计目标。以下是我们遵循的关键原则：

### 1. LLM 不直接控制硬件 / LLM Does Not Directly Control Hardware

LLM（大语言模型）输出仅作为意图解析，不直接生成或传递硬件控制指令。所有硬件操作必须通过以下链路验证：

```
LLM 输出 → 意图解析 → 安全验证 → 权限检查 → 硬件控制服务 → 执行
```

### 2. 命令确认机制 / Command Confirmation

- 任何可能导致机器人移动、急停、恢复导航的操作，必须先经过用户确认
- 关键操作（如急停恢复）需要多次确认
- 语音命令在关键操作上要求语音二次确认

### 3. 边界隔离 / Boundary Isolation

Askme 采用三层边界架构（核心 / 感知 / 插件）：

- **核心层**：语音交互、记忆管理、工具注册——不直接访问硬件
- **感知层**：传感器数据处理——只读，不控制
- **插件层**：外部技能和工具——通过定义好的端口和协议与核心交互

详见 [docs/ASKME_BOUNDARY.md](docs/ASKME_BOUNDARY.md)（项目级文档路径：`tools/askme/docs/ASKME_BOUNDARY.md`）。

### 4. 最小权限 / Least Privilege

- 外部技能和插件运行在受限的环境中
- API 访问基于角色和权限校验
- 敏感操作需要显式授权

### 5. 审计日志 / Audit Trail

- 所有关键操作（硬件控制、配置变更、权限修改）记录审计日志
- 日志不可篡改，支持事后追溯

### 6. 输入验证 / Input Validation

- 所有外部输入（语音、文本、API 请求）经过格式校验和内容过滤
- SQL 注入、提示注入等攻击向量有专门防御措施

---

## 安全相关文档 / Security-Related Documentation

- [ASKME_BOUNDARY.md](docs/ASKME_BOUNDARY.md) — 三层边界架构和安全隔离设计
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — 系统整体架构
