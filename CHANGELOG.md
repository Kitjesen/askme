# Askme Changelog

## 4.1.0 (2026-05-31)

### 交付运营平台产品化
- 多客户项目交付工具链：行业模板市场、客户项目目录、管理对象目录
- 交付资源注册中心：视觉模型、传感器协议、技能包、验收测试统一注册
- 客户提案包导出：JSON + 可打印 HTML，含验收证据清单和 SHA-256 哈希
- 验收闭环：onsite evidence 注册、acceptance review、customer signoff
- 交付资源治理请求：SLA 追踪、逾期升级、审批队列

### 运行时诊断加固
- Audio routing 诊断增强：USB-direct 路径、Sunrise MCP01 声卡容错
- 运行时诊断对话烟雾测试：dialogue smoke、runtime roundtrip
- 多 Agent 审查后测试覆盖率加固

### 所有权边界
- Module ownership 可执行化：明确包所有权、依赖方向、验证命令
- 六层架构边界测试：ports → providers → runtime → blueprints

### 产品模型路由
- 可审计场景意图路由：规则优先 + LLM 兜底，每次命中带 scenario_id
- 技能可调用性暴露：产品模型路由、skill callability

## 4.0.0 (2026-05-10)

### 交付运营基础
- 客户项目、行业模板、管理对象、导入导出基础
- 能力中心和技能包产品结构
- 任务确认、安全预检和运行调度闭环
- 审计与交付验收入口

### 语音交互
- Voice Mission Center 三栏 Dashboard
- Interaction Gate：多模态交互准入判断
- MiniMax TTS 链路

### 记忆与 RAG
- L0-L6 记忆分层
- KnowledgeCatalog 知识生命周期管理
- RAG Trust 离线评测

### 安全
- TaskHandoff → SafetyPreflight → runtime arbiter 链路
- 操作员目录和 RBAC 基础
- 统一审计查询和签名导出

## 3.x (2026-Q1)

- RobotMem (BM25+向量) 混合检索
- 首批 built-in 园区场景技能
- MCP server 工具层接入
- S100P 现场部署
- Voice Blueprint 跑通
