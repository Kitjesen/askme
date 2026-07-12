# GitHub Skill Research

日期：2026-06-05

状态：外部 skill 调研记录。本文只记录已在 GitHub 上读到 README 或 `SKILL.md` 的候选和采用建议，不表示已经安装或信任这些 skill。安装前必须逐个审查 `SKILL.md`、附带脚本、权限边界、许可证和维护状态。

## 调研目标

为 AskMe 当前“从产品需求出发，继续清理整理，并建立高级软件架构”的工作寻找可借鉴的 GitHub skill。重点不是找更多自动化，而是找能提升以下产出的结构化方法：

- 市场调研、竞品调研和替代方案矩阵。
- 方案商 ICP、访谈假设、JTBD、PRD、用户故事和验收标准。
- 需求到软件架构的追踪。
- 架构评审、边界守护和验证计划。

## 已验证候选

| 候选 | GitHub 证据 | 适合 AskMe 的点 | 风险 / 不直接安装原因 | 建议 |
| --- | --- | --- | --- | --- |
| OpenAI 官方 skills | https://github.com/openai/skills；`skills/.curated/notion-research-documentation/SKILL.md`、`skills/.curated/security-threat-model/SKILL.md`、`skills/.curated/pdf/SKILL.md` | 官方 Codex skill 格式和工具型能力来源。`notion-research-documentation` 适合把外部访谈/客户材料整理成带引用的 brief；`security-threat-model` 适合后续对 Field Delivery Domain、Dashboard/API、credential surface 做威胁建模；`pdf` 适合验收包、客户资料和供应商 PDF 的抽取/审阅。 | 官方 curated 目录偏工具链和文档处理，不提供 AskMe 所需的 PM 市场调研主流程；GitHub API 本轮触发 rate limit，只用 git sparse checkout 和仓库页面验证。 | 作为可信格式和工具型补充。当前不安装，后续若要处理 Notion/PDF 或做安全威胁模型，再按单个 skill 审查。 |
| Anthropic 官方 skills 示例库 | https://github.com/anthropics/skills；`template/SKILL.md` | 作为 `SKILL.md` 结构基准：folder + YAML frontmatter + instructions。 | 主要是示例和通用能力，不是 AskMe 的产品调研包；其中部分文档技能是 source-available，不等于可直接复用。 | 只作为格式和安装审查基准，不作为调研主方法。 |
| Mehdibargach/claude-code-pm-skills | https://github.com/Mehdibargach/claude-code-pm-skills；`skills/market-sizing/SKILL.md`、`skills/competitor-scan/SKILL.md`、`skills/user-interview-prep/SKILL.md`、`skills/feedback-analyzer/SKILL.md`、`skills/persona/SKILL.md`、`skills/product-teardown/SKILL.md` | 覆盖 PM 调研的最小闭环：产品拆解、竞品扫描、TAM/SAM/SOM、访谈设计、反馈归类、行为 persona。它的输出短、结构清晰，适合直接借到 AskMe 文档。 | 原本面向 Claude Code，英文输出为主；不能直接替代本项目的 AGENTS.md/OMX 路由。 | 首选借鉴对象。先不安装，把流程吸收到 `docs/MARKET_RESEARCH.md`、`docs/INTERVIEW_GUIDE_SOLUTION_PROVIDER.md` 和 `docs/COMPETITIVE_REPLACEMENT_MATRIX.md`。 |
| deanpeters/Product-Manager-Skills | https://github.com/deanpeters/Product-Manager-Skills；`skills/company-research/SKILL.md`、`skills/jobs-to-be-done/SKILL.md`、`skills/prd-development/SKILL.md`、`skills/tam-sam-som-calculator/SKILL.md`、`skills/positioning-statement/SKILL.md` | 结构比 Mehdibargach 更产品管理化，覆盖公司/竞品研究、JTBD、PRD、市场规模和定位语句。对 AskMe 最有用的是把“方案商交付负责人”的 job、pain、gain 和 PRD success criteria 连接起来。 | 内容偏通用 PM workshop；若直接套用，容易把机器人方案商交付中台写成泛 SaaS。 | 作为 PRD 和需求表达补强。优先借 `jobs-to-be-done`、`prd-development`、`positioning-statement` 的格式来收紧 `docs/PRODUCT_REQUIREMENTS.md`。 |
| mohitagw15856/pm-claude-skills | https://github.com/mohitagw15856/pm-claude-skills；`plugins/pm-discovery/skills/assumption-mapper/SKILL.md`、`plugins/pm-discovery/skills/user-interview-synthesis/SKILL.md`、`plugins/pm-engineering/skills/architecture-decision-record/SKILL.md`、`plugins/pm-engineering/skills/microservices-decomposition/SKILL.md`、`plugins/pm-engineering/skills/security-threat-model/SKILL.md` | 覆盖从发现到工程治理的桥接：假设风险图、访谈综合、ADR、服务边界、威胁建模。它比纯 PM skills 更适合 AskMe 当前“产品需求 -> 高级软件架构”的衔接。 | 仓库很大，插件/skill 分层复杂；不同 skill 可能触发面过宽。`microservices-decomposition` 只能作为边界思考，不代表 AskMe 必须微服务化。 | 不整包安装。把 `assumption-mapper` 用于需求风险台账，把 `architecture-decision-record` 用于未来架构变更记录。 |
| w95/awesome-claude-corporate-skills | https://github.com/w95/awesome-claude-corporate-skills；`04-marketing/market-research/SKILL.md`、`09-product-management/user-research-synthesizer/SKILL.md`、`09-product-management/prd-writer/SKILL.md`、`05-sales/competitive-intelligence/SKILL.md`、`12-procurement-supply-chain/vendor-evaluation/SKILL.md`、`08-it-engineering/software-architecture/SKILL.md` | 企业场景覆盖更宽，适合补市场研究、用户研究综合、销售 battlecard、供应商评估、软件架构检查清单。`vendor-evaluation` 对 AskMe 的 VMS/CMMS/OEM fleet 集成采购和 TCO 很有价值。 | 仓库很大，166 个 skill；不同 skill 质量、来源和许可证不完全一致；直接安装会增加触发噪声。 | 作为补充模板库。只挑 `market-research`、`user-research-synthesizer`、`vendor-evaluation`、`software-architecture` 的方法，不整包安装。 |
| w95 deep-research | `01-executive-leadership/deep-research/SKILL.md` | 能做多步长报告，适合后续严肃市场/技术 due diligence。 | 需要 `GEMINI_API_KEY`，声明每次约 2-10 分钟和 $2-5 成本；属于外部付费/凭据路径。 | 当前不采用。除非要做付费长报告，否则用本地 skill + web source-backed 调研即可。 |
| wshobson/agents marketplace | https://github.com/wshobson/agents；`docs/agent-skills.md`、`docs/harnesses.md`、`docs/plugin-eval.md` | 大型跨 harness marketplace，README 显示包含 84 plugins、192 agents、156 skills，并说明 Codex CLI 支持。`plugin-eval` 可作为未来评估自定义 AskMe skill 的质量框架。 | 规模很大，安装和路由成本高；多数 skill 是工程/SEO/平台能力，不是 AskMe 当前调研的主路径。 | 不安装。只借鉴 `plugin-eval` 的质量维度和 Codex harness 限制，用于后续自研 AskMe skill。 |

## 建议采用的调研组合

### 1. 产品和市场发现

使用 Mehdibargach 的 PM skills 作为主流程骨架，并用 deanpeters 的 JTBD/PRD/定位 skills 补表达质量：

1. `product-teardown`：拆解 OEM fleet、VMS、CMMS、通用 Agent、人工运营等替代方案的业务模型、核心循环、弱点和护城河。
2. `competitor-scan`：把直接竞争、间接竞争和替代方案落到定位轴、定价模型、白空间。
3. `market-sizing`：等定价单位明确后再做 TAM/SAM/SOM，避免把“机器人市场总规模”误当 AskMe 可服务市场。
4. `persona`：把“机器人方案商/集成商交付负责人”拆成 primary persona、secondary persona 和 anti-persona。
5. `jobs-to-be-done`：把客户需求写成“当机器人项目从 Demo 进入试点验收时，方案商需要降低现场交付不确定性”的 job，而不是泛泛写“需要智能机器人平台”。
6. `prd-development`：要求每个需求都带 success criteria、non-goals 和 verification target，再回写到 `docs/PRODUCT_REQUIREMENTS.md`。

### 2. 访谈和证据归类

用 `user-interview-prep` 和 `feedback-analyzer` 收紧访谈质量：

- 访谈问题要问过去行为和真实交付阻塞，不问“你会不会买 AskMe”。
- 每条访谈记录按 theme、sentiment、urgency、type 归类。
- 访谈结果回写到 `docs/INTERVIEW_GUIDE_SOLUTION_PROVIDER.md`，再反哺 `docs/SOLUTION_PROVIDER_ICP.md`。

### 3. 竞品和销售场景

用 w95 的 `market-research`、`competitive-intelligence` 和 `vendor-evaluation` 补 AskMe 的替代矩阵：

- `docs/COMPETITIVE_REPLACEMENT_MATRIX.md` 继续覆盖 OEM fleet、VMS、CMMS、人工运营、定制脚本、通用 Agent、AskMe。
- `vendor-evaluation` 用来记录客户为什么不愿意再买一个平台：实施成本、集成成本、数据迁移、SLA、内部项目管理成本、供应商锁定。
- 销售 battlecard 只作为内部思路，不要把 AskMe 包装成万能机器人平台。

### 4. 架构和 skill 质量

架构层仍以本地 `AGENTS.md`、`ai-slop-cleaner`、`analyze`、`code-review` 和文档契约测试为准；外部 `software-architecture` 只作为检查清单：

- AskMe 的业务边界继续围绕 Field Delivery Domain。
- Runtime / Safety / Hardware 不能被产品包装和调研话术偷偷吞掉。
- `plugin-eval` 的静态维度可用于后续自研 skill：frontmatter quality、progressive disclosure、structural completeness、token efficiency、harness portability。
- OpenAI `security-threat-model` 和 mohitagw 的 `architecture-decision-record` 可作为安全/架构治理补充，但必须围绕现有 `docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`，不能因为外部 skill 模板而引入不必要的服务拆分。

## 不采用 / 降级候选

- 直接整包安装大型 marketplace：当前脏工作区和路由已经复杂，整包安装会增加噪声。
- 需要外部 API key 或付费调用的 deep-research：本轮目标是明确需求和边界，不是自动生成大报告。
- 只在聚合站看到但本轮没有读到 `SKILL.md` 的候选：不作为当前依据。
- 泛 B2B SaaS 竞品模板：可以借格式，但不能直接套到“机器人现场运营交付中台”。

## 下一步建议

1. 不直接安装外部 skill；先把 Mehdibargach PM skills 的流程手工吸收到现有文档。
2. 用 `user-interview-prep` 的原则收紧 `docs/INTERVIEW_GUIDE_SOLUTION_PROVIDER.md`：过去行为、真实预算、真实验收阻断、forced choice。
3. 用 `feedback-analyzer` 的分类结构设计访谈记录表，再做 20-30 人访谈综合。
4. 用 w95 `vendor-evaluation` 补 `docs/COMPETITIVE_REPLACEMENT_MATRIX.md` 的采购/TCO/集成成本维度。
5. 用 deanpeters `jobs-to-be-done` 和 `prd-development` 复核 `docs/PRODUCT_REQUIREMENTS.md`，把 R1-R7 的 problem、non-goal、acceptance criteria 写得更像可交付 PRD。
6. 用 mohitagw `assumption-mapper` 把需求假设分成 high-impact / low-confidence 验证队列，写回 `docs/DEMAND_EVIDENCE_LEDGER.md`。
7. 如果未来要沉淀 AskMe 自有 skill，先按 Anthropic `template/SKILL.md` 写最小 skill，再用 wshobson `plugin-eval` 的质量维度审查。
