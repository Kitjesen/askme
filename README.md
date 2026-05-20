# Askme

## Contributor Start Here

For parallel development, start with the ownership and workflow contracts before
editing code:

- `docs/MODULE_OWNERSHIP.md` maps each collaboration lane to its package scope,
  exclusions, and required verification command.
- `docs/MULTI_AGENT_WORKFLOW.md` explains how the lead agent assigns independent
  work, reserves shared files, and integrates worker results.
- Boundary-sensitive changes should always run
  `pytest tests/test_six_layer_package_boundaries.py tests/test_package_migration_compat.py -q`.

Askme 是面向园区、厂区、仓储和景区巡检机器人的现场任务平台。它把语音、文本、知识库、技能、现场事件、运行交接和审计记录组合成一个可交付的机器人产品入口。

Askme 的产品边界很明确：

- 用户可以用语音或文本发起问路、巡检、异常处置和知识问答。
- 系统会把自然语言转成可确认、可审计、可中断的任务或回答。
- 大模型和语音层不直接控制硬件；机器人动作必须经过 TaskHandoff、SafetyPreflight 和 runtime arbiter。
- 客户知识必须经过上传、治理、检索和证据展示；没有依据时应要求确认或拒答。

## 当前产品能力

| 能力 | 当前状态 | 说明 |
| --- | --- | --- |
| 语音任务中心 | 可用 | 麦克风输入、ASR、LLM、TTS、打断和文本兜底由 `voice` 运行时承载。 |
| 园区巡检机器人运行时 | 可用作试点验证 | 覆盖语音、感知、现场事件、运行交接、控制适配、灯光状态和主动监测。 |
| 客户知识库 | 已有基础闭环 | 支持上传、预览、审批、重建索引、检索证据和过期/冲突治理。 |
| 现场事件 | 已有产品链路 | 支持摔倒无法恢复、卡住、电机故障、违停、烟火、垃圾桶、人群聚集、陌生人和问询点触发。 |
| 空间认知 | 已有园区点位模型 | 支持点位、别名、服务点、路线说明、问路和带路任务基础能力。 |
| 能力中心 | 已有目录和准入 | 把底层 skills 映射成客户可见能力包、场景蓝图、风险等级和审批依赖。 |
| MCP 工具服务 | 已有入口 | 可向 MCP 客户端暴露受控工具和资源，不提供原始硬件控制权限。 |

## 快速启动

安装依赖：

```powershell
cd D:\inovxio\tools\askme
pip install -e ".[dev]"
```

查看可交付运行蓝图：

```powershell
python -m askme.cli runtime blueprints --customer-visible
```

启动文本运行时：

```powershell
python -m askme.blueprints.presets.text
```

启动语音任务中心：

```powershell
python -m askme.blueprints.presets.voice
```

启动园区巡检机器人运行时：

```powershell
python -m askme.blueprints.presets.edge_robot
```

启动 Web Dashboard：

```powershell
python scripts/dev/run_dashboard_only.py --host 127.0.0.1 --port 8766
```

然后打开：

```text
http://127.0.0.1:8766/dashboard
```

## 运行蓝图

蓝图是 Askme 的产品运行组合。代码位置：

- `askme/blueprints/catalog/`：蓝图目录、客户可见描述、readiness、交付包。
- `askme/blueprints/presets/`：具体运行时组合。
- `askme/blueprints/runner/`：统一启动辅助。

客户可见蓝图：

| 蓝图 | 启动命令 | 用途 |
| --- | --- | --- |
| 语音任务中心 | `python -m askme.blueprints.presets.voice` | 客户演示、语音问答、语音任务确认，不直接接硬件。 |
| 语音感知运行时 | `python -m askme.blueprints.presets.voice_perception` | 在语音基础上接入感知 freshness、交互准入和安全状态。 |
| 园区巡检机器人运行时 | `python -m askme.blueprints.presets.edge_robot` | 面向园区/厂区试点，接入现场事件、控制适配和主动监测。 |
| 灵途语音导航适配器 | `python -m askme.blueprints.presets.lingtu_voice` | 面向灵途导航项目的站点定制入口。 |

内部蓝图：

| 蓝图 | 启动命令 | 用途 |
| --- | --- | --- |
| 文本运营控制台 | `python -m askme.blueprints.presets.text` | 研发、交付、CI 和无音频环境调试。 |
| MCP 工具服务 | `python -m askme.mcp.server` | 向 MCP 客户端提供受控工具能力。 |

导出某个蓝图交付包：

```powershell
python -m askme.cli runtime blueprints --name park --delivery-package --json
```

## Dashboard 页面

Dashboard 是客户和交付人员查看产品能力的入口。当前重点页面：

- `/dashboard`：总览、对话、现场事件、运行状态。
- `/dashboard/knowledge`：客户知识库，查看已有知识、上传、预览、审批、重建索引和证据。
- `/dashboard/capabilities`：机器人能力中心，查看巡检、安防、访客服务、空间认知、语音、审计等能力。
- `/dashboard/projects`：客户项目、对象目录、模板和交付包。
- `/dashboard/voice`：语音状态、音色、播放策略和语音健康。

客户演示时优先走这条顺序：

1. 打开总览，确认系统在线。
2. 打开客户知识库，确认机器人“知道什么”和“依据在哪里”。
3. 打开能力中心，说明机器人能做哪些业务动作、哪些需要审批。
4. 打开现场事件，演示违停、烟火、垃圾桶、机器人故障等场景。
5. 进入对话，演示问路、巡检和知识问答。

## 典型业务场景

| 场景 | 产品行为 |
| --- | --- |
| 游客问路 | 在服务点识别停留和交互意图，解析目的地，给出语音指路；必要时转带路任务。 |
| 游客带路 | 先确认目的地，再检查路线是否可通行，低速引导并记录服务结果。 |
| 车辆违停 | 检测非停车区停车，拍照、附地点、通知保安并归档事件。 |
| 烟火监测 | 接入温度、烟雾或视觉烟火证据，播报风险，通知保安并归档。 |
| 垃圾桶满溢 | 定点检测垃圾桶状态，通知保洁并生成事件记录。 |
| 夜间陌生人 | 识别窗边、角落等重点区域陌生人，拍照并通知保安。 |
| 机器人异常 | 摔倒无法恢复、卡住、电机故障时播报、通知、归档并等待处理。 |
| 突发巡检 | 管理员发起临时任务，系统中断或暂停当前巡检，交接给 runtime。 |

## 知识库和记忆

Askme 把两类记忆分开：

- 客户知识库：园区点位、路线、SOP、设备说明、FAQ，用于回答和证据展示。
- 机器人行为记忆：长期行为、任务偏好、历史运行经验，默认不和客户知识混在一起。

配置建议：

```yaml
memory:
  enabled: true
  customer_knowledge_backend: vector
  robot_behavior_memory_backend: robotmem
  robot_behavior_memory_enabled: false
```

产品原则：

- 过期、冲突、未审批知识不能直接进入回答。
- 回答气泡应展示引用依据。
- 没有证据时，系统应要求确认或拒答。
- 每条知识需要责任人、来源、版本和有效期。

## 语音链路

推荐国产低延迟链路：

```text
实时 ASR -> MiniMax-M2.7-highspeed -> TaskHandoff / SafetyPreflight / runtime arbiter -> MiniMax Speech 2.8 TTS
```

语音体验需要关注：

- 什么时候可以说话：UI 必须显示“正在听 / 正在思考 / 正在播报 / 可打断”。
- 为什么慢：拆分 ASR、LLM、TTS、播放和打断延迟。
- 为什么误触发：InteractionGate 需要结合服务点、声源、视觉、距离、停留和感知 freshness。
- 为什么断断续续：需要检查 TTS 分片、播放缓冲、声卡采样率和回声门限。

常用诊断：

```powershell
python -m askme.cli runtime audio-devices
python -m askme.cli runtime voice-health --json
python -m askme.cli runtime voice-online-smoke
python -m askme.cli runtime sunrise-voice-readiness --json
```

## 现场交付门禁

客户试点前至少要提供这些证据：

- 蓝图交付包：`runtime blueprints --delivery-package`
- 现场 readiness：`runtime field-readiness --json`
- 语音健康：`runtime voice-health --json`
- 现场事件冒烟：`runtime field-smoke-suite`
- DingTalk 通知预检：`runtime field-notification-preflight`
- 审计完整性：`runtime field-audit-integrity`

示例：

```powershell
python -m askme.cli runtime field-readiness --json
python -m askme.cli runtime field-smoke-suite --json
python -m askme.cli runtime field-audit-integrity --json
```

## 测试

默认 pytest 分片只跑快测；`pyproject.toml` 通过 `-m "not slow"` 排除慢测。
`tests/conftest.py` 会自动把 `tests/scenario_tests/`、`*e2e*` 和 benchmark
测试标为慢测分片。

```powershell
python -m pytest tests -q
python -m pytest tests -m "slow" -q
python -m pytest tests -m "scenario" -q
python -m pytest tests -m "e2e or benchmark" -q
```

常用快速回归：

```powershell
python -m pytest tests/test_blueprints_catalog.py tests/test_cli.py -q
python -m pytest tests/test_capability_center.py tests/test_memory_bridge.py -q
python -m pytest tests/test_api_route_dependency_injection.py -q
```

本地修改蓝图、API、记忆或能力中心后，至少运行：

```powershell
python -m pytest tests/test_package_migration_compat.py -q
```

## 项目结构

```text
askme/
  api/            FastAPI 路由和产品 API 表面
  audit/          审计查询、导出、完整性和复核
  blueprints/     产品运行蓝图
  cognition/      认知规划、任务理解和上下文
  memory/         客户知识库和机器人行为记忆
  pipeline/       现场事件、空间认知、交付 readiness
  runtime/        Runtime 模块组合和运行服务
  skills/         技能、能力包和技能准入
  static/         Dashboard 前端
  voice/          语音输入、ASR、TTS、播放和诊断
docs/
  PRODUCT.md      产品手册和路线图
  ARCHITECTURE.md 架构说明
  OPERATIONS.md   运维和交付说明
```

## 交付边界

可以对客户承诺：

- 支持受控场景下的语音问路、知识问答、巡检任务和现场事件处理。
- 支持试点项目的能力中心、知识库、现场事件和审计闭环。
- 支持通过配置和交付包复制到不同客户项目。

不能在没有现场证据前承诺：

- 无人值守生产运行。
- 任意开放域问答。
- 大模型直接控制机器人硬件。
- 未配置真实传感器、通知和机器人控制网关时的真实处置效果。

## 进一步文档

- [产品手册](docs/PRODUCT.md)
- [架构说明](docs/ARCHITECTURE.md)
- [运维交付](docs/OPERATIONS.md)
