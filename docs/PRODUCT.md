# AskMe 产品手册

版本：V0.8 试点交付版  
适用对象：客户决策人、现场运营、安保/保洁主管、交付工程师、售前销售、研发测试  
产品定位：面向园区、厂区、仓储、景区等场景的“机器人现场任务与智能交互平台”

## 1. 产品概览

AskMe 不是一个普通聊天框，也不是机器人底层控制系统。它是部署在机器人和客户现场系统之间的“任务大脑与交互入口”。

用户可以通过中文语音或文本表达需求，例如“巡检 A 区”“带我去咖啡店”“垃圾桶满了”“前面有人挡路”。系统会判断这句话是普通问询、现场事件、巡检任务、知识问答，还是需要拒绝或二次确认。对于可执行任务，AskMe 会生成结构化任务，经过安全预检、权限判断、运行调度，再进入演示、仿真、实验室或真实机器人运行链路。

一句话价值：AskMe 让机器人从“只能执行固定路线的设备”，升级为“能听懂现场需求、按规则执行任务、能留痕、可验收、可复制交付的现场服务终端”。

## 2. 产品边界

AskMe 的核心原则是：大模型不直接控制硬件。

大模型负责理解、规划、解释和对话；任务仲裁、安全预检、技能包、运行时和机器人控制系统负责真正执行。这样做的目的是避免“听错一句话就直接让机器人动作”的风险。

当前版本适合以下交付口径：

- 客户演示：展示语音交互、事件处置、知识问答、空间问路、任务流转和审计记录。
- 试点项目：围绕有限服务点、有限点位、有限路线、有限异常类型做现场试点。
- 生产上线：需要补齐客户身份系统、真实硬件运行链路、现场传感器、机器人控制适配、正式验收用例和安全责任边界后才能承诺。

不建议把当前版本描述为“完全无人值守生产系统”。更准确的说法是：AskMe 已具备产品化试点交付骨架和多个核心功能闭环，生产上线需要按客户现场继续接入和验收。

## 3. 典型客户与场景

AskMe 面向方案商和机器人集成商。不同客户的现场对象不同，但产品底层能力可以复用。

| 客户类型 | 典型对象 | 重点场景 |
| --- | --- | --- |
| 创意园区/商业园区 | 楼宇、商户、道路、卫生间、停车区、服务点 | 巡检、游客问路、带路、夜间异常、车辆违停、人群聚集 |
| 工厂/厂区 | 产线、设备、仓库、危化区、消防点、通道 | 设备巡检、烟火监测、人员闯入、故障上报、异常播报 |
| 仓储物流 | 货架、库区、装卸区、通道、充电点 | 通道阻塞、货物异常、人员聚集、机器人调度、库存辅助巡查 |
| 景区/文旅 | 景点、出入口、服务台、卫生间、停车区、游客动线 | 游客问路、应急广播、人流聚集、带路服务、遗失物线索记录 |
| 园区安保 | 门岗、围栏、窗户、死角、道路、重点点位 | 夜间陌生人拍照、恶意挡路、巡逻异常、保安通知、事件归档 |

产品设计必须支持“一套平台、多客户项目、多对象目录、多行业模板”的交付方式，而不是为每个客户重新写一套程序。

## 4. 用户角色

| 角色 | 关心的问题 | 在 AskMe 中的能力 |
| --- | --- | --- |
| 客户负责人 | 版本解决什么业务问题，能不能验收 | 看产品总览、场景清单、事件记录、交付状态和验收结果 |
| 现场主管 | 今天发生了什么，谁处理了，证据在哪里 | 查看现场事件、通知记录、处理状态和审计报告 |
| 安保/保洁人员 | 接到通知后要去哪里、处理什么 | 接收通知，查看位置、照片、事件类型和处理建议 |
| 交付工程师 | 新客户怎么配置、怎么复制项目 | 管理客户项目、对象目录、行业模板、导入导出包和验收用例 |
| 运营人员 | 知识库怎么维护，问路点位怎么改 | 上传知识、审批发布、维护点位别名、查看问询记录 |
| 研发测试 | 功能是否可测，接口是否稳定 | 使用 API、审计日志、运行状态、测试用例和模拟事件 |

## 5. 产品模块

### 5.1 语音与文本交互中心

用于承接用户自然语言输入，支持麦克风语音、文本输入、实时状态展示和语音播报。

核心能力：

- 显示“正在听、识别中、思考中、播报中、等待确认、已拒绝”等明确状态。
- 支持中途打断、重新说、取消任务、继续任务。
- 对游客问路、管理员任务、现场异常、普通问答做不同处理。
- 回复内容保持中文、短句、现场可听懂，不把内部术语直接说给客户或游客。
- 支持配置不同语音音色、播报策略和延迟指标。

验收重点：

- 用户能清楚知道什么时候可以说话。
- 没听清时会要求复述，而不是乱执行。
- 播报不卡顿、不重复、不把内部状态当成客户话术。

### 5.2 交互准入门

真实环境中不能“听到人声就回答”。AskMe 使用交互准入门判断是否应该进入对话。

判断依据包括：

- 是否位于服务点或允许交互区域。
- 是否检测到有人靠近、停留、看向机器人或发出明确问询。
- 声源方向与视觉目标是否一致。
- 距离是否在可交互范围内。
- 感知数据是否新鲜，是否过期。
- 是否多人同时说话，需要澄清或忽略。
- 当前是否正在处理更高优先级任务，例如火灾、故障、紧急巡检。

可能结果：

- 回复：确认对方正在和机器人说话。
- 澄清：不确定是否在问机器人。
- 记录但不回复：环境声音或旁人闲聊。
- 忽略：噪声、过远、无视觉目标、声画不一致。
- 中断当前服务：出现更高优先级安全事件。

### 5.3 现场事件处置

AskMe 支持把真实传感器、摄像头、机器人状态或人工上报统一转成现场事件。

首批场景包括：

- 机器人摔倒无法恢复。
- 机器人卡住无法运动。
- 人为恶意挡路。
- 关节电机故障。
- 夜间陌生人在窗户、角落、围栏等区域拍照或停留。
- 车辆违停，占用普通道路、主通道或禁停区。
- 火灾、烟雾、温度异常。
- 垃圾桶满溢。
- 突发任务巡检。
- 人群聚集。

标准处理流程：

1. 接收事件：来自摄像头、传感器、机器人状态、地图规则或人工触发。
2. 识别场景：判断事件类型、风险等级、地点、证据和响应组。
3. 生成播报：面向现场人员播放固定或大模型润色后的中文提示。
4. 通知响应人：按规则通知安保、保洁、运维或管理员。
5. 归档证据：保存时间、地点、图片、设备状态、处理记录和通知结果。
6. 跟踪闭环：支持确认、重新通知、申请关闭、主管审批关闭。
7. 生成报告：形成客户可读的事件报告。

### 5.4 园区空间认知与问路带路

用于让机器人理解园区地点、商户、道路、功能区和常用别名。

核心对象：

- 点位：楼宇、商户、卫生间、出口、楼梯、停车区、服务台、打卡点。
- 别名：咖啡店、咖啡馆、喝咖啡的地方、西门、停车的地方等。
- 服务点：入口、路牌、核心商户区、停车区入口等。
- 路线：可语音指路路线、可机器狗带路路线、禁行区域、楼梯/窄道/施工区。

典型流程：

1. 机器人巡检经过服务点。
2. 识别访客是否停留。
3. 主动问候：“你好，请问需要指路吗？”
4. 访客说出目的地。
5. 系统解析目的地并二次确认。
6. 根据路线和通行条件选择语音指路或带路。
7. 带路结束后记录服务结果，并恢复原巡检。

验收建议：

- 首期配置 3-5 个问询服务点。
- 首期维护核心点位和常见别名。
- 首期交付 3-5 条可演示带路路线。
- 超出知识库范围时明确说“不确定”，而不是编造路线。

### 5.5 任务运行与安全预检

用户确认任务后，AskMe 不会直接调用机器人控制。系统会先生成结构化任务并进入运行闭环。

任务生命周期：

草稿 → 等待确认 → 已确认 → 安全预检 → 排队/执行 → 暂停/阻塞/完成/失败/取消 → 复盘归档

安全预检检查：

- 操作人是否有权限。
- 任务是否属于当前客户项目。
- 点位、路线和对象是否存在。
- 所需技能包是否启用。
- 当前环境是否允许执行。
- 是否需要主管二次确认。
- 机器人运行模式是否匹配演示、仿真、实验室或现场。

客户可见表达应使用“等待确认、正在检查、准备执行、执行中、已暂停、已完成”等词，不直接展示内部状态名。

### 5.6 知识库与证据回答

AskMe 的回答应尽量来自客户已审批知识，而不是开放式编造。

知识管理能力：

- 上传知识：支持 Markdown、纯文本、CSV、JSON、JSONL/NDJSON；PDF、DOCX、XLSX、图片等非结构化文件需要先转换为可解析格式。
- 预览知识：查看解析结果、文件类型、预览方式、质量状态、可见范围和客户/项目关联。
- 审批知识：未审批内容不能进入正式回答。
- 发布知识：进入可检索知识库。
- 删除/恢复：支持运营维护。
- 版本治理：保留修订、字段差异和回滚入口。
- 重建索引：知识更新后重新生成检索索引。
- 冲突检测：同一问题存在多个相互矛盾答案时提示处理。
- 过期控制：过期、待复核、仅内部、冲突、删除知识不应继续作为客户回答依据。
- 关联管理：每条知识可绑定客户、项目、产品模块、推进事项和对象 ID。
- 证据展示：回答气泡展示引用来源、版本、更新时间和命中原因。

问答规则：

- 有可靠证据：回答并展示依据。
- 仅内部资料：可存档和检索治理状态，但不能进入客户回答 prompt。
- 待复核资料：进入审批队列，不参与正式回答。
- 证据过期：提示知识已过期，要求确认。
- 证据冲突：提示存在冲突，不直接给唯一结论。
- 无证据：拒答或转人工，不编造。
- 游客问路：只回答园区空间范围内的问题，不误触发机器人任务。

### 5.7 客户项目与对象目录

AskMe 面向多个客户交付，需要把不同客户的现场对象、技能包、传感器、验收用例分开管理。

客户项目包含：

- 客户名称、项目名称、现场名称。
- 行业模板：园区、工厂、仓储、景区等。
- 管理对象：楼宇、设备、点位、路线、停车区、垃圾桶、消防点、服务点等。
- 绑定资源：视觉模型、传感器协议、技能包、通知组、验收用例。
- 交付命名空间：确保不同客户数据隔离。
- 导入导出包：方便交付团队复制一个新客户项目。
- 实施交接清单：现场还缺什么、谁负责、什么时候验收。

产品目标：

- 不同客户可以复用同一平台。
- 交付人员不需要改代码就能创建客户项目。
- 每个对象都能关联检测模型、传感器、技能和验收标准。

### 5.8 能力中心与技能包

AskMe 把机器人能力抽象成可管理的技能包，而不是散落在代码里的函数。

技能包示例：

- 语音输入包：麦克风、ASR、唤醒、打断。
- 语音播报包：TTS、音色、固定话术、紧急播报。
- 视觉识别包：人、车、烟火、垃圾桶、陌生人、设备状态。
- 导航带路包：点位导航、路线规划、访客跟随、返回巡检。
- 现场事件包：异常识别、证据归档、通知响应组。
- 知识问答包：RAG、证据展示、冲突/过期拒答。
- 客户项目包：对象目录、模板、导入导出、验收用例。

每个技能包应具备：

- 客户可见说明。
- 输入输出定义。
- 风险等级。
- 启停状态。
- 所属客户项目。
- 审批记录。
- 调用审计。
- 测试用例。

## 6. 管理平台页面

AskMe 管理平台不应把所有功能揉在一个页面。推荐按产品任务拆分为多个页面。

| 页面 | 面向对象 | 主要用途 |
| --- | --- | --- |
| 产品总览 | 客户负责人、销售、交付 | 看版本能力、运行状态、今日事件、交付进度 |
| 对话中心 | 运营、测试、客户演示 | 语音/文本提问、查看证据、确认任务、观察状态 |
| 现场事件 | 安保、保洁、现场主管 | 查看异常事件、通知、处理状态、证据和报告 |
| 空间认知 | 运营、交付 | 管理点位、别名、服务点、路线、带路记录 |
| 知识库 | 运营、客户管理员 | 上传、审批、发布、冲突处理、过期维护 |
| 客户项目 | 交付、项目经理 | 管理客户、现场、对象目录、行业模板、资源绑定 |
| 能力中心 | 产品、交付、研发 | 管理技能包、风险、启停审批和调用审计 |
| 语音配置 | 测试、交付 | 选择音色、测试麦克风/扬声器、查看延迟 |
| 交付验收 | 交付、客户负责人 | 查看部署检查、验收用例、未完成项 |
| 审计报告 | 客户、主管、交付 | 导出事件、任务、知识、权限和通知记录 |

### 6.1 客户视角接口原则

这里的“接口”不只指 HTTP API，也包括客户看到的每个页面、按钮、语音提示、错误提示和导出的证据包。每个接口都必须让客户能回答四个问题：

1. 我现在看到的是什么现场对象或业务场景？
2. 我可以做什么，做了以后会不会触发机器人动作？
3. 系统为什么这么判断，依据和证据在哪里？
4. 出错、证据不足或权限不够时，下一步该找谁或补什么？

| 接口面 | 客户第一反应 | 产品必须给出的答案 | 风险兜底 |
| --- | --- | --- | --- |
| 对话入口 | 我现在能不能说话，说完会发生什么 | 语音状态、识别文本、回答依据、任务确认状态 | 听不清先澄清；高风险任务必须确认 |
| 现场事件 | 这件事严重吗，谁会处理 | 事件类型、地点、证据、通知对象、处理状态 | 高风险事件不能无审批关闭 |
| 空间问路 | 机器人是否真的知道我要去哪 | 候选点位、别名命中、路线说明、是否可带路 | 点位不存在或路线不可通行时不编造 |
| 知识库 | 这句话依据哪里，能不能对客户说 | 资料来源、版本、质量状态、可见范围、审批状态 | 仅内部、待复核、过期、冲突资料不进入回答 |
| 客户项目 | 这是哪个客户、哪个现场、哪些对象 | 客户、项目、对象目录、资源绑定、验收用例 | 跨客户数据隔离，复制项目带边界和缺口 |
| 能力中心 | 机器人到底会做什么，还缺什么 | 能力包、场景技能、启停状态、风险等级 | 新技能先进入审批，不直接执行 |
| 语音音色 | 客户和游客听到的声音是否合适 | 场景音色、播报策略、延迟、打断状态 | 紧急告警优先，夜间低扰可配置 |
| 交付检查 | 现在能否演示、试点或上线 | 门禁状态、阻塞项、依赖、下一步 | 不把演示能力说成生产上线 |
| 审计证据 | 发生过什么，能否验收追责 | 事件、任务、知识、权限、通知和导出哈希 | 待复核证据不能进入客户验收包 |
| HTTP API | 集成方调用失败时怎么处理 | 稳定字段、错误码、公开错误、下一步建议 | 内部异常不泄露，外部响应可复现 |

## 7. 标准业务流程

### 7.1 游客问路

1. 游客在服务点停留。
2. 机器人判断是否可以主动问询。
3. 机器人播报：“你好，请问需要指路吗？”
4. 游客说出目的地。
5. AskMe 在点位和别名库中检索。
6. 找到唯一目标后进行确认。
7. 简单路线走语音指路，复杂路线可进入带路。
8. 系统记录服务点、目的地、确认结果、是否完成。

异常处理：

- 听不清：要求游客再说一遍。
- 目标模糊：列出候选项让游客确认。
- 目标不存在：说明暂未收录，不编造。
- 路线不可通行：只提供语音建议或转人工。
- 当前有紧急事件：暂停问路，优先处理安全事件。

### 7.2 管理员发起突发巡检

1. 管理员在对话中心说“去 A 区巡检一下”。
2. 系统识别为任务请求。
3. 展示任务目标、位置、预计动作和风险提示。
4. 管理员确认。
5. 系统进行安全预检。
6. 运行调度进入演示、仿真、实验室或现场模式。
7. Dashboard 展示进度和结果。
8. 完成后生成任务记录和报告。

异常处理：

- 权限不足：拒绝执行。
- 点位不存在：要求补充或选择点位。
- 机器人不可用：进入排队或失败。
- 现场风险高：要求主管确认。
- 用户取消：停止任务并记录原因。

### 7.3 现场异常处置

1. 视觉/传感器/机器人状态上报异常。
2. AskMe 标准化事件类型和地点。
3. 系统判断响应组和风险级别。
4. 播放现场提示语。
5. 通知钉钉群或响应人。
6. 保存图片、传感器值、机器人状态和通知结果。
7. 现场人员确认或处理。
8. 高风险事件需要主管审批关闭。

建议固定话术：

- 机器人摔倒：“机器人发生跌倒，正在停止运动，请现场人员协助处理。”
- 卡住无法运动：“机器人检测到运动受阻，已暂停任务，请注意避让。”
- 人为挡路：“请不要阻挡机器人通行，现场已记录。”
- 电机故障：“机器人检测到关节异常，已停止相关动作并通知维护人员。”
- 烟火异常：“检测到疑似烟雾或火情，请现场人员立即确认。”
- 垃圾桶满溢：“检测到垃圾桶可能已满，已通知保洁处理。”

固定话术用于高风险场景。大模型可以用于生成更自然的说明，但不能改变风险结论、地点、责任人和处理动作。

### 7.4 知识运营

1. 运营上传客户知识。
2. 系统解析并生成预览。
3. 管理员审批。
4. 发布后进入检索。
5. 用户提问时返回答案和证据。
6. 知识过期、冲突或删除后不再作为正式依据。
7. 系统保留知识版本和操作记录。

### 7.5 新客户项目交付

1. 选择行业模板。
2. 创建客户项目和现场。
3. 导入或录入对象目录。
4. 绑定视觉模型、传感器协议、技能包和通知组。
5. 维护点位、别名、服务点和路线。
6. 导入客户知识并审批。
7. 运行验收用例。
8. 导出交付包和验收报告。

## 8. 数据与配置

AskMe 的产品数据建议分为以下几类：

- 客户项目：客户、项目、现场、行业、交付命名空间。
- 管理对象：楼宇、设备、点位、路线、区域、传感器、服务点。
- 知识数据：资料、版本、状态、证据、过期时间、冲突信息。
- 事件数据：类型、地点、风险、证据、通知、处理人、关闭记录。
- 任务数据：任务目标、确认状态、安全预检、运行状态、报告。
- 技能数据：技能包、输入输出、风险、启停、审批、调用记录。
- 审计数据：谁在什么时候做了什么，影响哪个客户项目和对象。

最低产品要求：

- 每条事件必须有时间、地点、类型、状态和来源。
- 每次高风险操作必须有操作人。
- 每个客户项目必须有独立命名空间。
- 每个可执行任务必须经过确认和安全预检。
- 每个知识回答应能追溯证据来源。

## 9. 部署与验收

### 9.1 部署准备

交付前需要准备：

- 客户项目基本信息。
- 现场地图或点位资料。
- 首批服务点和路线。
- 首批巡检对象和异常场景。
- 语音设备、摄像头、传感器和机器人连接方式。
- 钉钉或其他通知渠道。
- 客户知识资料和审批人。
- 验收用例和演示脚本。

### 9.2 试点验收标准

| 验收项 | 通过标准 |
| --- | --- |
| 语音交互 | 能清楚显示听、想、说、等待确认等状态 |
| 游客问路 | 首批点位和别名范围内可正确识别并回答 |
| 带路服务 | 首批演示路线可完成低速引导或给出不可带路原因 |
| 事件处置 | 指定异常能生成事件、通知、播报和归档 |
| 知识回答 | 回答展示证据，过期/冲突/无依据时不编造 |
| 任务确认 | 管理员任务必须确认后才进入运行 |
| 安全预检 | 权限、点位、技能、运行模式不满足时拒绝 |
| 审计报告 | 能查询和导出事件、任务、知识和操作记录 |
| 客户项目 | 能通过模板创建项目，并导入/导出交付包 |

### 9.3 生产上线准入

生产上线前必须进一步确认：

- 企业账号和 SSO/IAM 是否接入。
- RBAC 是否覆盖所有高风险操作。
- 真实机器人控制链路是否通过现场测试。
- 真实摄像头、传感器和模型是否稳定。
- 钉钉或其他通知渠道是否完成真实发送验证。
- 客户现场安全责任边界是否签署。
- 验收用例是否由客户、交付和研发共同确认。

## 10. 权限、安全与审计

AskMe 的权限策略应按客户项目和风险等级设计。

关键规则：

- 普通游客只能问路和咨询，不得触发机器人任务。
- 普通运营可查看记录和处理低风险事件。
- 管理员可发起巡检、维护知识和点位。
- 主管可审批高风险关闭、启停高风险技能。
- 交付工程师可配置客户项目、对象目录和验收用例。
- 研发调试权限不得默认开放给客户现场账号。

高风险操作必须记录：

- 操作人。
- 客户项目。
- 目标对象。
- 操作前状态。
- 操作后状态。
- 审批人或拒绝原因。
- 时间戳和审计链路。

## 11. 当前版本能力状态

已具备产品化基础的能力：

- 多页面 Dashboard 的产品框架。
- 语音/文本交互入口。
- 交互准入门设计与感知快照接口。
- 现场事件场景、通知、归档和报告链路。
- 知识库上传、审批、发布、检索、证据和冲突/过期治理基础。
- 客户项目、行业模板、管理对象、导入导出和实施交接基础。
- 能力中心和技能包产品结构。
- 任务确认、安全预检和运行调度的基础闭环。
- 审计与交付验收入口。

仍需继续补齐的生产级能力：

- 企业级账号、SSO、租户隔离和完整 RBAC。
- 真实机器人运行时的现场联调。
- 真实视觉模型、传感器协议和地图导航的批量适配。
- 更完整的技能包市场和在线增长机制。
- 知识版本差异、回滚、发布日历和过期提醒。
- 更成熟的语音端到端延迟优化和设备选择体验。
- 客户项目规模化交付工具链和安装包。

## 12. 客户沟通口径

推荐说法：

- “AskMe 是机器人现场任务与智能交互平台。”
- “它能把语音、知识、现场事件、空间地图和机器人任务连接起来。”
- “当前版本适合做客户试点和场景共创，生产上线需要按现场硬件和安全要求完成验收。”
- “系统回答会尽量基于已审批知识，并展示证据，避免乱答。”
- “游客问路和管理员任务是两条不同链路，不会因为游客随便一句话就触发机器人执行。”

避免说法：

- “机器人可以随便聊天，什么都能答。”
- “大模型可以直接控制机器人。”
- “不用现场配置，任何园区拿来就能跑。”
- “现在已经可以完全无人值守生产上线。”
- “视觉、语音、导航在所有现场都能百分百准确。”

## 13. 常见问题

### Q1：AskMe 和机器人底盘系统是什么关系？

AskMe 不替代机器人底盘系统。底盘系统负责运动控制、避障、导航和硬件状态。AskMe 负责自然语言交互、任务理解、安全预检、事件管理、知识回答、客户项目配置和审计闭环。

### Q2：游客说话会不会误触发机器人任务？

不应该。游客问路走交互准入门和空间问询链路。需要机器人执行的任务必须经过权限、确认和安全预检。

### Q3：没有真实机器人时能不能演示？

可以。演示和仿真模式可以展示任务确认、事件流转、知识回答、通知、审计和 Dashboard 效果。但真实运动能力必须等机器人和现场设备接入后验收。

### Q4：知识库能不能保证不乱答？

产品策略是尽量让回答来自已审批知识，并展示证据。过期、冲突、未审批或无依据时应拒答或要求确认。要达到生产级可靠性，还需要客户知识运营流程配合。

### Q5：不同客户项目怎么复用？

通过行业模板、客户项目、对象目录、技能包、知识库、通知组和验收用例实现复用。交付团队可以基于模板创建新项目，再导入客户点位、设备、路线和知识。

## 14. 版本路线

近期优先级：

1. 把客户项目、对象目录、模板市场和导入导出做成稳定交付工具。
2. 把知识库继续产品化：已补齐资料类型预览、质量状态、可见范围、客户/项目关联、版本差异和回滚；下一步补发布日历、定时过期提醒和更友好的非结构化文件转换。
3. 把现场事件处置接入更多真实设备和模型，包括摄像头、烟感、温度、电机故障、地图区域规则。
4. 把语音交互体验打磨到客户能直观看懂状态、能顺畅对话、能选择音色。
5. 把园区空间认知做成可配置产品，包括点位、别名、服务点、路线和带路验收。
6. 把能力中心升级为技能包市场，支持客户项目级启停、审批和调用审计。
7. 引入企业身份系统和租户隔离，为生产部署做准备。

中期方向：

- 多机器人调度。
- 移动端/远程指挥入口。
- 更完整的多模态感知融合。
- 现场数字孪生与地图可视化。
- 行业模板市场。
- 客户成功数据复盘。

---

# askme Product Brief

## Solution Provider Customer Project Layer

AskMe is now treated as a repeatable solution product, not a one-off park demo.
The delivery source of truth is still a validated site profile, but the profile
now has a customer/project/object boundary:

- `customer`: customer id, customer name, industry, project id, and delivery model.
- `site`: site id, site name, map version, zones, devices, thresholds, and responder groups.
- `managed_objects`: the customer's real managed objects, such as vehicles, visitors,
  equipment, shelves, trash bins, smoke/fire risk areas, crowd zones, or custom assets.

Current implementation:

- Industry templates live under `deploy/customer-project-templates`. Each
  template is a versioned delivery package with version, publish status,
  release channel, owner, upgrade policy, and minimum runtime metadata.
- Customer project instances live under
  `deploy/site-profiles/{tenant_id}/{delivery_namespace}/{customer_id}/{project_id}.yaml`
  when a non-default delivery scope is configured. Legacy/default projects still
  resolve from `{customer_id}/{project_id}.yaml`.
- `GET /api/field/customer-project-templates` lists reusable factory, park, warehouse,
  and scenic-area starter templates.
- `GET /api/field/customer-project-templates/{template_id}/history` lists release
  governance revisions for one reusable template.
- `GET /api/field/customer-project-template-release-requests` lists pending and
  reviewed reusable-template release requests.
- `GET /api/field/customer-project-template-release-notes` lists approved,
  published template releases that can appear in customer-facing sales and
  delivery materials.
- `POST /api/field/customer-project-template-release-notes/export` returns a
  portable JSON + printable HTML proposal bundle for those approved release
  notes, with optional customer/project context and a controlled
  `proposal_insert` section for sales/solution material.
- `POST /api/field/customer-project-templates/{template_id}/release-requests`
  creates a pending release request. A published release must be requested here
  before it can be applied.
- `POST /api/field/customer-project-template-release-requests/{request_id}/review`
  approves or rejects a pending release request. Self-approval is rejected.
- `POST /api/field/customer-project-templates/{template_id}/release` demotes,
  deprecates, or blocks a reusable template package with a required operator and
  reason. Direct `published` writes are rejected; they must use the release
  request and second-approver review path. Any applied write stores a JSON
  revision snapshot before changing the YAML template.
- `GET /api/field/customer-project-acceptance-registry` lists every managed-object
  acceptance reference used by customer projects and industry templates, with linked,
  manual-review, and blocked counts.
- `GET /api/field/customer-project-resource-catalog` lists every product resource
  referenced by managed objects across customer projects and industry templates:
  vision models, sensor protocols, skill packages, and acceptance tests. The
  registry reports registered versus unregistered bindings, consumers, project
  usage, and template usage so delivery can catch "string-only" bindings before
  customer signoff.
- `GET /api/field/delivery-resource-registry` lists the shared solution-provider
  resource registry under `deploy/delivery-resources`. These resources are
  reusable across customer projects when their tenant/project scope allows it.
- `POST /api/field/delivery-resource-registry` registers or updates one shared
  vision model, sensor protocol, skill package, or acceptance test with owner,
  version, scope, update reason, and an automatic revision snapshot before
  overwrite.
- `GET /api/field/delivery-resource-registry/history` lists registry revisions
  for audit review before rollback.
- `GET/POST /api/field/delivery-resource-governance-requests` exposes the
  shared-resource governance queue. Disable and rollback requests are created
  as pending records with preview impact, requester, reason, and current
  registry hash before any shared resource changes. Disable requests include
  impact analysis from the customer-project resource catalog: affected projects,
  managed objects, templates, consumer count, generated time, and a short
  reviewer message.
- `POST /api/field/delivery-resource-governance-requests/{request_id}/review`
  approves or rejects a pending shared-resource change. The requester cannot
  approve their own request; approval applies the disable or rollback operation,
  rejection leaves the registry unchanged.
- `POST /api/field/delivery-resource-registry/{resource_type}/{resource_id}/disable`
  remains as an approval-gated administrative path for disabling one shared
  resource. The normal Dashboard product flow now creates a governance request
  first. Managed-object readiness blocks bindings that still reference a
  disabled or blocked model, protocol, skill package, or acceptance test.
- `POST /api/field/delivery-resource-registry/rollback` supports dry-run and
  approval-gated rollback to a previous registry revision. Rollback requires an
  unrestricted delivery approver because it can affect multiple customer
  projects.
- `/dashboard/projects` includes a delivery resource registry form that writes
  the shared registry instead of editing individual project YAML. Delivery
  users can register reusable vision models, sensor protocols, skill packages,
  and acceptance tests, optionally scope them to a project, then bind those
  resource IDs from the managed-object editor.
- The same resource section now includes a governance panel for history,
  rollback dry-run, disable requests, rollback requests, and the approval queue.
  The queue shows the impact summary so the approver can see how many projects,
  objects, templates, and consumers would be affected before deciding. Customer
  object readiness reflects resource publish status instead of treating all
  registered IDs as safe.
- Delivery resource governance requests now carry an explicit review SLA:
  `sla_target_s`, `due_at`, `review_sla.state`, remaining/overdue seconds, and
  escalation policy. `/dashboard/projects` can show active, due-soon, and
  overdue requests, and `/api/field/delivery-resource-governance-requests`
  supports `overdue_only=true` so delivery owners can operate the queue instead
  of treating it as a passive audit log.
- Overdue delivery resource governance requests can now be escalated through
  `POST /api/field/delivery-resource-governance-requests/escalate-overdue`.
  The escalation is recorded on each request with escalation id, operator,
  overdue duration, delivery owner payload, delivery mode, sent channels, and
  per-channel delivery report. Local/demo deployments keep the escalation in a
  delivery-owner queue. Production deployments can enable
  `field_operations.delivery_resource_governance.delivery_owner_notifications`
  and route overdue reviews to webhook, DingTalk, WeCom, Feishu, or log
  channels through the shared `AlertDispatcher`.
- Project-level `delivery_resources` remains supported as an override for
  customer-specific assets, but the normal product path is now shared registry
  first, project override second.
- `deploy/delivery-resources/resources.yaml` seeds the shared registry for
  solution-provider delivery. It covers the default vision models, sensor
  protocols, skill packages, and customer-facing acceptance test references used
  by the demo project plus the factory, park, warehouse, and scenic-area
  templates. This lets delivery teams start from registered product resources
  instead of relying on Python built-ins.
- The managed-object quick editor includes a registered-resource picker. It
  appends a selected resource ID to the matching binding field, so delivery
  users can bind known models, protocols, skill packages, and acceptance tests
  without copying IDs from the resource catalog by hand.
- The resource card also renders a resource binding action plan. If bindings
  are still string-only references, it lists the missing registrations; when the
  visible project scope is clean, it tells delivery they can continue with
  package export, onsite evidence, and acceptance review.
- `POST /api/field/customer-projects/from-template` creates a new customer project profile.
- `GET /api/field/customer-projects` lists customer/project/site/object coverage.
- `GET /api/field/customer-project-workbench` returns one solution-provider
  workbench payload with delivery readiness, customer project catalog, industry
  template market, managed-object directory, delivery resources, and package
  delivery-gate surfaces.
- `GET /api/field/customer-projects` also returns a catalog-level
  `delivery_acceptance_gate` and per-project `product_acceptance_gate`. These
  gates summarize customer scope, site profile validity, managed-object catalog,
  vision/sensor/skill/acceptance bindings, object-change audit policy, and
  handoff artifacts so a solution-provider delivery lead can tell whether a
  customer project is blocked, needs manual review, or can move toward signoff
  without reading raw YAML.
- The same catalog endpoint supports delivery-directory filters:
  `tenant_id`, `delivery_namespace`, `customer_id`, `project_id`, `site_id`,
  `industry`, `gate_status`, and `deployment_stage`. Filtered responses
  recompute customers, project counts, managed-object counts, and the aggregate
  acceptance gate for the visible project set.
- `GET /api/field/solution-delivery-readiness` returns one product-facing
  delivery gate that rolls up customer project acceptance, template market
  readiness, resource binding readiness, and shared-resource governance queue
  state. Dashboard renders it as "客户交付总门禁" so delivery and product leads
  can see the current customer-facing claim boundary before drilling into
  separate project, template, resource, or audit panels.
- `GET /api/field/product-launch-readiness` returns one customer-facing launch
  decision across enterprise identity, field runtime readiness, solution
  delivery readiness, and the customer-project workbench. Dashboard renders it
  as "客户上线准入总览" on `/dashboard/delivery`, so sales, delivery, and the
  customer can see whether the product is still demo/integration only, ready for
  pilot/site trial, or ready for production acceptance. Demo operator identity
  blocks only the production claim; it does not hide trial/demo capability.
- `GET /api/field/customer-projects/{identifier}` returns one project profile and object catalog.
- `GET /api/field/customer-projects/{identifier}/execution-bindings` returns a
  customer-project execution plan for every managed object. The plan now binds
  each object to concrete ingest inputs, matched devices, protocol adapters,
  vision model references, skill package routes, acceptance tests, and runtime
  callback boundaries. Skill package routes expose the normalized capability
  name, installed contract status, safety level, approval policy, required
  inputs, output contract, tool route, and hardware boundary so delivery can
  audit whether an object is actually executable or still only configured.
  Adapter contracts also expose the field-ingest bridge, normalizer module,
  supported JSON/JSONL formats, dry-run command, live-post command, signing
  command, device secret env vars, sample fixture, and verification outputs.
  This makes the handoff usable by a delivery engineer without reading Python
  internals.
- Managed objects can now carry optional `tenant_ids`, `delivery_namespaces`,
  `customer_ids`, `project_ids`, and `site_ids`. Field ingest will not bind a
  device event to an object whose scope constraints do not match the active
  customer project, even if the device payload claims that object id. This is
  required for solution-provider deployments where templates and capability
  packages are reused across many customer sites.
- `/dashboard/projects` includes a customer-readable Managed Object Directory
  summary for solution delivery teams: total objects, deliverable objects,
  manual-check objects, blocked objects, acceptance-test count, scoped-object
  count, per-object resource/acceptance checks, and JSON/CSV exports for all or
  deliverable objects. This turns per-customer objects from hidden YAML into an
  auditable delivery surface.
- `GET /api/field/customer-projects/managed-object-directory` exposes the same
  directory as a scoped API for delivery tooling. It returns tenant, namespace,
  customer, project, site, object, resource binding, acceptance check, and
  derived delivery status fields, and it reuses the customer-project operator
  scope filters.
- The managed-object directory now returns an `action_plan` per object. The
  plan turns missing resources, unregistered catalog entries, blocked resource
  versions, missing acceptance requirements, and unsafe acceptance references
  into concrete owner/next-step items for delivery and QA.
- `POST /api/field/customer-projects/{identifier}/execution-bindings/{object_id}/rehearsal`
  runs a lab-only object接入演练. In `dry_run` mode it merges the object sample
  payload with project scope, matched device ids, and the selected adapter
  contract, then returns the normalized `/api/field/ingest` payload without
  creating a field event. In `shadow_post` mode it requires explicit
  confirmation before sending the rehearsal payload into the ingest service.
  Every response states `production_claim_allowed=false`: this proves adapter
  parsing and object binding only, not customer signoff or unattended production
  go-live.
- Rehearsal evidence now has an explicit product boundary. `dry_run` can never
  register onsite acceptance evidence. A confirmed `shadow_post` may register a
  `device_ingest` receipt only as `evidence_tier=acceptance_candidate` with
  `production_eligible=false`; it remains `manual_check` unless trusted device
  signature and runtime completion evidence are present. Dashboard shows this
  as an acceptance candidate, not production go-live proof.
- `GET /api/field/customer-projects/{identifier}/acceptance-report` returns a customer-readable
  delivery gate report with site profile, managed-object evidence, credential, and onsite
  acceptance boundaries. The report now also embeds a compact field-readiness snapshot
  from scenario evaluation, ingest smoke, voice smoke, notification smoke, runtime
  roundtrip, audit, and device-trust evidence.
- The acceptance report also embeds `execution_bindings` and a
  `managed_object_execution_bindings` gate, so customer signoff cannot ignore
  whether managed objects have executable ingest plans and audited skill
  boundaries. The report includes compact object-level adapter contracts so a
  reviewer can see which objects still need live bridge evidence or trusted
  device signatures before customer signoff.
- The acceptance report auto-surfaces read-only onsite evidence receipts from
  real-link readiness outputs only when the source proves a non-local deployment
  path: trusted device events plus real hardware flags for ingest, live TTS and
  non-local voice smoke for playback, external services plus non-local smoke for
  DingTalk delivery, and trusted non-local runtime callbacks with a verified
  final status. Local-server, mock, recorded, or missing evidence remains a
  manual-check gap and cannot satisfy customer signoff.
- Acceptance reports and dossiers now include a `site_acceptance_checklist` that
  turns the same gates into customer-delivery work items: site profile, managed
  object bindings, deployment credentials, device ingest, live voice, external
  notifications, runtime roundtrip, and audit/operator review. Dashboard renders
  this as a customer-site acceptance checklist instead of asking a customer to
  interpret raw readiness gates.
- Acceptance closure treats the same checklist as its own gate, and acceptance
  dossier manifests expose checklist status plus ready/manual/blocked counts for
  proposal and signoff review.
- `POST /api/field/customer-projects/{identifier}/managed-objects/{object_id}` upserts one managed object.
- `DELETE /api/field/customer-projects/{identifier}/managed-objects/{object_id}` removes one managed object.
- `GET /api/field/customer-projects/{identifier}/execution-bindings` turns
  managed-object bindings into a customer-project execution plan. For every
  object it shows which camera/sensor/robot sources are registered, which ingest
  adapter/protocol should be used, which vision models and skill packages are
  bound, which acceptance tests cover the object, the `/api/field/ingest` sample
  payload, the field-ingest bridge dry-run/live/signing commands, and the runtime
  callback boundary. This is the product bridge between "objects have resources"
  and "objects can drive a real field event."
- `GET /api/field/customer-projects/{identifier}/export` exports a reusable customer project package.
- `POST /api/field/customer-projects/package/verify` verifies a package manifest without writing.
- `POST /api/field/customer-projects/package/diff` previews import changes without writing.
- `POST /api/field/customer-projects/import` imports a previously exported package.
- `GET /api/field/customer-projects/{identifier}/history` lists saved customer
  project revisions.
- `POST /api/field/customer-projects/{identifier}/rollback` restores a customer
  project from a saved revision, with dry-run support before writing.
- `GET /api/field/customer-projects/{identifier}/acceptance-dossier` exports a
  customer handoff JSON dossier and a printable HTML dossier with the acceptance
  report, field-readiness evidence inventory, per-file SHA-256 hashes, and optional
  HMAC manifest signature. The dossier also carries project-level
  `launch_readiness` so the customer can see whether the project is demo only,
  pilot/site-trial ready, or ready for controlled production acceptance.
- `GET /api/field/customer-projects/{identifier}/proposal-bundle` exports a
  customer-facing proposal bundle that binds the customer project package,
  acceptance dossier, approved template release notes, controlled sales claims,
  launch-readiness boundary, and delivery boundaries into JSON + printable HTML.
- `POST /api/field/customer-projects/proposal-bundle/verify` verifies that a
  customer proposal bundle still matches its manifest, tenant, delivery namespace,
  handoff package, acceptance dossier, approved template release notes, and
  controlled delivery boundary before a customer or delivery team relies on it.
- `POST /api/field/customer-projects/acceptance-dossier/verify` verifies a
  pasted/exported acceptance dossier before customer signoff. It recomputes the
  dossier payload hash, enforces operator project scope, and rejects tampered
  handoff material without writing anything.
- `GET/POST /api/field/customer-projects/{identifier}/onsite-evidence` lets a
  delivery lead register real onsite evidence receipts for device ingest, voice
  playback, external notification delivery, runtime roundtrip callbacks, and
  customer review. These receipts are revisioned in the customer profile, surfaced
  in the acceptance report, and carried into the acceptance dossier manifest and
  evidence inventory. Unsupported evidence types/statuses are rejected before any
  profile write; the latest failed required receipt blocks acceptance until a
  later passing receipt is registered.
- Auto-surfaced readiness receipts are intentionally read-only and deterministic:
  they appear in the report and exported dossier, but they do not mutate the
  customer project YAML or replace the manual onsite evidence registry.
- The Dashboard receipt rows label the source as system auto-surfaced readiness
  evidence versus manually registered delivery evidence, and show the evidence
  type, SHA-256 prefix, and external reference/path for customer review.
- `GET /api/field/customer-projects/{identifier}/onsite-evidence` uses the same
  evidence view by default, so the lifecycle panel and acceptance report do not
  disagree. Callers can set `include_readiness_auto=false` when they need the
  raw manual registry only.
- `GET /api/field/customer-projects/{identifier}/acceptance-closure` and
  `POST /api/field/customer-projects/{identifier}/acceptance-review` turn those
  receipts into a project-level closure state: acceptance report, onsite evidence,
  manual delivery-owner review, acceptance dossier verification, proposal bundle
  verification, scoped audit-export verification, timeline, customer claim,
  blockers, and next step.
- `GET /api/field/customer-projects/{identifier}/customer-signoff` and
  `POST /api/field/customer-projects/{identifier}/customer-signoff` add the
  customer-side signoff loop after internal delivery readiness. A customer
  signoff records the decision, signatory, organization, risk acknowledgement,
  evidence references, credential reference, credential SHA-256, acceptance gate
  snapshot, handoff-material snapshot, operator, timestamp, profile revision,
  and the signoff record SHA-256. Accepted signoff is blocked until the
  acceptance-closure gates are ready for customer review and the credential
  reference/hash are present, while `needs_fix` and `rejected` can be recorded
  to preserve customer feedback even before final readiness.
- Acceptance closure now includes a dedicated `customer_signoff` gate. The
  project can only claim customer acceptance after an accepted signoff with risk
  acknowledgement is archived; otherwise it stays in internal manual check or
  ready-for-customer-signoff state.
- Acceptance dossiers and their manifests now carry customer signoff history,
  the latest signoff decision, signoff count, signoff payload hash, credential
  hash, and integrity result so exported handoff packages can show whether the
  project is only internally ready or actually accepted by the customer.
- `/dashboard/projects` is the product-facing console for customer projects, managed
  objects, import dry-run, export feedback, and project-scoped event filtering.
- `/dashboard/projects` now exposes a customer-project workspace navigation so
  delivery users can jump between project catalog, template market, object
  directory, import/export, acceptance evidence, resource bindings, event scope,
  template release governance, and multi-site rollout instead of reading one long
  mixed control panel.
- `/dashboard/projects` now adds a customer-readable solution-provider workbench
  strip above the technical panels. It explains the five delivery surfaces:
  customer project catalog, industry template market, managed-object directory,
  delivery resources, and package delivery gate, with status badges from the
  same backend readiness payloads.
- The same page now consumes `GET /api/field/customer-project-workbench` as a
  golden-path strip: industry template -> customer project catalog -> managed
  object directory -> delivery resources -> package delivery gate. This gives
  customers and delivery leads one readable route through the product instead of
  forcing them to infer readiness from separate API panels.
- `/dashboard/projects` now includes a template release governance board with
  pending, approved, and rejected counts plus a central review queue. Product
  owners can see which reusable templates are waiting for a second approver
  before they appear in customer-facing release notes.
- The customer-project import area separates project handoff packages, customer
  proposal bundles, and acceptance dossiers into distinct verification inputs.
  Proposal and dossier exports now refill their own verifier text boxes instead
  of overwriting the project import payload.
- The managed-object editor in `/dashboard/projects` now edits the full customer
  object delivery contract: object labels, scenario scope, zone/device source,
  responder group, required evidence, vision models, sensor protocols, skill
  packages, and acceptance-test references. It no longer silently leaves vision
  models empty or guesses sensor protocols from device-source strings.
- The acceptance review form now lets a delivery lead attach evidence references
  to the review decision, so customer signoff can point to onsite receipts,
  acceptance reports, dossiers, or audit/export records instead of submitting an
  empty review trail.
- The acceptance review form can now load onsite evidence receipts into a picker
  and append selected receipt references into the review evidence list. This
  keeps customer signoff tied to visible delivery evidence instead of forcing
  the delivery lead to copy receipt IDs by hand.
- Customer project read APIs require `field:project:read`; write/import/archive
  APIs require `field:project:write`.
- Template release APIs require product-owner permissions:
  `template:release:write` for pilot/deprecated/blocked changes and
  release-request creation; `template:release:approve` for reviewing requests.
- Operator identities can carry `customer_ids`, `project_ids`, and `site_ids`
  under `project_scope`; customer project and site catalogs are filtered by this
  scope, and out-of-scope project detail/export requests return 403.
- Operator identities can also carry `tenant_ids` and `delivery_namespaces`.
  These claims are evaluated with customer/project/site claims, so an operator
  assigned to one customer delivery namespace cannot read, verify, import, or
  mutate a same-named project in another namespace.
- Template catalogs also apply the operator's tenant and delivery-namespace
  scope. Shared default templates remain visible as common product starters,
  while future tenant-specific templates can be hidden from unrelated delivery
  spaces.
- Customer project write APIs also enforce `project_scope`: creating from a
  template, upserting a profile, importing a package, archiving a project, and
  changing managed objects are rejected when the operator is not assigned to the
  target customer, project, or site.
- Customer project profiles and handoff packages now carry `tenant_id` and
  `delivery_namespace`. Package import/diff matches an existing profile only
  when tenant, delivery namespace, customer, and project/site identity match;
  same-name projects in another delivery namespace are reported as collision
  candidates instead of being overwritten.
- Customer proposal bundles now carry the same delivery scope in their manifest.
  Dashboard verification rejects manifest scope tamper and delivery-boundary
  changes, so proposal material can be checked before sending or importing.
- Customer acceptance dossier verification now rejects payload-hash tamper,
  missing HMAC verification secrets when a signature is present, bad signatures,
  and out-of-scope operator attempts.
- Each managed object now exposes `acceptance_status` for required vision model,
  sensor protocol, skill package, and acceptance test bindings.
- Each managed object also exposes `resource_binding_status`, which checks those
  bindings against the delivery resource catalog. Built-in product resources and
  profile-level `delivery_resources` are treated as registered; unknown model,
  protocol, or skill references become manual-check catalog gaps instead of
  silently looking ready in the project UI.
- `acceptance_status.acceptance_checks` resolves local acceptance references to
  repository evidence: missing files block acceptance, unresolved nodes require
  manual review, and configured scenario aliases link to the deterministic
  scenario/pytest evidence already in the repository.
- The acceptance registry and resource catalog make those references inspectable
  across projects and templates, so delivery can see which customer objects still
  depend on missing resources or manual-review evidence before signoff.
- Reusable customer project packages carry `package_schema`,
  `reuse_assessment`, `deployment_dependencies`, `resource_catalog_summary`,
  `binding_readiness_summary`, `managed_object_action_plan`, and
  `package_delivery_gate`. The package manifest mirrors the resource binding
  status, action-plan counts, and delivery-gate state, so package diff/verify
  can reject tampered handoff metadata and Dashboard can show incoming/current
  resource gaps before import, including the object/resource IDs for
  unregistered bindings. This turns export/import into a customer-project
  handoff check: ready packages can seed another site, manual-check packages
  require onsite credential/evidence review, and blocked packages cannot be
  imported until the object bindings or acceptance evidence are fixed.
- Customer project writes now create local revision snapshots before overwriting
  the current profile. The project console exposes revision history and rollback
  dry-run/apply actions, so delivery teams can recover from mistaken onsite
  project edits without manually searching YAML backups.
- Industry template release writes now use a governed pattern: templates can be
  marked pilot, deprecated, or blocked directly by a product owner, while
  published promotion requires a release request and second product-owner
  approval. Every applied change keeps the previous package metadata in
  `_template_revisions`; every proposal is stored under
  `_template_release_requests`.

Product boundary:

- This is configuration-level tenant and delivery separation, not full SaaS tenant isolation.
  It prevents package/profile overwrite across delivery namespaces, but it does
  not replace enterprise identity, database-level row isolation, or customer KMS.
- Templates generate project profiles; templates do not directly drive robot runtime.
- Managed object bindings are declarative references to vision models, sensor protocols,
  skill packages, and acceptance tests. Runtime execution still goes through the existing
  scenario, safety, and runtime arbiter controls.
- Acceptance checks prove that a delivery profile points to local test/scenario
  evidence; they do not prove that a physical camera, sensor, notification robot,
  or robot runtime passed onsite acceptance.
- This still does not replace enterprise SSO/IAM tenant isolation; production
  deployments must bind operator identities and project scopes through the
  customer identity gateway.
- `GET /api/governance/identity-readiness` exposes that production boundary as a
  customer-facing gate. It reports whether the deployment is still using the
  demo operator directory, whether trusted IAM/SSO gateway headers are enabled,
  which operator/role/scope claims are configured, and whether the current
  release can claim enterprise identity readiness. The Dashboard overview shows
  the same gate as "企业身份准入" so delivery cannot accidentally present a demo
  operator directory as production tenant isolation.

Next product step:

- Bind real customer IAM/SSO gateways in deployment and capture the gateway
  verification evidence for operator id, roles, tenant/project/site scopes, and
  high-risk action approval.
- Add PDF rendering and proposal-template insertion for release-note bundles so
  sales and delivery can attach approved template packages to formal proposals.
- Bind field ingest events back to `managed_object_id` by source, zone, label, and scenario.
- Add browser/PDF export from the printable acceptance dossier.
- Replace local/lab smoke evidence with live onsite smoke evidence for camera, sensor,
  DingTalk, voice, and robot runtime.

2026-05-14 implementation update:

- Field events now inherit `customer_id`, `project_id`, `site_id`, `site_name`, and `industry` from the active site profile.
- Field events now resolve `managed_object_id` from explicit payload data first, then from scenario, detection labels, zone type, and source.
- Event list filtering accepts project and managed-object scope filters, and summaries include `by_project` and `by_managed_object`.
- Field action audit records now include customer/project/site/object identifiers.
- Field event detail/report and event write actions now enforce project scope.
  Operators cannot acknowledge, request close, close, resend notifications, or
  read event reports for events outside their assigned customer/project/site.
- Device ingest no longer trusts client-supplied `customer_id`, `project_id`,
  `site_id`, or `project_scope`; ingested events use the server/site-profile
  scope so a camera or sensor payload cannot reassign itself to another
  customer project.
- Customer project exports now include a manifest with `payload_sha256`, `profile_sha256`, managed-object count, and optional HMAC signature metadata.
- Customer project imports support dry-run verification and diff before writing, and existing profiles are matched by project/site identity before creating a new file.
- Customer project packages now include `acceptance_summary`, and package manifests include the overall acceptance status plus ready/manual/blocked object counts.
- Customer project import dry-runs now return incoming/current acceptance summaries so delivery can see whether a copied package is actually ready for signoff.
- Customer project packages now have standalone verify and diff APIs, so delivery tools can validate and preview a package without using the import endpoint.
- Customer project packages now include a reusable handoff schema,
  `reuse_assessment`, and `deployment_dependencies`. Export and import dry-run
  results tell delivery whether a package is ready to seed another customer,
  requires onsite manual checks, or is blocked by profile/acceptance errors.
- Package manifests now include reuse status plus dependency counts, and
  verification rejects manifest reuse-status tampering.
- Customer project packages now include a package delivery gate. The gate
  exposes `delivery_gate_status`, `delivery_gate_reasons`, `export_allowed`,
  `import_allowed`, and `customer_handoff_ready`. Import dry-runs still show
  blocked packages for diagnosis, but actual imports reject `blocked` gates
  with `package_delivery_gate_blocked` instead of writing an unsafe customer
  project profile.
- Customer project profile writes now save revision snapshots before template
  overwrite, profile upsert, package import, managed-object edit/delete, and
  rollback. History and rollback endpoints use the same project-scope permission
  checks as the rest of the customer project API.
- Customer project packages now include `tenant_id` and `delivery_namespace` in
  the customer payload and manifest. Verification rejects manifest scope tamper,
  and import/diff uses the delivery namespace to avoid overwriting another
  customer space with the same project id.
- API project-scope enforcement now consumes `tenant_ids` and
  `delivery_namespaces` from the local operator directory or trusted IAM
  headers. The default dashboard operator is scoped to the demo project's
  `default/default` delivery space.
- `/dashboard/projects` now shows tenant and delivery namespace in customer
  project cards, package export results, and package import dry-run results.
  Import dry-run also renders `collision_candidates` so delivery teams can see
  same-name projects in other namespaces before writing anything.
- The industry-template create form now asks for `tenant_id` and
  `delivery_namespace` and sends those fields into the customer project create
  API, so delivery teams can create pilot, lab, and production customer spaces
  from the product UI instead of editing YAML paths by hand.
- Industry templates now expose a product-facing `delivery_summary` and
  `delivery_checklist`: default managed objects, scenario scope, device
  sources, responder groups, vision models, sensor protocols, skill packages,
  acceptance tests, and the rollout steps delivery must complete before a
  customer handoff.
- Industry templates now also expose `template_package`: package schema,
  semantic version, publish status, release channel, owner, upgrade policy,
  runtime floor, product status, blockers, manual checks, dependency counts, and
  source hash. This makes the template market a governed product surface rather
  than a folder of YAML starters.
- Industry templates and exported customer-project packages now also carry
  customer-readable delivery scope: `applicability_scope`, `out_of_scope`,
  `customer_prerequisites`, `scenario_acceptance_criteria`, and
  `dependency_matrix`. These fields explain which customers and scenarios the
  template fits, what the customer must prepare, how each scenario will be
  accepted, and what the package must not claim.
- `GET /api/field/customer-project-templates` supports template-market filters:
  `tenant_id`, `delivery_namespace`, `industry`, `publish_status`,
  `product_status`, `template_id`, `release_channel`, and `owner`. Filtered
  responses recompute template, tenant, industry, publish-state, product-state,
  and managed-object counts for the visible template set.
- `/dashboard/projects` now renders those industry templates as a template
  market. Delivery can inspect customer fit, default object coverage, runtime
  bindings, acceptance status, and the rollout checklist, then select a
  template directly into the customer-project creation form.
- The template market also renders applicability, prerequisites, scenario
  acceptance criteria, and delivery boundaries, so customers and delivery teams
  do not have to read YAML to understand whether a template fits the project.
- `/api/field/customer-project-workbench` now returns a customer-readable
  vocabulary, acceptance flow, and delivery contract for solution-provider
  projects. The Dashboard project page uses those labels to present "客户空间",
  "交付空间", "现场对象", "能力配置", "交付包预检", and "验收材料" instead of
  exposing implementation terms such as tenant, runtime, API scope, or managed
  object in the primary customer-facing flow.
- Customer project catalog and detail payloads now include `delivery_workflow`.
  This turns each project into a delivery checklist covering customer scope,
  managed objects, runtime bindings, site map/devices, responder credentials,
  acceptance evidence, and handoff package status.
- `/dashboard/projects` now shows that delivery workflow inside each customer
  project card, so delivery can answer "what is still blocking handoff" without
  reading YAML or acceptance JSON.
- Acceptance reports and exported acceptance dossiers now carry the same
  `delivery_workflow`, and the printable HTML dossier includes a delivery
  workflow table. This keeps Dashboard status, API evidence, and customer
  handoff artifacts aligned.
- Acceptance reports, exported acceptance dossiers, and proposal bundles now
  carry the same project-level `launch_readiness` decision. The JSON manifest
  records `launch_readiness_status`, `launch_stage`, and `production_ready`, and
  the printable HTML shows "上线准入" with customer-readable status, release
  claim, next step, and gate evidence. Any tampering with this readiness section
  is caught by the dossier/proposal payload hash verification.
- `/dashboard/projects` now includes a customer-project metadata editor for
  customer-facing fields such as customer name, industry, project name, site
  name, and object-scope note. The editor loads the full site profile before
  saving, so updating labels does not drop zones, devices, responder groups, or
  managed objects.
- `/dashboard/projects` now includes a Managed Object Directory. Delivery teams
  can browse every customer-visible object with its project scope, responder
  group, vision model bindings, sensor protocols, skill packages, and acceptance
  tests, then load an object directly into the quick editor without copying
  identifiers from YAML.
- The Managed Object Directory also has a backend API at
  `/api/field/customer-projects/managed-object-directory`, so delivery scripts,
  customer handoff tools, and acceptance checks can consume the same scoped
  object status that the Dashboard shows.
- Managed-object removal is now treated as an offline lifecycle action. The
  backend rejects removals without a customer-visible reason, returns the
  removed object snapshot plus `offline_reason`, and the Dashboard shows impact
  before allowing the operator to submit the removal.
- The managed-object editor is now grouped by product boundary: basic object
  identity, detection scope, runtime bindings, and acceptance evidence. This
  keeps customer-facing object setup understandable for delivery teams instead
  of exposing one flat engineering form.
- Managed-object create, update, and offline actions now append a compact
  `object_change_log` to the customer project profile. The log includes action,
  object id, operator, reason, and before/after binding summaries, and the
  project Dashboard shows recent changes for handoff review.
- Dashboard read requests now send the selected `X-Askme-Operator-Id` header,
  keeping list/detail reads aligned with the same operator scope used by write
  actions.
- Customer project detail now has an acceptance-report API and UI action. The report separates local object evidence from missing deployment credentials and onsite acceptance proof, so the product does not overclaim production readiness.
- Acceptance reports now include field readiness gates and evidence report paths for scenario evaluation, ingest smoke, voice smoke, notification smoke, and runtime roundtrip. A project is blocked when high-risk audit review or live-field evidence gates are unresolved, even if managed-object acceptance tests are linked.
- Customer projects can now export an acceptance dossier JSON file and a printable HTML dossier. The dossier records the same acceptance report plus an evidence inventory with SHA-256 hashes for every linked smoke/readiness artifact. If `ASKME_CUSTOMER_ACCEPTANCE_DOSSIER_HMAC_SECRET` is configured, the manifest is HMAC-signed.
- Dashboard navigation now separates customer projects into `/dashboard/projects` instead of burying the workflow in delivery diagnostics.
- The customer project page now exposes package import dry-run, managed-object quick edit, and project/object event-scope checks.
- The customer project page now exposes lifecycle operations for package export, project archive, and managed-object delete with explicit UI confirmation and server-side permission checks.
- The dashboard overview page now uses customer-readable Chinese for the first-screen product status, delivery gate, scenario coverage, and multi-site rollout summary.
- JSON request parsing accepts UTF-8, UTF-8 BOM, UTF-16, and GB18030 bodies so field delivery tools on Windows do not fail only because their JSON body encoding differs.
- Customer project read endpoints now enforce `field:project:read`, so unknown operators cannot list or export project packages.
- The demo operator directory now exposes project scope claims and uses them to filter customer-project/site catalogs.
- Customer project write endpoints now enforce the same project scope, so a
  scoped supervisor cannot import, archive, or mutate another customer's
  project even when that supervisor has the generic `field:project:write`
  permission.
- Managed-object catalogs now include an acceptance readiness summary, per-object missing binding evidence, and local acceptance reference checks.
- The customer project page now renders per-object acceptance gates so delivery teams can see linked/manual/blocked evidence without opening YAML.
- Unified audit records now carry tenant, delivery namespace, customer, project,
  site, and managed-object identifiers when the source record provides them.
- `/api/audit/events` and `/api/audit/export` accept customer-project scope
  filters. Scoped operators are automatically narrowed to their assigned single
  tenant/customer/project/site; out-of-scope audit reads or exports return 403,
  and multi-project operators must choose an explicit scope before querying.
- The Dashboard delivery audit panel now filters by customer project and managed
  object, and audit exports reuse the same visible scope so customer evidence
  packages do not mix records from another delivery project.

Remaining boundary: managed-object bindings are now attached to events and audit records, but they still do not execute real vision models, sensor adapters, or robot runtime skills automatically.

更新时间：2026-05-13

askme 是面向机器人现场任务的自然语言入口。它不是普通聊天框，也不是机器人底层控制器；它负责把人的语音或文字目标变成可解释、可确认、可审计、可评测的任务意图，再交给安全和 runtime 层处理。

当前只维护三个入口文档：

- `docs/PRODUCT.md`：产品定位、能力边界、路线图。
- `docs/ARCHITECTURE.md`：系统结构、模块边界、数据流。
- `docs/OPERATIONS.md`：配置、启动、验收、排障。

## 产品目标

现场用户应该可以直接说：

- “请问洗手间在哪里？”
- “开始 A 区巡检。”
- “暂停一下。”
- “刚才巡检结果怎么样？”
- “停下。”

系统要做到：

- 听得见：麦克风、ASR、VAD、打断链路可观测。
- 知道何时该回话：Interaction Gate 能区分问路、任务指令、旁观闲聊、多人不确定和噪声。
- 回答有依据：RAG evidence 能显示来源、状态、是否被采用。
- 不乱执行：机器人任务必须经过 TaskHandoff、SafetyPreflight、runtime arbiter。
- 可接管：任务运行中可以暂停、继续、取消、查询状态。
- 可审计：回答依据、任务计划、operator action、runtime event、报告都能留痕。

## 已有能力

### 语音交互

- Voice Mission Center 三栏 UI：语音状态、对话、当前任务/服务能力。
- MiniMax 文本和 TTS 基础链路。
- Voice Turn Trace：ASR、LLM、TTS、播放、打断延迟桶。
- Interaction Gate：判断 `respond`、`clarify`、`record_only`、`ignore`、`refuse`。
- 旁观者提到“机器狗”不会误唤醒；多人/声画不一致时优先澄清。

### 记忆与 RAG

- MemoryBridge 支持 `mem0`、`robotmem`、`vector fallback`。
- KnowledgeCatalog 是知识生命周期事实源。
- 支持 Markdown、TXT、JSON、JSONL、NDJSON、CSV 导入和 API 预览；不成熟的二进制资料会被明确拒绝并提示先转换。
- 每条知识有 `quality_status`、`visibility`、客户、项目、产品模块、推进事项和对象关联字段。
- 未发布、待复核、仅内部、过期、删除、冲突知识不会进入 prompt。
- 回答会返回 `evidence` 和 `rag.answer_policy`。
- Dashboard 气泡展示回答依据。

### 任务运行

- TaskHandoff、SafetyPreflight、TaskRun、RuntimeEvent、TaskReport 已有 fake/sim/shadow 基础。
- RuntimeArbiterClient 是 contract-only，external/lab 默认禁用，不直接触碰硬件。
- Dashboard runtime 控制动作会记录 `operator_id`、`reason`、`risk_acknowledgement`。

### 评测证据

- RAG Trust 离线评测：游客问路、过期知识、冲突位置、删除知识、未知位置。
- Voice E2E 离线评测：游客问路、未知地点拒答、巡检 SOP、设备位置、过期路线拒答、旁观噪声、多人澄清、急停。
- Health snapshot 与 Dashboard 运营诊断显示 Knowledge Trust 和 Voice E2E 结果。

### 能力中心与在线技能增长

- Dashboard `能力中心` 展示客户可读的能力分组、场景能力蓝图、缺口、Agent Profile、生成技能审批队列、技能包和调用审计。
- `在线增长候选` 从真实调用审计里聚合失败、阻断、未命中请求，帮助产品经理判断哪些重复需求值得沉淀成技能。
- 产品经理可从增长候选一键生成 `SKILL.md` 草稿；草稿仍然是 `pending_approval`，不会自动启用。
- LLM 生成的新 `SKILL.md` 默认进入草稿/待审批，不会自动变成生产可用能力。
- 生成技能必须通过结构校验、触发词冲突检查、工具边界检查和人工审批。
- 审批通过后还必须分配到客户/园区 `Skill Package`，同一套产品可针对不同项目启用不同能力。
- `Skill Package` 已升级为客户项目发布单元，支持版本快照、pilot/prod 发布通道、灰度比例和回滚。
- 灰度比例为 `0%` 时，该能力包内技能不会进入可触发状态；回滚会生成新的版本记录，保留谁在何时回滚到哪个版本。
- 支持项目级、用户级和 managed Agent Profile Markdown 配置，字段包含工具 allow/deny、可派生子 agent、预加载 skills、MCP server、hooks、模型、最大轮次、超时、隔离方式、记忆范围和风险等级。
- Agent Profile 的 hooks 已支持产品级声明式拦截：`PreToolUse` 可在工具调用前拒绝，`PostToolUse` 可在结果返回前阻断敏感输出；系统不会执行任意 shell hook。
- `create_skill` 工具统一走 `SkillManager.create_generated_skill_draft`，所以语音/文本 Agent、Dashboard 候选生成和后端 API 都进入同一套待审批、禁用、校验、审计流程。
- 首批园区场景技能已从 planned 落成 built-in：`report_fall_unrecoverable`、`report_stuck`、`report_motor_fault`、`detect_night_intruder`、`detect_illegal_parking`、`detect_fire_smoke`、`inspect_trash_bin`、`offer_wayfinding_help`、`escort_visitor`。这些技能通过 `field_event_trigger` 进入 FieldOperationsService，生成事件、按策略通知、归档和审计，而不是只返回聊天文案。
- 操作员治理已从前端兜底推进到服务端目录：Dashboard 会读取 `/api/governance/current-operator`，未知操作员显示为未登记并且无权限；目录页面返回角色矩阵、SSO/IAM readiness 和生产阻塞原因。
- 企业账号接入采用网关验签模式：客户的 OIDC/IAM 网关验证登录 token 后注入受信身份头，askme 只根据已验证的 operator/roles 做权限和审计，不相信请求 body 里的 operator_id。
- 统一审计查询和导出已具备产品入口：`/api/audit/events` 汇总技能、现场事件和 runtime 审计；`/api/audit/export` 生成带 SHA-256/可选 HMAC 的 JSONL 证据包，并可投递到 SIEM/WORM webhook；`/api/audit/export/retry` 可查询和重放失败投递，Dashboard 交付页能看到待投递数量。

## 产品边界

askme 可以：

- 理解用户目标。
- 追问缺失信息。
- 生成高层任务计划。
- 检索知识并给出有依据的回答。
- 把确认后的计划交给 runtime arbiter。
- 展示任务状态、报告和审计证据。

askme 不可以：

- 直接控制电机、步态、机械臂、串口或 `cmd_vel`。
- 绕过 SafetyPreflight。
- 用过期或冲突知识驱动高风险任务。
- 在多人、声画不一致、感知过期时猜测说话对象。
- 在没有明确授权时执行真实硬件动作。

## 客户演示路径

1. 打开 Dashboard。
2. 用文本或语音输入“请问洗手间在哪里”。
3. 展示回答气泡中的依据。
4. 输入“开始 A 区巡检”。
5. 展示 Planning 卡片、确认动作、Runtime timeline。
6. 任务运行中输入“暂停”“继续”“取消”“现在执行到哪了”。
7. 打开运营诊断，看 Knowledge Trust、Voice E2E、Latency。

## 下一步路线

近期优先级：

1. 把 Voice E2E 从离线模拟升级为真实麦克风/录音回放评测。
2. 把 Knowledge Trust 与 Voice E2E 合并为统一 Readiness Evidence 页面。
3. Knowledge Console 增加审批、版本、冲突处理和异步重建索引 job。
4. TaskRunStore 持久化运行状态、runtime events、operator actions、reports。
5. Operator RBAC 下一步补企业登录页/会话 UI、审批流通知、审计导出重试任务和外部 SIEM/WORM 生产联调；当前已具备 OIDC/IAM 网关受信身份头适配、统一审计查询和签名导出。
6. Skill Package 增加客户项目验收状态、发布日历和字段级变更对比。
7. 接入真实感知 provider：pose/gaze、gesture、DOA、声画关联、接近/停留、多人仲裁。
8. external/lab runtime 只开放低风险 shadow/lab skill：status_report、capture_image、read_status_panel、generate_report、return_home。

本次新增的产品化增长能力：

- Agent Profile 可以通过 `POST /api/agent-profiles` 写入项目级配置，和 Claude Code 的项目 subagent 文件类似，但会进入 askme 审计链。
- 受控 agent 可调用 `create_agent_profile` 生成新的项目级 agent lane，用于沉淀“知识运营、问路、停车检测、垃圾桶巡检”等专职代理；工具权限由服务端 allowlist 校验，不能由前端或 LLM 自行扩权。
- Dashboard 能力中心已提供 Agent Profile 创建表单和预览按钮，产品经理可以在界面上填写角色边界、工具范围、可派生 agent 和预加载技能。
- 能力中心新增 `scenario_blueprints`：把机器人异常、夜间陌生人、违停、烟火、垃圾桶、突发巡检、人群聚集、问路和带路映射到 required skills、传感器/数据依赖、通知归档和验收标准。
- 园区问路已具备可调用技能入口：`lookup_place` 调用空间语义地图解析目的地，`recommend_route` 调用路线推荐服务生成语音指路或带路前 handoff 建议，`answer_wayfinding` 封装成游客可直接触发的语音指路能力；未知地点必须拒答或要求人工更新点位库。
- 人群聚集已具备可调用技能入口：`detect_crowd_gathering` 会在人数、停留时长或复巡证据满足策略时进入安保事件闭环，短暂停留不能被夸大成告警。
- 新 profile 只定义角色、工具边界、可派生 agent、预加载技能和风险等级；真实机器人动作仍必须经过 SkillGate、SafetyPreflight 和 runtime arbiter。

暂不做：

- 真实生产硬件动作默认开启。
- 机械臂抓取、靠近游客、开门、支付、删除数据等高风险动作。
- 让 LLM 直接输出底层控制命令。

## 15. 产品路线和近期打磨方向（2026-05-16）

AskMe 的产品方向明确为：面向方案商的多客户现场机器人任务平台，而不是单一项目的机器人聊天框。核心价值是把园区、厂区、仓储、景区等现场对象，统一变成可配置、可调用、可验收、可审计的能力包。

近期路线按四层推进：

1. 现场场景产品化：问路、带路、违停、烟火、垃圾桶、陌生人、机器人故障、人群聚集和突发巡检必须逐条形成验收卡片。每张卡片要展示触发来源、判断证据、调用技能、通知对象、语音播报、归档记录和未完成项。
2. 意图路由 2.0：不再只依赖 `voice_trigger` 硬匹配。路由顺序为急停规则、固定触发词、可审计场景语义规则、LLM 兜底。每次场景语义命中都必须带 `scenario_id`、置信度、命中词和规则编号，方便客户验收和事后追责。
3. 客户项目复制能力：行业模板、客户项目、现场对象、技能包、传感器协议、视觉模型和验收用例要绑定在一起。交付团队复制新客户项目时，不能只复制页面配置，还要复制验收边界和缺口清单。
4. 真实现场闭环：语音或内部事件可以触发能力，但高风险动作不能直接控制硬件。带路、巡检、回充、拍照取证、异常处置都必须进入 TaskHandoff、SafetyPreflight 和 runtime arbiter。

本阶段产品判断：

- 面向客户讲“现场任务平台”和“场景验收”，少讲 fake/sim/runtime 这类内部词。
- 面向交付讲“模板、对象、资源绑定、验收证据和未完成项”。
- 面向研发讲“场景入口、技能调用、事件闭环、审计证据和安全边界”。
- 面向销售只承诺可演示、可试点、可按现场验收扩展，不能承诺未接硬件的无人值守生产上线。

本次已经落地的细节打磨：

- 新增可审计场景意图层：常见说法如“车停在主通道中间了”“这边冒烟了”“垃圾桶快满了”“窗户旁边有人拍照”“机器狗倒地起不来了”可以路由到对应技能，而不是必须逐字命中固定触发词。
- 新增“人为恶意挡路”客户可见技能 `report_malicious_blocking`，进入 `field_event_trigger`，使用 `robot_abnormal_incident + malicious_blocking`，通知安保并归档证据。
- 新增 `/api/scenario-intents` 和 `/api/scenario-intents/preview`，用于 UI、测试和交付验收查看“这句话会不会触发、触发哪个技能、依据哪条规则”，预览接口不执行真实技能。
- 问路类问题允许问句触发；高风险事件问句默认不触发动作，避免“违停事件怎么处理吗”被误判成真实事件。
- 路由证据进入 trace payload，后续 Dashboard 可以直接显示“为什么系统判断这是烟火/违停/故障/挡路”。

下一步需要继续打磨：

- Dashboard 增加“场景验收矩阵”：每个场景一键模拟、真实触发、查看证据和导出验收结果。
- 增加真实语音 E2E 脚本：麦克风输入到 ASR、意图路由、技能调用、TTS 播报和归档必须全链路可复测。
- 把意图未命中、误命中和用户改口沉淀到增长候选，交给产品经理决定是否扩展技能包。
- 将客户项目模板与场景验收矩阵打通，让不同客户可以按行业模板复制能力、资源绑定和验收用例。
