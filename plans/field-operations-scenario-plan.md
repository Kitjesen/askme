# 园区机器狗场景产品计划

更新日期：2026-05-11

## 目标

把机器狗从“能对话、能巡检”推进到“能在园区真实值守”：能识别现场事件，知道什么时候该说话、什么时候只记录，知道通知安保还是保洁，知道什么时候打断当前任务，并且每个事件都有证据、地点、处理闭环和归档。

核心原则：

- 机器人不是听到声音就回答，而是先经过 Interaction Gate。
- 机器人不是看到异常就冲过去，而是先保留安全距离、拍照、上报、等待授权。
- 所有现场事件必须有地点、时间、证据、处理对象和归档记录。
- 游客服务和安防告警要分开，避免把问路做成安全事件，也避免把闲聊误触发任务。

## 场景矩阵

| 场景 | 优先级 | 触发条件 | 机器人动作 | 通知对象 | 归档 |
| --- | --- | --- | --- | --- | --- |
| 异常情况 | P0 | 摔倒无法恢复、卡住、恶意挡路、关节电机故障 | 停止危险动作，固定语音播报，等待处理 | 钉钉安保群 | 必须 |
| 夜间陌生人拍照 | P0 | 夜间重点区域出现陌生人并停留 | 保持距离，拍照标地点，通知安保 | 钉钉安保群 | 必须 |
| 车辆违停检测 | P0 | 车辆停在道路/主通道且不在停车区 | 拍照标地点，必要时语音提醒 | 钉钉安保群 | 必须 |
| 火灾及烟雾监测 | P0 | 温度、烟雾、视觉烟火任一高置信或多源联合触发 | 播放疏散提醒，拍照上传 | 钉钉安保群 | 必须 |
| 垃圾桶监测 | P1 | 定点垃圾桶满溢或外溢 | 拍照记录，通知保洁 | 钉钉保洁群 | 必须 |
| 突发任务巡检 | P0 | 管理员派遣到指定地点巡检 | 暂停当前自动巡检，前往目标点，支持实时画面 | 运营/安保 | 必须 |
| 人群聚集检测 | P1 | 人数 > 5 且持续 > 30 分钟，复巡仍存在 | 先记录，复巡后提醒，必要时通知安保 | 钉钉安保群 | 必须 |
| 路人指路 | P1 | 5 个固定路引点有人停留并面向路牌/机器人 | 主动询问是否需要指路，只回答地图库地点 | 不通知 | 可选 |
| 路人带路 | P2 | 游客明确请求，目的地在地图数据库且路线安全 | 确认目的地，低速引导，到达播报结束 | 不通知 | 必须 |

## 产品分层

### 1. 感知层

需要接入或预留：

- 摄像头：人、车、垃圾桶、烟火、陌生人、人数统计、照片证据。
- 夜间模式：红外/低照度摄像头、夜间时段配置、重点区域配置。
- 地图区域：停车区、道路、主通道、窗户、角落、路引点、禁行区、垃圾桶点位。
- 传感器：烟雾、温度、机器人姿态、电机故障、底盘卡住。
- freshness：每个感知结果必须带更新时间，过期结果不能直接触发高风险动作。

### 2. 规则层

需要建立 FieldScenarioRegistry + RuleEngine：

- 每个场景有固定触发规则。
- 每个规则有阈值、证据要求、去抖时间、复核策略。
- 每个场景决定是否通知、通知谁、是否语音播报、是否打断当前任务。
- 安防场景优先级高于游客服务，突发巡检高于自动巡检。

### 3. 交互层

路人问路不能靠“听到声音就回”：

- 固定 5 个路引帮助点位。
- 识别有人停留、朝向路牌或机器人、距离合适。
- 机器人先主动问一句：“你好，请问有什么需要指路的吗？”
- 用户说出地点后，从园区地图和知识库回答。
- 地点未知、路线过期、知识冲突时拒绝编路线，提示咨询人工。

### 4. 任务层

突发巡检和带路都必须走任务系统：

- 管理员突发巡检：暂停当前任务，记录中断原因，派遣到指定点。
- 带路：确认目的地，生成安全路线，低速移动，支持取消和结束。
- 火灾、故障、恶意挡路等高风险事件不应继续执行原任务。
- 所有任务切换都要进入审计日志。

### 5. 通知与归档层

通知消息必须给人可处理的信息：

- 事件类型。
- 当前地点和地图区域。
- 图片或视频证据。
- 触发原因和置信度。
- 建议处理动作。
- 机器人当前状态。
- 处理人、处理结果、关闭时间。

## 已落地的代码基线

- `askme.pipeline.field_scenarios`：新增产品场景注册表，覆盖 9 个场景。
- `askme.pipeline.incident_alerts`：扩展固定告警话术，覆盖夜间陌生人、违停、火灾烟雾、垃圾桶满溢、人群聚集、突发巡检。
- `askme.pipeline.field_operations`：新增真实事件执行链路，支持场景判定、证据校验、钉钉分组路由、JSONL 归档、事件关闭。
- `AlertDispatcher`：已有语音、钉钉、日志和 JSONL 事件归档能力。
- HTTP 已新增 `/api/field/scenarios`、`/api/field/events`、`/api/field/events/{event_id}/close`。
- Dashboard 已新增“现场事件”区域，可选择场景、触发样例、查看通知对象和归档结果。
- `InteractionGate`：已有交互准入基础，可以继续接路引点、多人、距离、朝向和声画一致性。

## 接下来实施顺序

### Phase 1：事件闭环最小可用

目标：不用真实算法，也能通过模拟事件验证播报、通知、归档。

- 给 9 个场景补模拟事件入口。
- Dashboard 增加“现场事件”列表。
- 事件卡片展示地点、证据、通知对象、处理状态。
- 钉钉通知支持安保群和保洁群分流。
- 增加事件关闭/备注接口。

验收：

- 模拟“垃圾桶满溢”只通知保洁。
- 模拟“夜间陌生人”通知安保并归档照片路径。
- 模拟“突发巡检”能暂停当前巡检并记录原因。

### Phase 2：地图和区域配置

目标：让系统知道哪里能停车、哪里是主通道、哪里是路引点。

- 建立 SiteMap 数据模型。
- 配置停车区、道路、主通道、窗户、角落、垃圾桶、路引点。
- 每个点位支持别名和中文地名。
- RAG/记忆系统只回答已发布、未过期、无冲突的地点知识。

验收：

- 停车区内车辆不报警。
- 主通道车辆停留超时报警。
- 未知地名问路拒答，不编路线。

### Phase 3：感知接入和 freshness

目标：把 mock 事件替换成真实摄像头/传感器输入。

- 接人、车、火、烟、垃圾桶满溢、人数统计。
- 接姿态、电机、卡住、烟感、温度。
- 每个感知输入必须有 timestamp、confidence、source。
- 过期传感器结果只能提示“需要复核”，不能直接触发确定告警。

验收：

- 旧照片不能触发新告警。
- 烟感离线时，火灾事件必须显示“传感器缺失”。
- 夜间陌生人需要夜间时段 + 重点区域 + 停留证据。

### Phase 4：游客指路和带路

目标：让机器人像园区服务人员，而不是路过就乱说话。

- 配置 5 个固定路引帮助点。
- Interaction Gate 接入路引点和停留判断。
- 地图数据库支持目的地、路线、别名、不可达状态。
- 带路任务支持低速、取消、到达、跟随丢失处理。

验收：

- 人在路引点停留时，机器人可主动询问。
- 人只是经过时不打扰。
- 带路过程中游客不跟随，机器人停下确认。

### Phase 5：客户演示包

目标：形成可给客户看的演示，而不是工程控制台。

- 现场事件看板。
- 模拟触发按钮。
- 地图点位和证据照片展示。
- 钉钉通知预览。
- 事件归档查询。
- 游客问路和带路演示脚本。

验收：

- 客户能看懂“发生了什么、通知了谁、机器人说了什么、证据在哪里、谁处理了”。

## 重要边界

- LLM 不直接判断“犯罪”或“恶意”，只能说“疑似异常/需要保安核实”。
- 没有证据照片和位置的事件不能进入高优先级告警。
- 没有地图知识不能带路。
- 夜间陌生人、违停、火灾烟雾、人群聚集都必须允许人工复核。
- 真实机器人动作上线前必须先通过模拟、shadow、实验室三阶段。

## 2026-05-11 推进记录

已补齐三条产品闭环：

1. 原始现场事件入口：新增 `/api/field/ingest`，可接摄像头检测、烟感/温度、机器人故障、地图区域配置。入口会先归一化为 `scenario_id`，再进入通知、归档和 Dashboard。
2. 低风险 LLM 现场播报：游客服务、设施服务等低风险场景可启用 `ASKME_FIELD_LLM_NARRATIVE=1`，由大模型把固定事实改写成更自然的一句话；P0 告警和 error 级别事件仍使用固定话术，避免临场发挥。
3. 语音风格：新增可选语音档位，包括巡检播报、游客服务、安防提醒、紧急告警、夜间低扰。Dashboard 可查看、切换、试听，部署时可把每个档位绑定到不同 MiniMax voice_id 或克隆音色。

下一步仍然要做真实设备接入验证：

- 摄像头检测服务把 `detections`、`zone_id`、`image_path` POST 到 `/api/field/ingest`。
- 烟感/温度服务把 `sensor.temperature_c`、`sensor.smoke_level` POST 到 `/api/field/ingest`。
- 机器人底盘/关节服务把 `robot.fault_type` POST 到 `/api/field/ingest`。
- 园区地图配置需要持久化 `site_map.zones`，让违停、路引点、夜间重点区域不靠硬编码。
- 高风险事件还需要 freshness 校验，过期感知只能要求复核，不能直接触发告警。

## 2026-05-11 二次反思与收口

骨架和产品级能力的区别：

- 骨架只会“收到事件就触发”；产品级系统必须先判断这条数据是否可信。
- 骨架只会“通知一次”；产品级系统必须避免同一辆车、同一个故障、同一个区域在短时间内刷屏。
- 骨架只会“字段齐就通过”；产品级系统必须把 freshness、confidence、source、zone、evidence 都作为审计事实留下。
- 骨架把 UI 当调试面板；产品级 UI 应该让客户看懂“为什么触发、为什么没触发、通知了谁、证据是什么”。

本轮新增硬约束：

- `/api/field/ingest` 会记录 `_ingested`、`source`、`_ingest_received_at`。
- P0/P1 真实感知事件必须通过 freshness 校验，默认 `max_input_age_s=30`。
- 摄像头识别类事件必须满足最低置信度，默认 `min_detection_confidence=0.55`。
- 同一 dedupe key 在 `dedupe_window_s=120` 秒内不会重复通知，只会归档为 duplicate。
- `FieldEventRecord` 记录 `freshness_status`、`freshness_age_s`、`confidence`、`dedupe_key`、`duplicate_of`，方便 UI 和审计系统解释。

下一步真实接入优先顺序：

1. 摄像头检测服务接入：先接车辆违停、垃圾桶满溢、人群聚集三类低动作风险场景。
2. 烟感/温度接入：必须包含传感器 ID、读数、时间戳、位置和照片证据；缺照片时只能待复核。
3. 机器人故障接入：底盘卡住、摔倒无法恢复、关节电机故障进入 P0，但真实硬件动作仍由 runtime/safety 接管。
4. 地图区域持久化：把 `site_map.zones` 从配置样例升级为可维护的园区地图数据源。
5. Dashboard 显示“未触发原因”：stale、low confidence、duplicate、missing evidence 都要给客户看懂。

## 2026-05-18 产品化推进记录：场景验收矩阵接口和页面

本次把“客户场景是否能验收”从脚本验证推进到产品接口和 Dashboard 页面：

- 新增 `/api/field/scenario-acceptance`，返回 9 个客户场景的验收矩阵：场景状态、自然语言入口、设备/传感器入口、通知对象、归档要求、现场依赖、验收标准和客户下一步。
- Dashboard `/dashboard/scenarios` 已接入该接口，页面明确展示“当前证明的是演示与集成验收，不等于无人值守生产上线”。
- 每张场景卡片增加“真实接入还缺什么”，让客户和交付团队看到 smoke/temperature sensor、camera/VMS、runtime arbiter、robot navigation gateway 等现场依赖。
- 场景页面仍保留一句话触发预览，预览只判断场景、技能、风险和依据，不会直接派发机器人任务。

验证证据：

- `python -m py_compile askme\pipeline\field\field_operations.py askme\api\routes\field_events.py` -> passed。
- `node --check askme\static\dashboard\app.js` -> passed。
- `python -m pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests\test_field_event_routes.py tests\test_field_operations.py::test_field_operations_http_endpoints -q` -> 3 passed。
- `python -m pytest tests\test_field_scenarios.py tests\scenario_tests\test_field_operations_evaluation.py tests\test_field_event_skills.py tests\test_capability_scenario_intent_routes.py -q` -> 15 passed。
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed；新增 `scenario_acceptance_page` 视觉门禁，桌面和移动无横向溢出，9 个场景全部覆盖，无 console/page/response errors。

仍然不能对客户宣称的内容：

- 不能宣称无人值守生产上线。
- 不能宣称已经接入真实摄像头、烟感、机器人底盘或钉钉生产 webhook。
- 不能把视觉冒烟脚本等同于现场验收；现场仍需要真实设备、生产凭证、运行回调和客户签收证据。

## 2026-05-18 Product checkpoint: admission decision evidence

Goal: make the product explain why a field event was triggered, blocked, deduplicated, or kept for human review.

Implemented:

- Field event views now include `admission_decision` with `blocked`, `customer_status`, `technical_reasons`, `evidence_facts`, and `next_step`.
- Dashboard field-event detail now renders an admission card before evidence, delivery, workflow, and audit sections.
- The card covers stale sensor input, low-confidence detections, duplicate events, and missing required evidence.
- The same card now exposes resource-binding gaps such as `no_managed_object_matched`; triggered events may continue handling, but production acceptance must bind devices, vision models, sensor protocols, skill packages, and acceptance tests to customer managed objects.
- Browser smoke now seeds a blocked customer-visible event and verifies the Dashboard shows the blocking reason, evidence fact, next step, and event context.

Validation:

- `python -m py_compile askme\pipeline\field\field_operations.py scripts\eval\check_dashboard_visual.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `python -m pytest tests\test_field_operations.py::test_ingest_does_not_bind_managed_object_when_project_scope_mismatches tests\test_field_operations.py::test_p0_event_missing_evidence_is_not_dispatched tests\test_field_operations.py::test_stale_sensor_ingest_is_archived_without_dispatch tests\test_field_operations.py::test_low_confidence_camera_ingest_requires_review tests\test_field_operations.py::test_duplicate_ingest_does_not_notify_twice tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 6 passed.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed. New interaction gate: `field_admission_decision`.

Remaining delivery risk:

- Rejected device-ingest events are customer-visible only when the field service has a customer project scope or the device payload binds to a managed object/project. Browser smoke now validates the scoped demo case. Production still needs explicit device/project binding instead of relying on demo defaults.

## 2026-05-18 Product checkpoint: device ingest scope contract

Goal: make every real camera, sensor, or robot ingest response explain which customer project and managed object it belongs to, and whether the event can be used as site-validation evidence.

Implemented:

- `/api/field/ingest` responses now include `ingest_scope_contract`.
- The contract records device trust, server-side customer project scope, managed-object binding, resource execution readiness, production gate, and audit facts.
- Client-supplied `customer_id`, `project_id`, `site_id`, and `managed_object_id` are not accepted as proof for ingested device events. The contract uses the server-side project scope and the event's managed-object binding result.
- Field event list/detail views now derive the same `ingest_scope_contract` for archived ingested events, so the evidence remains visible after the original HTTP response is gone.
- The production gate distinguishes `bound_ready`, `unbound_managed_object`, `blocked_device_trust`, `no_matching_scenario`, and resource-binding review states.

Validation:

- `python -m py_compile askme\pipeline\field\field_operations.py` -> passed.
- `python -m pytest tests\test_field_operations.py -q` -> 61 passed.

Remaining delivery risk:

- This proves server-side scope and object-binding evidence in software. It still needs real device payloads from the customer site, registered device identities, and acceptance-test artifacts before production signoff.

## 2026-05-18 Product checkpoint: Dashboard ingest binding evidence

Goal: make the customer-facing field event detail page show the same device/project/object binding evidence that `/api/field/ingest` returns.

Implemented:

- Field event detail now renders a device-ingest scope card directly under the admission decision card.
- The card shows device trust, server-side customer project scope, managed-object binding, selected capability or skill package, production gate, evidence count, freshness, and confidence.
- Visual smoke verifies the card, the four binding fact blocks, and the production gate copy on a blocked/stale device-ingest event.

Validation:

- `node --check askme\static\dashboard\app.js` -> passed.
- `python -m py_compile scripts\eval\check_dashboard_visual.py` -> passed.
- `python -m pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests\test_field_operations.py::test_ingest_infers_managed_object_with_matching_project_scope tests\test_field_operations.py::test_ingest_does_not_bind_managed_object_when_project_scope_mismatches tests\test_field_operations.py::test_field_operations_http_endpoints -q` -> 4 passed.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed; `field_admission_decision` now includes `has_ingest_scope_card`, `has_ingest_scope_grid`, and `has_ingest_scope_gate`.

## 2026-05-18 Product checkpoint: device onboarding readiness

Goal: give delivery and customer teams a direct report for whether real field devices are registered, trusted, seen recently, and bound to customer managed objects before site validation.

Implemented:

- Added `/api/field/device-onboarding`.
- The report derives from registered and observed device status, then adds managed-object candidates and an onboarding gate per device.
- Each device now reports whether it is ready, blocked, or still needs manual review for registration, live callback, signature, source policy, and customer-object binding.
- Dashboard delivery view now renders a device-onboarding card with readiness metrics, device rows, binding evidence, and concrete next actions.
- Static Dashboard checks now assert the endpoint, renderer, and CSS contract.

Validation:

- `python -m py_compile askme\pipeline\field\field_operations.py askme\api\routes\field_internal.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `python -m pytest tests\test_field_operations.py::test_device_onboarding_payload_reports_object_binding_and_next_actions tests\test_health.py::TestHealthServer::test_field_device_onboarding_endpoint_returns_delivery_report tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 3 passed.
- `python -m pytest tests\test_field_operations.py -q` -> 62 passed.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed.

Remaining delivery risk:

- This is still a software-side readiness report. Production signoff requires real camera/sensor/robot payloads, configured HMAC secrets, confirmed allowed sources, and customer-site managed-object bindings.

## 2026-05-18 Product checkpoint: device onboarding API surface contract

Goal: prevent real-device onboarding from being treated as ordinary customer copy or unmanaged Dashboard glue.

Implemented:

- `askme.api.composition.API_SURFACES` now declares `device onboarding evidence` under the `internal` surface.
- The internal surface manifest explicitly includes `askme.api.routes.field_internal`.
- `/api/surfaces` route inventory now verifies `/api/field/device-onboarding` remains classified as `internal`.
- Migration tests verify `register_field_routes` still delegates internal machine/device routes to `register_field_internal_routes`.

Validation:

- `python -m py_compile askme\api\composition.py askme\api\routes\field_internal.py` -> passed.
- `python -m pytest tests\test_package_migration_compat.py::test_api_surface_manifest_is_product_boundary_contract tests\test_package_migration_compat.py::test_field_route_registration_delegates_to_split_route_modules -q` -> 2 passed.
- `python -m pytest tests\test_health.py::TestHealthServer::test_api_surfaces_endpoint_returns_customer_boundary_map -q` -> 1 passed.

## 2026-05-18 Product checkpoint: device onboarding gates field readiness

Goal: make real-device onboarding evidence affect deployment readiness, not only appear as a separate report.

Implemented:

- `build_field_deployment_readiness()` now accepts a `device_onboarding` report and exposes dedicated gates:
  `device_onboarding_report_available`, `device_onboarding_no_blockers`,
  `device_onboarding_has_ready_device`, and `device_onboarding_all_ready`.
- `FieldOperationsService.readiness_payload()` now passes the live device-onboarding report into field readiness.
- Production readiness now requires registered devices to be observed, trusted, unblocked, fresh, and bound to customer managed objects.
- Delivery brief safety/ops checklist now includes the device-onboarding readiness gate.

Validation:

- `python -m py_compile askme\pipeline\field\field_deployment_readiness.py askme\pipeline\field\field_operations.py` -> passed.
- `python -m pytest tests\test_field_operations.py::test_readiness_payload_includes_device_onboarding_gates tests\test_field_operations.py::test_device_onboarding_payload_reports_object_binding_and_next_actions -q` -> 2 passed.
- `python -m pytest tests\test_field_deployment_readiness.py::test_field_deployment_readiness_uses_device_onboarding_report tests\test_field_deployment_readiness.py::test_field_deployment_readiness_can_be_production_ready -q` -> 2 passed.
- `python -m pytest tests\test_field_deployment_readiness.py -q` -> 12 passed.
- `python -m pytest tests\test_field_operations.py -q` -> 63 passed.

Remaining delivery risk:

- This still verifies the readiness contract in software. Site signoff still requires real customer-site camera, sensor, and robot payloads, real HMAC secrets, and acceptance artifacts archived from the deployed environment.

## 2026-05-18 Product checkpoint: device onboarding enters customer acceptance

Goal: make customer project acceptance and launch reports reflect whether real devices are onboarded, not only whether software smoke tests passed.

Implemented:

- Customer project field readiness now preserves the device-onboarding gates from Field readiness:
  `device_onboarding_report_available`, `device_onboarding_no_blockers`,
  `device_onboarding_has_ready_device`, and `device_onboarding_all_ready`.
- Customer acceptance gates now include `field_device_onboarding`, so trial reports show whether device readiness is complete, partial, or still manual.
- Launch readiness now has a separate `field_device_onboarding` gate. Production launch is not ready until all registered camera, sensor, and robot devices are observed, unblocked, fresh, and bound to customer managed objects.
- Auto-created onsite `device_ingest` evidence now requires a ready onboarded device, not only a generic trusted event archive.
- The real-link acceptance tests now seed signed demo device secrets and a fresh trusted camera event, then assert the report shows `ready=1`, `manual=3`, `blocked=0`, and keeps launch readiness at site-trial/manual until every registered device is onboarded.

Validation:

- `python -m py_compile askme\pipeline\field\customer_project_acceptance.py tests\test_field_site_profile.py` -> passed.
- `python -m pytest tests\test_field_site_profile.py::test_customer_project_acceptance_report_summarizes_delivery_gates tests\test_field_site_profile.py::test_acceptance_report_auto_backfills_required_onsite_evidence_from_real_link_reports tests\test_field_site_profile.py::test_acceptance_report_auto_backfill_is_read_only_and_idempotent -q` -> 3 passed.
- `python -m pytest tests\test_field_site_profile.py -q` -> 58 passed.
- `python -m pytest tests\test_field_deployment_readiness.py tests\test_field_operations.py -q` -> 75 passed.
- `python -m pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_field_customer_project_acceptance_routes.py -q` -> 8 passed.

Remaining delivery risk:

- This makes reports honest about device onboarding. It still does not replace real customer-site payload collection, production HMAC secret configuration, or a customer-signed acceptance dossier.
