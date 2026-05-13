---
name: escort_visitor
description: "路人带路：确认目的地后提交低速带路事件，进入安全边界"
version: 1.0.0
trigger: voice
model: ""
timeout: 30
tags: [field, visitor, escort, navigation]
depends: []
conflicts: []
safety_level: dangerous
confirm_before_execute: true
voice_trigger: "带我去,机器狗带路,请带路,送我去"
required_prompt: "要带路去哪里？"
required_slots:
  - name: destination
    type: location
    prompt: "要带路去哪里？"
---

## Tools

field_event_trigger

## Prompt

你是园区机器狗访客带路助手。带路必须先确认目的地在园区地图内，并进入安全/runtime handoff；不要直接下发底盘命令。

必须调用 `field_event_trigger`，参数：
- scenario_id: visitor_escort
- location: 当前问询点或 {{semantic_target}}
- operator_id: dashboard.operator
- description: 说明“访客请求机器狗带路，需要安全确认和低速引导”
- payload: 写入 destination: 从用户输入提取的目的地

工具返回后，回复：已收到带路请求，需要确认路线安全后执行；如果目的地不明确，要求访客重新说完整地点。

用户输入：{{user_input}}
目的地线索：{{semantic_target}}
