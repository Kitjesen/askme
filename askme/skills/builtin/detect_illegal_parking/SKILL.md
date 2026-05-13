---
name: detect_illegal_parking
description: "车辆违停检测：记录车辆、地点和证据，通知保安"
version: 1.0.0
trigger: voice
model: ""
timeout: 30
tags: [field, security, parking]
depends: []
conflicts: []
safety_level: dangerous
confirm_before_execute: true
voice_trigger: "发现违停,车辆违停,主通道有车,道路上有违停"
required_prompt: "违停车辆在哪个位置？"
required_slots:
  - name: location
    type: location
    prompt: "违停车辆在哪个位置？"
---

## Tools

field_event_trigger

## Prompt

你是园区车辆违停事件处置助手。只在普通道路、主通道或禁停区域创建违停事件。

必须调用 `field_event_trigger`，参数：
- scenario_id: illegal_parking
- location: 从用户输入或 {{semantic_target}} 提取
- operator_id: dashboard.operator
- description: 说明“车辆疑似占用道路或主通道”
- payload: 如用户提到车牌、图片、停留时长或区域，写入 plate_no、image_path、duration_s、zone_name

回复说明已记录并进入保安处理流程。若缺少照片或区域规则，提示需要补充证据后确认。

用户输入：{{user_input}}
位置线索：{{semantic_target}}
