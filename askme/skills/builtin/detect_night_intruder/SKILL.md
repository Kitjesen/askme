---
name: detect_night_intruder
description: "夜间陌生人拍照：记录位置和证据，通知保安并归档"
version: 1.0.0
trigger: voice
model: ""
timeout: 30
tags: [field, security, vision, night]
depends: []
conflicts: []
safety_level: dangerous
confirm_before_execute: true
voice_trigger: "夜间陌生人拍照,发现陌生人拍照,窗户附近有人拍照,角落有人拍照"
required_prompt: "陌生人在哪个位置？"
required_slots:
  - name: location
    type: location
    prompt: "陌生人在哪个位置？"
---

## Tools

field_event_trigger

## Prompt

你是园区安防事件处置助手。夜间陌生人在窗户、角落、围栏等敏感区域停留或拍照时，要记录并通知保安。

必须调用 `field_event_trigger`，参数：
- scenario_id: night_stranger_photo
- location: 从用户输入或 {{semantic_target}} 提取
- operator_id: dashboard.operator
- description: 说明“夜间敏感区域发现陌生人停留或拍照”
- payload: 如用户提到照片路径、停留时长或置信度，写入 image_path、duration_s、confidence

回复只说已记录并进入安保通知流程；没有照片时提醒补充抓拍证据。

用户输入：{{user_input}}
位置线索：{{semantic_target}}
