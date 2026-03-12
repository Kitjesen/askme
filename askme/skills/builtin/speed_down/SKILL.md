---
name: speed_down
description: 降低TTS语速
version: 1.0.0
trigger: voice
model: ""
timeout: 5
tags: [voice, control, speed]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 说慢点,说慢一点,语速慢点,讲慢点,说话慢点,太快了,听不懂
---

## Prompt

用一句话确认已降低语速。直接输出，不需要工具。

用户指令：{{user_input}}
