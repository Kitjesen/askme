---
name: speed_reset
description: 恢复TTS语速到默认值
version: 1.0.0
trigger: voice
model: ""
timeout: 5
tags: [voice, control, speed]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 恢复语速,正常语速,语速还原,默认语速,语速恢复正常
---

## Prompt

用一句话确认已恢复默认语速。直接输出，不需要工具。

用户指令：{{user_input}}
