---
name: volume_down
description: 调小TTS输出音量
version: 1.0.0
trigger: voice
model: ""
timeout: 5
tags: [voice, control, volume]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 声音小点,小声点,调小音量,音量小点,声音小一点,小声一点,声音太大了,声音太吵了
---

## Prompt

用一句话确认已调小音量。直接输出，不需要工具。

用户指令：{{user_input}}
