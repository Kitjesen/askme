---
name: volume_reset
description: 恢复TTS输出音量到默认值
version: 1.0.0
trigger: voice
model: ""
timeout: 5
tags: [voice, control, volume]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 恢复音量,正常音量,音量还原,默认音量,音量恢复正常
---

## Prompt

用一句话确认已恢复默认音量。直接输出，不需要工具。

用户指令：{{user_input}}
