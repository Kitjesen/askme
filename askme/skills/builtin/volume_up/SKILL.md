---
name: volume_up
description: 调大TTS输出音量
version: 1.0.0
trigger: voice
model: ""
timeout: 5
tags: [voice, control, volume]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 声音大点,大声点,调大音量,音量大点,声音大一点,大声一点,声音太小了
---

## Prompt

用一句话确认已调大音量。直接输出，不需要工具。

用户指令：{{user_input}}
