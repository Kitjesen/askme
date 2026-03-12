---
name: stop_speaking
description: 立即停止TTS语音播放——清空播放队列，助手停止说话
version: 1.0.0
trigger: voice
model: ""
timeout: 5
tags: [voice, control, tts]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 静音,别说了,不说了,停止说话,停止播放,不要说了
---

## Prompt

用户要求停止当前语音播放。

用简短的中文确认：**"好的。"**

不需要任何工具调用，直接输出确认语句。

用户指令：{{user_input}}
