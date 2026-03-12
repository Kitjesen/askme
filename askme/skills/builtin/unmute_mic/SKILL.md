---
name: unmute_mic
description: 重新开启麦克风监听——从静默模式恢复正常语音响应
version: 1.0.0
trigger: voice
model: ""
timeout: 5
tags: [voice, control, mute]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 开麦,打开麦克风,开启麦克风,重新开启,恢复监听
---

## Prompt

用户要求重新开启麦克风监听。

用简短的中文确认：**"好的，已重新开启。"**

不需要任何工具调用，直接输出确认语句。

用户指令：{{user_input}}
