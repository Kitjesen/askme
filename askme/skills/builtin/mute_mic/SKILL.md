---
name: mute_mic
description: 关闭麦克风监听——助手进入静默模式，不再响应语音指令直到说"开麦"
version: 1.0.0
trigger: voice
model: ""
timeout: 5
tags: [voice, control, mute]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 闭麦,关闭麦克风,关麦,闭嘴停止,进入静默
---

## Prompt

用户要求关闭麦克风监听。

用简短的中文确认：**"好的，已关闭麦克风。说"开麦"来重新打开。"**

不需要任何工具调用，直接输出确认语句。

用户指令：{{user_input}}
