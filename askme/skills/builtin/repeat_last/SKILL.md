---
name: repeat_last
description: 重复上一条语音回复内容
version: 1.0.0
trigger: voice
model: ""
timeout: 5
tags: [voice, control, tts]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 再说一遍,重复一遍,再说一次,你刚才说什么
---

## Prompt

用户想要重复刚才的语音回复内容。

直接回复：**"好的，再说一遍。"**

不需要工具调用。

用户指令：{{user_input}}
