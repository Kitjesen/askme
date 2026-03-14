---
name: workspace_info
description: 查看 agent 工作区的文件列表
version: 1.0.0
trigger: voice
model: ""
timeout: 10
tags: [agent, workspace]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 工作区有什么,看看工作区,工作区文件,工作区有哪些文件,工作区状态
---

## Tools

list_directory

## Prompt

用户想查看 agent 工作区里有哪些文件。

步骤：
1. 用 list_directory 列出目录 "data/agent_workspace" 的内容
2. 如果目录为空，回复"工作区目前是空的"
3. 如果有文件，简洁列出文件名（最多说前10个，超出告知"还有更多文件"）

用户指令：{{user_input}}
