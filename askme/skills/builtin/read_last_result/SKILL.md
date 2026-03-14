---
name: read_last_result
description: 读取上次 agent 任务的完整结果文件并朗读
version: 1.0.0
trigger: voice
model: ""
timeout: 15
tags: [agent, workspace, result]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 读上次结果,上次的结果,完整结果是什么,结果文件,读工作区结果,上次任务结果
---

## Tools

read_file

## Prompt

用户想听上次 agent 任务保存的完整结果。

步骤：
1. 用 read_file 读取文件 "data/agent_workspace/last_result.txt"
2. 如果文件不存在或为空，回复"还没有保存过任务结果"
3. 如果文件有内容，朗读其内容（控制在200字以内，超出部分提示"内容较长，已省略后续"）

用户指令：{{user_input}}
