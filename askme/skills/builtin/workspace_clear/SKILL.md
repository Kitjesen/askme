---
name: workspace_clear
description: 清空 agent 工作区（data/agent_workspace/）中的所有文件
version: 1.0.0
trigger: voice
model: ""
timeout: 20
tags: [agent, workspace, cleanup]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 清空工作区,删除工作区文件,清理工作区,工作区清空,清除工作区,工作区太乱了
---

## Tools

bash
list_directory

## Prompt

用户想清空 agent 工作区（data/agent_workspace/）中的所有文件。

步骤：
1. 先用 list_directory 查看 data/agent_workspace 目录内容
2. 如果目录为空或没有文件，直接回复"工作区已经是空的，没有文件需要清理"
3. 如果有文件，用 bash 执行以下命令删除所有文件（保留子目录结构）：
   `find . -maxdepth 3 -type f -delete`
4. 再次 list_directory 确认清理结果
5. 告知用户清理完成，例如"已清空工作区，删除了之前保存的所有文件"

注意：只删除文件，不删除目录结构；bash 工作区已沙箱隔离，操作安全。

用户指令：{{user_input}}
