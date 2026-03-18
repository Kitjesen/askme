---
name: list_skills
description: 列出当前已加载的所有语音技能及其触发词
version: 1.0.0
trigger: voice
model: ""
timeout: 30
tags: [utility, meta]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 列出技能,有哪些技能,技能列表,帮我看看技能,你的技能,你的功能,能力清单,功能介绍,有哪些功能,帮我列出技能
---

## Tools

list_directory

## Prompt

用户想知道你能做什么，或者有哪些已加载的技能。

用 list_directory 工具列出 askme/skills/builtin/ 目录的子目录（每个子目录是一个技能）。

然后用中文口语告诉用户：
1. 你当前有哪些主要技能（按功能分组说，不要逐条念）
2. 用几句话概括每类能力

示例回答风格：
"我现在主要有这几类能力：
一是机器人控制，比如让它站立、巡逻、回充；
二是自主任务，你让我研究什么或写代码，我可以自己上网查、写脚本；
三是文件操作，查看目录、读文件；
四是实时查询，比如查导航状态、查传感器数据。
有什么具体需要吗？"

保持口语化，不要用 markdown 或条目列表。
