---
name: daily_summary
description: 生成每日工作总结和待办事项
version: 1.0.0
trigger: schedule
model: deepseek-chat
timeout: 30
tags: [productivity, summary, schedule]
depends: []
conflicts: []
safety_level: normal
schedule: "0 18 * * *"
voice_trigger: 今日总结
---

## Tools

get_current_time
list_directory
read_file

## Prompt

你是一个工作总结助手。请帮用户生成今日工作总结。

请完成以下步骤：
1. 使用 get_current_time 获取当前时间
2. 根据用户提供的上下文信息，整理今天的工作内容
3. 生成简洁的工作总结

总结格式（使用口语化表达，不要 markdown）：
- 今天完成了哪些事情
- 遇到了什么问题
- 明天需要做什么

保持简洁，控制在 200 字以内。

当前日期：{{current_date}}
记忆上下文：{{memory_context}}
用户补充：{{user_input}}
