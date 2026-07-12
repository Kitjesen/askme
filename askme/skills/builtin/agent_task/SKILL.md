---
name: agent_task
description: 旧后台专家任务兼容入口——研究/写脚本/分析数据/自动化/查资料需转入 MCP/ZeroClaw 受控接入
version: 2.0.0
trigger: voice
timeout: 120
tags: [agentic, research, automation, coding]
depends: []
conflicts: []
safety_level: dangerous
execution: agent_shell
voice_trigger: 帮我研究,研究一下,写一个脚本,写个脚本,写一段代码,能不能写,帮我分析,能帮我分析,查资料,自主完成,帮我写代码,帮我写个,执行复杂任务,写脚本,分析数据,帮我调查,做个自动化,自动化处理,数据分析,帮我配置,帮我整理,做一个工具,帮我规划,帮我做一个,写个程序,帮我查一下,帮我搜一下,跑一个脚本,帮我跑,执行脚本,写段代码,写点代码,帮我测试,帮我验证,自动完成,帮我优化,帮我创建,创建一个,创建技能,新建技能,新建一个,帮我新建
---

## Tools

bash
write_file
edit_file
read_file
list_directory
http_request
robot_api
get_current_time
speak_progress
web_search
web_fetch
spawn_agent
create_skill

## Prompt

此技能保留为旧后台专家任务兼容入口，不再代表 Askme 本地可调用的自主执行循环。
真实多步 agent 决策应通过 MCP/ZeroClaw 受控接入，再进入 TaskHandoff、SafetyPreflight 和 runtime arbiter。
本 Prompt 节不直接使用，但 voice_trigger 和 safety_level 由 SkillManager 读取用于路由和确认。

任务：{{user_input}}
