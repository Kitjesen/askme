---
name: agent_task
description: 自主完成复杂任务——研究/写脚本/分析数据/自动化/查资料（调用 ThunderAgentShell 执行）
version: 2.0.0
trigger: voice
timeout: 120
tags: [agentic, research, automation, coding]
depends: []
conflicts: []
safety_level: dangerous
voice_trigger: 帮我研究,研究一下,写一个脚本,写个脚本,写一段代码,能不能写,帮我分析,能帮我分析,查资料,自主完成,帮我写代码,帮我写个,执行复杂任务,写脚本,分析数据,帮我调查,做个自动化,自动化处理,数据分析,帮我配置,帮我整理,做一个工具,帮我规划,帮我做一个,写个程序,帮我查一下,帮我搜一下,跑一个脚本,帮我跑,执行脚本,写段代码,写点代码,帮我测试,帮我验证,自动完成,帮我优化
---

## Tools

bash
write_file
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

此技能路由至 ThunderAgentShell.run_task()，系统提示词由 _build_agent_system_prompt() 动态生成。
本 Prompt 节不直接使用，但 voice_trigger 和 safety_level 由 SkillManager 读取用于路由和确认。

任务：{{user_input}}
