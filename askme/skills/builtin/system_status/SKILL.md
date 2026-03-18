---
name: system_status
description: 查看机器人系统状态：CPU、内存、磁盘、运行时间
version: 1.0.0
trigger: voice
model: ""
timeout: 15
tags: [utility, system, monitoring]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 系统状态,内存多少,磁盘空间,cpu占用,服务器状态,系统资源,内存够吗,存储空间,硬盘多少,运行多久了,运行时间,系统负载
---

## Tools

bash

## Prompt

用户想了解机器人系统当前的运行状态。

用 bash 工具依次执行以下只读命令（不会改变任何系统状态）：

1. `uptime` — 获取运行时间和系统负载
2. `free -h` — 获取内存使用情况（如果不可用，跳过）
3. `df -h / /data 2>/dev/null | head -5` — 获取根目录和数据目录磁盘占用

将三个命令的输出整合，用口语中文告诉用户：
- 系统已运行多久
- 内存剩余多少（总量/已用/可用）
- 磁盘占用情况

示例：
"系统已经运行了2天3小时。内存总共8GB，目前用了3.2GB，还剩4.8GB。
根目录磁盘用了60%，数据目录用了45%，空间还比较充裕。"

保持口语化简洁，不用 markdown，数字保留1位小数。
