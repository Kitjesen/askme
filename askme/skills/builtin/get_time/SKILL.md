---
name: get_time
description: 获取当前系统时间并以自然语言回复
version: 1.0.0
trigger: auto
model: ""
timeout: 10
tags: [utility, time]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 现在几点,几点了,什么时间,星期几,今天几号,几月几号,今天日期
---

## Tools

get_current_time

## Prompt

你是一个时间助手。用户想知道当前时间。

请调用 get_current_time 工具获取系统时间，然后用自然口语化的中文回答用户。

例如：现在是下午三点二十分。

用户说的是：{{user_input}}
