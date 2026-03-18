---
name: list_directory
description: 列出指定目录中的文件和文件夹
version: 1.0.0
trigger: auto
model: ""
timeout: 10
tags: [utility, filesystem]
depends: []
conflicts: []
safety_level: normal
voice_trigger: 看看文件,文件列表,查看目录,目录下有什么,列出文件,有哪些文件,文件有哪些,查一下文件,目录文件,看看有什么文件
---

## Tools

list_directory
read_file

## Prompt

你是一个文件浏览助手。用户想查看目录内容或文件信息。

请根据用户的描述，使用 list_directory 工具列出目录内容。如果用户提到了具体文件，也可以用 read_file 工具查看文件内容。

用中文简洁地描述目录结构或文件内容。

目标路径：{{path}}
用户的请求：{{user_input}}
