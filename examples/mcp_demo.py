# -*- coding: utf-8 -*-
"""
Askme MCP Server — 端到端 Demo 脚本

演示如何通过 MCP 客户端 SDK 调用 askme 的所有功能：
  1. 健康检查
  2. 查看技能列表
  3. 执行技能 (get_time)
  4. 机器人状态查询
  5. 语音合成 (TTS)
  6. 语音识别 (ASR)

使用方式:
    # 先在另一个终端启动 MCP 服务器
    python -m askme --transport sse --port 8080

    # 然后运行此 demo
    python examples/mcp_demo.py
"""

import asyncio
import io
import sys

# Fix Windows console encoding for Chinese output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


BASE_URL = "http://localhost:8080"


async def demo_health():
    """1. 健康检查"""
    print("\n" + "=" * 60)
    print("1. 健康检查 (askme://health)")
    print("=" * 60)

    print("  -> MCP 资源: askme://health")
    print("  <- 返回服务器状态、版本、子系统信息")
    print()
    print("  在 Claude Desktop/Code 中，你可以直接问：")
    print('  "askme 服务器状态怎么样？"')
    print("  Claude 会自动调用 health_check 资源并展示结果")


async def demo_skills():
    """2. 查看技能列表"""
    print("\n" + "=" * 60)
    print("2. 技能目录 (askme://skills)")
    print("=" * 60)

    print("  → MCP 资源: askme://skills")
    print("  ← 返回所有可用技能的名称、描述、触发词")
    print()
    print("  内置技能:")
    print("  -get_time      — 获取当前时间 (voice: '现在几点')")
    print("  -daily_summary — 每日总结")
    print("  -run_command   — 执行系统命令 (safety: dangerous)")
    print("  -list_directory— 列出目录内容")
    print("  -web_search    — 网络搜索")
    print("  -robot_move    — 控制机械臂移动")
    print("  -robot_grab    — 机械臂抓取")
    print("  -robot_home    — 机械臂回原点")
    print("  -robot_estop   — 紧急停止")


async def demo_execute_skill():
    """3. 执行技能"""
    print("\n" + "=" * 60)
    print("3. 执行技能 — execute_skill('get_time', '现在几点了？')")
    print("=" * 60)

    print("  → MCP 工具: execute_skill")
    print("    参数: skill_name='get_time', user_input='现在几点了？'")
    print()
    print("  执行流程:")
    print("  1. SkillManager 加载 SKILL.md 模板")
    print("  2. 注入变量: {{user_input}}, {{current_time}}")
    print("  3. SkillExecutor 调用 DeepSeek API")
    print("  4. 如果技能定义了 Tools，LLM 可调用对应工具")
    print("  5. 返回最终回复文本")
    print()
    print("  在 Claude Desktop 中使用:")
    print('  "帮我执行 get_time 技能"')
    print('  "现在几点了？" (如果配了 voice_trigger 自动匹配)')


async def demo_robot():
    """4. 机器人操作"""
    print("\n" + "=" * 60)
    print("4. 机器人控制")
    print("=" * 60)

    print("  MCP 工具 (7 个):")
    print()
    print("  robot_state()              — 查看当前状态")
    print("  robot_move(x, y, z)        — 移动到坐标 (mm)")
    print("  robot_pick(target)         — 抓取物体")
    print("  robot_place(location)      — 放置物体")
    print("  robot_home()               — 回初始位置")
    print("  robot_wave()               — 挥手")
    print("  robot_estop()              — 紧急停止")
    print()
    print("  MCP 资源 (3 个):")
    print("  robot://status             — 连接状态")
    print("  robot://joint/{id}/state   — 关节信息 (0-15)")
    print("  robot://safety/config      — 安全配置")
    print()
    print("  在 Claude Desktop 中使用:")
    print('  "机械臂现在什么状态？"')
    print('  "把机械臂移动到 (100, 200, 50)"')
    print('  "紧急停止！"')


async def demo_voice():
    """5. 语音 I/O"""
    print("\n" + "=" * 60)
    print("5. 语音输入/输出")
    print("=" * 60)

    print("  MCP 工具 (2 个):")
    print()
    print("  voice_listen()             — 打开麦克风 → VAD → ASR → 返回文字")
    print("    -100ms 块读取")
    print("    -VAD 检测语音活动")
    print("    -检测到语音端点后返回识别文字")
    print("    -30 秒超时")
    print()
    print("  voice_speak(text)          — TTS 合成 → 播放")
    print("    -MiniMax API 流式合成")
    print("    -MP3 解码 → float32 PCM")
    print("    -sounddevice 播放")
    print("    -自动去除 emoji/markdown")
    print()
    print("  完整语音对话流程 (Claude Desktop):")
    print("  1. Claude 调用 voice_listen() → 用户说话 → 返回文字")
    print('  2. Claude 看到文字，思考回复')
    print("  3. Claude 调用 voice_speak('你好！') → 扬声器播放")
    print()
    print("  在 Claude Desktop 中使用:")
    print('  "听我说话" → Claude 调用 voice_listen()')
    print('  "把这段话说出来：今天天气真好" → Claude 调用 voice_speak()')


async def demo_full_flow():
    """6. 完整对话流程"""
    print("\n" + "=" * 60)
    print("6. 完整端到端流程示例")
    print("=" * 60)

    print("""
  ┌──────────────┐    MCP     ┌──────────────┐
  │ Claude       │◄──────────►│ Askme MCP    │
  │ (大脑/思考)   │  JSON-RPC  │ Server       │
  │              │            │              │
  │ 1. 调用      │────────────►│ voice_listen │ → 麦克风 → ASR
  │    voice_    │            │              │
  │    listen()  │◄────────────│ "你好小智"   │
  │              │            │              │
  │ 2. LLM 思考  │            │              │
  │    生成回复   │            │              │
  │              │            │              │
  │ 3. 调用      │────────────►│ voice_speak  │ → TTS → 扬声器
  │    voice_    │            │ ("你好！")    │
  │    speak()   │◄────────────│ "[Spoken]"   │
  │              │            │              │
  │ 4. 需要技能？ │────────────►│ execute_     │ → SKILL.md
  │              │            │ skill()      │ → DeepSeek API
  │              │◄────────────│ "现在3:14PM" │
  │              │            │              │
  │ 5. 需要机器人 │────────────►│ robot_move   │ → 串口 → 机械臂
  │              │            │ (100,200,50) │
  │              │◄────────────│ {"status":"ok"}│
  └──────────────┘            └──────────────┘

  关键点:
  -Claude 是"大脑" — 决定何时听、何时说、何时调用技能
  -Askme 是"能力服务" — 提供语音、机器人、技能执行
  -通过 MCP 协议连接，Claude 自动发现并使用这些工具
""")


async def main():
    print("============================================================")
    print("           Askme MCP Server -- End-to-End Guide")
    print()
    print("  Quick start:")
    print("  1. cp .env.example .env   (fill in API keys)")
    print("  2. pip install -e .")
    print("  3. python -m askme        (start MCP server)")
    print("  4. Use in Claude Desktop/Code")
    print("============================================================")

    await demo_health()
    await demo_skills()
    await demo_execute_skill()
    await demo_robot()
    await demo_voice()
    await demo_full_flow()

    print("\n" + "=" * 60)
    print("部署方式")
    print("=" * 60)
    print("""
  本地 (Claude Desktop/Code):
    python -m askme                  # stdio 模式

  网络 (SSE):
    python -m askme --transport sse --port 8080

  Docker:
    docker build -t askme:4.0 .
    docker run --env-file .env -p 8080:8080 askme:4.0

  Docker Compose:
    docker compose up -d

  自定义配置:
    python -m askme --config /path/to/config.yaml

  功能标志:
    ASKME_FEATURE_ROBOT=0 python -m askme   # 禁用机器人
    ASKME_FEATURE_VOICE=0 python -m askme   # 禁用语音

  Legacy CLI (独立运行，不需要 Claude):
    python -m askme --legacy --text         # 文本模式
    python -m askme --legacy                # 语音模式
    python -m askme --legacy --robot        # 带机器人
""")


if __name__ == "__main__":
    asyncio.run(main())
