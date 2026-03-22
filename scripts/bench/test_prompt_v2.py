"""Test prompt strategies v2 — work WITH the relay, not against it."""

import asyncio
import io
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from askme.brain.llm_client import LLMClient

MODEL = "claude-opus-4-6"
QUESTIONS = [
    "你好，你是谁？",
    "讲个笑话",
    "什么是强化学习？简单说",
    "四足机器人用什么算法好？",
]

# Strategy A: Role-play framing (cooperative, not adversarial)
SYSTEM_A = (
    "你现在扮演一个名叫Thunder的中文语音助手角色。"
    "Thunder是穹沛科技研发的机器人AI伙伴。\n"
    "输出规则（因为你的回答会通过TTS语音合成播放）：\n"
    "- 只用中文回答\n"
    "- 不用任何markdown符号，不用列表、加粗、标题\n"
    "- 像朋友聊天一样口语化\n"
    "- 每次回答控制在50字以内\n"
    "- 可以讲笑话、聊天、回答知识问题"
)

# Strategy B: Same but instructions in user preamble (system empty)
SYSTEM_B_MSGS = lambda q: [
    {"role": "user", "content": "请扮演Thunder语音助手"},
    {"role": "assistant", "content": "好的，我是Thunder，穹沛科技的语音助手。请问有什么想聊的？"},
    {"role": "user", "content": (
        f"[TTS输出模式：只用中文，不用markdown，口语化，50字以内]\n{q}"
    )},
]

# Strategy C: Strong behavior constraints without identity fight
SYSTEM_C = (
    "OUTPUT FORMAT RULES (CRITICAL — responses will be played via TTS):\n"
    "1. Chinese only. No English unless user asks.\n"
    "2. NO markdown: no **, no -, no #, no ```, no lists.\n"
    "3. Maximum 50 Chinese characters per response.\n"
    "4. Conversational tone, like chatting with a friend.\n"
    "5. You can tell jokes, chat casually, answer any question.\n"
    "6. Your name is Thunder (语音助手)."
)

# Strategy D: System + fake turn + format enforcement in user msg
SYSTEM_D = (
    "RESPONSE FORMAT: Plain Chinese text only. No markdown. Max 50 chars. "
    "Conversational tone. Your nickname is Thunder."
)


async def test_strategy(client, name, make_msgs):
    print(f"=== {name} ===")
    for q in QUESTIONS:
        msgs = make_msgs(q)
        t0 = time.monotonic()
        ttft = None
        full = ""
        try:
            async for chunk in client.chat_stream(msgs):
                d = chunk.choices[0].delta
                if d.content:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    full += d.content
        except Exception as e:
            full = f"[ERROR: {e}]"

        dur = time.monotonic() - t0
        ft = f"{ttft:.1f}" if ttft else "N/A"

        has_md = any(c in full for c in ["**", "- ", "```"])
        is_cn = not any(c in full for c in ["I'm", "I can", "help you"])
        short = len(full) <= 80
        flags = []
        if has_md: flags.append("MD!")
        if not is_cn: flags.append("EN!")
        if not short: flags.append(f"LONG({len(full)})")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""

        print(f"  Q: {q}")
        print(f"  A: {full[:120]}{'...' if len(full)>120 else ''}")
        print(f"     TTFT={ft}s Total={dur:.1f}s{flag_str}")
    print()


async def main():
    client = LLMClient(model=MODEL)
    print(f"Model: {MODEL}\n")

    await test_strategy(client, "A: roleplay_system",
        lambda q: [{"role": "system", "content": SYSTEM_A},
                    {"role": "user", "content": q}])

    await test_strategy(client, "B: fake_turn_seed",
        lambda q: SYSTEM_B_MSGS(q))

    await test_strategy(client, "C: english_format_rules",
        lambda q: [{"role": "system", "content": SYSTEM_C},
                    {"role": "user", "content": q}])

    await test_strategy(client, "D: minimal_system",
        lambda q: [{"role": "system", "content": SYSTEM_D},
                    {"role": "user", "content": q}])


if __name__ == "__main__":
    asyncio.run(main())
