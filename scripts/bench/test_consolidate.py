#!/usr/bin/env python3
"""Test memory consolidation on S100P."""
import asyncio
import os
from askme.memory.robotmem_backend import RobotMemBackend


class SimpleLLMClient:
    """Minimal LLM client for consolidation test."""

    def __init__(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url="https://api.minimax.chat/v1",
        )

    async def async_chat_completion(self, messages):
        resp = await self._client.chat.completions.create(
            model="MiniMax-M2.7-highspeed",
            messages=messages,
        )
        return resp.choices[0].message.content


async def main():
    backend = RobotMemBackend(
        mem_cfg={"robotmem_collection": "askme_consolidate_test", "retrieve_timeout": 5.0},
        brain_cfg={},
    )
    await backend.warmup()
    print(f"Backend ready: {backend.available}")

    # Save some raw conversations
    conversations = [
        ("仓库A温度传感器报警了", "收到，立刻检查仓库A温度传感器。"),
        ("温度传感器校准完成了", "好的，已记录温度传感器恢复正常。"),
        ("3号巡检点发现漏水", "安排人员去3号巡检点检查。"),
        ("操作员要求增加仓库A巡检频率", "已记录，巡检频率将调整。"),
        ("仓库B一切正常", "收到，仓库B状态正常。"),
    ]
    for user, assistant in conversations:
        await backend.save(user, assistant)
    print(f"Saved {len(conversations)} raw conversations")

    # Consolidate
    llm = SimpleLLMClient()
    n = await backend.consolidate(llm, batch_size=10)
    print(f"Consolidated: {n} facts extracted")

    # Verify consolidated facts are searchable
    for q in ["温度", "漏水", "巡检频率"]:
        result = await backend.retrieve(q)
        lines = [l for l in result.split("\n") if l.strip()] if result else []
        print(f"  [{q}]: {len(lines)} results")
        for l in lines[:2]:
            print(f"    {l[:70]}")

    backend.close()
    print("\nConsolidation E2E passed!")


asyncio.run(main())
