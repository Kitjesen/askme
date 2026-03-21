"""Time tool — decorator-based implementation.

Replaces the old GetTimeTool(BaseTool) class with a single @tool function.
This is the first tool migrated to the decorator architecture as a proof
of concept.
"""

import datetime

from askme.core.decorators import tool


@tool(
    name="get_current_time",
    description="获取当前系统时间",
    agent_allowed=True,
    voice_label="获取时间",
)
def get_current_time() -> str:
    """返回当前系统日期和时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
