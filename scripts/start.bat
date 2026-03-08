@echo off
REM Start askme MCP server
REM Usage:
REM   scripts\start.bat                             -- stdio (default)
REM   scripts\start.bat --transport sse --port 8080 -- SSE mode
REM   scripts\start.bat --legacy --text             -- legacy CLI
cd /d %~dp0..
python -m askme %*
