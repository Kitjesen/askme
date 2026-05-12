@echo off
REM Start askme MCP server
REM Usage:
REM   scripts\dev\start.bat                             -- stdio (default)
REM   scripts\dev\start.bat --transport sse --port 8080 -- SSE mode
REM   scripts\dev\start.bat --legacy --text             -- legacy CLI
cd /d %~dp0..\..
python -m askme %*
