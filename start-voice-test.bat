@echo off
chcp 65001 >nul 2>&1
title Askme Voice Test - Thunder

echo ============================================
echo   Askme Voice Test - Thunder
echo ============================================
echo.

cd /d "%~dp0"

echo [INFO] Starting askme voice pipeline...
echo [INFO] Log file: dev-loop-main.log
echo [INFO] Speak after "Listening for speech..." appears
echo [INFO] Press Ctrl+C to stop
echo.

python -m askme --legacy

pause
