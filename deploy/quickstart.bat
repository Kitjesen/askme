@echo off
REM Askme + ZeroClaw quick deployment helper for Windows.

setlocal
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"

if "%1"=="docker" goto :docker
if "%1"=="local" goto :local
if "%1"=="stop" goto :stop
if "%1"=="setup" goto :setup
if "%1"=="" goto :menu

:menu
echo Usage: deploy\quickstart.bat [docker^|local^|setup^|stop]
echo.
echo   docker   - Start Askme + ZeroClaw with Docker Compose
echo   local    - Start local Askme runtime and ZeroClaw gateway
echo   setup    - Create docker\.env from local Askme config
echo   stop     - Stop local/Docker services
goto :end

:setup
echo [1/3] Configuring ZeroClaw API key...
python scripts\dev\setup_zeroclaw.py
if %errorlevel% neq 0 goto :err

echo [2/3] Creating Docker .env file...
python -c "from askme.config import get_config; c=get_config(); b=c['brain'] if isinstance(c,dict) else c.brain; k=b.get('minimax_api_key','') if isinstance(b,dict) else getattr(b,'minimax_api_key',''); open('docker/.env','w',encoding='utf-8').write(f'MINIMAX_API_KEY={k}\n')"
if %errorlevel% neq 0 goto :err

echo [3/3] Done. Run: deploy\quickstart.bat docker
goto :end

:docker
docker compose --env-file docker\.env -f docker\docker-compose.yml up -d
if %errorlevel% neq 0 goto :err
echo Askme:    http://localhost:8765
echo ZeroClaw: http://localhost:8080
goto :end

:local
start "Askme" python -m askme.blueprints.presets.edge_robot
timeout /t 5 /nobreak >nul
start "ZeroClaw" zeroclaw gateway --host 127.0.0.1 --port 8080
echo Services started locally
goto :end

:stop
docker compose --env-file docker\.env -f docker\docker-compose.yml down 2>nul
taskkill /f /im zeroclaw.exe 2>nul
echo Stopped
goto :end

:err
echo Deployment failed. Check the error above.
exit /b 1

:end
endlocal
