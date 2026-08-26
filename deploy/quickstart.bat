@echo off
REM AskMe + LiteLLM quick deployment helper; ZeroClaw is experimental opt-in only.

setlocal DisableDelayedExpansion
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"
set "APP_ENV_FILE=docker\.env"
set "LITELLM_ENV_FILE=docker\.env.litellm"
set "RUNTIME_DIR=%REPO_ROOT%\data\runtime"
set "ASKME_PID_FILE=%REPO_ROOT%\data\runtime\askme-local.pid"
set "ZEROCLAW_PID_FILE=%REPO_ROOT%\data\runtime\zeroclaw-local.pid"

if "%1"=="docker" goto :docker
if "%1"=="docker-zeroclaw" goto :docker-zeroclaw
if "%1"=="local" goto :local
if "%1"=="local-zeroclaw" goto :local-zeroclaw
if "%1"=="stop" goto :stop
if "%1"=="setup" goto :setup
if "%1"=="" goto :menu

:menu
echo Usage: deploy\quickstart.bat [docker^|local^|docker-zeroclaw^|local-zeroclaw^|setup^|stop]
echo.
echo   docker           - Unsupported on Windows: edge Docker requires Linux /dev/snd
echo   local            - Start LiteLLM, then local AskMe
echo   docker-zeroclaw  - Unsupported on Windows: edge Docker requires Linux /dev/snd
echo   local-zeroclaw   - EXPERIMENTAL: also start local ZeroClaw without MCP integration
echo   setup            - Create LiteLLM and application env templates
echo   stop             - Stop local/Docker services
goto :end

:setup
echo [1/2] Preparing application environment template...
if not exist "%APP_ENV_FILE%" copy /Y docker\.env.example "%APP_ENV_FILE%" >nul
echo [2/2] Preparing LiteLLM sidecar environment template...
if not exist "%LITELLM_ENV_FILE%" copy /Y docker\litellm.env.example "%LITELLM_ENV_FILE%" >nul
call :require_env_files
if errorlevel 1 goto :err
echo Fill both files and follow docs\LITELLM_GATEWAY.md to generate the AskMe scoped key.
goto :end

:docker
echo [ERROR] Docker edge runtime requires Linux /dev/snd; no service was started. 1>&2
echo Use deploy/quickstart.sh on the target Linux robot host, or use local on Windows. 1>&2
goto :err

:docker-zeroclaw
echo [ERROR] Docker edge runtime requires Linux /dev/snd; no service was started. 1>&2
echo Use deploy/quickstart.sh on the target Linux robot host. 1>&2
goto :err

:local
call :load_local_environment
if errorlevel 1 goto :err
call :start_litellm
if errorlevel 1 goto :err
python -m askme.llm.key_policy
if errorlevel 1 goto :err
call :start_askme
if errorlevel 1 goto :err
echo AskMe started locally
goto :end

:local-zeroclaw
call :load_local_environment
if errorlevel 1 goto :err
call :start_litellm
if errorlevel 1 goto :err
python -m askme.llm.key_policy --require-zeroclaw
if errorlevel 1 goto :err
python scripts\dev\setup_zeroclaw.py
if errorlevel 1 goto :err
call :start_askme
if errorlevel 1 goto :err
timeout /t 5 /nobreak >nul
call :start_zeroclaw
if errorlevel 1 goto :err
echo AskMe + experimental ZeroClaw started locally; MCP integration is unavailable
goto :end

:stop
set "STOP_FAILED=0"
call :stop_tracked_process "%ZEROCLAW_PID_FILE%" "zeroclaw gateway" "ZeroClaw"
if errorlevel 1 set "STOP_FAILED=1"
call :stop_tracked_process "%ASKME_PID_FILE%" "askme.blueprints.presets.edge_robot" "AskMe"
if errorlevel 1 set "STOP_FAILED=1"
if exist "%APP_ENV_FILE%" if exist "%LITELLM_ENV_FILE%" docker compose --env-file docker\.env --env-file docker\.env.litellm -f docker\docker-compose.yml down 2>nul
if "%STOP_FAILED%"=="1" (
  echo [ERROR] One or more recorded local processes could not be stopped. 1>&2
  goto :err
)
echo Stopped
goto :end

:stop_tracked_process
setlocal DisableDelayedExpansion
set "TRACKED_PID_FILE=%~1"
set "TRACKED_PROCESS_MARKER=%~2"
set "TRACKED_PROCESS_NAME=%~3"
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"
if errorlevel 1 (
  echo [ERROR] Could not create runtime directory %RUNTIME_DIR%. 1>&2
  endlocal & exit /b 1
)
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "$pidFile = $env:TRACKED_PID_FILE; $marker = $env:TRACKED_PROCESS_MARKER; $processName = $env:TRACKED_PROCESS_NAME;" ^
  "function Get-OwnedProcess([int] $candidate, [string] $expectedMarker, [string] $expectedIdentity) {" ^
  "  $process = Get-CimInstance -ClassName Win32_Process -Filter ('ProcessId = ' + $candidate) -ErrorAction SilentlyContinue;" ^
  "  if ($null -eq $process -or [string]::IsNullOrWhiteSpace($process.CommandLine) -or [string]::IsNullOrWhiteSpace($expectedIdentity)) { return $null };" ^
  "  $creationIdentity = $process.CreationDate.ToUniversalTime().ToString('o');" ^
  "  if ($process.CommandLine.IndexOf($expectedMarker, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $creationIdentity -ceq $expectedIdentity) { return $process };" ^
  "  return $null;" ^
  "};" ^
  "$lock = $null; $lockFile = $pidFile + '.lock';" ^
  "try {" ^
  "  $lock = [IO.File]::Open($lockFile, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None);" ^
  "  if (-not (Test-Path -LiteralPath $pidFile)) { Write-Host ('[SKIP] ' + $processName + ' has no recorded local PID.'); exit 0 };" ^
  "  $record = ([string] (Get-Content -LiteralPath $pidFile -Raw)).Trim() -split '\|', 2; $trackedPid = 0;" ^
  "  $expectedIdentity = if ($record.Count -eq 2) { $record[1] } else { '' };" ^
  "  if (-not [int]::TryParse($record[0], [ref] $trackedPid) -or $trackedPid -le 4 -or $null -eq (Get-OwnedProcess $trackedPid $marker $expectedIdentity)) {" ^
  "    Remove-Item -LiteralPath $pidFile -Force;" ^
  "    Write-Host ('[SKIP] Removed stale ' + $processName + ' PID file without signaling a process.');" ^
  "    exit 0;" ^
  "  };" ^
  "  Stop-Process -Id $trackedPid -ErrorAction Stop;" ^
  "  for ($attempt = 0; $attempt -lt 50 -and $null -ne (Get-OwnedProcess $trackedPid $marker $expectedIdentity); $attempt++) { Start-Sleep -Milliseconds 100 };" ^
  "  if ($null -ne (Get-OwnedProcess $trackedPid $marker $expectedIdentity)) { Stop-Process -Id $trackedPid -Force -ErrorAction SilentlyContinue };" ^
  "  for ($attempt = 0; $attempt -lt 10 -and $null -ne (Get-OwnedProcess $trackedPid $marker $expectedIdentity); $attempt++) { Start-Sleep -Milliseconds 100 };" ^
  "  if ($null -ne (Get-OwnedProcess $trackedPid $marker $expectedIdentity)) { throw ($processName + ' PID ' + $trackedPid + ' is still running; PID file was kept.') };" ^
  "  Remove-Item -LiteralPath $pidFile -Force;" ^
  "  Write-Host ('[OK]   Stopped ' + $processName + ' PID ' + $trackedPid + '.');" ^
  "} finally { if ($null -ne $lock) { $lock.Dispose() } }"
set "TRACKED_RESULT=%errorlevel%"
endlocal & exit /b %TRACKED_RESULT%

:require_env_files
if not exist "%APP_ENV_FILE%" (
  echo [ERROR] Missing %APP_ENV_FILE%; run deploy\quickstart.bat setup first. 1>&2
  exit /b 1
)
if not exist "%LITELLM_ENV_FILE%" (
  echo [ERROR] Missing %LITELLM_ENV_FILE%; run deploy\quickstart.bat setup first. 1>&2
  exit /b 1
)
call :secure_env_file "%APP_ENV_FILE%"
if errorlevel 1 exit /b 1
call :secure_env_file "%LITELLM_ENV_FILE%"
if errorlevel 1 exit /b 1
exit /b 0

:secure_env_file
icacls "%~1" /inheritance:r /grant:r "%USERDOMAIN%\%USERNAME%:(M)" "*S-1-5-18:(F)" "*S-1-5-32-544:(F)" >nul
if errorlevel 1 (
  echo [ERROR] Could not restrict the ACL on %~1. 1>&2
  exit /b 1
)
exit /b 0

:load_env_file
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~1") do set "%%A=%%B"
exit /b 0

:load_local_environment
call :require_env_files
if errorlevel 1 exit /b 1
call :load_env_file "%LITELLM_ENV_FILE%"
if errorlevel 1 exit /b 1
call :load_env_file "%APP_ENV_FILE%"
exit /b %errorlevel%

:start_litellm
call :require_env_files
if errorlevel 1 exit /b 1
docker compose --env-file docker\.env.litellm -f docker\docker-compose.litellm.yml up -d --wait litellm
exit /b %errorlevel%

:start_zeroclaw
if not defined ZEROCLAW_LITELLM_VIRTUAL_KEY (
  echo [ERROR] ZEROCLAW_LITELLM_VIRTUAL_KEY is required for experimental ZeroClaw. 1>&2
  exit /b 1
)
setlocal DisableDelayedExpansion
set "ZEROCLAW_API_KEY=%ZEROCLAW_LITELLM_VIRTUAL_KEY%"
set "TRACKED_PID_FILE=%ZEROCLAW_PID_FILE%"
set "TRACKED_PROCESS_MARKER=zeroclaw gateway"
set "TRACKED_EXECUTABLE=zeroclaw"
set "TRACKED_ARGUMENTS=gateway --host 127.0.0.1 --port 8080"
call :start_tracked_process
set "TRACKED_RESULT=%errorlevel%"
endlocal & exit /b %TRACKED_RESULT%

:start_askme
setlocal DisableDelayedExpansion
set "TRACKED_PID_FILE=%ASKME_PID_FILE%"
set "TRACKED_PROCESS_MARKER=askme.blueprints.presets.edge_robot"
set "TRACKED_EXECUTABLE=python"
set "TRACKED_ARGUMENTS=-m askme.blueprints.presets.edge_robot"
call :start_tracked_process
set "TRACKED_RESULT=%errorlevel%"
endlocal & exit /b %TRACKED_RESULT%

:start_tracked_process
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"
if errorlevel 1 (
  echo [ERROR] Could not create runtime directory %RUNTIME_DIR%. 1>&2
  exit /b 1
)
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "$pidFile = $env:TRACKED_PID_FILE; $marker = $env:TRACKED_PROCESS_MARKER;" ^
  "function Get-MarkedProcess([int] $candidate, [string] $expectedMarker) {" ^
  "  $process = Get-CimInstance -ClassName Win32_Process -Filter ('ProcessId = ' + $candidate) -ErrorAction SilentlyContinue;" ^
  "  if ($null -ne $process -and -not [string]::IsNullOrWhiteSpace($process.CommandLine) -and $process.CommandLine.IndexOf($expectedMarker, [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $process };" ^
  "  return $null;" ^
  "};" ^
  "function Get-OwnedProcess([int] $candidate, [string] $expectedMarker, [string] $expectedIdentity) {" ^
  "  $process = Get-MarkedProcess $candidate $expectedMarker;" ^
  "  if ($null -ne $process -and -not [string]::IsNullOrWhiteSpace($expectedIdentity) -and $process.CreationDate.ToUniversalTime().ToString('o') -ceq $expectedIdentity) { return $process };" ^
  "  return $null;" ^
  "};" ^
  "$lock = $null; $lockFile = $pidFile + '.lock';" ^
  "try {" ^
  "  $lock = [IO.File]::Open($lockFile, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None);" ^
  "  if (Test-Path -LiteralPath $pidFile) {" ^
  "    $record = ([string] (Get-Content -LiteralPath $pidFile -Raw)).Trim() -split '\|', 2; $trackedPid = 0;" ^
  "    $expectedIdentity = if ($record.Count -eq 2) { $record[1] } else { '' };" ^
  "    if ([int]::TryParse($record[0], [ref] $trackedPid) -and $trackedPid -gt 4 -and $null -ne (Get-OwnedProcess $trackedPid $marker $expectedIdentity)) { Write-Host ('[SKIP] Local process is already running with PID ' + $trackedPid + '.'); exit 0 };" ^
  "    Remove-Item -LiteralPath $pidFile -Force;" ^
  "  };" ^
  "  $child = Start-Process -FilePath $env:TRACKED_EXECUTABLE -ArgumentList $env:TRACKED_ARGUMENTS -WorkingDirectory $env:REPO_ROOT -WindowStyle Hidden -PassThru;" ^
  "  $markedChild = $null;" ^
  "  for ($attempt = 0; $attempt -lt 10 -and $null -eq $markedChild; $attempt++) { $markedChild = Get-MarkedProcess $child.Id $marker; if ($null -eq $markedChild) { Start-Sleep -Milliseconds 50 } };" ^
  "  if ($null -eq $markedChild) { Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue; throw ('Could not verify local process PID ' + $child.Id + '.') };" ^
  "  $creationIdentity = $markedChild.CreationDate.ToUniversalTime().ToString('o');" ^
  "  $tempFile = $pidFile + '.' + [guid]::NewGuid().ToString('N') + '.tmp';" ^
  "  try {" ^
  "    [IO.File]::WriteAllText($tempFile, ([string] $child.Id) + '|' + $creationIdentity + [Environment]::NewLine);" ^
  "    Move-Item -LiteralPath $tempFile -Destination $pidFile -Force;" ^
  "  } catch {" ^
  "    Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue;" ^
  "    Remove-Item -LiteralPath $tempFile -Force -ErrorAction SilentlyContinue;" ^
  "    throw;" ^
  "  };" ^
  "  Write-Host ('[OK]   Started local process with PID ' + $child.Id + '.');" ^
  "} finally { if ($null -ne $lock) { $lock.Dispose() } }"
exit /b %errorlevel%

:err
echo Deployment failed. Check the error above.
exit /b 1

:end
endlocal
