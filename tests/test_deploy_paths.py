from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUNRISE_ASKME_DIR = "/home/sunrise/data/inovxio/askme"
LEGACY_ASKME_DIR = "/home/sunrise/askme"


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_production_service_defaults_to_sunrise_data_path() -> None:
    service = read("deploy/askme.service")

    assert f"WorkingDirectory={SUNRISE_ASKME_DIR}" in service
    assert f"Environment=ASKME_DIR={SUNRISE_ASKME_DIR}" in service
    assert "EnvironmentFile=-/etc/default/askme" in service
    assert "askme.blueprints.presets.edge_robot" in service
    assert LEGACY_ASKME_DIR not in service


def test_sunrise_service_files_default_to_sunrise_data_path() -> None:
    service_files = [
        "scripts/runtime/services/askme.service",
        "scripts/runtime/services/askme-frame-daemon.service",
        "scripts/runtime/services/brainstem-ros2-bridge.service",
        "scripts/runtime/services/rerun-bridge.service",
    ]

    for relpath in service_files:
        service = read(relpath)
        assert f"Environment=ASKME_DIR={SUNRISE_ASKME_DIR}" in service
        assert "EnvironmentFile=-/etc/default/askme" in service
        assert LEGACY_ASKME_DIR not in service


def test_deploy_scripts_default_to_sunrise_data_path_with_env_override() -> None:
    install = read("deploy/install.sh")
    sync = read("scripts/dev/sync_sunrise.sh")
    agentic = read("scripts/dev/deploy_agentic_shell.sh")

    assert f'ASKME_DIR="${{ASKME_DIR:-{SUNRISE_ASKME_DIR}}}"' in install
    assert f'REMOTE_DIR="${{REMOTE_DIR:-{SUNRISE_ASKME_DIR}}}"' in sync
    assert f'RPATH="${{RPATH:-{SUNRISE_ASKME_DIR}}}"' in agentic


def test_sync_sunrise_never_pushes_remote_secrets_or_device_config() -> None:
    sync = read("scripts/dev/sync_sunrise.sh")

    assert "--exclude='.env'" in sync
    assert "--exclude='config.yaml'" in sync
    assert "for f in pyproject.toml requirements.txt README.md" in sync
    assert "$LOCAL_DIR/prompts/SOUL.md" in sync
    assert "$REMOTE_DIR/prompts/SOUL.md" in sync
    assert "config.yaml" not in "pyproject.toml requirements.txt README.md"


def test_deploy_surface_documents_assets_and_keeps_helpers_portable() -> None:
    readme = read("deploy/README.md")
    quickstart_sh = read("deploy/quickstart.sh")
    quickstart_bat = read("deploy/quickstart.bat")

    for token in (
        "`askme.service`",
        "`install.sh`",
        "`quickstart.sh`",
        "`quickstart.bat`",
        "`site-profiles/`",
        "`customer-project-templates/`",
        "`delivery-resources/`",
        "`security/`",
    ):
        assert token in readme

    assert "docker/docker-compose.yml" in quickstart_sh
    assert "docker\\docker-compose.yml" in quickstart_bat
    assert "D:\\inovxio" not in quickstart_bat
    assert "%~dp0.." in quickstart_bat


def test_quickstart_uses_two_stage_litellm_environment_without_direct_llm_keys() -> None:
    for relpath in ("deploy/quickstart.sh", "deploy/quickstart.bat"):
        script = read(relpath)
        normalized = script.replace("\\", "/").lower()

        assert "docker/.env.litellm" in normalized
        assert "--env-file docker/.env --env-file docker/.env.litellm" in normalized
        assert "litellm.env.example" in normalized
        assert "minimax_api_key" not in normalized
        assert "brain.get(" not in normalized


def test_quickstart_waits_for_litellm_and_loads_local_environments_before_consumers() -> None:
    shell = read("deploy/quickstart.sh").replace("\\", "/")
    batch = read("deploy/quickstart.bat").replace("\\", "/").lower()

    for script in (shell, batch):
        assert "docker/docker-compose.litellm.yml" in script
        assert "up -d --wait litellm" in script
        assert "load_env_file" in script
        assert "docker/.env.litellm" in script
        assert "docker/.env" in script
        assert "askme.llm.key_policy" in script
        assert "--require-zeroclaw" in script
        assert "docker-zeroclaw" in script
        assert "local-zeroclaw" in script

    assert "experimental-zeroclaw" in shell
    assert "experimental-zeroclaw" not in batch
    assert shell.count("start_litellm") >= 3
    assert batch.count("start_litellm") >= 3

    shell_docker = shell.split("  docker)", 1)[1].split("  docker-zeroclaw)", 1)[0]
    shell_local = shell.split("  local)", 1)[1].split("  local-zeroclaw)", 1)[0]
    batch_docker = batch.split("\n:docker\n", 1)[1].split("\n:docker-zeroclaw\n", 1)[0]
    batch_local = batch.split("\n:local\n", 1)[1].split("\n:local-zeroclaw\n", 1)[0]

    for default_path in (shell_docker, shell_local, batch_docker, batch_local):
        assert "zeroclaw gateway" not in default_path
        assert "--require-zeroclaw" not in default_path


def test_local_quickstarts_use_repo_scoped_pid_lifecycle() -> None:
    shell = read("deploy/quickstart.sh").replace("\\", "/")
    batch = read("deploy/quickstart.bat").replace("\\", "/").lower()

    for script in (shell, batch):
        assert "data/runtime" in script
        assert "askme-local.pid" in script
        assert "zeroclaw-local.pid" in script
        assert "start_askme" in script
        assert "start_tracked_process" in script

    shell_local = shell.split("  local)", 1)[1].split("  local-zeroclaw)", 1)[0]
    batch_local = batch.split("\n:local\n", 1)[1].split(
        "\n:local-zeroclaw\n",
        1,
    )[0]
    assert "python -m askme.blueprints.presets.edge_robot &" not in shell_local
    assert 'start "askme" python -m askme.blueprints.presets.edge_robot' not in batch_local
    assert "start_askme" in shell_local
    assert "start_askme" in batch_local


def test_local_quickstart_stop_only_targets_verified_recorded_processes() -> None:
    shell = read("deploy/quickstart.sh")
    batch = read("deploy/quickstart.bat").lower()

    assert "pkill -f" not in shell
    assert "process_matches" in shell
    assert 'kill -TERM "$tracked_pid"' in shell
    assert shell.count("stop_tracked_process") >= 3

    assert "taskkill /f /im" not in batch
    assert "get-ciminstance -classname win32_process" in batch
    assert "[int]::tryparse" in batch
    assert "stop-process -id $trackedpid" in batch
    assert batch.count("stop_tracked_process") >= 3

    shell_stop = shell.split("  stop)", 1)[1].split("  *)", 1)[0]
    batch_stop = batch.split("\n:stop\n", 1)[1].split("\n:require_env_files\n", 1)[0]
    for stop_block in (shell_stop, batch_stop):
        assert "ASKME_PID_FILE".lower() in stop_block.lower()
        assert "ZEROCLAW_PID_FILE".lower() in stop_block.lower()


def test_local_quickstart_pid_creation_is_locked_atomic_and_restart_safe() -> None:
    shell = read("deploy/quickstart.sh")
    batch = read("deploy/quickstart.bat").lower()

    assert "acquire_pid_lock" in shell
    assert 'mkdir -- "$lock_dir"' in shell
    assert 'mv -f -- "$temp_file" "$pid_file"' in shell
    assert "process_matches" in shell
    assert "already running with PID" in shell
    assert "Removed stale" in shell

    assert "[io.fileshare]::none" in batch
    assert "start-process" in batch
    assert "-passthru" in batch
    assert "move-item -literalpath $tempfile -destination $pidfile -force" in batch
    assert "already running with pid" in batch
    assert "removed stale" in batch


def test_local_quickstart_pid_record_prevents_pid_reuse_from_claiming_ownership() -> None:
    shell = read("deploy/quickstart.sh")
    batch = read("deploy/quickstart.bat").lower()

    assert "process_identity" in shell
    assert "/proc/${pid}/stat" in shell
    assert 'local expected_identity="$3"' in shell
    assert '[[ "$current_identity" == "$expected_identity" ]]' in shell
    assert "printf '%s\\n%s\\n'" in shell

    assert ".creationdate.touniversaltime().tostring('o')" in batch
    assert "-split '\\|', 2" in batch
    assert "$expectedidentity" in batch
    assert "$creationidentity" in batch


def test_deploy_readme_documents_owned_local_process_lifecycle() -> None:
    readme = read("deploy/README.md")
    normalized = " ".join(readme.split())

    assert "`data/runtime/askme-local.pid`" in readme
    assert "`data/runtime/zeroclaw-local.pid`" in readme
    assert "Repeated `local` starts" in readme
    assert "stale PID" in readme
    assert "never stops a process by image name or command-pattern search" in normalized


def test_docker_quickstarts_fail_closed_before_control_plane_on_unsupported_hardware() -> None:
    shell = read("deploy/quickstart.sh")
    batch = read("deploy/quickstart.bat").lower()
    deploy_readme = read("deploy/README.md")

    shell_docker = shell.split("  docker)", 1)[1].split("  docker-zeroclaw)", 1)[0]
    shell_experimental = shell.split("  docker-zeroclaw)", 1)[1].split("  local)", 1)[0]
    for block in (shell_docker, shell_experimental):
        assert block.index("prepare_linux_edge_audio") < block.index("start_litellm")
    assert '[[ "$(uname -s)" != "Linux" ]]' in shell
    assert "[[ ! -d /dev/snd ]]" in shell
    assert "getent group audio" in shell
    assert "export ASKME_AUDIO_GID" in shell

    batch_docker = batch.split("\n:docker\n", 1)[1].split("\n:docker-zeroclaw\n", 1)[0]
    batch_experimental = batch.split("\n:docker-zeroclaw\n", 1)[1].split("\n:local\n", 1)[0]
    for block in (batch_docker, batch_experimental):
        assert "no service was started" in block
        assert "start_litellm" not in block
        assert "docker compose" not in block
    assert "windows local-development helper" in deploy_readme.lower()
    assert "before starting services" in deploy_readme


def test_experimental_local_zeroclaw_maps_virtual_key_only_to_child_process() -> None:
    shell = read("deploy/quickstart.sh")
    batch = read("deploy/quickstart.bat").lower()

    assert (
        'ZEROCLAW_API_KEY="$ZEROCLAW_LITELLM_VIRTUAL_KEY" \\' + "\n    zeroclaw gateway"
    ) in shell
    assert "export ZEROCLAW_API_KEY" not in shell

    batch_helper = batch.split("\n:start_zeroclaw\n", 1)[1].split("\n:err\n", 1)[0]
    assert "setlocal disabledelayedexpansion" in batch_helper
    assert 'set "zeroclaw_api_key=%zeroclaw_litellm_virtual_key%"' in batch_helper
    assert 'set "tracked_executable=zeroclaw"' in batch_helper
    assert 'set "tracked_arguments=gateway --host 127.0.0.1 --port 8080"' in batch_helper
    assert "start-process -filepath $env:tracked_executable" in batch_helper
    assert "endlocal" in batch_helper
    start_line = next(line for line in batch_helper.splitlines() if "start-process" in line)
    assert "%zeroclaw_litellm_virtual_key%" not in start_line


def test_quickstart_secures_live_environment_files_before_use() -> None:
    shell = read("deploy/quickstart.sh")
    batch = read("deploy/quickstart.bat").lower()

    assert 'chmod 600 "$env_file"' in shell
    assert shell.count("secure_env_file") >= 2
    assert "icacls" in batch
    assert "/inheritance:r" in batch
    assert batch.count("secure_env_file") >= 3
    assert "if errorlevel 1" in batch


def test_deployment_troubleshooting_never_renders_or_prints_secrets() -> None:
    guide = read("docs/DEPLOYMENT.md")

    assert "config --quiet" in guide
    assert "run --rm --no-deps litellm-key-policy" in guide
    assert "run --rm askme env" not in guide
    assert 'grep -E "^(LITELLM_' not in guide
    assert "run --rm askme env" not in guide
    for line in guide.splitlines():
        if "docker compose" in line and " config" in line:
            assert "config --quiet" in line


def test_deployment_windows_acl_is_applied_and_checked_per_secret_file() -> None:
    guide = read("docs/DEPLOYMENT.md")

    principal = '"$env:USERDOMAIN\\${env:USERNAME}:(M)"'
    assert "icacls docker\\.env docker\\.env.litellm" not in guide
    assert f"icacls docker\\.env /inheritance:r /grant:r {principal}" in guide
    assert f"icacls docker\\.env.litellm /inheritance:r /grant:r {principal}" in guide
    assert guide.count("if ($LASTEXITCODE -ne 0)") >= 2


def test_deployment_default_and_experimental_zeroclaw_contracts_are_separate() -> None:
    guide = read("docs/DEPLOYMENT.md")
    required = guide.split("### 必需", 1)[1].split("### LiteLLM 控制面", 1)[0]
    experimental = guide.split("### 实验 ZeroClaw 凭据", 1)[1].split(
        "### 条件启用的语音供应商",
        1,
    )[0]
    standard = guide.split("### 标准部署", 1)[1].split(
        "### 实验 ZeroClaw 启动",
        1,
    )[0]

    assert "ZEROCLAW_LITELLM_VIRTUAL_KEY" not in required
    assert "ZEROCLAW_LITELLM_VIRTUAL_KEY" not in standard
    assert "ZEROCLAW_LITELLM_VIRTUAL_KEY" in experimental
    assert "experimental-zeroclaw" in experimental
    assert "仅" in experimental and "才必填" in experimental


def test_deployment_only_documents_consumed_container_environment() -> None:
    guide = read("docs/DEPLOYMENT.md")
    robot = guide.split("### 机器人服务（可选）", 1)[1].split(
        "### 运行时认证",
        1,
    )[0]

    for unsupported in (
        "TTS_VOICE_ID",
        "TTS_SPEED",
        "TTS_EMOTION",
        "OTA_SERVER_URL",
        "ROBOT_SERIAL_PORT",
    ):
        assert unsupported not in guide
    for localhost_url in (
        "http://localhost:8088",
        "http://localhost:5080",
        "http://localhost:5070",
    ):
        assert localhost_url not in robot
    assert "默认留空" in robot
    assert "容器可解析" in robot


def test_deployment_backup_and_image_claims_match_compose() -> None:
    guide = read("docs/DEPLOYMENT.md")

    assert "askme-litellm_askme_data" in guide
    assert "askme-litellm_litellm_db" in guide
    assert "tar -czf askme-backup-" not in guide
    assert "Askme 业务容器当前未启用 `read_only`" in guide
    assert "rootfs 只读" not in guide
    assert "/opt/askme/models/" in guide
    assert "askme-edge" not in guide
    assert "proxy_pass http://askme:8765;" in guide
    assert (
        "docker volume inspect askme-litellm_askme_data "
        "askme-litellm_litellm_db askme-litellm_zeroclaw_workspace"
    ) not in guide


def test_deployment_marks_provider_credentials_as_conditional() -> None:
    guide = read("docs/DEPLOYMENT.md")
    required = guide.split("### 必需", 1)[1].split("### LiteLLM 控制面", 1)[0]
    conditional = guide.split("### 条件启用的语音供应商", 1)[1].split(
        "### 语音配置",
        1,
    )[0]

    assert "MINIMAX_API_KEY" not in required
    assert "DASHSCOPE_API_KEY" not in required
    assert "MINIMAX_API_KEY" in conditional
    assert "DASHSCOPE_API_KEY" in conditional
    assert "启用" in conditional and "才必填" in conditional


def test_runtime_container_does_not_claim_an_unmounted_mcp_route() -> None:
    readme = read("README.md")

    assert "MCP SSE: `http://localhost:8765/mcp`" not in readme
    assert "runtime 容器当前未挂载 FastMCP" in readme


def test_docker_surface_documents_compose_entrypoints() -> None:
    readme = read("docker/README.md")

    for token in (
        "`docker-compose.yml`",
        "`docker-compose.edge-linux.yml`",
        "`docker-compose.prod.yml`",
        "`Dockerfile.askme`",
        "`Dockerfile.zeroclaw`",
        "`docker-entrypoint.sh`",
        "-f docker/docker-compose.edge-linux.yml",
    ):
        assert token in readme


def test_deployment_guide_uses_repo_root_docker_env_and_compose_file() -> None:
    guide = read("docs/DEPLOYMENT.md")

    assert "cp docker/.env.example docker/.env" in guide
    assert "vi docker/.env" in guide
    assert "--env-file docker/.env" in guide
    assert "--env-file docker/.env.litellm" in guide
    assert "-f docker/docker-compose.yml" in guide
    assert "--env-file .env" not in guide
    assert "cp .env .env.backup" not in guide
    assert "up -d --wait litellm" in guide
    assert "http://askme:8765" in guide
    assert "ASKME_LITELLM_BASE_URL=" not in guide
    assert "${ASKME_LITELLM_BASE_URL" not in guide
    assert "启动 ZeroClaw 进程不等于 AskMe MCP 集成可用" in guide
