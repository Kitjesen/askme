#!/bin/bash
# AskMe + LiteLLM quick deployment helper; ZeroClaw is experimental opt-in only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

APP_ENV_FILE="docker/.env"
LITELLM_ENV_FILE="docker/.env.litellm"
RUNTIME_DIR="${ASKME_QUICKSTART_RUNTIME_DIR:-${REPO_ROOT}/data/runtime}"
ASKME_PID_FILE="${RUNTIME_DIR}/askme-local.pid"
ZEROCLAW_PID_FILE="${RUNTIME_DIR}/zeroclaw-local.pid"
ACTIVE_PID_LOCK=""

release_pid_lock() {
  if [[ -n "$ACTIVE_PID_LOCK" && -d "$ACTIVE_PID_LOCK" ]]; then
    rm -f -- "${ACTIVE_PID_LOCK}/owner"
    rmdir -- "$ACTIVE_PID_LOCK" 2>/dev/null || true
  fi
  ACTIVE_PID_LOCK=""
}

trap release_pid_lock EXIT

acquire_pid_lock() {
  local pid_file="$1"
  local lock_dir="${pid_file}.lock"
  local lock_owner=""
  local attempt

  mkdir -p -- "$RUNTIME_DIR"
  for attempt in 1 2; do
    if mkdir -- "$lock_dir" 2>/dev/null; then
      ACTIVE_PID_LOCK="$lock_dir"
      printf '%s\n' "$$" > "${lock_dir}/owner"
      return 0
    fi

    if [[ -r "${lock_dir}/owner" ]]; then
      IFS= read -r lock_owner < "${lock_dir}/owner" || lock_owner=""
    fi
    if [[ "$lock_owner" =~ ^[0-9]+$ ]] && (( lock_owner > 1 )) \
      && kill -0 "$lock_owner" 2>/dev/null; then
      echo "[ERROR] Another quickstart operation owns $pid_file." >&2
      return 1
    fi

    rm -f -- "${lock_dir}/owner"
    rmdir -- "$lock_dir" 2>/dev/null || true
  done

  echo "[ERROR] Could not acquire the process lock for $pid_file." >&2
  return 1
}

process_identity() {
  local pid="$1"
  local stat_line=""
  local identity=""
  local -a stat_fields

  [[ "$pid" =~ ^[0-9]+$ ]] && (( pid > 1 )) || return 1
  if [[ -r "/proc/${pid}/stat" ]]; then
    IFS= read -r stat_line < "/proc/${pid}/stat" || return 1
    read -r -a stat_fields <<< "${stat_line##*) }"
    (( ${#stat_fields[@]} >= 20 )) || return 1
    printf 'proc:%s\n' "${stat_fields[19]}"
    return 0
  fi

  identity="$(ps -p "$pid" -o lstart= 2>/dev/null)" || return 1
  [[ -n "$identity" ]] || return 1
  printf 'ps:%s\n' "$identity"
}

process_matches() {
  local pid="$1"
  local marker="$2"
  local expected_identity="$3"
  local command_line=""
  local current_identity=""

  [[ "$pid" =~ ^[0-9]+$ ]] && (( pid > 1 )) || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -n "$expected_identity" ]] || return 1
  current_identity="$(process_identity "$pid")" || return 1
  [[ "$current_identity" == "$expected_identity" ]] || return 1

  if [[ -r "/proc/${pid}/cmdline" ]]; then
    command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  else
    command_line="$(ps -p "$pid" -o command= 2>/dev/null)" || return 1
  fi
  [[ "$command_line" == *"$marker"* ]]
}

start_tracked_process() {
  local pid_file="$1"
  local marker="$2"
  shift 2
  local tracked_pid=""
  local tracked_identity=""
  local child_pid
  local child_identity
  local temp_file

  acquire_pid_lock "$pid_file" || return 1
  if [[ -f "$pid_file" ]]; then
    {
      IFS= read -r tracked_pid || tracked_pid=""
      IFS= read -r tracked_identity || tracked_identity=""
    } < "$pid_file"
    if process_matches "$tracked_pid" "$marker" "$tracked_identity"; then
      echo "[SKIP] Local process is already running with PID $tracked_pid."
      release_pid_lock
      return 0
    fi
    rm -f -- "$pid_file"
  fi

  "$@" &
  child_pid=$!
  if ! kill -0 "$child_pid" 2>/dev/null; then
    wait "$child_pid" 2>/dev/null || true
    echo "[ERROR] Local process failed before its PID could be recorded." >&2
    release_pid_lock
    return 1
  fi
  child_identity="$(process_identity "$child_pid")" || child_identity=""
  if [[ -z "$child_identity" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
    echo "[ERROR] Could not identify local process PID $child_pid." >&2
    release_pid_lock
    return 1
  fi

  temp_file="${pid_file}.$$.$RANDOM.tmp"
  if ! (umask 077 && printf '%s\n%s\n' "$child_pid" "$child_identity" > "$temp_file") \
    || ! mv -f -- "$temp_file" "$pid_file"; then
    rm -f -- "$temp_file"
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
    echo "[ERROR] Could not record local process PID $child_pid." >&2
    release_pid_lock
    return 1
  fi

  echo "[OK]   Started local process with PID $child_pid."
  release_pid_lock
}

start_askme() {
  start_tracked_process \
    "$ASKME_PID_FILE" \
    "askme.blueprints.presets.edge_robot" \
    python -m askme.blueprints.presets.edge_robot
}

stop_tracked_process() {
  local pid_file="$1"
  local marker="$2"
  local process_name="$3"
  local tracked_pid=""
  local tracked_identity=""
  local attempt

  acquire_pid_lock "$pid_file" || return 1
  if [[ ! -f "$pid_file" ]]; then
    echo "[SKIP] $process_name has no recorded local PID."
    release_pid_lock
    return 0
  fi

  {
    IFS= read -r tracked_pid || tracked_pid=""
    IFS= read -r tracked_identity || tracked_identity=""
  } < "$pid_file"
  if ! process_matches "$tracked_pid" "$marker" "$tracked_identity"; then
    rm -f -- "$pid_file"
    echo "[SKIP] Removed stale $process_name PID file without signaling a process."
    release_pid_lock
    return 0
  fi

  if ! kill -TERM "$tracked_pid" 2>/dev/null; then
    echo "[ERROR] Could not stop $process_name PID $tracked_pid." >&2
    release_pid_lock
    return 1
  fi
  for attempt in {1..50}; do
    process_matches "$tracked_pid" "$marker" "$tracked_identity" || break
    sleep 0.1
  done

  if process_matches "$tracked_pid" "$marker" "$tracked_identity"; then
    kill -KILL "$tracked_pid" 2>/dev/null || true
    for attempt in {1..10}; do
      process_matches "$tracked_pid" "$marker" "$tracked_identity" || break
      sleep 0.1
    done
  fi
  if process_matches "$tracked_pid" "$marker" "$tracked_identity"; then
    echo "[ERROR] $process_name PID $tracked_pid is still running; PID file was kept." >&2
    release_pid_lock
    return 1
  fi

  rm -f -- "$pid_file"
  echo "[OK]   Stopped $process_name PID $tracked_pid."
  release_pid_lock
}

secure_env_file() {
  local env_file="$1"
  if ! chmod 600 "$env_file"; then
    echo "[ERROR] Could not restrict permissions on $env_file." >&2
    return 1
  fi
}

require_env_files() {
  local missing=0
  for env_file in "$APP_ENV_FILE" "$LITELLM_ENV_FILE"; do
    if [[ ! -f "$env_file" ]]; then
      echo "[ERROR] Missing $env_file; run 'bash deploy/quickstart.sh setup' first." >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    return 1
  fi
  for env_file in "$APP_ENV_FILE" "$LITELLM_ENV_FILE"; do
    secure_env_file "$env_file"
  done
}

load_env_file() {
  local env_file="$1"
  local raw_line key value
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    raw_line="${raw_line%$'\r'}"
    [[ -z "$raw_line" || "$raw_line" == \#* ]] && continue
    key="${raw_line%%=*}"
    value="${raw_line#*=}"
    if [[ "$key" == "$raw_line" || ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "[ERROR] Invalid environment assignment in $env_file." >&2
      return 1
    fi
    export "$key=$value"
  done < "$env_file"
}

load_local_environment() {
  require_env_files
  load_env_file "$LITELLM_ENV_FILE"
  load_env_file "$APP_ENV_FILE"
}

prepare_linux_edge_audio() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[ERROR] Docker edge runtime requires a Linux host with /dev/snd." >&2
    return 1
  fi
  if [[ ! -d /dev/snd ]]; then
    echo "[ERROR] /dev/snd is missing; refusing to start the edge runtime." >&2
    return 1
  fi
  if [[ -z "${ASKME_AUDIO_GID:-}" ]]; then
    if ! command -v getent >/dev/null 2>&1; then
      echo "[ERROR] getent is required to resolve the host audio group GID." >&2
      return 1
    fi
    ASKME_AUDIO_GID="$(getent group audio | cut -d: -f3)"
  fi
  if [[ -z "$ASKME_AUDIO_GID" ]]; then
    echo "[ERROR] Host audio group GID is unavailable; set ASKME_AUDIO_GID." >&2
    return 1
  fi
  export ASKME_AUDIO_GID
}

start_litellm() {
  require_env_files
  docker compose --env-file docker/.env.litellm -f docker/docker-compose.litellm.yml up -d --wait litellm
}

start_zeroclaw() {
  if [[ -z "${ZEROCLAW_LITELLM_VIRTUAL_KEY:-}" ]]; then
    echo "[ERROR] ZEROCLAW_LITELLM_VIRTUAL_KEY is required for experimental ZeroClaw." >&2
    return 1
  fi
  start_tracked_process \
    "$ZEROCLAW_PID_FILE" \
    "zeroclaw gateway" \
    env ZEROCLAW_API_KEY="$ZEROCLAW_LITELLM_VIRTUAL_KEY" \
    zeroclaw gateway --host 127.0.0.1 --port 8080
}

case "${1:-menu}" in
  setup)
    echo "[1/2] Preparing application environment template..."
    if [[ ! -f "$APP_ENV_FILE" ]]; then
      cp docker/.env.example "$APP_ENV_FILE"
      echo "[OK]   Created $APP_ENV_FILE"
    else
      echo "[SKIP] $APP_ENV_FILE already exists"
    fi

    echo "[2/2] Preparing LiteLLM sidecar environment template..."
    if [[ ! -f "$LITELLM_ENV_FILE" ]]; then
      cp docker/litellm.env.example "$LITELLM_ENV_FILE"
      echo "[OK]   Created $LITELLM_ENV_FILE"
    else
      echo "[SKIP] $LITELLM_ENV_FILE already exists"
    fi
    require_env_files
    echo "Fill both files and follow docs/LITELLM_GATEWAY.md to generate the AskMe scoped key."
    ;;

  docker)
    prepare_linux_edge_audio
    start_litellm
    docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml -f docker/docker-compose.edge-linux.yml up -d
    echo "Askme: http://localhost:8765"
    ;;

  docker-zeroclaw)
    prepare_linux_edge_audio
    start_litellm
    docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml -f docker/docker-compose.edge-linux.yml --profile experimental-zeroclaw up -d
    echo "Askme:                 http://localhost:8765"
    echo "ZeroClaw experimental: http://localhost:8080 (MCP integration unavailable)"
    ;;

  local)
    load_local_environment
    start_litellm
    python -m askme.llm.key_policy
    start_askme
    echo "AskMe started locally"
    ;;

  local-zeroclaw)
    load_local_environment
    start_litellm
    python -m askme.llm.key_policy --require-zeroclaw
    python scripts/dev/setup_zeroclaw.py
    start_askme
    sleep 5
    start_zeroclaw
    echo "AskMe + experimental ZeroClaw started locally; MCP integration is unavailable"
    ;;

  stop)
    stop_failed=0
    stop_tracked_process "$ZEROCLAW_PID_FILE" "zeroclaw gateway" "ZeroClaw" || stop_failed=1
    stop_tracked_process "$ASKME_PID_FILE" "askme.blueprints.presets.edge_robot" "AskMe" || stop_failed=1
    if [[ -f "$APP_ENV_FILE" && -f "$LITELLM_ENV_FILE" ]]; then
      docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml down 2>/dev/null || true
    fi
    if [[ "$stop_failed" -ne 0 ]]; then
      echo "[ERROR] One or more recorded local processes could not be stopped." >&2
      exit 1
    fi
    echo "Stopped"
    ;;

  *)
    echo "Usage: bash deploy/quickstart.sh [setup|docker|local|docker-zeroclaw|local-zeroclaw|stop]"
    echo "  *-zeroclaw commands are experimental and do not provide AskMe MCP integration."
    ;;
esac
