#!/usr/bin/env python3
"""Minimal helper for Qwen3.5-2B on the sunrise S100P board.

This script assumes the board-side helper scripts have been staged under:
  /home/sunrise/data/llm/
"""

from __future__ import annotations

import argparse
import base64
import shlex
import sys

import paramiko


DEFAULT_HOST = "192.168.66.190"
DEFAULT_USER = "sunrise"
DEFAULT_PASS = "sunrise"
REMOTE_LLM_DIR = "/home/sunrise/data/llm"
REMOTE_MODEL = "/home/sunrise/data/models/qwen35-2b/Qwen3.5-2B-Q4_K_M.gguf"


def build_client(host: str, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=15)
    return client


def exec_remote(client: paramiko.SSHClient, command: str) -> int:
    stdin, stdout, stderr = client.exec_command(command)
    out = stdout.read()
    err = stderr.read()
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write(out)
        sys.stdout.buffer.flush()
    else:
        sys.stdout.write(out.decode(errors="ignore"))
    if err:
        if hasattr(sys.stderr, "buffer"):
            sys.stderr.buffer.write(err)
            sys.stderr.buffer.flush()
        else:
            sys.stderr.write(err.decode(errors="ignore"))
    return stdout.channel.recv_exit_status()


def cmd_status(client: paramiko.SSHClient) -> int:
    command = f"""sh -lc '
ls -lh {shlex.quote(REMOTE_MODEL)} 2>/dev/null || true
printf "====\\n"
tail -n 20 {shlex.quote(REMOTE_LLM_DIR)}/download_qwen35_q4km.log 2>/dev/null || true
printf "====\\n"
ps -ef | grep -E "curl .*Qwen3.5-2B-Q4_K_M|download_qwen35_q4km" | grep -v grep || true
'"""
    return exec_remote(client, command)


def cmd_download(client: paramiko.SSHClient) -> int:
    command = f"""sh -lc '
if pgrep -af "curl .*Qwen3.5-2B-Q4_K_M" >/dev/null; then
  echo "download already running"
else
  nohup {shlex.quote(REMOTE_LLM_DIR)}/download_qwen35_q4km.sh > {shlex.quote(REMOTE_LLM_DIR)}/download_qwen35_q4km.log 2>&1 < /dev/null &
  echo "download started"
fi
'"""
    return exec_remote(client, command)


def cmd_run(client: paramiko.SSHClient, prompt: str) -> int:
    prompt_b64 = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
    command = f"""python3 - <<'PY'
import base64
import subprocess

prompt = base64.b64decode({prompt_b64!r}).decode("utf-8")
raise SystemExit(subprocess.run(
    [{REMOTE_LLM_DIR!r} + "/run_qwen35_q4km.sh", prompt],
    check=False,
).returncode)
PY"""
    return exec_remote(client, command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Operate Qwen3.5-2B on sunrise.")
    parser.add_argument(
        "action",
        choices=["status", "download", "run"],
        help="status: inspect download/progress; download: start background model download; run: execute one prompt",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt used by the run action")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASS)
    args = parser.parse_args()

    client = build_client(args.host, args.user, args.password)
    try:
        if args.action == "status":
            return cmd_status(client)
        if args.action == "download":
            return cmd_download(client)
        prompt = " ".join(args.prompt).strip() or "用中文介绍一下你自己，并简要说明你能帮机器人做什么。"
        return cmd_run(client, prompt)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
