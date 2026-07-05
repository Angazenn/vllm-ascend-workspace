#!/usr/bin/env python3
"""Launch and verify the GLM5 SFA prefill offload vLLM server.

Progress is written to stderr as JSON lines. The final result is written to
stdout as one JSON object.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


DEFAULT_HOST = os.environ.get("GLM5_SFA_OFFLOAD_HOST", "192.168.13.165")
DEFAULT_SSH_USER = os.environ.get("GLM5_SFA_OFFLOAD_SSH_USER", "")
DEFAULT_SSH_PORT = env_int("GLM5_SFA_OFFLOAD_SSH_PORT", 22)
DEFAULT_CONTAINER = os.environ.get("GLM5_SFA_OFFLOAD_CONTAINER", "zyj_offload")
DEFAULT_SCRIPTS_DIR = os.environ.get("GLM5_SFA_OFFLOAD_SCRIPTS_DIR", "/home/zyj/scripts")
DEFAULT_LAUNCH_SCRIPT = os.environ.get(
    "GLM5_SFA_OFFLOAD_LAUNCH_SCRIPT",
    "/home/zyj/scripts/launch_vllm_offload.sh",
)
DEFAULT_REQUEST_SCRIPT = os.environ.get(
    "GLM5_SFA_OFFLOAD_REQUEST_SCRIPT",
    "/home/zyj/scripts/curl.sh",
)
DEFAULT_PORT = env_int("GLM5_SFA_OFFLOAD_PORT", 8900)
ERROR_PATTERNS = "Traceback\\|AssertionError\\|RuntimeError\\|ValueError\\|StopIteration"
PROGRESS_PREFIX = "__GLM5_SFA_PREFILL_OFFLOAD_PROGRESS__="


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


def emit_progress(phase: str, message: str, **fields: Any) -> None:
    payload = {"phase": phase, "message": message, **fields}
    sys.stderr.write(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stderr.flush()


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def launch_basename(launch_script: str) -> str:
    return posixpath.basename(launch_script.rstrip("/")) or "launch"


def default_launch_log_path(scripts_dir: str, launch_script: str) -> str:
    name = launch_basename(launch_script)
    if name.endswith(".sh"):
        name = name[:-3]
    return scripts_dir.rstrip("/") + f"/{name}.nohup.log"


def default_process_pattern(launch_script: str) -> str:
    launch_name = re.escape(launch_basename(launch_script))
    if launch_name:
        launch_name = f"[{launch_name[0]}]{launch_name[1:]}"
    return "[v]llm serve|" + launch_name


def ssh_destination(host: str, ssh_user: str) -> str:
    return f"{ssh_user}@{host}" if ssh_user else host


def run_ssh(
    host: str,
    remote_cmd: str,
    *,
    timeout: int,
    ssh_user: str,
    ssh_port: int,
) -> CmdResult:
    cmd = ["ssh"]
    if ssh_port != 22:
        cmd.extend(["-p", str(ssh_port)])
    cmd.extend([ssh_destination(host, ssh_user), remote_cmd])
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return CmdResult(proc.returncode, proc.stdout, proc.stderr)


def docker_bash(container: str, script: str) -> str:
    return f"docker exec {shlex.quote(container)} bash -lc {shlex.quote(script)}"


def run_container(
    host: str,
    container: str,
    script: str,
    *,
    timeout: int,
    ssh_user: str,
    ssh_port: int,
) -> CmdResult:
    wrapped = f"source ~/.bashrc >/dev/null 2>&1; {script}"
    return run_ssh(
        host,
        docker_bash(container, wrapped),
        timeout=timeout,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
    )


def health_code(
    host: str,
    container: str,
    port: int,
    *,
    ssh_user: str,
    ssh_port: int,
) -> str:
    result = run_container(
        host,
        container,
        f"curl --max-time 2 -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/health || true",
        timeout=45,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
    )
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "000"


def launch_server(
    host: str,
    container: str,
    scripts_dir: str,
    launch_script: str,
    launch_log_path: str,
    *,
    ssh_user: str,
    ssh_port: int,
) -> tuple[str | None, CmdResult]:
    script = (
        f"cd {shlex.quote(scripts_dir)}; "
        f"nohup bash {shlex.quote(launch_script)} "
        f"> {shlex.quote(launch_log_path)} 2>&1 "
        f"< /dev/null & echo $!"
    )
    result = run_container(
        host,
        container,
        script,
        timeout=90,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
    )
    pid = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else None
    return pid, result


def process_snapshot(
    host: str,
    container: str,
    process_pattern: str,
    *,
    ssh_user: str,
    ssh_port: int,
) -> CmdResult:
    return run_container(
        host,
        container,
        f"pgrep -af -- {shlex.quote(process_pattern)} || true",
        timeout=45,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
    )


def wait_for_health(
    host: str,
    container: str,
    port: int,
    *,
    timeout_seconds: int,
    poll_interval: int,
    log_path: str,
    ssh_user: str,
    ssh_port: int,
) -> CmdResult:
    script = f"""
deadline=$((SECONDS + {timeout_seconds}))
while [ "$SECONDS" -lt "$deadline" ]; do
  code=$(curl --max-time 2 -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/health || true)
  if [ "$code" = "200" ]; then
    echo READY
    exit 0
  fi
  if grep -q "{ERROR_PATTERNS}" {shlex.quote(log_path)} 2>/dev/null; then
    echo ERROR
    grep -n -B 30 -A 90 "{ERROR_PATTERNS}" {shlex.quote(log_path)} | tail -260
    exit 1
  fi
  sleep {poll_interval}
done
echo TIMEOUT
tail -220 {shlex.quote(log_path)} 2>/dev/null || true
exit 124
"""
    return run_container(
        host,
        container,
        script,
        timeout=timeout_seconds + 90,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
    )


def fetch_models(
    host: str,
    container: str,
    port: int,
    *,
    ssh_user: str,
    ssh_port: int,
) -> tuple[dict[str, Any] | None, CmdResult]:
    result = run_container(
        host,
        container,
        f"curl --max-time 10 -s http://127.0.0.1:{port}/v1/models || true",
        timeout=60,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
    )
    text = result.stdout.strip()
    if not text:
        return None, result
    try:
        return json.loads(text), result
    except json.JSONDecodeError:
        return None, result


def run_request(
    host: str,
    container: str,
    scripts_dir: str,
    request_script: str,
    *,
    timeout_seconds: int,
    ssh_user: str,
    ssh_port: int,
) -> CmdResult:
    script = (
        f"cd {shlex.quote(scripts_dir)}; "
        f"if [ -x {shlex.quote(request_script)} ]; then "
        f"{shlex.quote(request_script)}; "
        f"else bash {shlex.quote(request_script)}; fi"
    )
    return run_container(
        host,
        container,
        script,
        timeout=timeout_seconds,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
    )


def json_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for idx, char in enumerate(text):
        if start is None:
            if char == "{":
                start = idx
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                snippet = text[start : idx + 1]
                try:
                    value = json.loads(snippet)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(value, dict):
                        candidates.append(value)
                start = None
    return candidates


def parse_request_response(result: CmdResult) -> dict[str, Any] | None:
    stdout_candidates = json_candidates(result.stdout)
    if stdout_candidates:
        return stdout_candidates[-1]
    combined_candidates = json_candidates(result.stdout + "\n" + result.stderr)
    return combined_candidates[-1] if combined_candidates else None


def response_summary(response: dict[str, Any] | None) -> dict[str, Any]:
    if response is None:
        return {"parsed": False}
    if "error" in response:
        return {"parsed": True, "error": response.get("error")}

    choices = response.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning")
    returned_text = content if content not in (None, "") else reasoning
    return {
        "parsed": True,
        "id": response.get("id"),
        "model": response.get("model"),
        "finish_reason": first.get("finish_reason"),
        "content": content,
        "reasoning": reasoning,
        "returned_text": returned_text,
        "usage": response.get("usage"),
        "system_fingerprint": response.get("system_fingerprint"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--host", default=DEFAULT_HOST, help="remote SSH host/IP")
    parser.add_argument(
        "--ssh-user",
        default=DEFAULT_SSH_USER,
        help="optional SSH username",
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=DEFAULT_SSH_PORT,
        help="remote SSH port",
    )
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--scripts-dir", default=DEFAULT_SCRIPTS_DIR)
    parser.add_argument("--launch-script", default=DEFAULT_LAUNCH_SCRIPT)
    parser.add_argument(
        "--launch-log-path",
        default=os.environ.get("GLM5_SFA_OFFLOAD_LAUNCH_LOG_PATH"),
        help="remote nohup log path; defaults to <scripts-dir>/<launch-script-basename>.nohup.log",
    )
    parser.add_argument("--request-script", default=DEFAULT_REQUEST_SCRIPT)
    parser.add_argument(
        "--log-path",
        default=os.environ.get("GLM5_SFA_OFFLOAD_LOG_PATH"),
        help="remote vLLM log path inspected while waiting for readiness",
    )
    parser.add_argument(
        "--process-pattern",
        default=os.environ.get("GLM5_SFA_OFFLOAD_PROCESS_PATTERN"),
        help="remote pgrep -af pattern used to detect an existing launch",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--health-timeout", type=int, default=900)
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument(
        "--request-only",
        action="store_true",
        help="skip launch and only run curl.sh",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = args.log_path or args.scripts_dir.rstrip("/") + "/log.txt"
    launch_log_path = args.launch_log_path or default_launch_log_path(args.scripts_dir, args.launch_script)
    process_pattern = args.process_pattern or default_process_pattern(args.launch_script)
    base_url = f"http://{args.host}:{args.port}"

    output: dict[str, Any] = {
        "status": "unknown",
        "host": args.host,
        "ssh_user": args.ssh_user,
        "ssh_port": args.ssh_port,
        "ssh_target": ssh_destination(args.host, args.ssh_user),
        "container": args.container,
        "base_url": base_url,
        "port": args.port,
        "scripts_dir": args.scripts_dir,
        "launch_script": args.launch_script,
        "launch_log_path": launch_log_path,
        "request_script": args.request_script,
        "log_path": log_path,
        "process_pattern": process_pattern,
    }

    try:
        emit_progress("health", "checking existing service")
        initial_health = health_code(
            args.host,
            args.container,
            args.port,
            ssh_user=args.ssh_user,
            ssh_port=args.ssh_port,
        )
        output["initial_health"] = initial_health

        launch_pid = None
        launched = False
        if initial_health != "200" and not args.request_only:
            emit_progress("process", "checking for existing launch or vLLM process")
            process_result = process_snapshot(
                args.host,
                args.container,
                process_pattern,
                ssh_user=args.ssh_user,
                ssh_port=args.ssh_port,
            )
            output["existing_processes"] = process_result.stdout.strip().splitlines()
            if process_result.stdout.strip():
                output["launch"] = {
                    "started": False,
                    "reason": "existing_process_not_yet_healthy",
                }
            else:
                emit_progress("launch", "starting vLLM server")
                launch_pid, launch_result = launch_server(
                    args.host,
                    args.container,
                    args.scripts_dir,
                    args.launch_script,
                    launch_log_path,
                    ssh_user=args.ssh_user,
                    ssh_port=args.ssh_port,
                )
                launched = launch_result.returncode == 0
                output["launch"] = {
                    "started": launched,
                    "pid": launch_pid,
                    "returncode": launch_result.returncode,
                    "stdout": launch_result.stdout.strip(),
                    "stderr_tail": "\n".join(launch_result.stderr.splitlines()[-20:]),
                }
                if launch_result.returncode != 0:
                    output["status"] = "launch_failed"
                    print_json(output)
                    return 2
        else:
            output["launch"] = {
                "started": False,
                "reason": "already_ready" if initial_health == "200" else "request_only",
            }

        if initial_health != "200":
            if args.request_only:
                output["status"] = "not_ready"
                print_json(output)
                return 2
            emit_progress("health", "waiting for service readiness", timeout=args.health_timeout)
            wait_result = wait_for_health(
                args.host,
                args.container,
                args.port,
                timeout_seconds=args.health_timeout,
                poll_interval=args.poll_interval,
                log_path=log_path,
                ssh_user=args.ssh_user,
                ssh_port=args.ssh_port,
            )
            output["health_wait"] = {
                "returncode": wait_result.returncode,
                "stdout_tail": "\n".join(wait_result.stdout.splitlines()[-40:]),
                "stderr_tail": "\n".join(wait_result.stderr.splitlines()[-20:]),
            }
            if wait_result.returncode != 0:
                output["status"] = "health_failed"
                print_json(output)
                return 2

        final_health = health_code(
            args.host,
            args.container,
            args.port,
            ssh_user=args.ssh_user,
            ssh_port=args.ssh_port,
        )
        output["final_health"] = final_health
        if final_health != "200":
            output["status"] = "not_ready"
            print_json(output)
            return 2

        emit_progress("models", "checking served models")
        models, models_result = fetch_models(
            args.host,
            args.container,
            args.port,
            ssh_user=args.ssh_user,
            ssh_port=args.ssh_port,
        )
        model_ids = [
            item.get("id")
            for item in (models or {}).get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]
        output["models"] = {
            "ids": model_ids,
            "raw": models,
            "returncode": models_result.returncode,
        }

        emit_progress("request", "running remote curl.sh")
        request_result = run_request(
            args.host,
            args.container,
            args.scripts_dir,
            args.request_script,
            timeout_seconds=args.request_timeout,
            ssh_user=args.ssh_user,
            ssh_port=args.ssh_port,
        )
        response = parse_request_response(request_result)
        output["request"] = {
            "returncode": request_result.returncode,
            "summary": response_summary(response),
            "response": response,
            "stdout_tail": "\n".join(request_result.stdout.splitlines()[-40:]),
            "stderr_tail": "\n".join(request_result.stderr.splitlines()[-40:]),
        }

        if request_result.returncode != 0:
            output["status"] = "request_failed"
            print_json(output)
            return 2
        if response and response.get("error"):
            output["status"] = "request_error"
            print_json(output)
            return 1

        output["status"] = "ok"
        output["launched"] = launched
        output["launch_pid"] = launch_pid
        print_json(output)
        return 0
    except subprocess.TimeoutExpired as exc:
        output["status"] = "timeout"
        output["error"] = str(exc)
        print_json(output)
        return 124
    except Exception as exc:
        output["status"] = "failed"
        output["error"] = str(exc)
        print_json(output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
