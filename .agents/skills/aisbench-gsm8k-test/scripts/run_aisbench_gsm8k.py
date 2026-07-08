#!/usr/bin/env python3
"""Run AISBench GSM8K via a remote Docker container.

Progress is written to stderr. The final stdout payload is JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import PurePosixPath


DEFAULT_HOST = "192.168.13.157"
DEFAULT_CONTAINER = "zyj_aisbench"
DEFAULT_RUN_CWD = "/workspace"
PACKAGE_NAME = "ais_bench_benchmark"
MODEL_NAME = "vllm_api_general_chat"
DATASET_NAME = "gsm8k_gen_0_shot_cot_chat_prompt"


@dataclass
class RemoteResult:
    returncode: int
    stdout: str
    stderr: str


def log(message: str) -> None:
    print(f"__AISBENCH_GSM8K_PROGRESS__={json.dumps({'message': message})}",
          file=sys.stderr,
          flush=True)


def run_local(cmd: list[str], timeout: int | None = None) -> RemoteResult:
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return RemoteResult(proc.returncode, proc.stdout, proc.stderr)


def ssh_target(args: argparse.Namespace) -> str:
    return f"{args.ssh_user}@{args.host}" if args.ssh_user else args.host


def remote_exec(args: argparse.Namespace,
                command: str,
                timeout: int | None = None) -> RemoteResult:
    shell_flag = "-ic" if args.container_shell == "interactive" else "-lc"
    remote_command = (
        "source ~/.bashrc >/dev/null 2>&1 || true; " + command
    )
    docker_command = (
        f"docker exec {shlex.quote(args.container)} bash {shell_flag} "
        f"{shlex.quote(remote_command)}"
    )
    ssh_cmd = ["ssh"]
    if args.ssh_port:
        ssh_cmd += ["-p", str(args.ssh_port)]
    ssh_cmd += [ssh_target(args), docker_command]
    return run_local(ssh_cmd, timeout=timeout)


def require_ok(result: RemoteResult, step: str) -> str:
    if result.returncode != 0:
        payload = {
            "status": "failed",
            "step": step,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(result.returncode or 1)
    return result.stdout


def parse_pip_show(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def discover_install(args: argparse.Namespace) -> dict[str, str]:
    log("checking ais_bench_benchmark editable install")
    out = require_ok(
        remote_exec(args, f"python3 -m pip show {PACKAGE_NAME}"),
        "pip_show",
    )
    fields = parse_pip_show(out)
    editable_root = fields.get("Editable project location")
    if not editable_root:
        payload = {
            "status": "failed",
            "step": "pip_show",
            "error": "pip show did not report an editable project location",
            "pip_show": fields,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(1)
    return {
        "package_location": fields.get("Location", ""),
        "editable_project_location": editable_root,
        "version": fields.get("Version", ""),
    }


def path_info(editable_root: str) -> dict[str, str]:
    root = PurePosixPath(editable_root)
    return {
        "model_config": str(root / "ais_bench" / "benchmark" / "configs" /
                            "models" / "vllm_api" /
                            "vllm_api_general_chat.py"),
        "dataset": str(root / "ais_bench" / "datasets" / "gsm8k" /
                       "test.jsonl"),
    }


def shell_quote_path(path: str) -> str:
    return shlex.quote(path)


def validate_paths(args: argparse.Namespace, paths: dict[str, str]) -> None:
    log("validating model config and dataset paths")
    cmd = (
        f"test -f {shell_quote_path(paths['model_config'])} && "
        f"test -f {shell_quote_path(paths['dataset'])}"
    )
    require_ok(remote_exec(args, cmd), "validate_paths")


def dataset_line_count(args: argparse.Namespace, dataset_path: str) -> int:
    out = require_ok(
        remote_exec(args, f"wc -l < {shell_quote_path(dataset_path)}"),
        "dataset_line_count",
    )
    return int(out.strip() or "0")


def remote_python(args: argparse.Namespace,
                  code: str,
                  timeout: int | None = None) -> str:
    cmd = "python3 -c " + shlex.quote(code)
    return require_ok(remote_exec(args, cmd, timeout=timeout), "remote_python")


def patch_max_out_len(args: argparse.Namespace,
                      config_path: str,
                      max_out_len: int) -> dict[str, str | int]:
    log(f"patching max_out_len to {max_out_len}")
    code = f"""
from pathlib import Path
import re
import time
p = Path({config_path!r})
s = p.read_text()
backup = p.with_name(p.name + ".bak." + time.strftime("%Y%m%d_%H%M%S"))
backup.write_text(s)
patterns = [
    (r'(?m)^(\\s*max_out_len\\s*=\\s*)\\d+', r'\\g<1>{max_out_len}'),
    (r'(?m)^(\\s*["\\']max_out_len["\\']\\s*:\\s*)\\d+', r'\\g<1>{max_out_len}'),
]
for pattern, repl in patterns:
    new_s, count = re.subn(pattern, repl, s, count=1)
    if count:
        p.write_text(new_s)
        print(str(backup))
        raise SystemExit(0)
raise SystemExit("max_out_len pattern not found in " + str(p))
"""
    backup_path = remote_python(args, code).strip()
    return {"config_backup": backup_path, "max_out_len": max_out_len}


def limit_dataset(args: argparse.Namespace,
                  dataset_path: str,
                  limit: int) -> dict[str, str | int]:
    log(f"reducing dataset to first {limit} samples")
    code = f"""
from pathlib import Path
import time
p = Path({dataset_path!r})
lines = p.read_text().splitlines(True)
backup = p.with_name(p.name + ".bak." + time.strftime("%Y%m%d_%H%M%S"))
backup.write_text("".join(lines))
p.write_text("".join(lines[:{limit}]))
print(str(backup))
"""
    backup_path = remote_python(args, code).strip()
    return {"dataset_backup": backup_path, "limit_samples": limit}


def run_aisbench(args: argparse.Namespace) -> RemoteResult:
    log("running ais_bench GSM8K")
    command = (
        f"cd {shell_quote_path(args.run_cwd)} && "
        f"ais_bench --models {MODEL_NAME} --dataset {DATASET_NAME}"
    )
    return remote_exec(args, command, timeout=args.timeout)


def parse_summary_path(text: str) -> str | None:
    match = re.search(r"write markdown summary to\s+(\S+)", text)
    return match.group(1) if match else None


def read_remote_file(args: argparse.Namespace, path: str) -> str:
    return require_ok(
        remote_exec(args, f"cat {shell_quote_path(path)}"),
        "read_summary",
    )


def parse_accuracy(markdown: str, stdout: str) -> float | None:
    for text in (markdown, stdout):
        match = re.search(
            r"\|\s*gsm8k\s*\|[^\n]+?\|\s*accuracy\s*\|\s*gen\s*\|\s*([0-9.]+)\s*\|",
            text,
        )
        if match:
            return float(match.group(1))
    match = re.search(r"gsm8k\s+\S+\s+accuracy\s+gen\s+([0-9.]+)",
                      stdout)
    return float(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--ssh-user", default="")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--container-shell",
                        choices=("interactive", "noninteractive"),
                        default="noninteractive")
    parser.add_argument("--run-cwd", default=DEFAULT_RUN_CWD)
    parser.add_argument("--max-out-len", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    install = discover_install(args)
    paths = path_info(install["editable_project_location"])
    validate_paths(args, paths)

    modifications: dict[str, str | int] = {}
    if args.max_out_len is not None:
        modifications.update(
            patch_max_out_len(args, paths["model_config"], args.max_out_len))
    if args.limit_samples is not None:
        modifications.update(limit_dataset(args, paths["dataset"],
                                           args.limit_samples))

    dataset_lines_before_run = dataset_line_count(args, paths["dataset"])
    if args.inspect_only:
        print(json.dumps({
            "status": "ok",
            "install": install,
            "paths": {
                **paths,
                "run_cwd": args.run_cwd,
            },
            "modifications": modifications,
            "dataset_line_count": dataset_lines_before_run,
        }, indent=2, sort_keys=True))
        return 0

    result = run_aisbench(args)
    if result.returncode != 0:
        print(json.dumps({
            "status": "failed",
            "step": "ais_bench",
            "install": install,
            "paths": paths,
            "modifications": modifications,
            "dataset_line_count": dataset_lines_before_run,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-8000:],
            "stderr_tail": result.stderr[-8000:],
        }, indent=2, sort_keys=True))
        return result.returncode or 1

    combined_output = result.stdout + "\n" + result.stderr
    summary_path = parse_summary_path(combined_output)
    summary_markdown = read_remote_file(args, summary_path) if summary_path else ""
    accuracy = parse_accuracy(summary_markdown, combined_output)

    payload = {
        "status": "ok",
        "install": install,
        "paths": {
            **paths,
            "run_cwd": args.run_cwd,
            "summary_markdown": summary_path or "",
        },
        "modifications": modifications,
        "dataset_line_count": dataset_lines_before_run,
        "result": {
            "accuracy": accuracy,
            "summary_markdown": summary_markdown,
        },
        "stdout_tail": result.stdout[-8000:],
        "stderr_tail": result.stderr[-4000:],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
