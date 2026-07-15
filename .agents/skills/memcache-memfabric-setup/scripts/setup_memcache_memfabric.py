#!/usr/bin/env python3
"""Set up MemCache Hybrid and a special MemFabric Hybrid fork on a remote container."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
from typing import Any


PROGRESS_PREFIX = "__MEMCACHE_MEMFABRIC_PROGRESS__="
RESULT_PREFIX = "__MEMCACHE_MEMFABRIC_RESULT__="


def shell_quote(value: str) -> str:
    return shlex.quote(str(value))


def tail_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def make_remote_script(args: argparse.Namespace) -> str:
    workspace = args.workspace.rstrip("/")
    memcache_dir = args.memcache_dir or f"{workspace}/memcache"
    memfabric_dir = args.memfabric_dir or f"{workspace}/memfabric-hybrid_kvoffload"

    return f"""#!/usr/bin/env bash
set -euo pipefail

ACTION={shell_quote(args.action)}
WORKSPACE={shell_quote(workspace)}
MEMCACHE_REPO={shell_quote(args.memcache_repo)}
MEMCACHE_DIR={shell_quote(memcache_dir)}
MEMCACHE_BRANCH={shell_quote(args.memcache_branch)}
MEMCACHE_REF={shell_quote(args.memcache_ref)}
MEMFABRIC_REPO={shell_quote(args.memfabric_repo)}
MEMFABRIC_DIR={shell_quote(memfabric_dir)}
MEMFABRIC_BRANCH={shell_quote(args.memfabric_branch)}
MEMFABRIC_REF={shell_quote(args.memfabric_ref)}
LOCAL_CONF=/usr/local/memcache_hybrid/latest/config/mmc-local.conf
META_CONF=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf
BASHRC="$HOME/.bashrc"
SETUP_LOG="/tmp/memcache_memfabric_setup_${{ACTION}}_$(date +%Y%m%d%H%M%S).log"
MEMCACHE_BUILD_LOG=/tmp/memcache_build_and_pack_run.log
MEMFABRIC_BUILD_LOG=/tmp/memfabric_kvoffload_build_and_pack_run.log

progress() {{
  local step="$1"
  local detail="${{2:-}}"
  python - "$step" "$detail" <<'PY' >&2
import json
import sys
print("__MEMCACHE_MEMFABRIC_PROGRESS__=" + json.dumps({{"step": sys.argv[1], "detail": sys.argv[2]}}, sort_keys=True))
PY
}}

fail() {{
  local msg="$1"
  progress "failed" "$msg"
  echo "$msg" >>"$SETUP_LOG"
  exit 1
}}

require_cmd() {{
  command -v "$1" >>"$SETUP_LOG" 2>&1 || fail "missing command: $1"
}}

clone_or_update() {{
  local label="$1"
  local repo="$2"
  local dir="$3"
  local branch="$4"
  local ref="$5"

  mkdir -p "$(dirname "$dir")"
  if [ -d "$dir/.git" ]; then
    progress "$label" "updating existing checkout at $dir"
    cd "$dir"
    git remote set-url origin "$repo" >>"$SETUP_LOG" 2>&1 || true
    git fetch origin "$branch" >>"$SETUP_LOG" 2>&1
  else
    if [ -e "$dir" ] && [ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
      fail "$dir exists but is not an empty git checkout"
    fi
    progress "$label" "cloning $repo into $dir"
    git clone "$repo" "$dir" >>"$SETUP_LOG" 2>&1
    cd "$dir"
    git fetch origin "$branch" >>"$SETUP_LOG" 2>&1
  fi

  cd "$dir"
  if git show-ref --verify --quiet "refs/heads/$branch"; then
    git checkout "$branch" >>"$SETUP_LOG" 2>&1
  else
    git checkout -B "$branch" "origin/$branch" >>"$SETUP_LOG" 2>&1
  fi

  if [ -n "$ref" ]; then
    git reset --hard "$ref" >>"$SETUP_LOG" 2>&1
  else
    git reset --hard "origin/$branch" >>"$SETUP_LOG" 2>&1
  fi

  git status --short --branch >>"$SETUP_LOG" 2>&1
  git log -1 --oneline --decorate >>"$SETUP_LOG" 2>&1
}}

build_with_log() {{
  local label="$1"
  local log_path="$2"
  shift 2
  progress "$label" "$*"
  : >"$log_path"
  if "$@" >"$log_path" 2>&1; then
    tail -120 "$log_path" >>"$SETUP_LOG" 2>&1 || true
  else
    local rc=$?
    echo "== $label failed rc=$rc ==" >>"$SETUP_LOG"
    tail -240 "$log_path" >>"$SETUP_LOG" 2>&1 || true
    exit "$rc"
  fi
}}

configure_memcache() {{
  progress "configure-mmc-local" "$LOCAL_CONF"
  [ -f "$LOCAL_CONF" ] || fail "missing config: $LOCAL_CONF"
  cp -a "$LOCAL_CONF" "$LOCAL_CONF.bak.$(date +%Y%m%d%H%M%S)"
  python - "$LOCAL_CONF" >>"$SETUP_LOG" 2>&1 <<'PY'
import re
import sys

path = sys.argv[1]
protocol_key = "ock.mmc.local_service.protocol"
client_values = [
    ("ock.mmc.client.read_thread_pool.size", "12"),
    ("ock.mmc.client.write_thread_pool.size", "4"),
    ("ock.mmc.client.batch_option.chunk.size", "1MB"),
]

with open(path, "r", encoding="utf-8") as handle:
    lines = handle.read().splitlines()

out = []
protocol_seen = False
client_keys = {{key for key, _ in client_values}}

for line in lines:
    if re.match(rf"^\\s*{{re.escape(protocol_key)}}\\s*=", line):
        if not protocol_seen:
            out.append(f"{{protocol_key}} = device_sdma")
            protocol_seen = True
        continue
    if any(re.match(rf"^\\s*{{re.escape(key)}}\\s*=", line) for key in client_keys):
        continue
    out.append(line)

if not protocol_seen:
    if out and out[-1] != "":
        out.append("")
    out.append(f"{{protocol_key}} = device_sdma")

if out and out[-1] != "":
    out.append("")
for key, value in client_values:
    out.append(f"{{key}} = {{value}}")

with open(path, "w", encoding="utf-8") as handle:
    handle.write("\\n".join(out) + "\\n")
PY
}}

configure_bashrc() {{
  progress "configure-bashrc" "$BASHRC"
  touch "$BASHRC"
  sed -i '/^# BEGIN CODEX MEMCACHE_ENV$/,/^# END CODEX MEMCACHE_ENV$/d' "$BASHRC"
  cat >>"$BASHRC" <<'EOF'

# BEGIN CODEX MEMCACHE_ENV
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf
# END CODEX MEMCACHE_ENV
EOF
}}

verify_setup() {{
  progress "verify" "packages, config, imports, and bashrc"
  if [ -f /usr/local/memcache_hybrid/set_env.sh ]; then
    source /usr/local/memcache_hybrid/set_env.sh >>"$SETUP_LOG" 2>&1 || true
  fi
  if [ -f /usr/local/memfabric_hybrid/set_env.sh ]; then
    source /usr/local/memfabric_hybrid/set_env.sh >>"$SETUP_LOG" 2>&1 || true
  fi
  export MMC_META_CONFIG_PATH="$META_CONF"
  export MMC_LOCAL_CONFIG_PATH="$LOCAL_CONF"
  python - "$ACTION" "$LOCAL_CONF" "$META_CONF" "$SETUP_LOG" "$MEMCACHE_BUILD_LOG" "$MEMFABRIC_BUILD_LOG" "$MEMCACHE_DIR" "$MEMFABRIC_DIR" "$BASHRC" <<'PY'
import importlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys

action, local_conf, meta_conf, setup_log, memcache_build_log, memfabric_build_log, memcache_dir, memfabric_dir, bashrc = sys.argv[1:10]

def package_info(name):
    try:
        module = importlib.import_module(name)
        return {{"version": metadata.version(name), "file": getattr(module, "__file__", None)}}
    except Exception as exc:
        return {{"error": f"{{type(exc).__name__}}: {{exc}}"}}

def git_head(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception as exc:
        return f"unknown: {{type(exc).__name__}}: {{exc}}"

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except FileNotFoundError:
        return ""

conf = read_text(local_conf)
keys = [
    "ock.mmc.local_service.protocol",
    "ock.mmc.client.read_thread_pool.size",
    "ock.mmc.client.write_thread_pool.size",
    "ock.mmc.client.batch_option.chunk.size",
]
settings = {{}}
counts = {{}}
for key in keys:
    pattern = re.compile(rf"^\\s*{{re.escape(key)}}\\s*=\\s*(.*?)\\s*$", re.MULTILINE)
    matches = pattern.findall(conf)
    counts[key] = len(matches)
    settings[key] = matches[-1] if matches else None

bashrc_text = read_text(bashrc)
bashrc_block_present = all(item in bashrc_text for item in [
    "source /usr/local/memcache_hybrid/set_env.sh",
    "source /usr/local/memfabric_hybrid/set_env.sh",
    "export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf",
    "export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf",
])

packages = {{
    "memcache_hybrid": package_info("memcache_hybrid"),
    "memfabric_hybrid": package_info("memfabric_hybrid"),
}}
expected = {{
    "ock.mmc.local_service.protocol": "device_sdma",
    "ock.mmc.client.read_thread_pool.size": "12",
    "ock.mmc.client.write_thread_pool.size": "4",
    "ock.mmc.client.batch_option.chunk.size": "1MB",
}}
checks_ok = (
    all("error" not in value for value in packages.values())
    and all(counts[key] == 1 for key in expected)
    and all(settings[key] == value for key, value in expected.items())
    and bashrc_block_present
)
result = {{
    "status": "ok" if checks_ok else "needs_attention",
    "action": action,
    "packages": packages,
    "config": {{
        "local_path": local_conf,
        "meta_path": meta_conf,
        "settings": settings,
        "active_counts": counts,
    }},
    "environment": {{
        "MMC_META_CONFIG_PATH": os.environ.get("MMC_META_CONFIG_PATH"),
        "MMC_LOCAL_CONFIG_PATH": os.environ.get("MMC_LOCAL_CONFIG_PATH"),
        "bashrc": bashrc,
        "bashrc_block_present": bashrc_block_present,
    }},
    "sources": {{
        "memcache": {{"path": memcache_dir, "head": git_head(memcache_dir)}},
        "memfabric": {{"path": memfabric_dir, "head": git_head(memfabric_dir)}},
    }},
    "logs": {{
        "setup": setup_log,
        "memcache_build": memcache_build_log,
        "memfabric_build": memfabric_build_log,
    }},
}}
print("__MEMCACHE_MEMFABRIC_RESULT__=" + json.dumps(result, ensure_ascii=False, sort_keys=True))
PY
}}

: >"$SETUP_LOG"
progress "start" "$ACTION"
require_cmd git
require_cmd python
require_cmd bash

if [ "$ACTION" = "install" ]; then
  clone_or_update "memcache-source" "$MEMCACHE_REPO" "$MEMCACHE_DIR" "$MEMCACHE_BRANCH" "$MEMCACHE_REF"
  progress "memcache-submodules" "3rdparty plus memfabric_hybrid master"
  cd "$MEMCACHE_DIR"
  git submodule update --init 3rdparty/ >>"$SETUP_LOG" 2>&1
  git -c submodule.3rdparty/memfabric_hybrid.branch=master submodule update --remote 3rdparty/memfabric_hybrid >>"$SETUP_LOG" 2>&1
  git submodule status 3rdparty/ >>"$SETUP_LOG" 2>&1

  cd "$MEMCACHE_DIR"
  build_with_log "build-memcache" "$MEMCACHE_BUILD_LOG" bash script/build_and_pack_run.sh --build_mode RELEASE --build_test OFF
  progress "install-memcache" "output/memcache_hybrid-*_linux_aarch64.run"
  bash output/memcache_hybrid-*_linux_aarch64.run >>"$SETUP_LOG" 2>&1
  progress "uninstall-default-memfabric" "pip uninstall -y memfabric_hybrid"
  python -m pip uninstall -y memfabric_hybrid >>"$SETUP_LOG" 2>&1 || true

  clone_or_update "memfabric-source" "$MEMFABRIC_REPO" "$MEMFABRIC_DIR" "$MEMFABRIC_BRANCH" "$MEMFABRIC_REF"
  cd "$MEMFABRIC_DIR"
  build_with_log "build-memfabric" "$MEMFABRIC_BUILD_LOG" bash script/build_and_pack_run.sh
  progress "install-memfabric" "output/memfabric_hybrid-*_*_*.run"
  bash output/memfabric_hybrid-*_*_*.run >>"$SETUP_LOG" 2>&1

  configure_memcache
  configure_bashrc
fi

verify_setup
"""


def run_remote(args: argparse.Namespace, script: str) -> tuple[int, str, str]:
    target = f"{args.user}@{args.host}"
    remote_cmd = f"docker exec -i {shell_quote(args.container)} bash -se"
    cmd = [
        "ssh",
        "-p",
        str(args.port),
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
    ]
    if args.identity_file:
        cmd.extend(["-i", args.identity_file])
    cmd.extend([target, remote_cmd])

    print(
        PROGRESS_PREFIX
        + json.dumps(
            {"step": "connect", "detail": f"{target}:{args.port} container={args.container}"},
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def pump_stdout() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_chunks.append(line)

    def pump_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_chunks.append(line)
            print(line, file=sys.stderr, end="")

    out_thread = threading.Thread(target=pump_stdout, daemon=True)
    err_thread = threading.Thread(target=pump_stderr, daemon=True)
    out_thread.start()
    err_thread.start()

    assert proc.stdin is not None
    try:
        proc.stdin.write(script)
        proc.stdin.close()
    except BrokenPipeError:
        pass

    returncode = proc.wait()
    out_thread.join(timeout=5)
    err_thread.join(timeout=5)
    return returncode, "".join(stdout_chunks), "".join(stderr_chunks)


def parse_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "install"])
    parser.add_argument("--host", required=True, help="Remote SSH host or IP")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="root")
    parser.add_argument("--container", default="zyj_offload")
    parser.add_argument("--identity-file")
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--workspace", default="/home/zyj/codes/offload")
    parser.add_argument("--memcache-repo", default="https://gitcode.com/Ascend/memcache.git")
    parser.add_argument("--memcache-dir")
    parser.add_argument("--memcache-branch", default="develop")
    parser.add_argument("--memcache-ref", default="ebdb48")
    parser.add_argument(
        "--memfabric-repo",
        default="https://gitcode.com/wlwen/memfabric-hybrid_kvoffload.git",
    )
    parser.add_argument("--memfabric-dir")
    parser.add_argument("--memfabric-branch", default="release_kv_v2")
    parser.add_argument(
        "--memfabric-ref",
        default="",
        help="Optional commit/ref to reset to after checking out --memfabric-branch",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    script = make_remote_script(args)
    returncode, stdout, stderr = run_remote(args, script)
    result = parse_result(stdout)

    if result is None:
        result = {
            "status": "failed",
            "action": args.action,
            "returncode": returncode,
            "error": "remote command did not emit final result JSON",
            "stdout_tail": tail_text(stdout),
            "stderr_tail": tail_text(stderr),
        }
    else:
        result["returncode"] = returncode

    result["target"] = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "container": args.container,
    }

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if returncode == 0 else returncode


if __name__ == "__main__":
    raise SystemExit(main())
