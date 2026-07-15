---
name: glm5-sfa-prefill-offload-verification
description: Launch and verify a GLM5 SFA prefill offload vLLM service on a configurable remote SSH host/container. Use when Codex is asked to start or verify the GLM5 offload server, wait for readiness, run the remote request script, and report the returned response text.
---

# GLM5 SFA Prefill Offload Verification

## Overview

Verify the GLM5 SFA prefill offload path on a remote development target. The workflow starts or reuses the vLLM server, checks `/health` and `/v1/models`, sends the remote request script, and reports the returned assistant text.

The verifier is configurable. Defaults preserve the original ad-hoc endpoint:

- host: `192.168.13.165`
- SSH user: current SSH default
- SSH port: `22`
- container: `zyj_offload`
- scripts directory: `/home/zyj/scripts`
- launch script: `/home/zyj/scripts/launch_vllm_offload.sh`
- request script: `/home/zyj/scripts/curl.sh`
- launch log: `/home/zyj/scripts/launch_vllm_offload.nohup.log`
- readiness log: `/home/zyj/scripts/log.txt`
- port: `8900`
- expected served model: `glm5`
- container shell mode: `noninteractive` (`bash -lc`)

## Safety boundary

- Health probes, requests, and restarting vLLM or the selected target container
  do not require approval.
- Ask for explicit approval before resetting an NPU/host or killing a process
  outside the target container. If NPU occupancy cannot be cleared from inside
  the container, report the blocker and stop.

When the user gives a remote address, SSH user/port, container, scripts directory, launch script, request script, service port, or shell mode, pass those values to the verifier. Do not edit this skill just to change targets.

Use `--container-shell interactive` when the user says their manual launch works from an interactive docker shell, or when `.bashrc` gates important runtime setup behind an interactive-shell check. This makes container commands use `bash -ic` instead of the default `bash -lc`.

## Quick Start

Run the bundled verifier from the workspace root:

```bash
python3 .agents/skills/glm5-sfa-prefill-offload-verification/scripts/verify_glm5_sfa_prefill_offload.py
```

Run against a non-default target:

```bash
python3 .agents/skills/glm5-sfa-prefill-offload-verification/scripts/verify_glm5_sfa_prefill_offload.py \
  --host 192.168.13.157 \
  --ssh-user root \
  --ssh-port 22 \
  --container zyj_offload \
  --scripts-dir /home/zyj/scripts \
  --launch-script /home/zyj/scripts/launch_vllm_offload.sh \
  --request-script /home/zyj/scripts/curl.sh \
  --container-shell interactive \
  --port 8900
```

The script prints progress to stderr and a final JSON payload to stdout. In the final answer to the user, report:

- whether launch was skipped or started
- health status and base URL
- served model id from `/v1/models`
- request `finish_reason`
- returned text, preferring `choices[0].message.content` and falling back to `choices[0].message.reasoning`
- log path from the script output when diagnosis is needed

## Workflow

### 1. Check Existing Service

Probe the host and container:

```bash
ssh <ssh-target> "docker exec <container> bash -lc 'source ~/.bashrc >/dev/null 2>&1; curl --max-time 2 -s -o /dev/null -w \"%{http_code}\" http://127.0.0.1:<port>/health || true'"
```

With `--container-shell interactive`, the verifier uses `docker exec <container> bash -ic ...` for health, model, and request commands. The launch command also backgrounds `bash -ic 'source ~/.bashrc; cd <scripts-dir>; bash <launch-script>'` so the long-running server is started from an interactive shell environment.

If health is already `200`, reuse the running service. Do not start a duplicate vLLM process.

If health is not ready but `pgrep -af` shows an active launch or vLLM process in the container, wait for readiness instead of starting another copy. The script derives the launch-process pattern from `--launch-script`; override it with `--process-pattern` when needed.

### 2. Launch When Needed

When `/health` is not `200`, start the server from the configured remote scripts directory. The launch script is not necessarily executable, so run it with `bash`.

```bash
ssh <ssh-target> "docker exec <container> bash -lc 'source ~/.bashrc >/dev/null 2>&1; cd <scripts-dir>; nohup bash <launch-script> > <launch-log-path> 2>&1 < /dev/null & echo LAUNCH_PID=\$!'"
```

Always source `~/.bashrc` inside the container shell before launch. Expect sourcing to take roughly 15 seconds on this image.

### 3. Wait For Readiness

Poll `/health` until it returns `200`. While waiting, inspect the configured readiness log for fatal startup signatures:

- `Traceback`
- `AssertionError`
- `RuntimeError`
- `ValueError`
- `StopIteration`

If readiness times out, return the tail of the configured readiness log and do not claim success.

### 4. Verify Served Model

Check models:

```bash
ssh <ssh-target> "docker exec <container> bash -lc 'source ~/.bashrc >/dev/null 2>&1; curl --max-time 10 -s http://127.0.0.1:<port>/v1/models'"
```

The expected model id is `glm5`. If the request script returns `The model ... does not exist`, the configured remote request script is targeting the wrong model name; ask the user to fix it or edit the remote script only when explicitly requested.

### 5. Send Verification Request

Run the configured remote request script from the configured scripts directory:

```bash
ssh <ssh-target> "docker exec <container> bash -lc 'source ~/.bashrc >/dev/null 2>&1; cd <scripts-dir>; if [ -x <request-script> ]; then <request-script>; else bash <request-script>; fi'"
```

Parse the JSON response. For chat completions, returned text can be in `choices[0].message.content`; for this GLM5 reasoning path it may instead be in `choices[0].message.reasoning`.

## Automation Script

Use `scripts/verify_glm5_sfa_prefill_offload.py` for normal work. Useful options:

```bash
python3 .agents/skills/glm5-sfa-prefill-offload-verification/scripts/verify_glm5_sfa_prefill_offload.py \
  --host 192.168.13.165 \
  --ssh-user root \
  --ssh-port 22 \
  --container zyj_offload \
  --scripts-dir /home/zyj/scripts \
  --launch-script /home/zyj/scripts/launch_vllm_offload.sh \
  --request-script /home/zyj/scripts/curl.sh \
  --port 8900 \
  --container-shell noninteractive \
  --health-timeout 900
```

Use `--request-only` when the server is already known to be ready and the user only asks to resend `curl.sh`.

Supported environment overrides:

- `GLM5_SFA_OFFLOAD_HOST`
- `GLM5_SFA_OFFLOAD_SSH_USER`
- `GLM5_SFA_OFFLOAD_SSH_PORT`
- `GLM5_SFA_OFFLOAD_CONTAINER`
- `GLM5_SFA_OFFLOAD_SCRIPTS_DIR`
- `GLM5_SFA_OFFLOAD_LAUNCH_SCRIPT`
- `GLM5_SFA_OFFLOAD_LAUNCH_LOG_PATH`
- `GLM5_SFA_OFFLOAD_REQUEST_SCRIPT`
- `GLM5_SFA_OFFLOAD_LOG_PATH`
- `GLM5_SFA_OFFLOAD_PROCESS_PATTERN`
- `GLM5_SFA_OFFLOAD_PORT`
- `GLM5_SFA_OFFLOAD_CONTAINER_SHELL` (`noninteractive` or `interactive`)
