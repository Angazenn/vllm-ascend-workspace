from __future__ import annotations

import json
import posixpath
import re
import shlex
import sys
import contextlib
from dataclasses import dataclass
from typing import Any

RAW_REMOTE_RE = re.compile(r"(^|\s)(ssh|scp|sftp|rsync)\b")
PASSWORD_RE = re.compile(r"(?i)(sshpass|expect\b|--password(?:=|\s+)\S+|password=\S+|token=\S+|api[_-]?key=\S+)")
DISRUPTIVE_REMOTE_RE = re.compile(
    r"(?ix)(?:"
    r"\bnpu-smi\b[^\n;&|]*(?:\breset\b|(?:^|\s)-r(?:\s|$))|"
    r"\bdcmi[^\n;&|]*\breset\b|"
    r"(?:^|[;&|]\s*|\bsudo\s+)\b(?:reboot|shutdown\s+-r)\b|"
    r"\b(?:ssh\s+\S+[^\n]*|docker\s+exec\s+\S+[^\n]*|nsenter\b[^\n]*)\b(?:kill|pkill|killall)\b"
    r")"
)
REMOTE_PATH_FIELDS = ("file_path", "path", "cwd", "remote_path")
REMOTE_TOOL_PATH_FIELDS = {
    "remote.read": ("file_path",),
    "remote.write": ("file_path",),
    "remote.edit": ("file_path",),
    "remote.multi_edit": ("file_path",),
    "remote.bash": ("cwd",),
    "remote.glob": ("path",),
    "remote.grep": ("path",),
    "remote.ls": ("path",),
    "remote.monitor": ("cwd",),
    "remote.apply_patch": ("cwd",),
    "remote.artifact_manifest": ("remote_path",),
    "remote.artifact_pull": ("remote_path",),
    "remote.artifact_push": ("remote_path",),
    "remote.context_snapshot": (),
    "remote.probe": (),
}


@dataclass
class GuardDecision:
    action: str
    reason: str | None = None
    additional_context: str | None = None

    @property
    def blocked(self) -> bool:
        return self.action == "deny"


def read_hook_payload() -> dict[str, Any]:
    text = sys.stdin.read()
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return data if isinstance(data, dict) else {"value": data}


def extract_command(payload: dict[str, Any]) -> str:
    for key in ("command", "cmd"):
        if isinstance(payload.get(key), str):
            return payload[key]
    tool_input = payload.get("tool_input") or payload.get("input") or payload.get("arguments")
    if isinstance(tool_input, dict):
        for key in ("command", "cmd"):
            if isinstance(tool_input.get(key), str):
                return tool_input[key]
    if isinstance(payload.get("raw"), str):
        return payload["raw"]
    return ""


def extract_tool_name(payload: dict[str, Any]) -> str:
    raw = str(payload.get("tool_name") or payload.get("tool") or payload.get("name") or "")
    if raw:
        return raw
    tool_call = payload.get("tool_call")
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or tool_call.get("tool_name") or "")
    return ""


def extract_tool_input(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("tool_input", "input", "arguments"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    params = payload.get("params")
    if isinstance(params, dict):
        arguments = params.get("arguments")
        if isinstance(arguments, dict):
            return arguments
    tool_call = payload.get("tool_call")
    if isinstance(tool_call, dict):
        for key in ("arguments", "input"):
            value = tool_call.get(key)
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                with contextlib.suppress(json.JSONDecodeError):
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        return parsed
    return {}


def canonical_remote_tool_name(tool_name: str) -> str | None:
    if not tool_name:
        return None
    name = tool_name
    if "__" in name:
        name = name.split("__")[-1]
    if name.startswith("remote."):
        canonical = name
    elif name.startswith("remote_"):
        canonical = "remote." + name.removeprefix("remote_").replace("_", ".")
        if canonical not in REMOTE_TOOL_PATH_FIELDS:
            canonical = "remote." + name.removeprefix("remote_")
    else:
        return None
    return canonical if canonical in REMOTE_TOOL_PATH_FIELDS else None


def normalize_remote_path(root: str, cwd: str, value: str) -> str:
    candidate = value if value.startswith("/") else posixpath.join(cwd, value)
    return posixpath.normpath(candidate)


def is_under_root(root: str, path: str) -> bool:
    root_norm = posixpath.normpath(root)
    return path == root_norm or path.startswith(root_norm.rstrip("/") + "/")


def inspect_remote_tool_call(tool_name: str, tool_input: dict[str, Any]) -> GuardDecision:
    canonical = canonical_remote_tool_name(tool_name)
    if not canonical:
        return GuardDecision("allow")
    if canonical in {"remote.bash", "remote.monitor"}:
        command = str(tool_input.get("command") or "")
        if DISRUPTIVE_REMOTE_RE.search(command) and not bool(tool_input.get("approve_disruptive_action")):
            return GuardDecision(
                "deny",
                "NPU/host reset or outside-target process termination requires separate user approval; rerun with approve_disruptive_action=true only after confirmation.",
            )
    return GuardDecision("allow")


def shell_words(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def inspect_command(command: str) -> GuardDecision:
    if DISRUPTIVE_REMOTE_RE.search(command):
        return GuardDecision(
            "deny",
            "Use remote.bash and request separate user approval before an NPU/host reset or outside-target process termination.",
        )
    return GuardDecision("allow")


def inspect_payload(payload: dict[str, Any]) -> GuardDecision:
    tool_name = extract_tool_name(payload)
    tool_input = extract_tool_input(payload)
    if canonical_remote_tool_name(tool_name):
        return inspect_remote_tool_call(tool_name, tool_input)
    return inspect_command(extract_command(payload))


def codex_response(decision: GuardDecision) -> dict[str, Any]:
    if decision.blocked:
        return {"decision": "deny", "reason": decision.reason or "blocked by remote-dev guard"}
    if decision.additional_context:
        return {"decision": "allow", "additionalContext": decision.additional_context}
    return {"decision": "allow"}
