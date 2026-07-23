from __future__ import annotations

import importlib
import os
import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPO_ROOT = ROOT.parent

import core.state_store as state_store  # noqa: E402
import mcp.tools as mcp_tools  # noqa: E402
from core.endpoint import Endpoint  # noqa: E402
from core.ssh_transport import RemoteCompleted  # noqa: E402
from mcp.schemas import ALIASES, ENDPOINT_SELECTOR_ANY_OF, TOOL_SCHEMAS  # noqa: E402
from mcp.tools import list_resources, list_tools, read_resource  # noqa: E402


class McpSchemaTests(unittest.TestCase):
    def test_all_expected_tools_are_listed(self) -> None:
        names = {tool["name"] for tool in list_tools()}
        self.assertEqual(names, set(TOOL_SCHEMAS))
        for expected in (
            "remote.read",
            "remote.write",
            "remote.edit",
            "remote.multi_edit",
            "remote.bash",
            "remote.glob",
            "remote.grep",
            "remote.ls",
            "remote.monitor",
            "remote.apply_patch",
            "remote.job_status",
            "remote.job_tail",
            "remote.job_stop",
            "remote.artifact_manifest",
            "remote.artifact_pull",
            "remote.artifact_push",
            "remote.context_snapshot",
            "remote.probe",
        ):
            self.assertIn(expected, names)
            self.assertIn("inputSchema", next(tool for tool in list_tools() if tool["name"] == expected))

    def test_underscore_aliases_map_to_canonical_names(self) -> None:
        self.assertEqual(ALIASES["remote_read"], "remote.read")
        self.assertIn("remote.bash", TOOL_SCHEMAS)

    def test_normal_tools_express_endpoint_selector_anyof(self) -> None:
        job_tools = {"remote.job_status", "remote.job_tail", "remote.job_stop"}
        for name, schema in TOOL_SCHEMAS.items():
            if name in job_tools:
                self.assertNotIn({"anyOf": ENDPOINT_SELECTOR_ANY_OF}, schema.get("allOf", []))
            else:
                self.assertIn({"anyOf": ENDPOINT_SELECTOR_ANY_OF}, schema.get("allOf", []), name)
        self.assertIn({"required": ["host", "port"]}, ENDPOINT_SELECTOR_ANY_OF)
        self.assertIn({"required": ["alias"]}, ENDPOINT_SELECTOR_ANY_OF)
        self.assertIn({"required": ["session_id"]}, ENDPOINT_SELECTOR_ANY_OF)

    def test_moonshot_flavor_removes_combinators(self) -> None:
        from mcp.schemas import APPLY_PATCH_DESCRIPTION, ENDPOINT_SELECTOR_DESCRIPTION, resolve_tool_schemas

        flavored = resolve_tool_schemas("moonshot")
        self.assertEqual(set(flavored), set(TOOL_SCHEMAS))
        for name, schema in flavored.items():
            self.assertEqual(schema.get("type"), "object", name)
            for combinator in ("anyOf", "oneOf", "allOf"):
                self.assertNotIn(combinator, schema, name)
        self.assertIn(ENDPOINT_SELECTOR_DESCRIPTION, flavored["remote.read"].get("description", ""))
        apply_patch_description = flavored["remote.apply_patch"].get("description", "")
        self.assertIn(ENDPOINT_SELECTOR_DESCRIPTION, apply_patch_description)
        self.assertIn(APPLY_PATCH_DESCRIPTION, apply_patch_description)
        for name in ("remote.job_status", "remote.job_tail", "remote.job_stop"):
            self.assertNotIn("description", flavored[name], name)

    def test_standard_flavor_remains_the_default(self) -> None:
        from mcp.schemas import SCHEMA_FLAVOR_ENV, resolve_tool_schemas

        original = os.environ.pop(SCHEMA_FLAVOR_ENV, None)
        try:
            self.assertIs(resolve_tool_schemas(), TOOL_SCHEMAS)
            self.assertIs(resolve_tool_schemas("standard"), TOOL_SCHEMAS)
            self.assertIn({"anyOf": ENDPOINT_SELECTOR_ANY_OF}, TOOL_SCHEMAS["remote.read"].get("allOf", []))
            self.assertIn("anyOf", TOOL_SCHEMAS["remote.apply_patch"])
        finally:
            if original is not None:
                os.environ[SCHEMA_FLAVOR_ENV] = original

    def test_list_tools_honors_moonshot_flavor_env(self) -> None:
        from mcp.schemas import SCHEMA_FLAVOR_ENV

        original = os.environ.pop(SCHEMA_FLAVOR_ENV, None)
        os.environ[SCHEMA_FLAVOR_ENV] = "moonshot"
        try:
            schemas = {tool["name"]: tool["inputSchema"] for tool in list_tools()}
            self.assertEqual(set(schemas), set(TOOL_SCHEMAS))
            for name, schema in schemas.items():
                for combinator in ("anyOf", "oneOf", "allOf"):
                    self.assertNotIn(combinator, schema, name)
        finally:
            os.environ.pop(SCHEMA_FLAVOR_ENV, None)
            if original is not None:
                os.environ[SCHEMA_FLAVOR_ENV] = original

    def test_resources_include_endpoint_index(self) -> None:
        resources = {resource["uri"] for resource in list_resources()}
        self.assertIn("remote://endpoints", resources)
        content = read_resource("remote://endpoints")
        self.assertEqual(content["mimeType"], "application/json")
        self.assertIn("endpoints", json.loads(content["text"]))

    def test_resources_include_and_read_job_resources(self) -> None:
        original_state_root = state_store.substrate_root
        original_run_script = mcp_tools.run_script
        try:
            with tempfile.TemporaryDirectory() as tmp:
                state_store.substrate_root = lambda: Path(tmp)  # type: ignore[assignment]
                endpoint = Endpoint(host="127.0.0.1", port=46000, root="/vllm-workspace")
                state_store.ensure_endpoint_state(endpoint)
                job_id = "job-abc123"
                record = {
                    "schema_version": "remote-dev.job.v1",
                    "job_id": job_id,
                    "target": endpoint.to_result_target(),
                    "remote_dir": f"{endpoint.root}/.remote-dev/jobs/{job_id}",
                    "started_at": "2026-05-25T00:00:00Z",
                }
                state_store.atomic_write_json(state_store.job_record_path(endpoint, job_id), record)
                mcp_tools.run_script = lambda *_args, **_kwargs: RemoteCompleted(0, "log\n", "")  # type: ignore[assignment]

                base = f"remote://endpoint/{endpoint.endpoint_id}/job/{job_id}"
                resources = {resource["uri"] for resource in list_resources()}
                self.assertIn(base + "/status", resources)
                self.assertIn(base + "/stdout", resources)
                self.assertIn(base + "/stderr", resources)

                status = read_resource(base + "/status")
                self.assertEqual(json.loads(status["text"])["job_id"], job_id)
                stdout = read_resource(base + "/stdout")
                self.assertEqual(stdout["mimeType"], "text/plain")
                self.assertEqual(stdout["text"], "log\n")
        finally:
            state_store.substrate_root = original_state_root  # type: ignore[assignment]
            mcp_tools.run_script = original_run_script  # type: ignore[assignment]

    def test_resources_include_and_read_artifact_manifest(self) -> None:
        original_state_root = state_store.substrate_root
        try:
            with tempfile.TemporaryDirectory() as tmp:
                state_store.substrate_root = lambda: Path(tmp)  # type: ignore[assignment]
                endpoint = Endpoint(host="127.0.0.1", port=46000, root="/vllm-workspace")
                state_store.ensure_endpoint_state(endpoint)
                manifest = {
                    "schema_version": "remote-dev.artifact_manifest.v1",
                    "status": "ok",
                    "endpoint_id": endpoint.endpoint_id,
                    "file_count": 0,
                    "files": [],
                }
                artifact_id = "artifact-abc123"
                manifest_path = state_store.artifacts_dir(endpoint.endpoint_id) / artifact_id / "manifest.json"
                state_store.atomic_write_json(manifest_path, manifest)

                uri = f"remote://endpoint/{endpoint.endpoint_id}/artifacts/{artifact_id}/manifest"
                resources = {resource["uri"] for resource in list_resources()}
                self.assertIn(uri, resources)
                content = read_resource(uri)
                self.assertEqual(content["mimeType"], "application/json")
                self.assertEqual(json.loads(content["text"])["schema_version"], "remote-dev.artifact_manifest.v1")
        finally:
            state_store.substrate_root = original_state_root  # type: ignore[assignment]

    def test_context_snapshot_can_skip_live_probe(self) -> None:
        from mcp.tools import call_tool

        payload = call_tool(
            "remote.context_snapshot",
            {
                "host": "example.invalid",
                "port": 22,
                "root": "/vllm-workspace",
                "live_probe": False,
            },
        )
        self.assertEqual(payload["result"]["outcome"], "success")
        self.assertEqual(payload["result"]["tool"], "remote.context_snapshot")

    def test_server_supports_content_length_framing(self) -> None:
        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        encoded = json.dumps(request, separators=(",", ":")).encode("utf-8")
        framed = b"Content-Length: " + str(len(encoded)).encode("ascii") + b"\r\n\r\n" + encoded
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / ".remote-dev" / "mcp" / "server.py")],
            input=framed,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", errors="replace"))
        header, body = proc.stdout.split(b"\r\n\r\n", 1)
        self.assertIn(b"Content-Length:", header)
        response = json.loads(body.decode("utf-8"))
        self.assertEqual(response["id"], 1)
        self.assertIn("tools", response["result"])

    def test_mcp_server_sets_process_level_ledger_scope(self) -> None:
        from core.state_store import resolve_ledger_scope

        original = os.environ.pop("REMOTE_DEV_SESSION_ID", None)
        try:
            import mcp.server as mcp_server

            importlib.reload(mcp_server)
            scope = resolve_ledger_scope()
            self.assertTrue(scope.startswith("mcp-"))
            self.assertNotEqual(scope, "default")
        finally:
            if original is None:
                os.environ.pop("REMOTE_DEV_SESSION_ID", None)
            else:
                os.environ["REMOTE_DEV_SESSION_ID"] = original


if __name__ == "__main__":
    unittest.main()
