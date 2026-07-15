---
name: memcache-memfabric-setup
description: Build, install, verify, and configure MemCache Hybrid plus the special MemFabric Hybrid KV offload fork inside a remote Docker container. Use when Codex is asked to set up memcache_hybrid, replace memfabric_hybrid with the wlwen/memfabric-hybrid_kvoffload release_kv_v2 fork, edit mmc-local.conf for device_sdma, or persist MemCache/MemFabric environment variables in container bashrc.
---

# MemCache MemFabric Setup

Use this skill for the fragile source-build workflow needed by vLLM Ascend KV offload experiments. The workflow targets a remote Docker container and produces final JSON on stdout with progress on stderr.

Prefer the bundled script instead of recreating SSH, git, build, installer, and config-edit shell by hand.

## Entry Points

Check current package/config/env state without changing source trees or installed files:

```bash
python3 .agents/skills/memcache-memfabric-setup/scripts/setup_memcache_memfabric.py status \
  --host 192.168.13.165 \
  --container zyj_offload
```

Run the complete setup:

```bash
python3 .agents/skills/memcache-memfabric-setup/scripts/setup_memcache_memfabric.py install \
  --host 192.168.13.165 \
  --container zyj_offload
```

Defaults:

- SSH user: `root`
- SSH port: `22`
- container: `zyj_offload`
- workspace: `/home/zyj/codes/offload`
- MemCache repo: `https://gitcode.com/Ascend/memcache.git`
- MemCache branch/ref: `develop` at `ebdb48`
- MemFabric fork: `https://gitcode.com/wlwen/memfabric-hybrid.git`
- MemFabric branch: `develop`

## Install Workflow

The `install` action performs these steps in order:

1. Clone or update `/home/zyj/codes/offload/memcache`.
2. Check out MemCache `develop` and hard reset it to `ebdb48`.
3. Initialize `3rdparty/` submodules.
4. Update `3rdparty/memfabric_hybrid` from `master`.
5. Build MemCache with:

```bash
bash script/build_and_pack_run.sh --build_mode RELEASE --build_test OFF
```

6. Run `bash output/memcache_hybrid-*_linux_aarch64.run`.
7. Uninstall the wheel-provided `memfabric_hybrid`.
8. Clone or update `/home/zyj/codes/offload/memfabric-hybrid_kvoffload`.
9. Check out MemFabric fork branch `release_kv_v2`.
10. Build it with `bash script/build_and_pack_run.sh`.
11. Run `bash output/memfabric_hybrid-*_*_*.run`.
12. Configure `/usr/local/memcache_hybrid/latest/config/mmc-local.conf`.
13. Add the MemCache/MemFabric environment block to `/root/.bashrc`.
14. Verify packages, imports, config values, and bashrc block.

The script refuses to use a non-empty source directory that is not a Git checkout. Existing Git checkouts are updated and reset only inside the configured MemCache and MemFabric source directories.

## Config Contract

`mmc-local.conf` must end with exactly one active protocol key:

```conf
ock.mmc.local_service.protocol = device_sdma
```

The following active client settings must also be present exactly once:

```conf
ock.mmc.client.read_thread_pool.size = 12
ock.mmc.client.write_thread_pool.size = 4
ock.mmc.client.batch_option.chunk.size = 1MB
```

The script removes duplicate active lines for these keys and appends the desired client settings. It creates a timestamped backup beside `mmc-local.conf` before editing.

## Environment Contract

The container root `.bashrc` gets this managed block:

```bash
# BEGIN CODEX MEMCACHE_ENV
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf
# END CODEX MEMCACHE_ENV
```

On updates, replace only the managed block.

## Output Rules

- Report the JSON summary fields that matter: `status`, package versions, config values, source commits, and remote log paths.
- Do not paste full build logs unless a failure needs diagnosis.
- If SSH or network access is blocked by sandboxing, rerun the script command with approval.
