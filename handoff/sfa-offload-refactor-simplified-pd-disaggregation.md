# SFA Offload Refactor Simplified: PD Disaggregation

## Snapshot

- Date: 2026-07-15
- Workspace: `vllm-ascend-workspace`
- Repository: `vllm-ascend`
- Branch: `offload_merge`
- Current commit: `3196e228 Restrict direct SFA host offload to PD decode nodes`
- Model verified: GLM-5.2 w8a8
- Parallel configuration verified: TP16, EP enabled, MTP with one draft token
- P node: `192.168.13.157`, container `zyj_offload`, port `8100`
- D node: `192.168.13.165`, container `zyj_offload`, port `8200`
- Proxy: P node, port `8900`
- State: implemented, committed, and remotely verified end to end

This handoff documents the simplified SFA PD-disaggregation design after the
tail-free direct decode refactor. It supersedes the older decode-layout parts
of `sfa-offload-refactor-simplified.md` and
`sfa-offload-split-cache-refactor.md` for the current branch.

The relevant commit sequence is:

```text
f16150f6  SFA indexer spec refactor
c4341cdc  Refactor SFA layerwise prefill offload cache
70dfb804  Support split-cache SFA PD disaggregation
64cb5732  Refactor direct SFA decode offload without tail buffers
3196e228  Restrict direct SFA host offload to PD decode nodes
```

## Design Summary

P and D intentionally use different physical cache layouts while keeping one
compatible logical `UniformTypeKVCacheSpecs` group.

```text
P node
  use_offload=false
  normal paged main KV and real-indexer HBM caches
  layerwise physical reuse
  AscendStoreConnector stores prefill KV/indexer in MemCache
  SFAPDCpuOffloadConnector exposes the same HBM data to D through MemFabric

D node
  use_offload=true
  one logical Uniform cache group for scheduler block management
  paged HBM allocated only for real indexer owners
  main KV is authoritative in connector-owned CPU memory
  fixed per-layer NPU resident rows hold selected historical tokens and the
  current decode/MTP tokens
  no paged main-KV HBM cache and no separate tail buffers
```

The PD wire path adapts these local layouts. P does not need to allocate D's
resident workspace, and D does not need to reproduce P's layer-reuse pools.

## Mode Selection

Direct SFA host offload is enabled only when all of these are true:

```text
kv_connector == "SFAPDCpuOffloadConnector"
kv_role is consumer-only
additional_config.use_offload == true
additional_config.enable_sparse_c8 == false
```

This check is centralized in `is_direct_sfa_host_offload()`.

The connector enforces the asymmetric launch contract at startup:

```text
P producer: use_offload=false
D consumer: use_offload=true and use_layerwise=true
```

It also requires BF16 cache and `DCP * PCP == 1`. A local prefill batch on the
D direct-host path raises an error. This keeps the optimization specific to a
PD decode node and prevents the old colocated/standalone behavior from being
selected accidentally.

`use_offload=true` with another connector is no longer supported. The legacy
standalone `SFAKVOffloadConnector` connector and scheduler were deleted; the
remaining SFA host worker is an internal component of the D-side
`SFAPDCpuOffloadConnector`.

## One Logical Cache Group

The scheduler sees one `UniformTypeKVCacheSpecs` group containing:

```text
main specs:     one AscendMLAAttentionSpec for every main SFA layer
indexer specs:  one AscendSFAIndexerCacheSpec for every real indexer owner
```

Only real model indexer modules publish indexer specs. GLM-5.2 `skip_topk`
layers do not receive fake indexer owners, allocations, or transfer entries.

Keeping both spec types in one Uniform group gives main and indexer layers the
same scheduler block-ID namespace. Main specs still participate in scheduler
block allocation even though their D-side bytes do not live in paged HBM.

### D-Side Capacity

The D worker replaces normal physical allocation with:

```text
physical HBM tensors = real indexer tensors only
main KV bytes        = connector-owned host cache
```

The block count is bounded independently by HBM and host capacity:

```text
physical_page_bytes = sum(page_size of every real indexer spec)
host_page_bytes     = sum(page_size of every main SFA spec)

hbm_limit  = floor(available_hbm / physical_page_bytes)
host_limit = floor(128 GiB / host_page_bytes)
num_blocks = min(hbm_limit, host_limit)
```

An explicit block override is rejected when it exceeds either limit. The
startup max-length check counts only physical indexer HBM because main KV is
host-resident.

For the verified GLM-5.2 run, D allocated `1730` blocks. Its actual main CPU
pool was about `18.77 GiB`; the `128 GiB` constant is a capacity ceiling, not
an unconditional allocation.

## D Runtime Cache

Every main SFA attention implementation owns fixed-address resident tensors:

```text
resident K: [max_topk_rows, resident_capacity, 1, kv_lora_rank]
resident V: [max_topk_rows, resident_capacity, 1, qk_rope_head_dim]
```

where:

```text
decode_width = 1 + num_speculative_tokens
max_topk_rows = min(max_num_batched_tokens,
                    max_num_seqs * decode_width)
managed_capacity = resident_capacity - decode_width
```

The five-entry compatibility tuple is:

```text
0: block-shaped alias of resident K, used by the KV writer
1: block-shaped alias of resident V/rope, used by the KV writer
2: real indexer tensor when this layer owns one, otherwise an empty sentinel
3: row-shaped resident K workspace
4: row-shaped resident V/rope workspace
```

Entries `0/1` and `3/4` are views over the same storage. Real indexer cache is
bound independently and composed into tuple entry 2 only for an actual owner.
This compatibility tuple can be removed once the SFA kernel accepts separate
main and indexer handles directly.

### Resident Row Layout

Each query row is split into two regions:

```text
[0, managed_capacity)                    historical LRU-resident tokens
[managed_capacity, resident_capacity)    current decode/MTP fresh window
```

New KV is written directly into the fresh window. Slot selection uses stable
request-row metadata plus the token's offset within the current decode step:

```text
slot = request_row * resident_capacity
     + managed_capacity
     + fresh_offset
```

The model runner supplies `req_ids_tensor`, `token_to_req`, and
`tokens_per_req` only in direct D mode. These tensors make the path robust to
batch reordering and to multiple MTP rows per request.

### Tail-Free Sparse Attention

There is no separate two-block NPU tail cache and no two-attention LSE merge.
Top-k indices are partitioned into:

```text
historical tokens -> LRU lookup and CPU-to-NPU load into managed slots
current tokens    -> already present in the fresh window
```

The resulting resident slot IDs are passed to one
`npu_sparse_flash_attention` call. MTP query rows copy the request's canonical
fresh window into each query row before attention so every causal draft row
can address the current token window.

The resident tensors have stable addresses and fixed shapes, so the same path
works with `FULL_DECODE_ONLY` graph capture. The current implementation passes
holey sparse indices directly with `sparse_indices_discrete=true`.

### Token-Wise Host Save

After a decode step, new K/V rows are copied from the fresh resident window to
their final positions in the CPU block cache. Saving does not wait for a full
128-token block and does not copy data back through a paged HBM scratch cache.

For MTP, all newly generated rows use the request's canonical resident row and
different fresh offsets. The save metadata therefore repeats `source_rows`
and increments `source_slots` across the reserved fresh window.

The CPU cache remains authoritative for historical main KV. LRU misses load
selected token rows back to the managed resident prefix before sparse
attention.

## PD Transfer Flow

MemFabric stays in pull mode.

### P Registration

P runs the existing split prefill layout from `c4341cdc`. It registers one PD
transfer record per main transformer layer:

```text
normal owner layer:  main K, main V, indexer K
skip_topk layer:     main K, main V
```

Indexer owners do not create extra layer callbacks or transfer events. For the
verified model this produced:

```text
79 transfer layers
101 cache owners = 79 main + 22 real indexer
```

P registers the HBM source regions, sends `MF_META`, and then emits
`READ_READY_BATCH` when a layer's data is ready. D pulls from P's HBM source
tensors and replies with `READ_DONE` or `READ_FAILED`.

P physical-pool reuse remains gated by both children of `MultiConnector`:

```text
SFAPDCpuOffloadConnector must receive READ_DONE from D
AscendStoreConnector must finish its layerwise store/reuse step
```

This prevents P from overwriting a shared HBM pool before either destination
has consumed it.

### D Destinations

D allocates two destination classes for each request:

```text
main KV:     connector CPU block IDs from CPUBlockManager
indexer KV:  scheduler block IDs from the one Uniform group
```

The complete prompt main KV, including the final partial page, is pulled into
CPU memory. Real indexer pages are pulled into their resident paged HBM
tensors. Trailing rows in a partial main page remain unused until decode fills
them token by token.

P and D must run the same protocol version. Mixed-version compatibility is not
provided by this refactor.

## AscendStore Independence

P still includes `AscendStoreConnector` because PD transfer and prefill cache
storage use different host backends:

```text
AscendStoreConnector      layerwise prefill reuse in MemCache
SFAPDCpuOffloadConnector  P-HBM to D-CPU/HBM transfer through MemFabric
```

`use_offload=false` does not disable AscendStore. It only prevents the D-only
direct host runtime from replacing P's normal paged caches.

The final startup fix in `3196e228` scopes resident-buffer allocation and its
capacity validation under `use_direct_sfa_host_offload`. Without that guard,
P incorrectly validated the default `buffer_size=2048` against MTP's
`decode_width=2` before loading weights.

## Verified Launch Configuration

### P Node

Script:

```text
/home/zyj/scripts/disaggregated_prefill_v1/run_sfa_pd_prefill.sh
```

Essential settings:

```text
--additional-config '{"use_offload": false}'
--gpu-memory-utilization 0.95
--max-num-seqs 16
--max-num-batched-tokens 1024
--max-model-len 8192
--enforce-eager
--speculative-config '{"method":"mtp","num_speculative_tokens":1,"enforce_eager":true}'

kv_connector: MultiConnector
children:
  SFAPDCpuOffloadConnector, kv_role=kv_producer, use_layerwise=true
  AscendStoreConnector, kv_role=kv_producer, use_layerwise=true
```

`mmc_meta_service` must be running inside the P container. Start it with:

```bash
mmc_meta_service &
```

### D Node

Script:

```text
/home/zyj/scripts/disaggregated_prefill_v1/run_sfa_pd_decode.sh
```

Essential settings:

```text
--additional-config '{
  "use_offload": true,
  "lru_resident_cache_config": {
    "enabled": true,
    "buffer_size": 2176,
    "topk": 2048
  }
}'
--gpu-memory-utilization 0.95
--max-num-seqs 16
--max-num-batched-tokens 256
--max-model-len 8192
--speculative-config '{"method":"mtp","num_speculative_tokens":1,"enforce_eager":true}'
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'

kv_connector: SFAPDCpuOffloadConnector
kv_role: kv_consumer
use_layerwise: true
```

For MTP1, `decode_width=2`. The resident buffer must leave at least two fresh
slots after `topk=2048` and must be divisible by cache block size 128. The
smallest valid configured value is therefore `2176`, not `2050`.

At `gpu_memory_utilization=0.85`, D reported `-4.97 GiB` available KV memory.
Raising D to `0.95` produced `1.17 GiB` available KV memory and allowed cache
initialization to complete. P reported `5.69 GiB` available KV memory.

Start order remains:

```text
1. D
2. P
3. proxy
4. request
```

## Remote Verification Result

The final test used `/home/zyj/scripts/curl.sh` with an 18-token prompt and
requested up to 512 output tokens.

Observed result:

```text
HTTP status:             200
completion tokens:      357
total tokens:           375
final answer:           coherent greeting and offer to help
D external cache hit:   100%
D generation rate:      about 17.8 tokens/s after startup
P errors/timeouts:      0
D errors/timeouts:      0
```

All 16 D ranks received `MF_META` with 79 transfer layers. D registered 22 real
indexer owners plus 79 main layers, while P registered 101 total cache owners.
P, D, and proxy remained healthy after completion.

The final Python change passed remote `py_compile`. Earlier focused tests for
the refactor completed with `23 passed, 32 deselected`; the final producer-only
scope fix was then validated by the successful P startup and full PD request.

No C++ change or vLLM-Ascend reinstall was required for the final fix.

## Removed Legacy Paths

Commit `3196e228` deliberately removed code that no longer belongs to this
design:

- standalone `SFAKVOffloadConnector` and its scheduler;
- multi-group direct-decode cache planning;
- synthetic/placeholder indexer allocations for direct offload;
- normal paged main-KV HBM allocation on D;
- tail buffers and tail block tables;
- staging buffers used to bridge paged main KV to resident attention;
- legacy `indexer_slot_mapping` direct-offload metadata;
- full-block-only decode save logic; and
- old C8 compatibility branches inside direct offload.

The cleanup changed 26 files with 709 insertions and 2826 deletions. Most of
the deletion is obsolete cache-planning, connector, scheduler, and tail/staging
logic.

## Current Limitations

- BF16 cache only for direct D offload.
- `DCP=PCP=1`; CP-aware token routing and manager-page addressing are not
  implemented here.
- D supports decode batches only. Chunked/local prefill on D is rejected.
- The host cache has a fixed 128 GiB capacity ceiling per worker configuration.
- Main/indexer remain one logical Uniform group; they are not independently
  schedulable resources.
- P and D protocol versions must match.
- MLAPO was not enabled in the verification scripts and remains outside this
  handoff's tested scope.
- Long prompts crossing many main CPU pages and concurrent multi-request MTP
  deserve additional stress testing even though the short MTP request passed.

## Follow-Up Checklist

1. Run a prompt spanning multiple 128-token blocks and verify the final
   partial main page plus later token-wise decode writes.
2. Run concurrent requests and inspect stable request-row/LRU isolation after
   batch reordering.
3. Enable `VLLM_ASCEND_MF_VERIFY=1` for source/destination checksum comparison.
4. Benchmark TPOT against the previous tail-buffer decode implementation.
5. Add a focused unit test proving P does not allocate or validate direct
   resident state when `use_offload=false`.
6. Revisit CP and sparse C8 only as separate follow-up designs.

## Operational Safety

Keep service and process management inside the Docker containers. Do not issue
host-level NPU reset operations. If an NPU is occupied by a process that cannot
be identified or stopped from the target container, stop verification and
report the external ownership instead.
