# SFA Offload Refactor Simplified

## Snapshot

- Date: 2026-07-14
- Repository: `vllm-ascend`
- Branch: `offload_merge`
- Split-indexer baseline: `f16150f6aee535a099ee844c11c715acf0710ca1`
- Prefill refactor: `c4341cdc Refactor SFA layerwise prefill offload cache`
- PD compatibility: `70dfb804 Support split-cache SFA PD disaggregation`
- Decode owner sharing: `c9f41c22 Refactor SFA decode offload cache ownership`
- State: prefill and PD commits pushed; decode owner-sharing commit is local

This change lets layerwise `AscendStoreConnector` transfer the refactored SFA
main KV and real indexer caches directly. The scheduler still exposes them in
one `UniformTypeKVCacheSpecs` group, while the runtime keeps separate physical
reuse pools for main and indexer tensors.

## Change Size

```text
12 files changed, 611 insertions(+), 28 deletions(-)
Net change: +583 lines
```

The split between production code and tests is:

| Area | Files | Insertions | Deletions | Net |
| --- | ---: | ---: | ---: | ---: |
| Production | 9 | 481 | 28 | +453 |
| Tests | 3 | 130 | 0 | +130 |
| Total | 12 | 611 | 28 | +583 |

## File Breakdown

| File | Added | Removed | Purpose |
| --- | ---: | ---: | --- |
| `tests/ut/distributed/ascend_store/test_config_data.py` | 20 | 0 | Variable layer-transfer range coverage |
| `tests/ut/distributed/ascend_store/test_kv_transfer.py` | 39 | 0 | Split main/indexer batch construction coverage |
| `tests/ut/distributed/ascend_store/test_layerwise_config.py` | 71 | 0 | SFA transfer-plan and physical-pool coverage |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py` | 15 | 0 | Publish split-SFA layout metadata to AscendStore |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py` | 25 | 5 | Support per-layer transfer sizes and ranges |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py` | 28 | 14 | Build transfers from semantic main/indexer cache entries |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_config.py` | 170 | 1 | Shared split-SFA cache plan, ownership mapping, and byte accounting |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py` | 24 | 2 | Use split-SFA host-page sizing in scheduler allocation |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py` | 69 | 2 | Register and transfer separate main/indexer device buffers |
| `vllm_ascend/patch/platform/patch_kv_cache_utils.py` | 47 | 0 | Preserve complete split specs after scheduler-side spec collapsing |
| `vllm_ascend/worker/model_runner_v1.py` | 70 | 0 | Construct separate physical reuse pools for main and indexer caches |
| `vllm_ascend/worker/worker.py` | 33 | 4 | Apply byte-weighted cache sizing for the physical split layout |

## Why It Is Larger Than Expected

The core connector transfer change is only part of the diff. Directly sending
the split caches also requires all participants to agree on the same layout:

1. A shared plan pairs each transformer layer's main cache with its optional
   real indexer cache. Synthetic indexer ownership is not introduced.
2. Main and indexer caches have different page sizes, so physical reuse pools
   and available-memory accounting must be byte-aware.
3. vLLM collapses a uniform group to one representative scheduler spec. The
   original per-owner specs must be retained as Ascend metadata so scheduler
   and worker calculate the same GVA page size.
4. AscendStore transfer batches must support a variable byte range per layer:
   main-only for `skip_topk` layers and main-plus-indexer for real owners.
5. Focused unit coverage contributes 130 lines, about 21% of all insertions.

The largest individual change is `layerwise_config.py` at `+170/-1`. It holds
the shared interpretation of cache ownership and physical layout so the
connector, scheduler, worker, and model runner do not each infer the split in
different ways.

## Scope Boundaries

- This work targets layerwise prefill offload through `AscendStoreConnector`.
- Main KV and real indexer caches remain in one logical KV cache group.
- Main and indexer caches use separate physical reuse pools.
- Indexer storage is allocated and transferred only for real indexer owners.
- Decode offload and PD-disaggregated transfer are not part of the prefill
  refactor commit. The narrow PD compatibility extension is documented below.
- Sparse C8 plus offload is not addressed here.

## PD Disaggregation Compatibility Extension

Commit `70dfb804` adapts the existing `SFAPDCpuOffloadConnector` to the
simplified prefill layout without introducing a second cache-layout framework
or rewriting the PD protocol.

### Problem

After the prefill refactor, the P node no longer exposes a combined cache tuple
for every main layer. It has separate runtime cache bindings:

```text
main owner:     (main_k, main_v)
indexer owner:  (indexer_k)       # only when the model has a real indexer
```

The D node still uses the existing decode-offload layout. Its PD reader expects
one metadata record per main layer, with main K/V first and an optional indexer
tensor after them. Passing P's raw `kv_caches` dictionary directly would create
extra transfer layers for indexer owners and would break the one-callback-per-
transformer-layer ordering used by layerwise reuse.

### Deliberately Asymmetric P and D Layouts

The two nodes keep their local layouts because they serve different purposes:

```text
P node
  scheduler: one UniformType KV cache group
  owners:    all main layers plus real indexer owners
  runtime:   separate layer-reused main and indexer physical pools
  host path: AscendStore layerwise prefill offload

D node
  runtime:   existing decode-offload main tuples and indexer destinations
  host path: SFA decode CPU offload and LRU resident loading
```

PD transfer is therefore a compatibility adapter between local layouts, not a
requirement that P and D allocate identical cache tuples.

### P-Side Transfer Manifest

The producer reuses `SFALayerwiseCachePlan`, the same plan used by
AscendStore. It creates exactly one `LayerMetadata` entry for every main
attention callback:

```text
normal layer:      (main_k, main_v, indexer_k)
skip_topk layer:   (main_k, main_v)
```

The entry is keyed by the main layer name. Real indexer ownership comes from
the cache specs in the one scheduler group; no synthetic indexer layer or
hard-coded GLM layer interval is introduced.

For this BF16 scope, every transferred tensor must satisfy:

```text
tensor block count == scheduler num_blocks
stride of dimension 0 == one contiguous block
block_size_scale == 1
```

These checks keep the existing positional MemFabric protocol valid: D derives
the address of block `b` as `base_addr + b * block_len`.

### Main-Layer Scheduling and Reuse

The P cache dictionary contains both main and indexer owners, but only main
layers execute forward callbacks. The producer therefore records an ordered
main-layer list and sets:

```text
total transfer layers = number of main layers
```

Indexer owners do not allocate extra send events or advance the layer counter.
When vLLM passes an empty layer name, the producer resolves it from this ordered
main-layer list.

`AscendMultiConnector` uses the same main-layer count when deriving the GVA
reuse plan. This keeps the two P-side children aligned:

```text
SFAPDCpuOffloadConnector: wait until D replies READ_DONE
AscendStoreConnector:     wait until its layerwise transfer/reuse gate finishes
```

A reused physical pool is not overwritten until both connector paths have
finished the corresponding main-layer step.

### D-Side Compatibility

The D reader accepts both valid P manifests:

```text
2 tensors: main K/V only
3 tensors: main K/V plus indexer K
```

Two tensors are normal for a `skip_topk` layer and no longer produce a missing-
indexer error. When the third tensor exists, D routes it to the existing
resident indexer destination. Indexer-scale handling is attempted only when an
indexer leg is present, preserving the old path without making sparse C8 part
of this refactor.

The remainder of the D pull flow is unchanged:

- complete main blocks go to the decode CPU pool;
- the current partial main block goes to HBM;
- an optional real indexer block goes to its resident HBM cache; and
- `READ_DONE` releases the matching P-side reuse gate.

### Change Size

```text
5 files changed, 296 insertions(+), 36 deletions(-)
```

| File | Purpose |
| --- | --- |
| `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/worker.py` | Compose the P manifest and schedule only main-layer callbacks |
| `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/read_thread.py` | Accept main-only or main-plus-indexer P records |
| `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py` | Size the shared reuse plan from main-layer entries |
| `tests/ut/kv_offload/test_sfa_pd_cpu_offload_single_rank.py` | Cover manifest composition and main-only D resolution |
| `tests/ut/distributed/kv_transfer/test_ascend_multi_connector.py` | Cover reuse scheduling with indexer owners in the same group |

### Scope Boundaries

- BF16 only; sparse C8 plus offload remains out of scope.
- The P node must expose exactly one scheduler KV cache group for this
  simplified split-prefill path.
- The compatibility adapter does not redesign D's decode-offload cache layout.
- The PD wire protocol remains positional; semantic role strings and a new
  owner-set negotiation protocol are intentionally avoided.
- MLAPO-specific ordering is unchanged because the current launch scripts do
  not enable MLAPO.
- CP remains out of scope.

### Remote Verification

The committed code was verified with GLM-5.2 using TP16 and one MTP draft token:

```text
P node: 192.168.13.157, port 8100, 79 main layers, 22 real indexers
D node: 192.168.13.165, port 8200
proxy:  192.168.13.157, port 8900
```

The single end-to-end request produced a coherent 512-token completion from a
580-token prompt. All 16 D ranks received the 79-layer `MF_META`; D reported a
100% external-prefix cache hit and decode offload released completed blocks 3
through 6. P and D both remained healthy, and their recent logs contained no
traceback, runtime error, value error, or transfer timeout.

## Direct Decode Owner-Anchored HBM Reuse

Commit `c9f41c22` refactors direct BF16 decode offload through
`SFAKVOffloadConnector`. It keeps separate logical main and indexer groups but
allocates one physical K/V scratch pool per real indexer owner. This avoids
allocating fake indexer caches while retaining cross-layer HBM reuse for models
such as GLM-5.2.

### Logical Groups and Owner Plan

Direct decode offload now exposes exactly two semantic KV cache groups:

```text
main group:     every main SFA attention layer
indexer group:  real DeepseekV32IndexerCache owners only
```

`SFAOffloadSharedCachePlan` sorts the real indexer owners and assigns each main
layer to its nearest preceding owner. A GLM-5.2 layout therefore resembles:

```text
pool 0: indexer 0 + main 0
pool 1: indexer 1 + main 1
pool 2: indexer 2 + main 2, 3, 4, 5
pool 3: indexer 6 + main 6, 7, 8, 9
...
```

The plan rejects a main layer before the first owner. Every MTP main layer must
also own a real indexer, preventing an MTP layer from accidentally sharing a
transformer pool with incompatible lifetime semantics. GLM-5.1 remains valid:
because every main layer has an owner, it naturally receives one pool per
layer and gains little cross-layer reuse.

Groups are resolved by cache-spec type, not by positional assumptions:

- `OffloadMLAAttentionSpec` identifies the main group.
- `AscendSFAOffloadIndexerCacheSpec` identifies the real-indexer group.

### Compatible Manager Pages

Main and indexer groups use different logical block sizes but equal-byte
manager pages:

```text
indexer_block_size = main_block_size * kv_lora_rank / index_head_dim
indexer_pad_dim = index_head_dim * qk_rope_head_dim / kv_lora_rank

indexer_block_size * (index_head_dim + indexer_pad_dim)
    == main_block_size * (kv_lora_rank + qk_rope_head_dim)
```

For the verified GLM BF16 configuration, one 128-token main page contains the
same bytes as one 512-token indexer manager page. The runtime indexer tensor is
still a contiguous 128-dimension alias over K storage; the additional 16
dimensions exist only in page-size accounting.

The two groups retain independent block tables and slot mappings. vLLM's
shared global block pool gives them disjoint physical block IDs, while the
equal-byte pages make either ID address the same-size physical allocation.

### Physical Pool and Runtime Tuple

The model runner emits one `KVCacheTensor` per owner segment. Its `shared_by`
list contains exactly one real indexer owner and every main layer assigned to
that owner. Each physical pool allocates one aligned K tensor and one aligned V
tensor; the owner indexer is a contiguous alias into the K allocation.

Every main layer keeps the seven-entry runtime tuple:

```text
0: owner-shared main K scratch
1: owner-shared main V scratch
2: owner indexer alias
3: per-layer CPU/LRU resident K workspace
4: per-layer CPU/LRU resident V workspace
5: per-layer, per-request two-block K tail
6: per-layer, per-request two-block V tail
```

Non-owner main layers receive the owner alias in tuple slot 2 for a stable
runtime shape, but only the real indexer module writes through that alias.

### Capacity Accounting

Capacity is calculated from physical allocations rather than inflating a
normal-layout block count:

```text
fixed_hbm = resident workspaces + two-block tail buffers + alignment reserve

num_blocks = floor(
    (available_hbm - fixed_hbm)
    / (num_real_indexer_owners * main_page_bytes)
)
```

An explicit block-count override is validated against the same byte formula.
This makes fixed per-layer decode workspaces visible before paged cache blocks
are allocated and produces a clear error when they cannot fit.

On the remote GLM-5.2 run, 79 main layers and 22 owner pools required about
6.06 GiB of fixed HBM. The original `--gpu-memory-utilization 0.88` supplied
only about 1.11 GiB of available KV memory and was correctly rejected. A
temporary launch copy using `0.98` supplied about 7.23 GiB and allocated 469
blocks. The checked-in/remote original launch script was not modified.

### Runtime Data Flow

The worker receives generic per-group metadata:

```text
block_table_tensors_by_group
slot_mappings_by_group
```

Main K/V writes use the main group's mapping. Real indexer writes use the
indexer group's 512-token manager mapping. Request IDs are assigned stable tail
slots so batch reordering cannot mix requests; slots are released when a
request finishes, including scheduler cleanup-only steps.

Every produced main K/V token is scattered into its layer-private two-block
tail. Decode attention reads:

- tokens in the latest two logical blocks from the NPU tail;
- older selected tokens from CPU/LRU resident buffers; and
- combines the two attention results using their softmax LSE values.

Completed prefill blocks are copied from normal owner scratch. A block that
becomes full during decode is copied from the layer-private tail, because the
shared scratch may already contain another layer. Each owner pool has a reuse
gate that waits only for a pending normal-scratch HBM-to-CPU copy. Tail-source
copies do not block scratch reuse.

### Chunked Prefill Correction

Remote verification exposed a distinct chunked-prefill lifetime issue. The
580-token prompt was scheduled as 512 tokens followed by 68 tokens. On the
second chunk, an owner-shared scratch tensor no longer contains the previous
layer prefix, so the normal NPU prefill path cannot read the whole context from
scratch.

The corrected path preserves the count of CPU-offloaded prefix blocks during
owner-mode prefill and splits top-k attention into two sources:

```text
older completed prefix blocks -> CPU/LRU resident workspace
current prefill chunk          -> owner-shared NPU scratch
```

The resident workspace remains decode-sized. The prefill CPU leg processes
query rows in bounded slices, avoiding a 512-row resident allocation for every
layer.

### MTP Metadata

Speculative decode now rebuilds both semantic group mappings for each draft
position. Rejection compaction preserves or recomputes the grouped block-table
and slot-mapping metadata, so the main and real-indexer writes continue to use
their own address spaces during MTP execution.

### Verification

Static checks passed for all changed Python files:

```text
python3 -m py_compile <changed Python files>
git diff --check
```

The focused remote unit suite passed with `31 passed` (16 warnings). A
graph-mode GLM-5.2 service with TP16 and one MTP draft token was then launched
using the temporary `0.98` memory-utilization configuration:

- service became healthy and remained healthy after requests;
- two concurrent requests returned coherent answers (`56` and `Paris`), with
  no tail-slot cross-request contamination;
- a 580-token prompt plus 64 output tokens produced a coherent continuation;
- logs showed four initial full prefill blocks `[3, 4, 5, 6]` copied to CPU and
  a later decode-full block `[7]` copied from the tail path; and
- logs contained no traceback, timeout, cleanup crash, or reuse-gate failure.

### Scope Boundaries

- Direct BF16 `SFAKVOffloadConnector` mode with `use_offload=True` only.
- Tail width remains fixed at two blocks per request and main layer.
- Sparse C8, CP, `AscendStoreConnector`, combined connectors, and PD
  disaggregation are unchanged by this commit.
- The current remote service uses a temporary launch copy at
  `http://192.168.13.157:8900`; the original script still uses `0.88` and is
  expected to fail fixed-workspace capacity validation for this model.

### Change Size

```text
16 files changed, 2609 insertions(+), 128 deletions(-)
```

Most of the change is concentrated in SFA runtime attention, physical cache
allocation, grouped metadata propagation, and focused regression coverage. No
C++ files changed in this commit.
