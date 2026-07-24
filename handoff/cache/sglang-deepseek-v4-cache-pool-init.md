# SGLang DeepSeek V4 Cache-Pool Initialization

## Scope

- Workspace: `vllm-ascend-workspace`
- SGLang commit inspected:
  `3217b7e3ceef8a9ca59be7f3a0eb516fa1404dbb`
- Subject: DeepSeek V4 cache-memory budgeting, physical sub-pool creation,
  allocator selection, request-to-cache mappings, and HiSparse behavior
- Primary implementation:
  - `sglang/srt/model_executor/pool_configurator.py`
  - `sglang/srt/mem_cache/kv_cache_configurator.py`
  - `sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
  - `sglang/srt/mem_cache/allocator/swa.py`

This document focuses on cache initialization and address ownership. It assumes
the reader already understands the DeepSeek V4 SWA, C4, C128, compressor, and
indexer computations.

## Executive Summary

SGLang represents DeepSeek V4 with an explicit multi-pool cache:

```text
one raw-token request sequence
    |
    +-- SWA KV for every local layer
    +-- C4 KV for C4 layers
    +-- C4 indexer cache for C4 layers
    +-- C128 KV for C128 layers
    +-- C4 compressor and indexer state
    +-- C128 compressor state
```

Initialization proceeds as follows:

```text
profile available HBM
    -> construct DSV4PoolConfigurator
    -> normalize all variable pool costs to bytes per "full token"
    -> reserve fixed request-scoped C128 state
    -> solve for full-token capacity F
    -> derive each sub-pool's capacity
    -> instantiate DeepSeekV4TokenToKVPool
    -> instantiate the platform-appropriate allocator
    -> instantiate request mapping tables
    -> optionally wrap C4 with HiSparse
```

The "full-token capacity" is only a memory-accounting baseline. There is no
ordinary full-attention KV tensor containing `F` raw entries. `F` defines a
common logical horizon from which the physical pool capacities are derived.

The generic CUDA path uses one primary full logical namespace and derives most
compressed addresses from its page geometry. The Ascend NPU path is more
explicit: it independently allocates C4, C128, and state pages and records
their locations in auxiliary per-request tables.

## 1. Initialization Entry Point

`KVCacheConfigurator._resolve_memory_pool_config()` drives sizing:

```python
available_bytes = self._profile_available_bytes(pre_model_load_memory)
config = configurator.calculate_pool_sizes(available_bytes, page_size)
config.max_running_requests = self.resolve_max_num_reqs(
    config.max_total_num_tokens
)
config = configurator.finalize_with_max_running_requests(config)
```

The important phases are:

1. Measure the HBM budget available to KV-related pools.
2. Select `DSV4PoolConfigurator` for a DeepSeek V4 model.
3. Calculate all token-scaled pool sizes.
4. Resolve `max_running_requests`.
5. Finalize request-scoped C128 state capacity.

Relevant functions:

- `kv_cache_configurator.py::KVCacheConfigurator._profile_available_bytes`
- `kv_cache_configurator.py::KVCacheConfigurator._resolve_memory_pool_config`
- `pool_configurator.py::DSV4PoolConfigurator`

## 2. Available HBM Budget

SGLang records free device memory before loading the model and measures it
again after model/runtime initialization:

```python
slack_gb = pre_model_load_memory * (1 - mem_fraction_static)
rest_memory_gb = current_available_memory - slack_gb
available_bytes = int(rest_memory_gb * (1 << 30))
```

The post-load measurement already reflects model weights and runtime
allocations. The subtracted slack preserves the fraction of the original free
memory excluded by `mem_fraction_static`.

Conceptually:

```text
current free HBM
    - reserved runtime headroom
    = HBM available for cache-pool initialization
```

## 3. PP-Local Layer Inventory

`DSV4PoolConfigurator` slices the model's `compress_ratios` to the current
pipeline-parallel stage:

```python
compression_ratios = cfg.compress_ratios[start_layer:end_layer]
```

It counts:

```text
N     = number of all PP-local DSV4 layers
N4    = number of PP-local C4 layers
N128  = number of PP-local C128 layers
```

All memory coefficients are local to that worker's layer slice. This avoids
charging one PP rank for another PP rank's cache tensors.

## 4. The Full-Token Accounting Unit

Let `F` denote the proposed full-token capacity. The physical capacities are
defined as functions of `F`:

```text
SWA KV entries:       r * F
C4 KV entries:        F / (4 * H)
C4 indexer entries:   F / 4
C128 KV entries:      F / 128
```

where:

```text
r = swa_full_tokens_ratio, normally 0.1
H = HiSparse host_to_device_ratio, or 1 without HiSparse
```

SGLang divides the memory usage of every variable-sized pool by `F`. The result
is `bytes_per_full_token`.

This is amortized accounting. Fractional entries are not allocated. SGLang
first solves for an integer `F`, then creates integer page-aligned physical
capacities.

## 5. Bytes per Full Token

### 5.1 DSV4 KV entry

The packed DSV4 KV entry size is:

```text
K = qk_nope_head_dim
    + 2 * qk_rope_head_dim
    + 8 bytes of quantization-scale storage
```

For the normal DSV4 dimensions:

```text
K = 448 + 2 * 64 + 8 = 584 bytes
```

The NoPE portion is FP8, while the RoPE portion is BF16.

### 5.2 C4 indexer entry

The indexer entry size is:

```text
I = indexer_head_dim
    + 4 * floor(indexer_head_dim / 128)
```

This represents FP8 indexer data plus one FP32 scale per 128-element
quantization block.

### 5.3 State terms

Define:

```text
A       = qk_nope_head_dim + qk_rope_head_dim
R4      = C4 state ring size
W       = SWA window/page size
S4      = bytes per C4 compressor-state row
SI4     = bytes per C4 indexer-state row
```

The number of C4 state rows scales as:

```text
(r * F / W) * R4
```

so its amortized row count per full token is:

```text
r * R4 / W
```

### 5.4 Complete coefficient

Ignoring a later speculative-decoding multiplier:

```text
bytes_per_full_token =
      r             * K   * N
    + 1 / (4 * H)   * K   * N4
    + 1 / 128       * K   * N128
    + 1 / 4         * I   * N4
    + r * R4 / W    * S4  * N4
    + r * R4 / W    * SI4 * N4
```

The terms respectively represent:

1. SWA KV for every layer.
2. C4 KV for C4 layers.
3. C128 KV for C128 layers.
4. The full C4 indexer history.
5. C4 compressor state.
6. C4 indexer compressor state.

C128 state is intentionally absent from this coefficient because its capacity
is request-scoped instead of proportional to `F`.

With speculative decoding, SGLang reserves the equivalent of one draft layer:

```python
bytes_per_full_token *= (N + 1) / N
```

Implementation:

- `pool_configurator.py::DSV4PoolConfigurator._get_bytes_per_full_token`

## 6. Fixed C128 State Reservation

C128 state is sized from the number of concurrent request slots:

```text
C128 fixed bytes =
    state rows
    * state width
    * state dtype bytes
    * N128
```

If `max_running_requests` was supplied, SGLang uses the per-attention-DP-worker
value directly.

Otherwise, it first computes a preliminary token capacity:

```python
F0 = int(available_bytes / bytes_per_full_token)
```

It estimates the request count from `F0` and `context_len`, clamps the estimate,
calculates the fixed C128 reservation, and then recomputes the actual `F`.

Online C128 stores one `(max, sum, kv)` state per request slot. Offline C128
uses request-associated raw-state rings. Both remain fixed-cost terms with
respect to `F`.

Implementation:

- `pool_configurator.py::DSV4PoolConfigurator._get_c128_state_fixed_bytes`
- `pool_configurator.py::DSV4PoolConfigurator._get_c128_state_fixed_bytes_for_token_capacity`

## 7. Solving for Pool Capacities

The raw full-token capacity is:

```text
F_raw =
    floor(
        (available_bytes - fixed_C128_state_bytes)
        / bytes_per_full_token
    )
```

SGLang aligns `F_raw` down to the logical `page_size`:

```python
F = F_raw // page_size * page_size
```

It then derives:

```python
swa_tokens = align_down(int(F * r), page_size)
c4_tokens = F // (4 * H)
c128_tokens = F // 128
c4_state_rows = swa_tokens // W * R4
```

The resulting `MemoryPoolConfig` contains:

```text
max_total_num_tokens
full_max_total_num_tokens
swa_max_total_num_tokens
c4_max_total_num_tokens
c128_max_total_num_tokens
c4_state_pool_size
c128_state_pool_size
```

`c128_state_pool_size` is finalized after `max_running_requests` is known.

Implementation:

- `pool_configurator.py::DSV4PoolConfigurator.calculate_pool_sizes`
- `pool_configurator.py::DSV4PoolConfigurator._compute_dsv4_sizes`
- `pool_configurator.py::DSV4PoolConfigurator.finalize_with_max_running_requests`

## 8. Page-Size Defaults

The DSV4 argument override selects:

```text
CUDA/ROCm default logical page size: 256
Ascend NPU logical page size:        128
```

The configurator requires the logical page size to be divisible by 128.

In the generic separate-pool layout:

```text
C4 physical page size:   logical page size / 4
C128 physical page size: logical page size / 128
```

The request prefix cache is also page-aligned. Therefore reusable radix-cache
hit lengths are multiples of this logical page size. HiSparse currently
requires radix caching to be disabled.

## 9. Physical Pool Construction

`KVCacheConfigurator` instantiates `DeepSeekV4TokenToKVPool` with all derived
capacities.

### 9.1 SWA pool

The SWA pool:

- exists for every PP-local DSV4 layer;
- stores packed raw KV;
- has capacity `swa_max_total_num_tokens`;
- uses a separately allocated physical namespace;
- is related to the primary logical namespace through a full-to-SWA mapping.

### 9.2 C4 KV pool

The C4 KV pool:

- exists only for C4 layers;
- stores one compressed entry per four raw tokens;
- normally has capacity `F / 4`;
- has capacity `F / (4 * H)` when HiSparse is enabled.

### 9.3 C4 indexer pool

The C4 indexer pool:

- exists only for C4 layers;
- has logical capacity `F / 4`;
- remains full-sized when HiSparse is enabled;
- is required to score the complete C4 history before selected C4 KV entries
  can be retrieved.

### 9.4 C128 KV pool

The C128 pool:

- exists only for C128 layers;
- stores one compressed entry per 128 raw tokens;
- has capacity `F / 128`.

### 9.5 State pools

The pool object creates:

- a C4 compressor-state pool for each C4 layer;
- a C4 indexer-state pool for each C4 layer;
- a C128 compressor-state pool for each C128 layer.

The exact state addressing differs by platform:

- generic CUDA uses ring/request-derived addressing;
- Ascend NPU uses explicitly paged state allocators and per-request tables.

### 9.6 Layer mapping

`DeepSeekV4TokenToKVPool._init_compressed_layer_mapping()` records, for every
model layer:

```text
(compression ratio, sub-pool-local layer id, compressed KV pool)
```

This translates an absolute model layer ID into the appropriate C4/C128 pool
and its local buffer index.

Implementation:

- `deepseek_v4_memory_pool.py::DeepSeekV4TokenToKVPool`
- `deepseek_v4_memory_pool.py::DeepSeekV4TokenToKVPool._init_compressed_layer_mapping`

## 10. Generic Allocator Structure

The generic DSV4 path uses `SWATokenToKVPoolAllocator`.

It owns:

```text
full_attn_allocator: primary logical/raw-token page ownership
swa_attn_allocator:  SWA physical pages
```

The primary allocator may be logical-only for DSV4; there is no ordinary
full-attention KV tensor associated with every raw slot.

For each allocation, it obtains:

```text
full logical locations: f[]
SWA physical locations: s[]
```

and records:

```python
full_to_swa_index_mapping[f[i]] = s[i]
```

Only `f[]` is returned as the normal scheduler-facing `out_cache_loc` and
written into the base request-to-token table.

Old SWA mappings are freed once they fall outside the active sliding window.
The primary logical locations remain alive when long-context compressed cache
or prefix-cache ownership still requires them.

For generic C4/C128 storage, the compressed location can be projected from the
full-page geometry. A full logical page ID owns the corresponding C4 and C128
sidecar pages. These sidecar pages do not need independent free-list decisions.

Implementation:

- `allocator/swa.py::SWATokenToKVPoolAllocator`

## 11. Request-to-Token Mapping

The base `ReqToTokenPool` stores:

```text
req_to_token[request slot, raw sequence position]
    = primary full logical slot
```

It represents request ownership and sequence order. It does not itself encode:

- SWA eviction;
- compression boundaries;
- C4/C128 physical layouts;
- compressor-state rings;
- HiSparse residency.

The attention backend and pool/allocator mappings turn these logical locations
into write-side slot mappings and read-side page tables.

The useful conceptual split is:

```text
ReqToTokenPool:
    Which primary logical slot belongs to request token t?

Allocator / pool mapping:
    Which physical slots hold SWA, C4, C128, and state representations?

Attention metadata:
    Which slots are written now, and which pages are read by this kernel?
```

## 12. Write-Side and Read-Side Metadata

For a new raw token, DSV4 ultimately needs:

```text
raw/full out location
SWA out location
C4 out location at a C4 boundary
C128 out location at a C128 boundary
C4/C128 state locations
```

These are analogous to kernel `slot_mapping` values.

On the read side, metadata construction builds:

```text
SWA page indices
C4 sparse selected page indices
C128 page indices
compressed sequence lengths
```

The C4 indexer first scores the C4 history. C4 attention then reads only the
selected compressed KV entries. C128 and SWA use their own page metadata.

## 13. Ascend NPU Allocator

The Ascend backend uses `DSV4NPUTokenToKVPoolAllocator`, which extends the
generic SWA allocator with independent paged allocators:

```text
full logical allocator
SWA allocator
C4 KV allocator
C128 KV allocator
C4 state allocator
C128 state allocator
```

For an extension from raw prefix length `p` to sequence length `s`, it computes:

```text
new C4 entries   = floor(s / 4)   - floor(p / 4)
new C128 entries = floor(s / 128) - floor(p / 128)
```

It allocates only the compressed entries whose boundaries were completed by
the extension.

The return value is:

```python
DSV4OutCacheLoc(
    out_full_loc,
    out_swa_loc,
    out_c4_loc,
    out_c128_loc,
    out_c4_state_loc,
    out_c128_state_loc,
)
```

The scheduler continues to treat `out_full_loc` as its ordinary
`out_cache_loc`. The complete bundle is retained for the NPU DSV4 backend.

Implementation:

- `hardware_backend/npu/dsv4/dsv4_allocator.py::DSV4NPUTokenToKVPoolAllocator`

## 14. Ascend NPU Request Tables

Because C4/C128/state pages are independently allocated, their physical
locations cannot be reconstructed by dividing the primary full location.

`DSV4NPUReqToTokenPool` therefore extends the base request pool with:

```text
req_to_token               primary full logical slots
req_to_token_swa           SWA physical slots
req_to_token_c4            C4 KV physical slots
req_to_token_c128          C128 KV physical slots
req_to_token_c4_state      C4 state physical slots
req_to_token_c128_state    C128 state physical slots
```

It remains one central request-pool object, but it is not one universal mapping
tensor.

After a successful allocation, NPU hooks distribute the flattened
`DSV4OutCacheLoc` tensors across the appropriate per-request table rows.

On request completion, the allocator reads these tables and returns C4, C128,
and remaining state pages to their individual free lists.

Implementation:

- `hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py::DSV4NPUReqToTokenPool`
- `hardware_backend/npu/dsv4/dsv4_common_hooks.py`

## 15. HiSparse Modification

HiSparse changes only the long C4 KV residency model:

```text
full raw location
    -> compressed C4 logical location
    -> host C4 location
    -> current HBM hot-buffer location
```

With host-to-device ratio `H`:

```text
complete C4 logical/host capacity: F / 4
C4 HBM capacity:                  F / (4 * H)
```

The C4 indexer remains full-sized in HBM so it can score the complete C4
history. SWA, C128, and state sizing are not divided by `H`.

`DeepSeekV4HiSparseTokenToKVPoolAllocator` wraps the existing logical/SWA
allocator and adds:

- a smaller C4 HBM paged allocator;
- a C4-logical-to-current-device mapping;
- request-associated host mappings managed by `HiSparseCoordinator`;
- swap-in and LRU hot-buffer management during decode.

The original request/raw-token ownership remains intact. HiSparse adds a
residency translation for C4 rather than redefining the request sequence.

Current limitations include:

- radix prefix caching must be disabled;
- only supported DSA/DSV4 paths can enable HiSparse;
- the C4 HBM saving trades against host-memory consumption and transfer cost.

Implementation:

- `allocator/hisparse.py::DeepSeekV4HiSparseTokenToKVPoolAllocator`
- `managers/hisparse_coordinator.py::HiSparseCoordinator`
- `arg_groups/hisparse_hook.py::validate_hisparse`

## 16. Prefix-Cache Granularity

When radix caching is enabled, matching and insertion are aligned to the
logical `page_size`:

```text
usable prefix length = floor(candidate length / page_size) * page_size
```

Partially filled pages may remain private to an active request, but they are
not cross-request reusable radix-cache prefixes.

C4 and C128 compression boundaries do not independently lower this matching
granularity. They change the number of physical entries represented by a
logical page.

Implementation:

- `mem_cache/radix_cache.py::RadixCache.match_prefix`
- `mem_cache/radix_cache.py::RadixCache.insert`

## 17. Design Interpretation

SGLang's DSV4 design centralizes request ownership but explicitly separates
physical storage:

```text
request/raw-token namespace
    -> pool-specific address transformations
    -> physical multi-pool cache
```

Its advantages are:

- the request sequence has a clear primary logical representation;
- each physical pool and its capacity are visible;
- C4/C128 compression is easy to express as a ratio of `F`;
- platform-specific allocation can be substituted behind the same scheduler
  request.

Its costs are:

- capacities are statically partitioned at initialization;
- `swa_full_tokens_ratio` is a workload-sensitive policy;
- DSV4 requires custom pool, allocator, and metadata code;
- the Ascend path needs several auxiliary request mapping tables;
- HiSparse adds another C4 logical-to-resident translation.

The implementation is therefore clearer than a deeply shared cache-group
layout at the top-level accounting layer, but it is not complexity-free. Much
of the complexity lives in physical address translation and lifecycle
coordination.

## 18. Source Map

| Concern | Primary source |
|---|---|
| Available memory profiling | `mem_cache/kv_cache_configurator.py` |
| DSV4 capacity calculation | `model_executor/pool_configurator.py` |
| Physical DSV4 sub-pools | `mem_cache/deepseek_v4_memory_pool.py` |
| Generic full/SWA allocation | `mem_cache/allocator/swa.py` |
| Generic paged allocation | `mem_cache/allocator/paged.py` |
| Radix prefix alignment | `mem_cache/radix_cache.py` |
| HiSparse C4 allocation | `mem_cache/allocator/hisparse.py` |
| HiSparse host/device lifecycle | `managers/hisparse_coordinator.py` |
| Ascend DSV4 multi-pool allocator | `hardware_backend/npu/dsv4/dsv4_allocator.py` |
| Ascend auxiliary request tables | `hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py` |
| DSV4 attention metadata | `layers/attention/deepseek_v4_backend.py` |
| DSV4 compressor writes | `layers/attention/dsv4/compressor_v2.py` |
| C4 indexer selection | `layers/attention/dsv4/indexer.py` |

