# GLM SFA Offload Cache Refactor Handoff

## Status

- Workspace: `vllm-ascend-workspace`
- vLLM-Ascend branch: `offload_merge`
- Branch HEAD before these edits: `304ba9df`
- Implementation state: local, uncommitted changes in the `vllm-ascend` submodule
- Primary target: GLM-5/GLM-5.2 SFA KV offload
- Initial parallelism scope: DCP=1 and PCP=1
- Remote/NPU validation: intentionally not run for this change

This work depends on the split-cache design documented in
`handoff/sfa-indexer-cache-refactor.md`. Read that handoff first for the
non-offload motivation, upstream comparison, and the distinction between KV
cache groups, physical tensors, and cross-layer cache reuse.

## Problem

The original `offload_merge` branch represented every SFA transformer layer
with a combined main/indexer cache layout. It also synthesized missing
GLM-5.2 indexer cache modules by deep-copying layer 0 across the model.

That caused two problems:

1. `skip_topk` layers reserved indexer cache even though they consume a shared
   top-k result and do not execute indexer computation.
2. Layerwise reuse treated main KV and indexer ownership as the same reuse
   problem. Main KV needs explicit sequential overwrite gating, while indexer
   groups need ordinary independent block-ID namespaces.

The target layout has four physical paged-cache pools when the connector is
configured with:

```text
layerwise_num_shared_buffers = 2
layerwise_independent_layers = first,last  # default
```

The four pools are:

```text
pool 0: first main SFA layer, independent
pool 1: last main SFA layer, independent
pool 2: alternating interior main layers, layerwise reused
pool 3: remaining interior main layers, layerwise reused
```

Active indexer owners share these same four raw pools, but use separate KV
cache groups and therefore separate block tables/block-ID namespaces.

## Implemented Cache Model

### Main cache group

All main SFA layers remain in one logical group. They use the same block table
and block IDs. Physical storage is reduced to the four slots returned by
`get_layerwise_storage_indices()`.

Only the two reused interior slots receive entries in
`_layerwise_reuse_mate_map`. The independent first/last layers do not have a
reuse mate.

### Indexer cache groups

Only active indexer owners publish an indexer cache spec. An active owner has a
real indexer and executes indexer computation. A GLM-5.2 `skip_topk` layer:

- keeps its main SFA cache;
- receives no indexer cache spec or metadata;
- receives no indexer AscendStore entry;
- reads the shared top-k buffer in the existing SFA path.

The previous model-runner code that deep-copied the first indexer module into
missing layers was removed. `IndexerWrapper` now retains the original
`DeepseekV32IndexerCache` only for offload layers where `skip_topk` is false.

Active indexer names are sorted by transformer-layer ID, compacted, and split
into groups no wider than the physical-pool count. For `P=4` pools and `M`
active indexers:

```text
G = ceil(M / P)
indexer_group[g] = active_indexers[g::G]
```

For 20 active indexers this produces:

```text
[0, 5, 10, 15]
[1, 6, 11, 16]
[2, 7, 12, 17]
[3, 8, 13, 18]
[4, 9, 14, 19]
```

Position `p` from every indexer group is assigned to physical pool `p`.
Members of one indexer group therefore occupy distinct pools, while members
from different groups can share a pool because their block IDs belong to
different group namespaces.

### Physical `shared_by` rewrite

The upstream allocator initially emits tensors based on logical group width.
`_merge_kv_cache_tensors_for_layer_reuse()` replaces that layout with exactly
the layerwise storage slots. For each physical pool it combines:

- the main owners from that layerwise slot; and
- the same-position indexer owner from each indexer group.

The rewrite validates that every main/indexer cache owner appears exactly once
and that all source tensors have a compatible physical size.

With the default first/last independent configuration and two shared buffers,
`get_layerwise_num_tensors()` reports four, so memory-capacity inflation and
the final physical allocation use the same tensor count.

## New Indexer Spec and Backend

The implementation introduces:

- `AscendSFAOffloadIndexerCacheSpec`
- `AscendSFAOffloadIndexerBackend`
- `AscendSFAOffloadIndexerMetadata`
- `AscendSFAOffloadIndexerMetadataBuilder`

The existing non-offload `AscendSFAIndexerMetadataBuilder` remains cache-only
and continues to return `None`.

The new offload builder returns the indexer group's own:

- `block_table_tensor`
- `slot_mapping`

Request-level offload fields such as `num_offloaded_blocks`, request IDs, and
token-to-request mappings remain on main SFA metadata. The old duplicated
`indexer_block_table_tensor` and `indexer_slot_mapping` fields were removed
from main/common SFA metadata.

`AscendSFAImpl` obtains indexer metadata from:

```python
get_forward_context().attn_metadata[self.indexer.k_cache.prefix]
```

Only active indexer layers perform this lookup.

## Page Size and Kernel Block Size

The offload indexer spec separates its semantic indexer width from physical
page padding.

For the current GLM dimensions:

```text
main block size:       128 tokens
index head dimension:  128
index padding:          16
```

Non-C8:

```text
main page:     128 * (512 + 64) * 2 = 147456 bytes
indexer page:  512 * (128 + 16) * 2 = 147456 bytes
```

C8:

```text
main page:     128 * 1168 = 149504 bytes
indexer page: 1024 * 146 = 149504 bytes
```

The indexer backend deliberately keeps:

```python
get_supported_kernel_block_sizes() == [128]
```

The 512-token BF16 manager page is viewed as four virtual 128-token kernel
blocks. The 1024-token C8 manager page is viewed as eight virtual 128-token
kernel blocks. The model runner now uses normal backend block-size selection
instead of hardcoding 512/1024 as the kernel block size.

## Allocation, Reshape, and Runtime Composition

Each physical pool is allocated once using the existing offload raw split:

- `raw_k`
- `raw_v`
- optional C8 pad/scale storage

Owner-specific views are then created from that shared storage:

- main owner: main K/V views plus resident top-k work buffers;
- BF16 indexer owner: indexer K view;
- C8 indexer owner: indexer K and scale views.

Indexer reshape uses:

```text
[num_manager_blocks * blocks_per_manager_block, 128, 1, index_head_dim]
```

The SFA kernels still expect the legacy combined tuple. Main cache binding
therefore retains placeholder positions, and
`AscendSFAImpl._compose_offload_kv_cache()` substitutes the active indexer K
and optional scale from `self.indexer.k_cache.kv_cache` immediately before
execution. A skipped layer keeps placeholders and never accesses them.

This composition layer should be removed when the SFA kernel interface accepts
separate main/indexer cache handles.

## KV Group Planning

`patch_kv_cache_utils.py` now installs a GLM-specific group planner when the
cache spec set consists of offload main specs and offload indexer specs.

The planner:

1. rejects DCP or PCP greater than one;
2. unifies main/indexer page sizes;
3. emits one main group containing all SFA layers;
4. emits compact indexer groups with at most four owners for the current
   configuration.

Other models and cache layouts fall back to upstream grouping.

## AscendStore Changes

AscendStore no longer rejects layerwise mode solely because multiple KV cache
groups exist.

For split GLM SFA caches it now:

- treats the configuration as hybrid/multi-group;
- registers main K/V only for the main group, excluding placeholders and
  resident top-k work buffers;
- registers indexer K/scale under their indexer group IDs;
- records transformer-layer-to-`(group_id, group_layer_id)` ownership;
- uses group block sizes, keys, block IDs, and addresses for key-based
  layerwise save/load;
- queues a main owner and its active indexer owner(s) in the same transformer
  layer step;
- disables the single-group GVA layerwise path for multi-group caches and uses
  the group-aware key path instead.

Layerwise overwrite gating remains main-only. Indexer transfers are queued
with the corresponding main-layer step, so completion of that layer's FIFO
work protects the shared raw pool before a later main reuse mate overwrites it.

PD-disaggregation-specific workers/connectors were not refactored in this
change and remain out of scope.

## Files Changed

Core implementation:

- `vllm_ascend/core/kv_cache_interface.py`
- `vllm_ascend/attention/indexer.py`
- `vllm_ascend/attention/sfa_v1.py`
- `vllm_ascend/attention/utils.py`
- `vllm_ascend/ops/mla.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/patch/platform/patch_kv_cache_utils.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_config.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`

Tests:

- `tests/ut/core/test_kv_cache_interface.py`
- `tests/ut/attention/test_sfa_offload_indexer.py`
- `tests/ut/distributed/ascend_store/test_layerwise_config.py`

## Test Coverage Added

The new tests cover:

- two shared buffers plus default first/last independence producing four
  physical pools;
- all-active indexer grouping;
- compact grouping with skipped GLM-5.2 indexers;
- incomplete final indexer groups;
- duplicate-free physical owner coverage;
- indexer group positions mapping to distinct physical pools;
- BF16 and C8 indexer page accounting;
- semantic indexer head size versus physical padding;
- the fixed 128-token backend kernel block size;
- the unchanged non-offload metadata builder;
- offload metadata slicing from the indexer's own group.

## Validation Completed

Completed locally:

- package-wide Python bytecode compilation with the bytecode cache redirected
  to `/tmp`;
- `git diff --check`;
- targeted Ruff checks for the new spec/backend/grouping helpers and tests.

Not run:

- NPU-dependent unit tests;
- GLM-5 or GLM-5.2 service smoke tests;
- remote AscendStore round-trip tests;
- performance or memory-capacity measurements.

The workspace instructions prohibit running `torch_npu`-dependent tests on the
local Mac, and remote verification was explicitly skipped for this task.

## Remaining Work and Risks

1. Run the focused unit tests and GLM-5/GLM-5.2 service path on an Ascend
   container before merging.
2. Exercise both BF16 and C8 with an actual GLM-5.2 indexer-skip pattern.
3. Validate that reported KV capacity reflects exactly four physical paged
   pools for `layerwise_num_shared_buffers=2` with default independent layers.
4. Run AscendStore save/load round trips with several indexer groups and verify
   independent group block IDs address the expected shared raw pages.
5. Confirm prefix caching and preemption restore both main and indexer groups.
6. Validate PP behavior; the current implementation explicitly rejects DCP/PCP
   but does not add a new PP restriction.
7. Keep PD-disaggregation disabled until its workers are updated for split
   cache ownership; several PD paths still assume every main layer carries a
   five/six-tensor combined cache at registration time.
8. Remove the runtime placeholder/composition compatibility layer after the
   kernel accepts split handles.

## Review Checklist

- Confirm no code deep-copies an indexer into GLM-5.2 skipped layers.
- Confirm `skip_topk` layers publish only a main cache spec.
- Confirm the default first/last independent configuration plus two shared
  buffers yields four `KVCacheTensor` entries.
- Confirm every active indexer group has at most four owners.
- Confirm members of one indexer group map to distinct physical pools.
- Confirm indexer groups use their own block tables and slot mappings.
- Confirm only main reused layers appear in `_layerwise_reuse_mate_map`.
- Confirm BF16 indexer manager blocks are 512 tokens but kernel blocks are 128.
- Confirm C8 indexer manager blocks are 1024 tokens but kernel blocks are 128.
- Confirm main/indexer page sizes match before physical-pool rewriting.
- Confirm AscendStore excludes resident work buffers from persistent group
  registration.
- Confirm PD-disaggregation is not inferred to work from local offload or
  AscendStore behavior.
