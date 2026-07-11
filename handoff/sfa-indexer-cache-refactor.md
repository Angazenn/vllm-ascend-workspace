# SFA Indexer Cache Spec Refactor Handoff

## Status

- Workspace: `vllm-ascend-workspace`
- vLLM-Ascend branch: `indexer_reuse`
- Current vLLM-Ascend commit: `2f78eb8e SFA indexer spec refactor`
- Base at the time of this handoff: `origin/main`
- Current scope: non-connector SFA execution
- Primary target: GLM-5.x/GLM-5.2 SFA indexer cache reuse
- Pull request: <https://github.com/vllm-project/vllm-ascend/pull/11647>

The vLLM-Ascend submodule is clean. The workspace root records changed submodule
pointers, which is expected.

## Problem

Before this work, vLLM-Ascend described the SFA main MLA cache and indexer cache
with one `AscendMLAAttentionSpec`. The model runner then split that allocation
into multiple physical tensors:

- main MLA K/V or packed KV
- indexer K
- optional indexer scale

This made indexer storage part of every main SFA cache specification. For models
that reuse top-k indices and do not instantiate an indexer on every layer, such
as GLM-5.2, the combined spec could reserve indexer memory for layers that only
need the main MLA cache.

The allocation also diverged from upstream vLLM, where the main MLA cache and
`DeepseekV32IndexerCache` publish separate cache specifications.

## Cache Model

Three forms of cache sharing must remain distinct.

### KV cache group

Layers in one `KVCacheGroupSpec` share a logical block table and therefore use
the same block IDs. Main SFA and its indexer cache belong together here because
attention block `N` and indexer block `N` describe the same token page.

### Physical cache tensor

`KVCacheTensor.shared_by` describes physical storage aliasing. Different KV
cache groups can share one byte buffer because their live block IDs come from a
common pool and do not overlap.

Main SFA and its indexer must not use this form of sharing: they use the same
block IDs simultaneously, so aliasing one physical page would overwrite data.

### Cross-layer cache reuse

Explicit cross-layer reuse means two layers intentionally consume the same
logical cache content. This is separate from both grouping and physical memory
pooling.

The resulting SFA design is:

```text
same UniformTypeKVCacheSpecs / KV cache group
    -> same block table and block IDs

different cache specs and KVCacheTensor allocations
    -> different physical tensors

different attention backends / AttentionGroups
    -> main SFA metadata versus cache-only indexer metadata
```

## Implemented Design

### Separate cache specifications

`AscendMLAAttentionSpec` now describes only the main MLA cache.

`AscendSFAIndexerCacheSpec` describes:

- indexer K head size and dtype
- optional indexer scale dimension and dtype
- sparse C8 state
- DCP indexer replication factor
- indexer page-size accounting

Both specs can be collected into one `UniformTypeKVCacheSpecs`, preserving one
logical cache group while allowing vLLM to emit independent physical cache
tensors for the two members.

Indexer-specific fields were removed from the main spec. `scale_dim` and
`scale_dtype` remain on `AscendMLAAttentionSpec` because DeepSeek V4 compressed
MLA still uses them.

### Cache layouts

Non-C8 SFA:

```text
main spec:    (k_cache, v_cache)
indexer spec: (indexer_k_cache,)
kernel input: (k_cache, v_cache, indexer_k_cache)
```

Sparse C8 on A3 and A5 uses the unified packed layout introduced by upstream
vLLM-Ascend PR #11228:

```text
main spec:    (packed_kv_cache,)
indexer spec: (indexer_k_cache, indexer_scale_cache)
kernel input: (packed_kv_cache, indexer_k_cache, indexer_scale_cache)
```

Device-specific indexer dtypes remain:

- A3: indexer K `int8`, scale `float16`
- A5: indexer K `float8_e4m3fn`, scale `float32`

### Runtime cache composition

The current Ascend SFA kernels still accept the legacy combined cache tuple.
`AscendSFAImpl._compose_sfa_kv_cache()` therefore reads:

- the main cache from the SFA attention module argument
- the indexer cache from `self.indexer.k_cache.kv_cache`

It validates both tuple shapes and composes the kernel input shown above.
Layers that reuse top-k indices and have no local indexer return the main cache
unchanged.

This is an adaptation layer, not the desired final kernel API. A TODO records
that it should disappear once kernels accept separate main/indexer handles.

### Placeholder indexer backend

The split indexer cache needs its own backend so model-runner cache
initialization can create an AttentionGroup for its spec. The implementation is
in `vllm_ascend/attention/indexer.py`, mirroring upstream's `indexer.py` module.

`AscendSFAIndexerBackend` provides:

- backend identity
- cache shape
- supported block size
- metadata-builder selection

It is a cache placeholder and does not execute indexer computation.

`AscendSFAIndexerMetadataBuilder` deliberately returns `None`. The real SFA
attention layer continues to build and consume SFA metadata. Reusing
`AscendSFAMetadataBuilder` is not valid because its MLA base assumes
`layer_names[0]` resolves to a real `MLAAttention` object with
`prefill_backend`. The indexer layer resolves to `DeepseekV32IndexerCache`,
which does not have that attribute.

### Model-runner allocation and reshape

The model runner recognizes `AscendSFAIndexerCacheSpec` before the generic
attention path.

Allocation creates:

- one raw indexer K tensor
- one raw scale tensor when `scale_dim > 0`
- an aligned combined raw allocation for C8 K/scale when KV transfer alignment
  is active

Reshape independently creates indexer K and scale views. DCP replication is
owned by the indexer spec: its page size is multiplied by
`sfa_dcp_replicated_indexer_size`, and its logical block dimension is reshaped
to `num_blocks * replication_size`.

The main SFA allocation no longer creates indexer tensors.

### Cache binding

Splitting the cache creates two cache-owning module names under one transformer
layer:

```text
...self_attn.attn
...self_attn.indexer.k_cache
```

Upstream `bind_kv_cache` groups names by extracted transformer-layer index.
On the Ascend path, these two names create a duplicate layer-index situation.

The custom binding path therefore:

- binds both caches to their modules in `static_forward_context`
- keeps only real attention caches in the roleless runner `self.kv_caches`
- omits indexer entries from that flattened list

This does not mean one physical cache per transformer layer. It means one cache
per cache-owning module, while preserving the legacy meaning of
`self.kv_caches`.

### C8 ownership and selection

The upstream PR #11228 behavior is retained:

```python
use_sparse_c8_sfa = use_sparse_c8_indexer or (
    enable_sparse_c8 and not has_indexer and skip_topk
)
```

The ownership lookup was tightened:

- main SFA spec uses `impl.use_sparse_c8_sfa`
- runtime indexer C8 checks `self.indexer.k_cache.prefix`
- indexer spec checks its own `*.indexer.k_cache` layer name

PR #11228 checked the parent SFA layer name for indexer C8. The current
`AscendConfig.is_sparse_c8_layer()` strips indexer quantization suffixes to the
common parent prefix and also falls back to transformer-layer index matching.
Consequently, parent attention and indexer cache names currently produce the
same result. The new call sites make ownership explicit but do not intentionally
change C8 selection.

There is still duplicate determination: runtime initialization and cache-spec
construction each query C8 state. They should remain equivalent under the
current matcher, but this is worth consolidating in a future cleanup.

## Upstream Comparison

Upstream vLLM keeps indexer and sparse MLA as distinct internal operations under
one outer MLA-layer forward:

```text
MultiHeadLatentAttentionWrapper.forward
    -> Indexer.forward
        -> SparseAttnIndexer custom op
        -> update indexer cache and topk_indices_buffer
    -> MLAAttention.forward
        -> consume topk_indices_buffer
        -> run sparse attention
```

`DeepseekV32IndexerCache.forward()` is empty because the module owns cache
planning/binding, not indexer computation.

vLLM-Ascend still performs indexer selection and SFA attention inside
`AscendSFAImpl.forward()`. The split specs align cache ownership with upstream,
while `_compose_sfa_kv_cache()` preserves the existing combined Ascend kernel
contract.

## DeepSeek V4 Compatibility

DeepSeek V4 follows the `use_compress` path, while this change is guarded by
the SFA `use_sparse` path. Its DSA/indexer backends and real metadata builders
remain unchanged.

The main compatibility points are:

- compressed MLA keeps `scale_dim` and `scale_dtype`
- DeepSeek V4 indexer/cache modules do not receive
  `AscendSFAIndexerCacheSpec`
- the existing DeepSeek V4 binding/layout branch remains in control

A focused unit test protects the compressed MLA indexer layout. A full
DeepSeek V4 runtime validation has not been performed as part of the latest
rebased state.

## Scope Boundary

This phase intentionally targets the non-connector scenario.

Not yet established:

- connector registration semantics for split SFA main/indexer tensors
- connector transfer and restore of both cache owners
- disaggregated prefill/decode interoperability with the split ownership model
- offload behavior for every connector implementation

The existing aligned raw-allocation logic was preserved, but that alone is not
connector validation.

## Files Changed

Core implementation:

- `vllm_ascend/core/kv_cache_interface.py`
- `vllm_ascend/attention/indexer.py`
- `vllm_ascend/attention/sfa_v1.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/ops/mla.py`
- `vllm_ascend/utils.py`

Tests and CI selection:

- `tests/ut/worker/a2/test_model_runner_v1.py`
- `tests/ut/ops/test_mla.py`
- `tests/ut/attention/test_indexer.py`
- `.github/workflows/scripts/test_config.yaml`

## Validation Completed

After the latest upstream rebase:

- relevant pre-commit hooks passed
- focused remote tests passed: 20 tests
- CPU-only placeholder backend tests passed: 2 tests
- the CI coverage-config validator reported zero uncovered source/test files
- all changed Python files passed Ruff, formatting, spelling, forbidden-import,
  package-layout, and repository-specific checks

Before the latest upstream rebase, a full GLM-5 service validation also passed:

- TP16 service became healthy
- a real request returned a coherent response
- reported KV cache capacity was 55,808 tokens with 5.03 GiB available KV
  memory

That service result was useful evidence for the design, but the final squashed
commit should receive another full service run before merge because upstream
PR #11228 changed the packed C8 implementation during the rebase.

## Remaining Work and Risks

1. Run the final squashed commit through the full GLM-5/GLM-5.2 service path on
   the target A3/A5 matrix.
2. Add connector-aware design and validation before claiming connector support.
3. Remove `_compose_sfa_kv_cache()` when kernels accept separate cache handles.
4. Replace the custom binding path when upstream/Ascend binding becomes aware
   of cache-module roles instead of only transformer-layer indices.
5. Consolidate the duplicate C8 decision made during module initialization and
   indexer-spec construction.
6. Exercise DCP/PCP, prefix caching, offload, and disaggregated serving with the
   split tensors.
7. Run a full DeepSeek V4 regression after the final branch is published.

## Review Checklist

- Confirm main and indexer specs are combined into one logical KV cache group.
- Confirm they receive separate physical allocations.
- Confirm non-indexer/skip-topk layers allocate no indexer tensor.
- Confirm A3/A5 packed C8 main cache contains exactly one tensor.
- Confirm C8 indexer cache contains K and scale tensors.
- Confirm DCP replication is counted only by the indexer spec.
- Confirm both modules are bound through `static_forward_context`.
- Confirm indexer cache is absent from the flattened `self.kv_caches` list.
- Confirm DeepSeek V4 remains on its existing compressed-cache path.
- Confirm connector behavior is not inferred from non-connector validation.
