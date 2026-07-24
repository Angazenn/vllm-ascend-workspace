# vLLM DeepSeek V4 KV Cache Spec Initialization

## Scope

- Workspace: `vllm-ascend-workspace`
- vLLM commit inspected: `54503ecec0f3ac31e5ecfc5f28652e4cc42307b5`
- vLLM checkout state: detached `HEAD`
- Subject: upstream vLLM DeepSeek V4 cache-spec collection, cache-group
  construction, packed allocation, and `KVCacheTensor.shared_by` selection
- Primary implementation:
  `vllm/v1/core/kv_cache_utils.py`

This document focuses on cache initialization. It assumes the reader already
understands the DeepSeek V4 C4/C128 compression and attention computation.

## Executive Summary

DeepSeek V4 uses several cache lifetimes and token granularities:

```text
full compressed history: logical block size 256
raw SWA:                 block size 64
C4 compressor state:     block size 4, window 8
C128 compressor state:   block size 8, window 128
```

vLLM initializes these caches in four stages:

```text
cache-owning modules
    -> one KVCacheSpec per module
    -> KVCacheGroupSpecs sharing logical block tables
    -> KVCacheTensor descriptors sharing physical page slots
    -> one packed backing allocation with per-cache strided views
```

Two meanings of "sharing" must remain distinct:

1. Layers in one `KVCacheGroupSpec` share a logical block table. They use the
   same block IDs simultaneously and therefore require different physical page
   slots.
2. Layers in one `KVCacheTensor.shared_by` list alias the same physical page
   slot. They must belong to different cache groups, whose live block IDs are
   allocated from a common `BlockPool` and therefore do not overlap.

`shared_by` is physical pool multiplexing. It does not mean the listed layers
contain the same KV values.

## 1. Cache-Providing Modules

Each DeepSeek V4 cache-owning module implements `AttentionLayerBase`, registers
itself in `compilation_config.static_forward_context`, and exposes
`get_kv_cache_spec()`.

For an attention prefix such as `model.layers.0.attn`, the possible cache
modules are:

| Module suffix | Layers | Spec |
|---|---|---|
| `.swa_cache` | All | `SlidingWindowMLASpec(block_size=64, sliding_window=W)` |
| `.attn` | C4/C128 | `MLAAttentionSpec(block_size=256, compress_ratio=C)` |
| `.compressor.state_cache` | C4 | `SlidingWindowMLASpec(block_size=4, sliding_window=8)` |
| `.compressor.state_cache` | C128 | `SlidingWindowMLASpec(block_size=8, sliding_window=128)` |
| `.indexer.k_cache` | C4 only | `MLAAttentionSpec(block_size=256, compress_ratio=4)` |
| `.indexer.compressor.state_cache` | C4 only | `SlidingWindowMLASpec(block_size=4, sliding_window=8)` |

The main attention module returns no spec for `compress_ratio <= 1`; an
SWA-only layer still has its separate `.swa_cache`.

Relevant definitions:

- Main compressed cache:
  `vllm/models/deepseek_v4/attention.py::DeepseekV4Attention.get_kv_cache_spec`
- Indexer K cache:
  `vllm/models/deepseek_v4/attention.py::DeepseekV4IndexerCache`
- Raw SWA cache:
  `vllm/v1/attention/backends/mla/sparse_swa.py::DeepseekV4SWACache`
- Compressor state:
  `vllm/models/deepseek_v4/compressor.py::CompressorStateCache`

## 2. Spec Collection

`GPUModelRunner.get_kv_cache_spec()` scans all registered
`AttentionLayerBase` modules:

```python
attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
for layer_name, attn_module in attn_layers.items():
    if spec := attn_module.get_kv_cache_spec(vllm_config):
        kv_cache_spec[layer_name] = spec
```

For attention specs, the runner also asks the backend whether kernels index
pages by block stride and copies that capability into the spec.

The engine gathers the dictionaries from all workers, merges them by layer
name, constructs groups globally, and then projects the global groups back to
each worker. Global construction is important for pipeline parallelism:
grouping follows the whole model's layer ratios rather than one PP stage's
local subset.

Entry points:

- Collection:
  `vllm/v1/worker/gpu_model_runner.py::GPUModelRunner.get_kv_cache_spec`
- Multi-worker merge and projection:
  `vllm/v1/core/kv_cache_utils.py::get_kv_cache_configs`
- Main group dispatch:
  `vllm/v1/core/kv_cache_utils.py::get_kv_cache_groups`

## 3. Logical and Storage Block Sizes

`MLAAttentionSpec.block_size` remains in raw-token coordinates. Compression
changes only the number of stored entries inside a physical page:

```python
storage_block_size = block_size // compress_ratio
```

With the standard logical block size of 256:

| Cache | Logical tokens represented | Entries stored per page |
|---|---:|---:|
| C4 main KV | 256 | 64 |
| C4 indexer K | 256 | 64 |
| C128 main KV | 256 | 2 |

This lets the full-history caches share a raw-token block table even though
their physical page shapes differ.

The standalone cache groups use their natural raw-token block sizes:

| Cache group type | Block size |
|---|---:|
| Raw SWA | 64 |
| C4 main/indexer states | 4 |
| C128 states | 8 |

The scheduler block size is the LCM of the group block sizes, normally 256 for
DeepSeek V4.

## 4. Initial DeepSeek V4 Spec Categories

`group_and_unify_kv_cache_specs()` activates when any
`SlidingWindowMLASpec` is present. It first creates `UniformTypeKVCacheSpecs`
categories.

### Full MLA category

All `MLAAttentionSpec` instances enter one category:

```text
C4 main compressed KV
C4 indexer K
C128 main compressed KV
```

They all have logical block size 256 and full-history lifetime. The
`UniformTypeKVCacheSpecs` wrapper preserves each module's individual
compression ratio, page size, dtype, and backend layout.

### Sliding-window categories

All `SlidingWindowMLASpec` instances are keyed by:

```python
(spec.block_size, spec.sliding_window)
```

For DeepSeek V4 this normally gives:

```text
(64, W):   raw SWA caches
(4, 8):    C4 main compressor states + C4 indexer compressor states
(8, 128):  C128 compressor states
```

This key is a DSV4-specific convention in the current implementation. The code
itself notes that grouping only by block size and window is fragile.

## 5. Understanding `_get_kv_cache_groups_uniform_groups`

The function consumes:

```python
[
    full_mla_spec,
    raw_swa_spec,
    c4_state_spec,
    c128_state_spec,
]
```

and returns one unsplit full group plus one or more split groups for every
sliding-window category.

### 5.1 The full group is the anchor

The first element must contain only `MLAAttentionSpec`. It becomes one
`KVCacheGroupSpec` without being split:

```python
full_mla_group = KVCacheGroupSpec(
    layer_names=list(full_mla_spec.kv_cache_specs),
    kv_cache_spec=full_mla_spec,
)
```

It defines:

- the anchor number of layer tuples;
- the physical page-size classes allowed in the packed layout;
- the full-history block-table group.

### 5.2 Layer tuples

A layer tuple contains one layer for each distinct physical page size in a
`UniformTypeKVCacheSpecs`.

For example:

```text
11 C4 indexer caches: page size A
11 C4 main caches:    page size B
10 C128 main caches:  page size C
```

is interpreted as:

```text
tuple 0:  [C4I-0,  C4A-0,  C128-0]
...
tuple 9:  [C4I-9,  C4A-9,  C128-9]
tuple 10: [C4I-10, C4A-10, implicit padding]
```

`UniformTypeKVCacheSpecs.get_num_layer_tuples()` returns the most common number
of layers among its distinct page sizes. For counts `{A: 11, B: 11, C: 10}`,
the tuple count is 11.

This is a structural assumption about DSV4's repeated cache pattern, not a
general tuple-inference algorithm.

### 5.3 Target tuple count

The function computes the tuple counts of all input categories. A representative
case is:

```text
full MLA:   11
raw SWA:    21
C4 states:  11
C128 state: 10
```

It calls:

```python
target = _approximate_gcd(
    [11, 21, 11, 10],
    lower_bound=11,
)
```

Despite the name, `_approximate_gcd` does not calculate a mathematical GCD. It
tests every candidate `d` from `lower_bound` through the largest count and
minimizes:

```text
sum(round_up(x, d) - x)
```

Ties prefer larger `d`. For the example, `d=11` needs only two padded tuples:

```text
11 -> 11: 0
21 -> 22: 1
11 -> 11: 0
10 -> 11: 1
```

The full group's tuple count is the lower bound because the full group is the
unsplit anchor.

The function subsequently assigns rounded counts to
`num_layer_tuples_per_group`, but the reassigned list is not used afterward.
The actual split is driven by `target`, `cdiv`, and strided tuple distribution.
This dead assignment is one reason the code is difficult to follow.

### 5.4 Align sliding-window pages to full-group page sizes

The function obtains the full group's page sizes:

```python
all_page_sizes = full_mla_spec.get_page_sizes()
```

Each sliding-window/state page of size `P` is padded to:

```python
min(full_page_size for full_page_size in all_page_sizes
    if full_page_size >= P)
```

This restricts all categories to compatible physical slot sizes.

For the FlashMLA `fp8_ds_mla` layout, the important sizes are:

| Cache | Real bytes | Initial aligned bytes | Packed class |
|---|---:|---:|---:|
| C4 main KV | 37,376 | 37,440 | 37,440 |
| C4 indexer K | 8,448 | 8,640 | 8,640 |
| C128 main KV | 1,168 | 1,728 | 1,728 |
| Raw SWA | 37,376 | 37,440 | 37,440 |
| C4 main state | 32,768 | 32,832 | 37,440 |
| C4 indexer state | 8,192 | 8,640 | 8,640 |
| C128 state | 32,768 | 32,832 | 37,440 |

The first alignment comes from each DSV4 spec's 576-byte alignment. The final
column includes the second-stage padding to the nearest full-group class.
FlashInfer/BF16 layouts have different numbers but follow the same process.

The implementation mutates `page_size_padded` with `object.__setattr__` even
though the spec dataclasses are frozen.

### 5.5 Build tuples inside one category

After alignment, layers are collected by page size:

```python
layers_per_size[page_size].append(layer_name)
```

The function asserts that every page-size list inside one category has the same
length. For C4 state:

```text
8,640:  [IState-0, IState-1, ..., IState-10]
37,440: [AState-0, AState-1, ..., AState-10]
```

Then:

```python
layer_tuples = list(zip(*layers_per_size.values()))
```

produces:

```text
(IState-0, AState-0)
(IState-1, AState-1)
...
(IState-10, AState-10)
```

For raw SWA there is only one page size, so each tuple has one cache module.

### 5.6 Split and interleave tuples

For a category with `n` tuples:

```python
num_tuple_groups = cdiv(n, target)
```

The function distributes tuples with:

```python
group_layer_tuples = layer_tuples[i::num_tuple_groups]
```

This is strided distribution rather than contiguous slicing. With 21 raw-SWA
tuples, target 11, and two groups:

```text
raw-SWA group 0: tuple 0, 2, 4, ..., 20  # 11 tuples
raw-SWA group 1: tuple 1, 3, 5, ..., 19  # 10 tuples
```

Striding distributes sequential transformer layers more evenly across groups,
which avoids pathological padding after projection to PP workers.

Each selected tuple list is flattened and wrapped in a new
`UniformTypeKVCacheSpecs` and `KVCacheGroupSpec`.

### 5.7 Representative output

For 11 C4 layers, 10 C128 layers, and 21 total raw-SWA layers:

```text
Group 0: full MLA
    11 C4 indexer K
    11 C4 main KV
    10 C128 main KV
    logical block size 256

Group 1: raw SWA, even transformer layers
    11 cache modules
    block size 64

Group 2: raw SWA, odd transformer layers
    10 cache modules
    block size 64

Group 3: C4 states
    11 tuples x (indexer state, main state)
    22 cache modules
    block size 4

Group 4: C128 states
    10 cache modules
    block size 8
```

Every module in one group uses the same group block table.

## 6. Constructing `KVCacheTensor.shared_by`

Group creation and physical sharing are separate steps.

`_bucket_layers_by_page_size()` creates:

```python
buckets[page_size][slot_idx] = [layer_names]
```

For every cache group, it resets a counter per page size:

```python
slot_count = defaultdict(int)
```

The first layer of page size `P` in that group goes to slot 0, the second to
slot 1, and so forth. When another group is processed, its counters restart at
zero, so its page-`P` layers are appended to the existing slots.

This guarantees that a slot contains at most one layer from each logical cache
group.

Using the representative example:

```text
37,440-byte slot i, i=0..9:
    C4 main KV i          from full group
    raw SWA i             from raw-SWA group 0
    raw SWA i             from raw-SWA group 1
    C4 main state i       from C4-state group
    C128 state i          from C128-state group

8,640-byte slot i:
    C4 indexer K i        from full group
    C4 indexer state i    from C4-state group

1,728-byte slot i:
    C128 main KV i        from full group
```

Shorter groups simply do not appear in the final slot. For example, slot 10
has no member from a group containing only 10 tuples.

Each `buckets[P][i]` list becomes one `KVCacheTensor.shared_by` list.

## 7. Why Cross-Group Aliasing Is Safe

All cache-group managers use one common `BlockPool`. Consequently, physical
block IDs are exclusive across simultaneously live allocations:

```text
if group A owns block 7,
group B cannot simultaneously own block 7
```

Therefore two layers from different groups can map their block 7 view to the
same bytes: only the group currently owning block 7 may use those bytes.

Layers within one group cannot alias:

```text
same group
    -> same block table
    -> same block IDs at the same time
    -> require distinct page slots
```

Layers from different groups may alias:

```text
different groups
    -> independently managed block tables
    -> globally exclusive live block IDs
    -> may reuse one physical page slot
```

This is not the explicit cross-layer KV-content reuse represented by
`kv_sharing_target_layer_name`.

## 8. Packed Cache Planning

`_get_kv_cache_config_packed()` calculates bytes per global physical block:

```python
bytes_per_block = sum(
    page_size * number_of_slots_at_that_size
    for each page_size
)
```

Then:

```python
num_blocks = available_memory // bytes_per_block
total_size = num_blocks * bytes_per_block
```

It emits one descriptor for each `(page_size, slot_idx)`:

```python
KVCacheTensor(
    size=total_size,
    shared_by=buckets[page_size][slot_idx],
    offset=byte_offset,
    block_stride=bytes_per_block,
)
```

All descriptors report the same `total_size` because they are views into one
backing allocation. `offset` selects a slot within each packed block, while
`block_stride` advances to the next global block.

Conceptually:

```text
packed backing: [num_blocks, bytes_per_block]

block 0:
    [large slot 0][large slot 1]...[medium slot 0]...[small slot 0]...
block 1:
    [large slot 0][large slot 1]...[medium slot 0]...[small slot 0]...
...
```

## 9. Allocation and Binding

Because packed descriptors have `block_stride > 0`,
`GPUModelRunner._allocate_kv_cache_tensors()` creates one `torch.int8`
backing tensor and maps every listed layer name to it.

`_reshape_kv_cache_tensors()` builds a `(offset, block_stride)` lookup for each
layer. `_reshape_attention_kv_cache()` then creates the layer view:

```python
backing.view(-1, block_stride)[:, offset:offset + real_page_bytes]
```

and converts it to the backend's dtype and cache shape.

Two layers in the same `shared_by` list receive the same offset and therefore
alias the same page bytes. Layers with different descriptors share the backing
allocation but use different offsets.

Finally, `bind_kv_cache()` assigns every view to the corresponding registered
module's `.kv_cache`.

## 10. Worker and Scheduler Views

The worker-side `KVCacheConfig` retains `UniformTypeKVCacheSpecs` so allocation
and attention backends can see every module's actual page size and layout.

The scheduler does not need the per-layer physical details. When
`generate_scheduler_kv_cache_config()` encounters a uniform wrapper, it
replaces it with one representative inner spec. All members have the same
manager behavior by construction:

```text
full MLA group:  block 256
raw SWA group:   block 64, window W
C4 state group:  block 4, window 8
C128 state group:block 8, window 128
```

The scheduler builds one manager per group and all managers share one global
`BlockPool`.

## 11. Invariants and Fragile Assumptions

The DSV4 path depends on the following invariants:

1. The first grouped spec contains all and only full-history
   `MLAAttentionSpec`s.
2. All full specs use the same logical block size.
3. Sliding-window categories can be uniquely identified by
   `(block_size, sliding_window)`.
4. Within one sliding-window category, every aligned page-size class has the
   same number of layers.
5. Every sliding-window page can be padded to a full-group page size that is at
   least as large.
6. The most common per-page-size layer count is a valid estimate of the number
   of layer tuples.
7. The full group is a suitable unsplit anchor for the other categories.
8. Packed cache backends use a blocks-first layout compatible with
   `(offset, block_stride)` views.
9. One common `BlockPool` enforces exclusive live block IDs across cache
   groups.

Potential cleanup opportunities:

- Rename `_approximate_gcd` to describe its padding-minimization behavior.
- Remove or use the dead rounded `num_layer_tuples_per_group` assignment.
- Replace `(block_size, sliding_window)` DSV4 category detection with an
  explicit cache role.
- Represent tuples directly instead of inferring them from page-size counts
  and dictionary ordering.
- Avoid mutating frozen spec objects with `object.__setattr__`.
- Add a debug dump of groups, page classes, slots, offsets, and `shared_by`
  lists during cache initialization.

## Source Map

- Spec classes and storage block size:
  `vllm/v1/kv_cache_interface.py`
- DSV4 cache modules:
  `vllm/models/deepseek_v4/attention.py`
- Compressor state specs:
  `vllm/models/deepseek_v4/compressor.py`
- Raw SWA spec:
  `vllm/v1/attention/backends/mla/sparse_swa.py`
- Spec collection:
  `vllm/v1/worker/gpu_model_runner.py::get_kv_cache_spec`
- DSV4 category construction:
  `vllm/v1/core/kv_cache_utils.py::group_and_unify_kv_cache_specs`
- DSV4 tuple grouping:
  `vllm/v1/core/kv_cache_utils.py::_get_kv_cache_groups_uniform_groups`
- Physical slot bucketing:
  `vllm/v1/core/kv_cache_utils.py::_bucket_layers_by_page_size`
- Packed descriptors:
  `vllm/v1/core/kv_cache_utils.py::_get_kv_cache_config_packed`
- Raw backing allocation and reshape:
  `vllm/v1/worker/gpu_model_runner.py`
- Packed attention view:
  `vllm/v1/worker/gpu/attn_utils.py::_reshape_attention_kv_cache`
- Shared block pool:
  `vllm/v1/core/kv_cache_coordinator.py::KVCacheCoordinator`
