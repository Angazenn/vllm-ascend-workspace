# GLM SFA Split-Cache Offload Refactor Handoff

## Status

- Workspace: `vllm-ascend-workspace`
- vLLM-Ascend branch: `offload_merge`
- Current committed HEAD: `477ef238 Support split-cache SFA PD disaggregation`
- Requested reference commit: `db41cf6e SFA indexer spec refactor`
- Equivalent commit in the reconstructed branch: `c69ad816 SFA indexer spec refactor`
- Current PD-disaggregation adaptation: implemented, remotely verified,
  committed, and pushed to `anga/offload_merge`
- Primary models: GLM-5.1 and GLM-5.2 SFA
- Current offload scope: BF16, DCP=1, PCP=1
- Remotely verified PD configuration: TP16 with one MTP draft token

The current branch was reconstructed on top of an updated
`lf/feat/sfa-offload-rebase-on-main`. Therefore `db41cf6e` is not an ancestor
of the current HEAD. Its logical replacement is `c69ad816`; the small textual
differences between them are integration changes required by the newer MTP/PD
baseline.

This handoff covers all split-cache offload refactoring layered on that indexer
spec baseline:

```text
c69ad816  SFA indexer spec refactor
334b64bf  Refactor SFA layerwise prefill offload cache
5555bffd  Refactor SFA decode offload cache
477ef238  Support split-cache SFA PD disaggregation
```

The updated remote baseline also already contained:

```text
a98c3ecd  support PD layer reuse with MTP cache layout
1eb781ea  keep AscendStore layerwise reuse stable with MTP decode
946c17bb  size SFAPD transfer events for MTP layers
```

Those three commits are inherited context, not newly authored parts of this
refactor.

## Motivation

The old combined layout made every SFA main layer look as if it owned a local
indexer cache. That was tolerable for GLM-5.1, where indexer ownership is broad,
but it wasted cache and produced incorrect grouping assumptions for GLM-5.2,
where many layers reuse top-k results and have `skip_topk=true`.

The refactor separates two logical resources:

```text
main SFA cache:  compressed K/nope plus rope/PE
indexer cache:   local sparse-indexer K, plus scale when applicable
```

Only a real `DeepseekV32IndexerCache` module publishes an indexer cache spec.
A layer that reuses top-k output keeps its main SFA cache but owns no indexer
cache, metadata, transfer entry, or physical allocation.

The important invariant is that cache ownership comes from real model modules,
not a hard-coded layer interval. The same code therefore handles GLM-5.1 and
GLM-5.2 without creating placeholder indexer owners.

## Core Invariants

- Main and indexer cache specs are separate logical owners.
- Only real indexer modules publish indexer specs. `skip_topk` layers never
  receive synthetic cache ownership.
- Every cache group owns its own block table, slot mapping, and block-ID
  namespace. Generic per-group metadata is used instead of deriving indexer
  addresses from the main group.
- Cache groups, physical `KVCacheTensor.shared_by` pools, and cross-layer reuse
  are separate concepts. Sharing or aliasing bytes does not make two groups
  share scheduler metadata.
- Main/indexer physical aliasing is retained only where the decode layout
  requires it. Connector code still identifies tensors by semantic ownership.
- AscendStore and decode CPU offload remain distinct connector paths with
  different host-memory ownership and transfer behavior.
- SFA kernels still receive a compatibility tuple composed at runtime; this is
  not the scheduler or allocator's logical cache model.

## Layouts by Mode

The logical split is common, but the grouping and physical storage differ by
execution mode.

### Non-offload baseline

The `c69ad816` indexer-spec refactor gives main SFA and indexer modules separate
cache specs and bindings. The runtime composes those split handles into the
legacy tuple still expected by the SFA kernel.

Sparse C8 behavior remains available on this non-offload path. This offload
refactor does not attempt to support or validate C8 together with offload.

### Layerwise prefill offload

This path is selected by `AscendStoreConnector` with layerwise mode enabled.
It intentionally runs with `use_offload=false`; `use_offload` refers to the
direct decode CPU-offload path, not AscendStore prefill offload.

Its logical layout is:

```text
group 0: all main SFA layers
group 1..N: compact groups containing only real indexer owners
```

With the default first/last independent layers and two shared layerwise
buffers, the model runner allocates four physical pools:

```text
pool 0: first main layer, independent
pool 1: last main layer, independent
pool 2: one set of reused interior layers
pool 3: the other set of reused interior layers
```

For `P` physical pools and `M` real indexer owners, indexer groups are formed
with:

```text
G = ceil(M / P)
indexer_group[g] = real_indexer_owners[g::G]
```

Each indexer group is no wider than the physical-pool count. The final
`KVCacheTensor.shared_by` rewrite associates each logical owner with one of
the actual physical pools and validates that every owner appears exactly once.

### Direct decode offload

This path is selected by exact connector name `SFAKVOffloadConnector`,
layerwise mode, and `use_offload=true`.

Its logical group order is currently:

```text
group 0: resident real-indexer owners
group 1: all main SFA layers
```

The final-group placement preserves the existing connector contract in which
`SFAKVOffloadConnector` consumes the main group. New code resolves groups by
cache spec type whenever possible instead of depending on this position.

The physical K/indexer alias is preserved. The main layer receives a five-entry
cache tuple:

```text
0: main K/nope cache
1: main V/rope cache
2: empty legacy indexer placeholder
3: CPU/LRU resident K workspace
4: CPU/LRU resident V/rope workspace
```

A real indexer module is bound separately to a one-tensor alias view over the
main raw K pool. Immediately before SFA execution, `_compose_sfa_kv_cache()`
replaces tuple entry 2 with that layer's bound indexer cache. A `skip_topk`
layer has no local indexer binding and does not perform this composition.

This adapter exists only because the current SFA kernel still accepts a
combined tuple. It can be removed when the kernel accepts separate main and
indexer cache handles.

## Prefill Refactor

Commit `334b64bf` introduced the layerwise AscendStore split layout.

### Cache specifications and metadata

- Added `AscendSFALayerwiseIndexerCacheSpec`.
- Added its cache-only attention backend and metadata builder.
- Kept main SFA request/offload metadata on the main attention layer.
- Built indexer block tables and slot mappings from each indexer's own cache
  group.
- Removed the need to duplicate explicit indexer block-table fields in normal
  main metadata.
- Removed synthetic/deep-copied indexer modules for `skip_topk` layers.

The manager page is padded so main and indexer specs can share physical pool
sizes while retaining the kernel's 128-token virtual block size. For the
current BF16 GLM dimensions:

```text
main page:     128 * (512 + 64) * 2 = 147456 bytes
indexer page:  512 * (128 + 16) * 2 = 147456 bytes
```

Thus one indexer manager page corresponds to four 128-token kernel blocks.

### AscendStore changes

AscendStore became cache-group aware for this GLM layout:

- Main K/V entries are registered under the main group.
- Indexer K entries are registered only for real indexer owners and use their
  own group IDs.
- Per-layer transfers select keys, block IDs, addresses, block sizes, and slot
  mappings from the owning group.
- Main and indexer entries for one transformer layer are queued in the same
  layerwise transfer step.
- The single-group GVA shortcut is not used for this multi-group layout.
- Layerwise overwrite gating remains tied to main layers. Completion of the
  layer step also protects any indexer data sharing that physical pool.

### Allocation and memory accounting

- Startup cache-capacity inflation and final physical allocation use the same
  layerwise physical-pool count.
- Main and compact indexer owners are merged into the actual
  `KVCacheTensor.shared_by` pools.
- Indexer cache views are reshaped from their owner pool using manager-page to
  kernel-block scaling.
- Binding preserves both cache-owning module names while the runner's flattened
  cache list continues to represent real attention layers.
- Attention-backend initialization detects split indexer spec classes. The old
  name-based `indexer` skip is retained only for legacy non-split entries, so a
  real split indexer receives its cache-only backend and metadata builder.

## Decode Refactor

Commit `5555bffd` introduced the direct decode-offload split layout.

### Cache specifications and grouping

- Added `AscendSFAOffloadIndexerCacheSpec` and its cache-only backend and
  metadata builder.
- `OffloadMLAAttentionSpec` describes only real main KV.
- The planner emits one resident indexer group containing the real owner subset
  and one main group containing every SFA layer.
- The owner subset is validated against main transformer-layer IDs.
- GLM-5.1 and GLM-5.2 use the same planner; no modulo layer pattern is encoded.
- Direct decode split offload is constrained to BF16, DCP=1, and PCP=1 for this
  step.

### Connector scope

`SFAKVOffloadConnector` remains KV-only:

- It discovers and consumes the main `OffloadMLAAttentionSpec` group.
- It saves and loads only main K/V CPU blocks.
- It does not register, offload, or load resident indexer groups.
- Indexer caches remain on NPU and are addressed through their own metadata.

MTP draft layers are not scheduler cache owners. When a draft forward context
does not contain the indexer module key, the proposer carries that indexer
group's independently updated block table and slot mapping on the draft main
metadata. This is a narrow MTP compatibility path, not a return to deriving
indexer metadata from the main group.

This keeps the decode CPU-offload path independent from AscendStore and avoids
hard-coding indexer tensors into connector tuple positions.

### Asynchronous block-boundary correctness

Two timing guards were added for asynchronous layerwise saves and MTP:

1. `OffloadMLAAttentionManager` tracks finalized tokens per request. A newly
   completed HBM block is not released until its tokens are finalized and the
   connector can safely preserve it.
2. `SFAKVOffloadWorker` separates pending CPU blocks from completed CPU blocks.
   CPU/LRU attention sees only completed blocks; pending blocks become visible
   after the layerwise save wait finishes.

This prevents decode from reading an incompletely copied CPU block or freeing
the HBM source one speculative step too early.

## PD Disaggregation Refactor

Commit `477ef238` adapts `SFAPDCpuOffloadConnector` to the two different split
layouts used by the P and D nodes.

### P and D cache layouts

The P node runs nested `AscendStoreConnector` layerwise prefill reuse:

```text
one main group
compact real-indexer groups
four layer-reused physical pools with the default configuration
```

The D node runs decode CPU offload:

```text
one resident real-indexer group
one main group
separate indexer bindings
five-entry main cache tuples
```

The layouts are intentionally different. PD transfer code now maps semantic
tensor roles between them rather than assuming identical group positions or
tuple layouts.

### Semantic layout discovery

The new `sfa_pd_cpu_offload/layout.py` provides `SFASplitCacheLayout` and
`resolve_sfa_split_cache_layout()`.

It:

- resolves the one main group and all indexer groups by cache spec class;
- records every owner-to-group mapping;
- requires indexer owners to be a subset of main transformer layers;
- permits at most one real indexer per transformer layer; and
- never assumes that main or indexer groups have fixed numeric positions.

The shared cache-interface helpers now recognize:

- direct or nested layerwise `AscendStoreConnector` on P; and
- exact `SFAPDCpuOffloadConnector` consumer mode as the D split-decode layout.

`AscendMultiConnector` forwards the finalized `KVCacheConfig` so the nested PD
connector can resolve the physical and logical layout after pool rewriting.

### Protocol changes

P creates one transfer metadata entry per main transformer layer. Each entry
contains semantic roles:

```text
main_k
main_v
indexer_k       # only for a real indexer owner
indexer_scale   # only when present
```

`MF_META` now carries:

- tensor roles;
- P source group IDs;
- manager-page block-size scales;
- the exact set of P real-indexer owner names; and
- the P session information required by MemFabric pull mode.

D validates that its real-indexer owner set exactly matches P. A mismatch
reports both missing and extra owner names instead of proceeding with a
corrupt layer mapping.

`READ_READY_BATCH` carries P block IDs for every cache group. D chooses source
IDs from P's semantic group mapping and destination IDs from its own main or
indexer group.

For a tensor whose manager page spans multiple kernel blocks, the source
address is computed as:

```text
address = base + block_id * block_len * block_size_scale
```

### Destination routing

- Complete main pages are pulled into the D CPU offload pool on TP rank 0.
- The current partial main page is pulled into resident D HBM on every rank.
- Real indexer pages are pulled into their resident D HBM cache on every rank.
- A `skip_topk` layer transfers only `main_k` and `main_v`.
- D cache registration reads separate indexer cache bindings and never assumes
  that indexer data is `main_cache[2]`.

### Reuse synchronization and MTP

P-side physical reuse mates are derived from the final
`KVCacheTensor.shared_by` pools, not from a hard-coded layer interval.

Before a reused pool is overwritten, both conditions must hold:

```text
SFAPD transfer has received READ_DONE from D
AscendStore has completed its own layerwise save/load reuse gate
```

Coordination events are sized and indexed by main transformer layers,
including MTP. Indexer cache owners do not create extra transformer-layer
event slots.

### MLAPO ordering fix

Remote checksum verification exposed one additional race. In the MLAPO path,
`_sfa_preprocess_with_mlapo()` scattered K/V into the layer-reused paged cache
before connector synchronization. D could therefore pull data from a later
layer that had already overwritten the same physical pool.

The wait now occurs before the MLAPO cache write:

```text
wait_for_kv_layer_from_connector(layer_name)
_sfa_preprocess_with_mlapo(... writes paged cache ...)
```

This wait serially covers both nested connectors under `AscendMultiConnector`:
the SFAPD `READ_DONE` dependency and the AscendStore reuse dependency.

## Configuration Semantics

Layerwise prefill offload is enabled by connector configuration, not by
`use_offload`:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "backend": "memcache",
    "use_layerwise": true,
    "lookup_rpc_port": "0",
    "layerwise_num_shared_buffers": 2
  }
}
```

Direct decode CPU offload is selected separately:

```json
{
  "kv_connector": "SFAKVOffloadConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "use_layerwise": true
  }
}
```

For PD disaggregation, the existing launch/proxy configuration remains valid:

- P uses `AscendMultiConnector` with producer-side
  `SFAPDCpuOffloadConnector` and layerwise `AscendStoreConnector` children.
- D uses consumer-side `SFAPDCpuOffloadConnector` and `use_offload=true`.
- MemFabric remains in pull mode.

No launch-script changes were required by the PD protocol refactor.

## Verification

### Static and unit checks

- Python compilation passed for all changed Python files.
- `git diff --check` passed.
- Focused remote test set passed: `22 passed, 16 warnings`.
- Coverage includes split layout resolution, exact owner-set validation,
  manager-page scaling, D registration, skip-topk behavior, MTP handling,
  block-boundary finalization, and direct decode cache composition.

### Remote PD verification

Environment:

```text
P/proxy: 192.168.13.157, container zyj_offload
D:       192.168.13.165, container zyj_offload
model:   GLM-5.2 w8a8
layout:  TP16, DCP=1, PCP=1, MTP=1
```

Observed layout:

```text
P: 79 main layers, 22 real indexer owners, 4 physical pools
D: 79 main layers, 22 real indexer owners
P and D owner sets matched exactly
```

Verified cases:

- Partial-block prompt: partial main page and resident indexer pages reached D
  HBM; output was coherent.
- Long prompt/decode: four complete main pages reached the D CPU pool, the
  partial page remained in HBM, indexer pages reached resident HBM, and decode
  offload continued across generated block boundaries.
- Concurrent two-request prefill: requests used distinct source and destination
  block IDs, and both responses were coherent.
- Debug checksums: indexer checksums matched exactly. Main floating reductions
  differed only by expected small reduction-order noise after the MLAPO fix.
- No transfer failure, owner mismatch, checksum failure, or reuse timeout was
  observed.

## Important Files

Cache specs, mode detection, and managers:

- `vllm_ascend/core/kv_cache_interface.py`
- `vllm_ascend/core/single_type_kv_cache_manager.py`
- `vllm_ascend/patch/platform/patch_kv_cache_utils.py`

Allocation, binding, and runtime composition:

- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/worker/worker.py`
- `vllm_ascend/attention/indexer.py`
- `vllm_ascend/attention/sfa_v1.py`

Prefill AscendStore:

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`

Direct decode offload:

- `vllm_ascend/distributed/kv_transfer/sfa_kv_offload/`

PD disaggregation:

- `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/layout.py`
- `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/protocol.py`
- `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/scheduler.py`
- `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/worker.py`
- `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/send_thread.py`
- `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/read_thread.py`
- `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/connector.py`
- `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`

Focused tests:

- `tests/ut/kv_offload/test_sfa_pd_cpu_offload_single_rank.py`
- `tests/ut/core/test_kv_cache_interface.py`
- `tests/ut/patch/platform/test_sfa_decode_offload_groups.py`
- `tests/ut/attention/test_sfa_offload_indexer.py`

The earlier, narrower design notes remain useful background:

- `handoff/sfa-indexer-cache-refactor.md`
- `handoff/offload-cache-refactor.md`

## Patch Size Assessment

The verified implementation is correct, but its review surface is too broad.
The net diff from the reconstructed split-indexer baseline `c69ad816` through
`477ef238` is:

```text
production: +2256 -1141 = 3397 changed lines
tests:       +780  -294 = 1074 changed lines
total:      +3036 -1435 = 4471 changed lines across 33 files
```

The three feature commits contribute this churn:

```text
334b64bf prefill split cache: +913  -801 = 1714
5555bffd decode split cache:  +878   -45 =  923
477ef238 PD split cache:     +1284  -628 = 1912
```

The largest individual files are:

```text
748  sfa_pd_cpu_offload/read_thread.py
505  worker/model_runner_v1.py
460  test_sfa_pd_cpu_offload_single_rank.py
426  core/kv_cache_interface.py
266  sfa_pd_cpu_offload/worker.py
248  ascend_store/pool_worker.py
225  ascend_store/kv_transfer.py
```

This is line churn, not merely new logic. In particular, the PD change rewrites
large portions of descriptor construction, batching, logging, and tests while
the required semantic change is narrower: P and D need to translate between
different main/indexer cache layouts.

## Reduction Strategy

### Preserve the verified implementation

Keep `477ef238` and `anga/offload_merge` as the known-correct full-feature
reference. Do not simplify that branch in place before a replacement branch
passes the same remote verification. The existing commit is useful as:

- a behavioral reference;
- a source for focused cherry-picks;
- a checksum/debug comparison target; and
- a fallback if the reduced implementation misses a boundary condition.

### Rebuild as stacked scopes

Start the reduced series from `c69ad816`, the reconstructed equivalent of the
split-indexer baseline. Build independently runnable scopes in this order:

1. Common split-cache offload infrastructure.
2. Layerwise prefill offload only.
3. Direct decode offload only.
4. Decode asynchronous block-finalization correctness.
5. PD disaggregation compatibility.
6. MLAPO wait-before-overwrite correctness.

The first verification target should stop after step 2. It should launch
layerwise `AscendStoreConnector` with `use_offload=false` and pass GLM-5.1 and
GLM-5.2 prefill-offload checks before decode or PD code is added.

Stacked commits make review easier, but they do not reduce the total diff by
themselves. The following implementation changes are needed to reduce actual
LOC and coupling.

### Unify split-indexer infrastructure

The layerwise-prefill and resident-decode indexer implementations currently
duplicate cache specs, metadata classes, metadata builders, and backends even
though their page-size and addressing behavior are effectively the same.

Use one shared split-offload indexer implementation, for example:

```text
AscendSFASplitIndexerCacheSpec
AscendSFASplitIndexerMetadata
AscendSFASplitIndexerMetadataBuilder
AscendSFASplitIndexerBackend
```

The active connector and main-cache spec already distinguish prefill from
decode. Group planning can use that mode rather than requiring two nearly
identical indexer class families.

Also centralize real-indexer owner discovery. One helper should return:

```text
main layer names ordered by transformer layer ID
real indexer owner names ordered by transformer layer ID
main-name/indexer-name mapping by transformer layer ID
```

The group planner, model runner, AscendStore, decode connector, and PD adapter
should consume that result instead of repeating layer-name parsing and subset
validation.

### Isolate model-runner changes

`model_runner_v1.py` currently contains split-spec checks in several generic
allocation, reshape, binding, and attention-group paths. Replace those
scattered branches with a small split-SFA allocation adapter that owns:

- physical-pool owner rewriting;
- raw main/indexer alias selection;
- owner-specific reshape;
- cache binding; and
- legacy kernel-tuple composition metadata.

The generic model-runner loops should call the adapter at one or two explicit
extension points. They should not need to know every split-indexer spec class.

The five-entry direct-decode main tuple can remain for compatibility, but the
empty indexer placeholder and bound alias should be assembled in that helper
rather than handled in multiple allocation branches.

### Keep connector responsibilities narrow

Retain the existing ownership boundaries:

- `AscendStoreConnector` handles prefill/prefix host storage.
- `SFAKVOffloadConnector` handles main-KV decode CPU offload only.
- Resident indexer caches never enter the decode CPU pool.
- `SFAPDCpuOffloadConnector` translates between P and D layouts but does not
  redefine either connector's local cache ownership.

Introduce a small semantic registration record shared by connectors, such as:

```text
layer name
tensor role
cache group ID
physical tensor/view
manager-page scale
```

AscendStore and SFAPD can build these records once during cache registration.
Their save/load loops can then remain close to the original implementation.

### Minimize the PD protocol

The current PD implementation is deliberately generic, but that generality is
unnecessary for the present BF16-only scope. Most of the PD reduction should
come from restoring the original `read_thread.py` and adapting only metadata
intake and descriptor construction.

For the reduced protocol:

- Keep fixed BF16 roles: `main_k`, `main_v`, and optional `indexer_k`.
- Do not carry `indexer_scale`; C8 with offload remains out of scope.
- Send the sorted real-indexer owner set once per P session.
- Send source group mapping once per owner or group, not once per tensor on
  every layer.
- Send manager-page scale once per group when all entries in that group share
  it.
- Continue carrying all P block IDs by group in `READ_READY_BATCH`.
- Let D derive destination tensors and destination group IDs from its local
  `SFASplitCacheLayout`.

P and D run the same new protocol and model, so exact owner-set validation is
sufficient to establish the optional-indexer layout. Repeating role strings,
group IDs, and scales for every layer is avoidable.

The existing read thread should retain its original:

- request batching;
- descriptor coalescing;
- MemFabric execution;
- completion handling; and
- normal logging.

Only these areas should change:

1. Accept and validate compact split-layout metadata.
2. Select P source block IDs by source group.
3. Append main descriptors and an optional indexer descriptor.
4. Select the D CPU or HBM destination by semantic cache kind.

This compatibility-adapter approach should remove most of the current
748-line `read_thread.py` churn.

### Separate orthogonal correctness fixes

Several valuable fixes are not intrinsic to split-cache allocation and should
be reviewable independently:

- removal of unsupported C8/offload compatibility code;
- finalized-token tracking before HBM block release;
- pending versus completed CPU block visibility;
- MTP-specific indexer metadata fallback; and
- MLAPO connector wait before a reused cache write.

Keeping these in focused commits makes regressions easier to bisect and avoids
making cache-layout review depend on asynchronous decode details.

### Reduce test churn without reducing coverage

Do not shrink the patch by deleting behavioral tests. Instead:

- leave existing SFAPD tests unchanged where behavior is unchanged;
- add split-layout tests in a focused new file;
- share a compact fixture for P/D cache groups and real-indexer owners;
- parameterize GLM-5.1 all-indexer and GLM-5.2 sparse-owner cases; and
- keep block-boundary, MTP, partial-page, and multi-request cases attached to
  the commit that introduces the relevant behavior.

The current 460-line rewrite of
`test_sfa_pd_cpu_offload_single_rank.py` should become a small compatibility
update plus focused new tests for the adapter.

### Review-size target

Aim for no more than roughly 500-600 changed lines per functional review step.
This is a review target rather than a strict code metric. If the complete
initial refactor must remain below roughly 1,000 changed lines, PD support must
be deferred to a stacked follow-up; splitting commits cannot make its protocol
translation disappear.

## Reference Branch State

The full implementation is committed as:

```text
477ef238 Support split-cache SFA PD disaggregation
```

It is pushed to `anga/offload_merge`, and the `vllm-ascend` submodule worktree
is clean at that commit. The workspace root may still report `M vllm-ascend`
when its recorded gitlink has not been updated; that does not indicate dirty
files inside the submodule.

## Remaining Scope and Risks

- Build the reduced stacked implementation while preserving `477ef238` as the
  reference branch.
- Add a repeatable end-to-end PD integration test; current full validation is
  remote/manual plus focused unit tests.
- Benchmark latency and throughput after correctness is locked down.
- Mixed old/new P and D protocol versions are intentionally unsupported.
- Sparse C8 with offload is not implemented or verified. Non-offload C8 remains
  intact.
- DCP and PCP greater than one remain out of scope for the offload layouts.
- The runtime cache-composition adapter remains coupled to the current SFA
  kernel tuple ABI.
- Current PD verification used GLM-5.2. The ownership logic generalizes to
  GLM-5.1, but the reduced PD implementation should still receive a GLM-5.1
  remote regression before upstreaming.
