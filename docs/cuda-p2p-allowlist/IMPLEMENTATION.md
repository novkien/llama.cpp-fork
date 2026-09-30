# Implementation map and engineering notes

No code below is an implemented patch. This is the work map for the future coder, based on `526c43b8f7dfea9032e9f35e7a1be9183ca7cc20`. Read the current source before using these symbol names. The feature contract is in [DESIGN.md](DESIGN.md); sources are in [RESEARCH.md](RESEARCH.md).

## 1. Trace the whole layer handoff

```text
llama model tensor placement
  -> per-layer backend/buffer selection
  -> graph scheduler split and input-copy allocation
  -> ggml_backend_sched_compute_splits
       user input: synchronous tensor_copy directly
       ordinary split input: backend cpy_tensor_async directly
            success: compute consumer after backend's ordering
            false: synchronize producer + destination reuse
                   -> ggml_backend_tensor_copy
                        -> destination buffer cpy_tensor
                             success: stop
                             false: host allocation -> get -> set -> free
```

In `src/llama-model.cpp`, `get_layer_buft_list` selects a device with cumulative split weights and `std::upper_bound`; the repeating-layer loop and output-layer selection use it. Inspect actual selected backends rather than assuming the split percentages equal exact layer counts. Input/output placement, tensor overrides, CPU fallback, and the MTP graph may add transfers outside the three adjacent boundaries.

In `ggml/src/ggml-backend.cpp`, the scheduler calls the backend interface directly. It does not always pass through `ggml_backend_tensor_copy_async`. It also has an immediate-copy path for `GGML_TENSOR_FLAG_INPUT`. Existing destination-use events and producer completion must remain valid.

## 2. Exact change targets

All CUDA symbols in this table are in `ggml/src/ggml-cuda/ggml-cuda.cu` unless stated otherwise.

| Target | Existing behavior | Required future work | Primary failure risk |
|---|---|---|---|
| `ggml_cuda_init` | Presence of old flag enables all capable ordered physical pairs. | Parse/validate once; enable exactly the selected directions; publish immutable state. | Partial policy, accidentally enabled PCIe edge, recursive static initialization. |
| `ggml_cuda_get_physical_device` / `ggml_cuda_info` | Map logical backend IDs to runtime physical IDs. | Reuse after initialization; use local `info` during initialization. | Mixing host, runtime, virtual, and route-list indices. |
| `ggml_cuda_pool_vmm::alloc` | Old flag or NCCL build broadens peer descriptors. | Selective precedence; owner plus allowed readers for every new mapping. | NCCL macro silently widens policy; owner omitted; direction reversed. |
| `ggml_backend_cuda_cpy_tensor_async` | CUDA-to-CUDA copy with source event and destination wait. | Classify after validity/ownership checks; decline denied edge before work. | Copy enqueued before false; invalid lifetime or event ordering. |
| `ggml_backend_cuda_buffer_cpy_tensor` | Separate synchronous callback also calls peer copy. | Apply the same physical-pair exclusion. | Async fallback re-enters peer copy through this callback. |
| `ggml_backend_cuda_comm_init` and dispatch | Select NCCL/internal/meta reductions. | Explicitly reject unsupported restrictive collective use before setup. | Null/fallback is mistaken for rejection; external mapping bypass. |
| `ggml/src/ggml-cuda/common.cuh` | Internal device/context declarations. | Add only necessary internal policy representation/declarations. | Public ABI growth or unnecessary routing subsystem. |
| `ggml/src/ggml-backend.cpp` | Generic/scheduler fallback algorithms. | Review and test; initially no algorithm change. | Broad refactor obscures the focused CUDA change. |
| `docs/build.md` | Maintained build/config instructions. | Document the delivered option when it exists. | Docs advertise an unsupported binary. |
| Existing backend tests and test registration | Backend operation validation infrastructure. | Add focused policy/copy integration coverage using existing conventions. | Mock-only tests presented as physical GPU validation. |

Locate the relevant code without relying on stale line numbers:

```sh
git grep -n -E 'GGML_CUDA_P2P|cudaDeviceEnablePeerAccess|cuMemSetAccess|cudaMemcpyPeer|cudaMemcpyDefault|ncclCommInitAll' -- ggml/src
git grep -n -E 'get_layer_buft_list|LLAMA_SPLIT_MODE_TENSOR' -- src/llama-model.cpp
git grep -n -E 'ggml_backend_tensor_copy|ggml_backend_sched_compute_splits|GGML_TENSOR_FLAG_INPUT' -- ggml/src/ggml-backend.cpp
```

These are read-only inventory commands, not a guarantee that the search terms exhaust future CUDA APIs.

## 3. Initialization recipe

First resolve the physical runtime count and logical mapping. Parse the entire option and conflicting environment/build settings into temporary state. Reject invalid syntax/counts before CUDA peer side effects. Deduplicate undirected pairs, expand directions, and validate both capabilities.

Then select the correct CUDA current device and enable each allowed edge exactly once. Handle only documented `cudaErrorPeerAccessAlreadyEnabled` as an expected idempotent case. Unexpected errors must preserve the exact pair/API/error in the diagnostic and stop this opt-in initialization. Restore the previously current device as appropriate. Do not suppress every CUDA error or reset devices.

Do not expose a "configured" matrix as an "enabled" matrix before all requested directions succeed. Do not continue serving with half the requested policy. Policy is fixed for the process; no allocations or graph capture should precede its establishment.

A plain pair list is not proof of NVLink. Log identities and selected edges; topology preflight must verify the known-good physical pairs. The first implementation does not need auto-discovery or NVML as a new mandatory runtime dependency.

## 4. Allocation recipe

Prepare per-owner access lists from initialized policy. The local owner always has READWRITE. A remote reader gets READWRITE only for its allowed reader-to-owner direction. Keep the original allocation, reservation, mapping, and handle-release lifecycle.

Implement selective-mode precedence ahead of the legacy NCCL/P2P widening. Preserve the old code path exactly when the option is absent. Test growing a pool and remapping; an allocation-level assertion is stronger evidence than merely seeing the new environment variable in logs.

Ordinary tensor storage uses the device allocation helper while VMM temporary storage comes from the pool. A raw `cudaMalloc` probe is not coverage of the pool. Do not add a managed/async-pool allocation as an incidental optimization.

## 5. Copy recipe

Maintain existing backend/buffer ownership checks before interpreting CUDA contexts. Resolve valid views through backing storage. Do not widen accepted layouts or treat every cross-backend copy as cross-physical.

In the async callback: same physical means existing local D2D; selected distinct physical means existing peer copy and event handoff; denied distinct physical means false before enqueue. In the synchronous buffer callback: apply the identical denial before its peer API. That allows the existing generic fallback to run.

The public async wrapper synchronizes both backends when a copy is declined. The scheduler instead synchronizes the producer and the destination copy-slot reuse event when available. Preserve each caller's existing contract rather than adding an unconditional global device synchronization.

Returning false after allocating/enqueueing half of a staged copy risks duplicate transfers and buffer lifetime errors. The initial implementation avoids that by adding no async staging resources. In a later staging optimization, a partially submitted failure is an error to handle/drain, not a request to retry through another callback.

## 6. Sensitive implementation details

| Detail | Safe reasoning rule |
|---|---|
| Virtual aliases | Classify by backing physical GPU; preserve local D2D between different backend instances. |
| Event lifetime | Preserve source event record and destination wait; software call order does not prove CUDA completion timing. |
| Destination reuse | Honor scheduler copy-slot events, including pipeline copies and graph reuse. |
| Source lifetime | Do not release/reuse source data while a relevant copy still reads it. |
| Cancellation | Queued CUDA work persists; retain/drain resources rather than freeing staging immediately. |
| Views/layout | Test valid views with populated backing buffers, offsets, and guards; do not claim arbitrary malformed views are supported. |
| VMM permissions | Reader/owner direction differs from a casual source/destination naming convention. |
| Old flag | Presence, not a parsed boolean; conflicting old/new keys must be rejected. |
| Graph capture | No blocking fallback inside an already invalidated capture; inspect real transfer/capture boundaries. |
| Diagnostics | Counters/traces should be opt-in and bounded; no per-token logging overhead in production. |
| External libraries | Never assume another library respects this private policy. Guard unreviewed communicator use. |
| `NO_PINNED` | No hidden application pinned pool; driver-internal staging is a separate CUDA behavior. |

## 7. Work sequence

1. Inventory current code and instruction scope; preserve concurrent edits. Read the issue and results, including negative controls.
2. Implement internal policy and parser with focused pure tests, including all incompatibilities. Do not touch layer placement or request handling.
3. Gate runtime enablement and VMM access with one policy.
4. Gate both copy callbacks, retaining legacy/local/allowed paths. Exercise the generic and direct-scheduler entry points.
5. Add the collective rejection boundary; verify no transport setup occurs first.
6. Run native tests and profiling in [TESTING.md](TESTING.md). Add optimized staging only if its measured benefit is needed and its lifetime is proven.
7. Update maintained configuration docs and the narrowly scoped launcher dependency. Keep missing runtime evidence explicit.

## 8. Managed-route dependency

The reviewed proxy and bridge allow `GGML_CUDA_P2P` but not the proposed pair key. Before managed acceptance, a companion change must add the new key and matching bounded grammar/conflict checks to `src/llama_proxy/launcher_spec.py` and `bridge/server.py`, with tests in `tests/test_launcher_spec.py` and `tests/test_bridge_ports.py`. Recheck their current branch and instructions before that separate repository work.

Keep the child authoritative for its actual CUDA count/capability. Preserve full UUID visibility, split/MTP/RPC arguments, `NO_PINNED`, envelope hashing, and revision-checked configuration writes. Check the effective inherited child environment for an old/new conflict. Route readback alone does not prove the binary recognizes the option; require its policy diagnostic and the intended binary identity.

This docs PR makes none of those changes. It records the dependency so a CLI-only demonstration cannot be mistaken for end-to-end delivery.
