# Design: physical-pair allowlist with explicit host fallback

This is the proposed implementation contract for [issue #3](https://github.com/novkien/llama.cpp-fork/issues/3), not a description of an existing option. Source and primary-document references are indexed in [RESEARCH.md](RESEARCH.md).

## 1. Problem and non-goals

Recorded tests distinguish working NVLink transfers from a corrupt PCIe peer transfer on the same four-V100 system. The aim is to avoid the latter while retaining all four GPUs. A two-GPU workload is not acceptance. This feature does not repair IOMMU/ACS/switch firmware, rearrange layers for an alleged speedup, implement tensor parallelism, or monitor physical links continuously.

The pair selector operates in the CUDA backend. It does not belong in the HTTP request path, model token loop, or a proxy-side topology poll. No NVML or shell command is needed per copy.

## 2. What "return to the normal path" means

There are two different paths in the reviewed source:

1. Legacy P2P environment absent: CUDA copy callbacks still call `cudaMemcpyPeerAsync` when compiled in. Explicit peer enablement is absent, but the runtime selects the transport. The recorded peer-off experiment was correct; that does not prove every allocation type follows an identical route.
2. Proposed excluded pair: both CUDA callbacks decline before enqueueing work. The existing generic backend uses host storage and source get/destination set. CUDA buffer get/set use D2H/H2D and synchronize their streams.

The proposal deliberately chooses path 2 for excluded pairs. It preserves a correct fallback algorithm, not necessarily the exact latency, buffering, or overlap of path 1. Do not advertise identical performance or infer that every `cudaMemcpyPeerAsync` call itself proves physical P2P was active.

Layer split determines where weights and compute reside. The selector controls transfers between the resulting backend buffers; it must not change layer placement. A transfer can also connect nonadjacent GPUs, especially with draft/target graphs or outputs. Apply policy to actual endpoints, not layer numbers or adjacency.

## 3. Configuration and identity

Proposed environment variable:

```text
GGML_CUDA_P2P_PAIRS=0-1,2-3
```

Endpoints are physical CUDA runtime ordinals after `CUDA_VISIBLE_DEVICES`, with range `[0, physical_device_count)`. They are not host `nvidia-smi` indices, indices into a shortened `--device` list, virtual ggml device IDs, or RPC IDs. Use the existing logical-to-physical mapping for backend copies. Startup diagnostics must show runtime ordinal, full UUID, and PCI identity so an operator can check the interpretation.

The recorded UUID-prefix order is `5ca3,b2e,e149,d4d`; recover full UUIDs from current evidence rather than reconstructing them. If visibility changes, the same pair text can select different hardware. CUDA/NVML ordinals must not be equated.

Every pair is symmetric configuration expanded into two directional permissions. It is not transitive: `0-1,1-2` does not permit `0-2`. Reversed/repeated pairs normalize to one pair. Same-physical transfers always use local D2D, including different virtual backends sharing one GPU.

For this deployment:

| Physical source/destination | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | LOCAL | PEER | HOST | HOST |
| 1 | PEER | LOCAL | HOST | HOST |
| 2 | HOST | HOST | LOCAL | PEER |
| 3 | HOST | HOST | PEER | LOCAL |

There are four allowed and eight denied ordered distinct-device edges.

### Grammar

Parse a nonempty comma-separated list of decimal `a-b` pairs, allowing ASCII whitespace around tokens/separators. Accept decimal leading zeros and normalize them. Reject signs, overflow, self-pairs, empty entries, incomplete pairs, trailing garbage, and out-of-range endpoints. Validate everything before the first enable call. This resolves the earlier plan's optional wording about leading zeros into one testable proposed behavior.

An explicitly requested unsupported direction is an initialization error, not permission to enable a larger set. Check capability in both directions. Capability is necessary but not proof of a correct link.

### Compatibility decisions

| Setting | Proposed result |
|---|---|
| New key absent | Keep legacy behavior unchanged. |
| New key present and valid | Activate immutable restrictive policy. |
| Old and new P2P keys both present | Error, including old value `0` or empty. |
| New key empty | Error, not enable-all or legacy mode. |
| `GGML_CUDA_NO_PEER_COPY` build with new key | Error for unsupported opt-in. |
| Managed-memory opt-in with new key | Error in v1; not covered by the transport contract. |
| `GGML_CUDA_NO_PINNED` present | Preserve it; use the ordinary fallback, without an application pinned pool. |
| NVIDIA CUDA, layer split | Intended supported feature scope. |
| CUDA collective initialization with new key | Reject before NCCL/internal/meta transport setup in v1. |
| HIP/MUSA | Diagnose unsupported attempted opt-in; no behavior change when absent. |
| RPC | No remote transport/enumeration changes. |

The old key is presence-based: `GGML_CUDA_P2P=0` still activates its legacy branch. Do not silently reinterpret it in this feature.

## 4. One authoritative policy

Keep capability, requested permission, and successfully enabled state distinct. Publish the policy only after selected directions initialize successfully. Read it without mutation from allocation and copy paths. A fixed-size matrix and per-owner access lists are sufficient; avoid a generalized routing framework.

During `ggml_cuda_init`, build from the local `info` under construction. The existing physical-device helper calls `ggml_cuda_info`; calling it recursively during that same function-local static initialization is unsafe. After initialization, use the normal helper.

No per-request enable/disable, device reset, or hot environment reload. A new configuration requires a fresh child process. This is an application transport policy, not a security isolation boundary against arbitrary third-party CUDA code in the same process.

## 5. Memory access invariant

For a VMM mapping owned by physical GPU `owner`, grant READWRITE to the owner and only readers for which `policy(reader, owner)` is enabled. Always include the owner. Deduplicate physical IDs. This direction describes who may access the allocation, not simply the direction in which a tensor is being copied.

The selective branch takes precedence over the current `GGML_USE_NCCL` compile-time widening. Retain the original branch when selective mode is absent. Apply permissions to each new span before it is returned. Verify growth/remap behavior and query flags with `cuMemGetAccess` in native tests.

Omission from a new access list must not be confused with revoking previously granted access. Immutable pre-allocation policy avoids that ambiguity. GPU VMM allocations using `CU_MEM_ALLOCATION_TYPE_PINNED` with a DEVICE location are not host staging buffers.

Ordinary ggml tensor buffers and temporary VMM pools are different allocation families. Both need native tests. Managed memory, CUDA async memory pools, external mappings, and IPC need separate access review if introduced later.

## 6. Copy invariant and synchronization

For a denied distinct-device edge, both callbacks return false before any copy or completion event is submitted. The generic fallback then completes producer work, protects destination reuse according to its caller, reads into host memory, and writes to the destination. A return value of false must never mean "some work was already enqueued; try again elsewhere."

For allowed edges, retain the existing peer-copy source stream and event record/destination wait. For local edges, retain D2D. Do not replace a forbidden peer call with a cross-device `cudaMemcpyDefault`, D2D call, remote-pointer kernel, or external collective that bypasses the policy.

Do not globally disable events or CUDA graphs to implement selection. Scheduler input copies and the ordinary async split-input copies are separate entry paths. Their synchronization contracts are not interchangeable.

## 7. NCCL scope

The source initializes communicators through a separate communication context; a NCCL build alone does not prove a communicator runs in layer mode. Keep the existing build usable. However, actual NCCL transport can independently enable/map peers. In v1, a restrictive-mode collective attempt must fail explicitly before transport initialization. Returning null and silently running an unreviewed meta fallback is not that rejection.

`NCCL_P2P_LEVEL=NVL` only constrains NCCL topology selection; it does not implement this arbitrary pair matrix or control llama.cpp's copy callbacks. Internal AllReduce is not a four-GPU substitute in the reviewed implementation.

## 8. Performance and completion

Start with existing blocking host fallback. Only profile-driven work should add a bounded reusable staging pool, and it must respect `NO_PINNED`. Buffer reuse after D2H but before H2D completion is forbidden; cancellation does not cancel already queued CUDA operations. Keep bytes unchanged and avoid new precision conversions.

Correctness, selected transport, and workload speed are separate outcomes. Existing NVLink bandwidth does not forecast Qwen tokens/s. Native four-GPU evidence is required to promote the feature; the local software-double experiment only establishes branch feasibility. See [PREFLIGHT.md](PREFLIGHT.md) and [TESTING.md](TESTING.md).
