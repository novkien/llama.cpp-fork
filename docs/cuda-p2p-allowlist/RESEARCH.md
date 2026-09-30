# Research and source map

Reviewed for this documentation pack on 2026-09-30. API documentation establishes contracts; fork source establishes call sites; recorded host experiments establish only their tested conditions. None proves an unimplemented feature works on production.

## Fork source baseline

Base commit: `526c43b8f7dfea9032e9f35e7a1be9183ca7cc20` on the accessible fork's `master`. The existing implementation PR #2 concerns RAM prompt-cache retention, not selective P2P. Do not build this work on that unrelated feature branch or include its changes accidentally.

| Reference | Verified relevance |
|---|---|
| [S1: CUDA backend](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/ggml/src/ggml-cuda/ggml-cuda.cu) | Peer enable loop, VMM descriptors, ordinary allocation, both copy callbacks, communicator setup. Blob `b50d05d5f76fb7010d921b8e71bd4ff83af5b23c`. |
| [S2: Generic backend/scheduler](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/ggml/src/ggml-backend.cpp) | Async/sync fallback, direct scheduler callback, user-input immediate copy and reuse events. Blob `20bf965017e31338f579c1d8d1a64b0ab56c78ee`. |
| [S3: Model placement](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/src/llama-model.cpp) | `get_layer_buft_list`, cumulative split lookup, repeating and output layer assignment. Blob `ab5e744b5d1076c2ad9c4700051ae3bdb9edb024`. |
| [S4: Backend public API](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/ggml/include/ggml-backend.h) | Real scheduler assignment/allocation/compute/synchronization APIs; reset invalidates allocated tensors. |
| [S5: CUDA context](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/ggml/src/ggml-cuda/common.cuh) | Internal device/context declarations and VMM compile guards. |
| [S6: Existing tests](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/tests/CMakeLists.txt) | Existing backend test target and registration patterns to reuse. |

Read the applicable repository instructions. This pack adds no competing AGENTS policy and exports no ChatGPT project canonical into runtime files. It is limited to this feature's design and evidence.

## Primary API knowledge

| Source | Contract used | What not to infer |
|---|---|---|
| [N1: Runtime peer access](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html) | Enablement is directional; capability query and expected already-enabled error are documented. | Capability does not establish correctness, bandwidth, or NVLink topology. |
| [N2: CUDA multi-GPU guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html) | Cross-device event waits are supported; event recording/timing has device association constraints; PCIe/IOMMU caveats matter. | A software trace of wait calls does not establish real GPU completion order. |
| [N3: Volta NVLink section](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html#nvlink-interconnect) | Transfers between NVLink-connected endpoints use NVLink transparently within CUDA. | An allowlist is not a cable selector or continuous link-health monitor. |
| [N4: VMM driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html) and [VMM guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html) | Allocation location, mapping, and per-device access are separate operations. | Filtering runtime enable calls alone does not restrict all VMM allocations. |
| [N5: Synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html) | Pageable host copies may synchronize despite the Async suffix. | Explicit host fallback necessarily has the same performance as implicit runtime staging. |
| [N6: NCCL environment](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-level) | `NVL` is a NCCL P2P topology cutoff; direct-user-buffer access has a separate control. | NCCL settings apply to arbitrary llama.cpp CUDA copies or to every private endpoint allowlist. |
| [N7: Stream-ordered allocator](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html) | Async memory pools have their own access controls. | A future allocator can be substituted without revisiting policy coverage. |

The retrieved documentation versions need not match the installed V100 toolchain. Verify applicability during native testing; do not upgrade the host just to match a web document.

## Upstream implementation context

[Merged upstream PR #21891](https://github.com/ggml-org/llama.cpp/pull/21891) moved NCCL communicator management into a separately initialized context. This supports checking actual communicator use rather than treating a NCCL build as proof of all-device communicator creation.

[NVIDIA NCCL P2P source](https://github.com/NVIDIA/nccl/blob/master/src/transport/p2p.cc), inspected during the issue research, contains same-process peer-enable and VMM/IPC mapping paths. It motivates a guard around unreviewed collective use. The moving link is a reference, not a claim about the host's installed NCCL revision.

## Recorded host evidence, not repeated here

[Issue #3](https://github.com/novkien/llama.cpp-fork/issues/3) and [the direct copy result in proxy #470](https://github.com/novkien/llama-proxy/issues/470#issuecomment-5904023781) report zero mismatches for CUDA0->1 and CUDA2->3 with peer access, but severe mismatches for CUDA1->2 with peer access. Access was enabled symmetrically; the published data-transfer measurements cover those three directions only.

[Rollback evidence](https://github.com/novkien/llama-proxy/issues/470#issuecomment-5903851666) records recovery to about 45 tokens/s with the old flag removed. The exact IOMMU/ACS/switch/driver subcause is not isolated. No new host benchmark is claimed in this docs PR.

## Integration reference

[Proxy PR #472](https://github.com/novkien/llama-proxy/pull/472) demonstrates matching proxy/bridge validation for the old P2P flag. The previously read launcher and bridge lacked the new pair key. Recheck their live source before a companion change; adding docs here does not change their allowlists or deployed state.

## Design conclusions versus facts

The fail-closed parser, pair matrix, v1 collective guard, initial explicit host fallback, and decision to defer optimized staging are proposed engineering choices. Primary documentation makes them implementable; it does not mandate this exact UI or prove its speed. The local probe establishes reachability of the existing fallback under test-supplied guards. Native transport, actual VMM pool permissions, graph timing, and Qwen performance remain to be tested.
