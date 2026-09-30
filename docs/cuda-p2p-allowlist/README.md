# Selective CUDA peer transport

Status: the opt-in CUDA backend is implemented and has passed the bounded native four-V100 tests in [NATIVE-RESULTS.md](NATIVE-RESULTS.md). Managed model acceptance and deployment evidence are tracked in [issue #3](https://github.com/novkien/llama.cpp-fork/issues/3).

## Behavior

`GGML_CUDA_P2P_PAIRS=0-1,2-3` enables both directions of the two configured physical CUDA pairs. Endpoints use the runtime order after `CUDA_VISIBLE_DEVICES`; startup logs show each full UUID and PCI identity. Every excluded distinct-device edge declines both CUDA copy callbacks and reaches the existing explicit host get/set fallback. Local copies and virtual aliases keep D2D behavior. The VMM pool grants its owner and only permitted physical readers, including in an NCCL build.

For the verified four-V100 UUID order `5ca3,b2e,e149,d4d`, the forward and reverse layer handoffs are `PEER, HOST, PEER`. Resolve full current UUIDs and topology before configuring another process. Capability alone does not establish a correct peer link.

The old global `GGML_CUDA_P2P` key conflicts with the pair selector even when its value is `0` or empty. Empty/malformed lists, self-pairs, out-of-range/unsupported directions, managed-memory mode and peer-copy-disabled builds fail initialization. Selective collective initialization is unsupported in v1 and fails before NCCL/internal/meta transport setup. HIP/MUSA attempted opt-in is unsupported. With the new key absent, legacy behavior is retained. Configuration changes require a fresh process.

## Evidence and maintenance

| Document | Purpose |
|---|---|
| [DESIGN.md](DESIGN.md) | Physical identity, grammar, VMM/copy invariants and supported scope. |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Reviewed source paths, synchronization boundaries and managed-launch dependency. |
| [RESEARCH.md](RESEARCH.md) | Primary-source context and limits of the transport claims. |
| [PREFLIGHT.md](PREFLIGHT.md) | Experiments A/B/C and their different evidence levels. |
| [PREFLIGHT-RESULTS.md](PREFLIGHT-RESULTS.md) | Historical host-only experiment and negative controls. |
| [NATIVE-RESULTS.md](NATIVE-RESULTS.md) | Actual hardware, whole-scheduler, callback, pool and bandwidth results. |
| [TESTING.md](TESTING.md) | Regression matrix and separate four-GPU model acceptance. |

The initial source-branch experiment established only dispatch feasibility. Native testing subsequently exercised all 12 directed edges, offset views and physical aliases, the actual VMM pool, and a whole four-stage scheduler graph with the candidate's unmodified callbacks. The measured explicit pageable-host path is slower than the selected NVLink paths; copy bandwidth does not forecast Qwen token throughput.

The first implementation uses the existing blocking fallback. It adds no staging pool, model/layer-placement change, RPC transport change, precision conversion or NVML runtime dependency. A companion launcher change permits the new key at the proxy and bridge; effective child environment, binary identity and policy logs remain part of managed acceptance.

The current owner request also delivers PR #2's RAM-cache repair. Optional SSD spill is not implemented. Neither a merge nor successful model loading substitutes for the applicable runtime checks in the issues. Preserve concurrent changes and the exact reviewed candidate during task-scoped publication and deployment.
