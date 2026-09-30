# Native selective CUDA transport results

## Candidate and environment

These tests ran on 2026-09-30 on `jarvis-llm`, with four NVIDIA Tesla V100 32 GB GPUs, driver 580.178.04 and CUDA 12.9. The candidate CUDA source SHA256 was `cb675a0ceacc4c2ba3eb5fdff32eb13e529892ab9f84c3002203492a61296c23`; its built `libggml-cuda.so` SHA256 was `b42d9a0bad2a017a95db540ba0448ba39f73a3e93f1a00e30a9beb060429a540`. Tests identify these bytes rather than an eventual merge SHA. The server build subsequently included PR #2's RAM-cache repair; the CUDA candidate did not change.

The Release build used `/usr/local/cuda/bin/nvcc`, `CMAKE_CUDA_ARCHITECTURES=70-real`, `GGML_CUDA=ON`, `GGML_CUDA_NCCL=ON`, `GGML_RPC=ON` and `LLAMA_BUILD_TESTS=ON`. The CUDA test configuration kept `GGML_CUDA_NO_PINNED=1` and used the following full `CUDA_VISIBLE_DEVICES` order:

| Runtime device | UUID | PCI bus | Selected peer |
|---|---|---|---|
| CUDA0 | GPU-5ca3dbce-7f3a-eb29-7a10-4e4575f9404a | 0000:08:00 | CUDA1 |
| CUDA1 | GPU-b2e2716e-9619-1588-5e82-42fa348e000e | 0000:0c:00 | CUDA0 |
| CUDA2 | GPU-e149008e-c306-ac2a-d54e-bfc12d31bd52 | 0000:0d:00 | CUDA3 |
| CUDA3 | GPU-d4d77b42-58b5-94ed-fdac-8032f7cbae86 | 0000:09:00 | CUDA2 |

The selected pairs have NV6 links. All four selected directions and all eight excluded directions were tested. Tests did not enable the known unsafe inter-pair PCIe peer links. Driver, BIOS, IOMMU and ACS settings were unchanged. `GGML_CUDA_P2P` was absent except in deliberate startup-conflict tests that aborted before enables.

## Observed results

| Test | Actual coverage | Result |
|---|---|---|
| Standalone native transport | 192 cudaMalloc and 192 standalone VMM cases; eight nonzero sizes, guards, two patterns, all directed edges and GPU consumers. | Zero mismatches. Selected peer enables only. |
| Actual backend pool | 192 malloc and 192 VMM cases through `ggml_backend_cuda_context::new_pool_for_device`; allocation, growth and reuse. | Zero mismatches; actual `cuMemGetAccess` queries matched the owner/selected-reader policy. |
| Candidate copy callbacks | Both direct callbacks and generic sync/async copies; all 12 directed edges; six sizes from 1 to 1,048,576 floats; plain tensors and offset views. | 576 cases; zero mismatches, unchanged guards and zero denied-callback destination writes. |
| Physical aliases | Eight logical devices over four physical devices; same-physical copies and permitted/excluded physical edges. | 2,688 cases; zero mismatches. Local aliases retained D2D behavior. |
| Whole scheduler | Candidate's unmodified callbacks; forward, reverse and two nonadjacent permutations; four native GPU compute stages; input synchronization and copy-pipeline settings 1/4; three graph replays. | Eight graphs, each with four verified splits; independent numeric oracle passed. |
| Startup matrix | Three candidate native probes, normalized/duplicate pairs, 15 malformed/range/self-pair cases, old-key values `0`/empty, managed-memory conflict and three collective modes. | All 25 fresh processes gave the expected success or diagnostic. Invalid startup input made no peer enables. |
| Already enabled | Exactly the four permitted directions enabled before backend initialization. | Initialization succeeded and handled `cudaErrorPeerAccessAlreadyEnabled`. |
| Failure injection | Observer supplied reverse capability failure and reverse enable API failure. | Capability failure aborted before any enable. Enable failure aborted initialization; no serving state was published. These were simulated API results. |
| Physical reader count | `GGML_CUDA_DEVICES=1`, actual VMM owner CUDA0. | CUDA1 was still granted its selected physical-reader permission. |
| Legacy/reordered process | New key absent; then full UUID order reversed while preserving both NVLink pairs. | Whole-scheduler oracle passed in both fresh processes. |
| Maintained backend operations | `test-backend-ops`, CUDA0, F32 ADD/SCALE. | 60/60 passed in the NCCL-enabled build. |
| Combined RAM-cache repair | `test-server-prompt-cache` and focused HTTP cache-retention/idle-cache tests with the official stories260K F32 model. | 33 native checks and 14 HTTP tests passed; 40 unrelated HTTP tests were deselected. This is small-model coverage, not live Nex hybrid acceptance. |

The API observer was a secondary check. It aborted forbidden enables, peer copies, VMM grants, cross-device default copies and collective initialization, while delegating actual CUDA calls. The scheduler trace reported four enables and 24 permitted peer copies; the alias trace reported four enables and 768 permitted peer copies. Both had zero forbidden operations. Primary evidence also includes direct callback return values, untouched destination sentinels on declines, real VMM permission queries, source-path review and independent native graph outputs. Observer traces alone would not prove correctness or absence of an unobserved path.

The pipeline runs tested each configured copy count and graph replay. They did not establish rotation through every scheduler copy slot, concurrent graph races or cancellation. Zero-length tensors were not run in the native callback matrix. HIP, MUSA, NCCL-off and peer-copy-disabled builds were not compiled in this campaign; their startup guards were reviewed. These limits remain explicit in the acceptance matrix.

## Transfer measurements

Measurements used 16 MiB logical payloads, 16 repetitions, host monotonic elapsed time through actual completion and byte verification. GB/s counts logical payload bytes once, not the sum of D2H and H2D traffic. They are transfer probes with no claim about end-to-end model speed.

| Buffer/path | Directed measurements, GB/s | Correctness |
|---|---|---|
| Standalone cudaMalloc, selected peers disabled | 0->1: 6.211; 1->0: 5.524; 2->3: 7.528; 3->2: 7.563 | Zero wrong bytes |
| Standalone cudaMalloc, selected peers enabled | 0->1: 125.952; 1->0: 126.901; 2->3: 125.893; 3->2: 126.743 | Zero wrong bytes |
| Actual backend malloc pool, selected peer path | 0->1: 125.986; 1->0: 126.923; 2->3: 132.032; 3->2: 127.326 | Zero wrong bytes |
| Actual backend VMM pool, selected peer path | 0->1: 131.613; 1->0: 126.865; 2->3: 125.829; 3->2: 126.993 | Zero wrong bytes |
| Explicit pageable-host fallback, eight excluded directions | 3.374-4.382 | Zero wrong bytes |

The selected NVLink transfer path improved in this probe. The excluded explicit host fallback can cost more than the legacy implicit PCIe copy. Qwen inference must determine whether their combined effect improves the requested workload.

## Reproduction and remaining acceptance

The disposable C/CUDA harnesses, API observer, Python drivers and raw result logs are retained with the owner's acceptance artifact and summarized in [PR #4](https://github.com/novkien/llama.cpp-fork/pull/4). They do not add a maintained backend API or ship a runtime observer. The drivers launch fresh children with the full UUID order above, the selective key `0-1,2-3`, and expected success/abort assertions. Compile against the exact candidate's headers/libraries; an older library that ignores the new key cannot establish these results.

Native transport is PASS for the tested scope. Production acceptance remains separate: matching published native/RPC builds, proxy/bridge key validation and effective child mapping, unchanged four-GPU Qwen baseline/selective/rollback and four-slot requests, and real Qwen/Nex RAM-cache continuation checks. The current DFlash route and historical MTP fixture require separate matched comparisons. Optional SSD spill is not implemented. Issues stay open until their applicable deployed checks pass.
