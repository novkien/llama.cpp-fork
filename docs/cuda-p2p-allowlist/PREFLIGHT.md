# Pre-implementation experiment: falsify the layer-transfer design

## Question and evidence levels

Can the existing llama.cpp layer handoff retain the selected CUDA peer path and deliberately reach the normal generic host fallback for excluded physical pairs, without dropping GPUs or changing layer placement?

Three experiments answer different parts. Do not combine their verdicts into a false hardware PASS.

| Experiment | Can run before feature code? | What it establishes |
|---|---|---|
| A. Extracted source control flow with test doubles | Yes; executed locally. | Existing callback/fallback branches can implement the proposed dispatch. |
| B. Four-GPU CUDA/VMM transport microprobe | Yes, with a disposable native harness. | These physical pairs, explicit host fallback, and restricted VMM descriptors work on the actual host. |
| C. Whole ggml scheduler with actual CUDA backends | Yes using clearly identified test-only callback seams, or later using candidate code. | Real graph allocation, layer-like partitioning, producer/consumer ordering, and transfer paths work together. |

The eventual four-GPU Qwen/MTP acceptance remains separate. No unchanged binary implements an environment key it does not recognize. Test wrappers or standalone allocations must never be described as the production feature.

## A. Local branch experiment

[Results and limitations](PREFLIGHT-RESULTS.md) record the executed host-only experiment. It compiles the existing generic copy functions, both CUDA copy callback bodies, and two exact scheduler blocks with minimal types and mocked CUDA/memory/event helpers. External callback decorators supply the proposed deny decision without changing those source bodies.

Test six entry paths: generic async, generic sync, scheduler ordinary input with reuse event, scheduler ordinary input without reuse event, and scheduler user input with/without reuse event. Exercise all 12 ordered distinct-device pairs, valid populated-buffer views, local copies, and virtual aliases.

Negative controls intentionally apply only one decorator, or neither. Detect forbidden peer-call dispatch even if the software-copy output matches. This prevents a bytes-only checker from approving a routing error.

This does not execute full graph splitting, real CUDA event timing, VMM, initialization, NCCL, a model, or the proposed parser. Those are explicitly not established by A.

## B. Native four-GPU microprobe, without a feature patch

This is a test specification, not an existing executable or a command that was run in the ChatGPT container. Use a disposable process and small allocations; do not enable unsafe PCIe pairs even for a negative control.

### Preconditions and provenance

Record the exact source/build/toolkit/driver identities, full GPU UUIDs, PCI identities, and physical topology. Verify four GPUs and the two intended active NVLink pairs under the process's effective visibility. Preserve host settings. Do not change BIOS, IOMMU, ACS, driver, RPC, or a running production child.

Use the existing compatible V100 toolchain; documentation from a newer CUDA release is not an upgrade instruction. A missing GPU/library produces PARTIAL with the actual probe error, not PASS or a fabricated bandwidth result.

Capture selected runtime environment keys without secrets. The old global P2P variable must be absent. Do not rely on the unimplemented new environment key. The disposable harness explicitly enables the four selected directions through the CUDA runtime.

### Exact operation sequence

1. Call `cudaGetDeviceCount`, enumerate UUID/PCI identities, and validate the selected mapping. Check `cudaDeviceCanAccessPeer` in both directions of selected pairs only as capability evidence.
2. Enable only 0->1, 1->0, 2->3, 3->2 in their proper current-device contexts. Fail on unexpected errors. Never enable a denied direction, and never use a sample that loops over all capable pairs.
3. Allocate independent source/destination buffers on all four devices with `cudaMalloc`. Fill each source with a pattern encoding source, destination, offset, and iteration; prefill destination with another sentinel.
4. For a selected edge, invoke `cudaMemcpyPeerAsync` on the producer stream; record its completion event; make the destination stream wait; enqueue a destination-side consumer/checksum and verify output on the host.
5. For an excluded edge, explicitly wait for the producer and protect destination reuse, then D2H to ordinary host memory and H2D to destination. Synchronize completion before validation. Do not call any cross-device peer/default/D2D API on that edge.
6. Repeat the same matrix using separate native VMM buffers: DEVICE allocation properties, granularity-rounded `cuMemCreate`, reserve/map, owner plus permitted reader descriptors, then `cuMemSetAccess`. Query each physical device's flags with `cuMemGetAccess`. This tests the VMM concept, not the actual ggml pool.
7. Run the complete forward chain 0->1->2->3 and reverse chain 3->2->1->0 with a deterministic producer transformation and consumer validation at every boundary. Run nonadjacent edges too.
8. Synchronize owned work and release only the harness's resources. Exit the process. Do not reset devices to perform cleanup.

Use bounded sizes: zero-handling separately, 1, 15, 16, 17, 4096, 65537, 1048576, and 16777216 bytes. For VMM allocate at least the required granularity even for a small logical payload. Compare every logical byte plus guard ranges. Add repeated and alternating transfers; cap peak allocation and iteration counts.

### What counts as success

All tested bytes match, no invalid access/timeouts, and instrumentation shows no denied enable, grant, or peer-copy API. Both directions of each selected pair must be correct. Report one-directional throughput separately from bidirectional aggregate throughput, after warmup, without converting it to predicted tokens/s.

For cross-device end-to-end timing, use host monotonic timing around a completed operation or correctly scoped device timings. Do not subtract CUDA event timestamps from different devices. Preserve the route's `NO_PINNED` choice in the first comparison; a pinned-host variant is a separately named experiment.

If B passes, the conclusion is "native transport/VMM mechanism is feasible for this host," not "llama.cpp already implements it."

## C. Whole-scheduler native seam test

### Build the real graph path

Use actual ggml CUDA backends plus a CPU backend last, consistent with the scheduler contract. Construct a small four-stage graph using exactly representable FP32 values, for example each stage computes `y = 2*x + bias_i` with bounded integer inputs. Put each stage's operations and its bias on the intended GPU. Use `ggml_backend_sched_set_tensor_backend` before graph allocation.

Run `ggml_backend_sched_alloc_graph` and `ggml_backend_sched_graph_compute`; inspect `ggml_backend_sched_get_n_splits`, each node's chosen backend, and tagged transfer endpoints. Require that the four compute stages actually run on four distinct CUDA backends. A CPU fallback, a graph collapsed to one device, or only two devices is not a pass.

Trace actual inter-stage tensors. Expected adjacent activation handoffs are PEER/HOST/PEER. Do not infer this merely from their intended assignment or count unrelated CPU input/bias transfers as peer violations. Compare each stage and final output against an independently calculated CPU oracle.

Repeat reverse order, a nonadjacent cross-group edge, both input-copy paths, valid views, and repeated graph execution. Test the scheduler's pipeline-copy setting both ways; it is not the same setting as server request `--parallel 4`.

### Before feature implementation

A native test harness can decorate the existing backend `cpy_tensor_async` and buffer `cpy_tensor` interfaces, delegating selected/local edges to their original callbacks and declining excluded edges. It must cover buffers allocated by the scheduler, not only manually created tensors. Keep original interface entries and ownership intact. No process-global callback replacement or hot mutation during work is acceptable.

This uses internal APIs and requires a version-specific disposable adapter; none is claimed implemented here. It proves a seam, not a supported public customization API. The current VMM allocator's NCCL-driven widening remains a separate problem: a callback-only seam is NOT validation of the full VMM policy. Either instrument an explicitly identified test-only VMM seam in a disposable source copy, or mark actual-pool permission coverage PARTIAL. Do not quietly rebuild without NCCL/VMM and claim production-equivalent success.

Do not use `LD_PRELOAD` as the primary method: intercepted API visibility and linkage cannot by themselves prove every backend path was covered. Do not run a patched production server merely to test this seam.

### After feature implementation

Remove test-only policy decorators and run C directly against the exact candidate. Observe native enable/grant/copy operations and VMM pool behavior, not only debug statements that report intended policy. The experiment must still detect an async-only or buffer-only regression through safely mocked negative controls, not by enabling the known-bad PCIe hardware path.

## Decision rule before coding

A passing A with B/C unavailable means PARTIAL overall: the source path is suitable, but native integration is unconfirmed. A passing B plus actual-scheduler C is strong evidence to proceed with the bounded feature; it is still not final Qwen acceptance. Any mismatch or forbidden-path use invalidates the relevant candidate. Missing evidence is not an implementation defect and must not be converted into a hardware failure claim.

The current run stops at A because CUDA is absent. The next native experiment and its required evidence are completely specified above; there is no claim that a real GPU test ran in this environment.
