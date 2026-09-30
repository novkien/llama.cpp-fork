# Candidate regression and acceptance plan

These are requirements for future implementation. The docs-only PR satisfies documentation delivery, not the unchecked runtime requirements below. The pre-code test and actual local results are in [PREFLIGHT.md](PREFLIGHT.md) and [PREFLIGHT-RESULTS.md](PREFLIGHT-RESULTS.md).

## 1. Evidence must identify the candidate

Record source commit plus any local diff, compiled binary identity, architecture/build flags, CUDA driver/runtime/compiler, NCCL library version when relevant, full GPU UUID/PCI mapping, model identity, and exact launch configuration. Freshly resolve the intended remote/base branch; a previous SHA is not proof of current deployment.

Separate same-binary option-off/option-on tests from old-production-binary/new-candidate comparisons. Otherwise a source-version difference is confounded with P2P selection. Build/test results follow actual candidate bytes; rerun affected checks after relevant edits.

## 2. Focused test matrix

| Area | Required cases | Failure detector |
|---|---|---|
| Parser | Empty, signs, overflow, garbage, missing endpoint, duplicate/reversed pairs, decimal leading zeros, whitespace, count changes. | Complete parse or explicit error; zero enables before validation completes. |
| Policy | All 12 distinct directed edges, diagonal, physical aliases, non-transitivity, all denied cross-group edges. | Independent expected matrix, not production classifier reused as its own oracle. |
| Initialization | Both directions supported, one direction unsupported, expected already-enabled, unexpected failure. | No partial-serving state; exact failing pair/API; no forbidden enable. |
| VMM | Owner, selected readers, excluded readers, growth/remap, VMM off. | Native `cuMemGetAccess` and allocation/operation traces, plus correct consumers. |
| Copy dispatch | Both CUDA callbacks, generic sync/async, scheduler direct async, scheduler input sync. | Peer/host call counts and bytes by actual physical pair; no bypass. |
| Data | Odd/zero sizes, alignment/granularity boundaries, valid views, guards, different patterns and iterations. | Byte-for-byte raw copies; exact or justified numeric oracle for graph ops. |
| Lifecycle | Producer work, consumer waits, reuse slots, reverse traffic, graph replay, cancel, teardown. | Native completion correctness, no use-after-free/hang or unbounded memory. |
| Scope | NCCL build on/off, actual collective attempted, managed mode conflict, peer-copy-disabled build, HIP/MUSA compile guards. | Unsupported selective combinations rejected before unsafe work; legacy unaffected. |
| Integration | Route and bridge validation, launch hash/environment, binary recognizes option. | Exact readback plus effective child state and native policy evidence. |

For the current `0-1,2-3` matrix, inspect eight denied directions, not only 1->2. Independently count expected edges. Keep same-physical aliases local even across different backend instances.

For raw-copy tests, initialize destination sentinels each iteration and validate guards. Bytes alone are not sufficient to prove selected transport: the local negative controls deliberately demonstrate this limitation.

## 3. Real layer-like graph

Run experiment C from PREFLIGHT against real backends with no policy decorators. Force and verify four compute stages across four physical GPUs. Include independent expected numeric results and tagged per-boundary transfers. Check forward/reverse and nonadjacent edges.

Actual VMM scratch pools must be exercised separately with operations that allocate temporary storage; a four-stage add/scale graph may not use the CUDA VMM pool at all. Record observed pool allocations and permission queries rather than assuming coverage from a successful graph. Avoid adding a broad graph-optimizer rewrite solely for this test.

Exercise scheduler events and copy-slot reuse without replacing them with one global device synchronization. Test input flags separately because they select a different entry path. A host-only excerpt probe cannot establish asynchronous timing, capture compatibility, or race freedom.

## 4. Optional staging optimization

Only add a reusable pinned staging path if profiling justifies it. Respect `GGML_CUDA_NO_PINNED`; keep any pinning experiment separately named. Verify pool cap, allocation failure before submission, capacity pressure, large-copy chunking, slot/event generation reuse, opposite-direction dependencies, cancellation, and teardown.

The source data must outlive the D2H read; the staging slot must outlive H2D completion; the destination must not be consumed before completion. A source-side completion observer must retain the documented copy lifetime semantics. Avoid cycles when adding cross-stream waits. Return false only when nothing was submitted.

No FP16/BF16 compression, global graph disable, or global synchronization may conceal a correctness defect. Test data transfer byte-for-byte.

## 5. Four-GPU Qwen acceptance

Preserve the exact target workload from issue #3, after resolving the full current launch configuration:

```text
--device CUDA0,CUDA1,CUDA2,CUDA3
-sm layer
-ts 28,28,29,15
--spec-type draft-mtp
--spec-draft-device CUDA3
--spec-draft-n-max 3
--spec-draft-n-min 1
-c 400000
--parallel 4
```

Keep the actual model, UUID order, RPC configuration, host settings, other environment values, and prompt-cache/warmup policy constant. Do not replace acceptance with two GPUs, a smaller context/model, MTP off, or a different split. Diagnostic target-only/small-model runs are useful but separately labeled.

Use fresh children for baseline -> selective -> rollback. Do not add the already-broken global-P2P condition. First reproduce the bounded deterministic 256-token request; then run several repeats and the configured four-slot workload. Recover the original prompt from the recorded evidence; do not invent a prompt and compare its acceptance counts to the historical run as though they were identical.

Record individual outputs/token IDs when available, finish reasons, draft generated/accepted counts, target verification work, native decode and prompt throughput, TTFT, client elapsed, copy counts/bytes/time, host-memory usage, clocks/temperature/competing load, errors, cancellation, and cleanup. Report all repeats and their median/range rather than one selected run.

The historical healthy reference is approximately 44-45 decode tokens/s and 159/285 accepted draft tokens for the recorded request. These are context-specific observations, not universal required counts. Use a fresh healthy same-binary baseline. Establish baseline variability before interpreting a difference. Raw-copy mismatches always fail; large unexplained output/draft divergence is not dismissed as normal nondeterminism.

A transfer bandwidth gain need not improve token generation if those copies occupy little runtime or the explicit host fallback adds more cost. Report correctness, selected transport, and performance separately. Native correct routing with unresolved performance gain remains PARTIAL for production promotion.

## 6. Publication and rollback

A documentation PR or implementation merge does not close the issue. No production action is included in the current draft. When native execution is authorized, use existing managed drain/unload/reload controls rather than killing a busy bridge child. Preserve other routes and rollback to the verified healthy binary/environment. Remove new/old P2P keys as required; setting the old key to `0` is not disabling it.

The companion proxy/bridge change needs matching validation and launch-hash tests. Verify the effective child, not only route JSON. An old binary ignoring the new key cannot pass activation.

## 7. Completion checklist for the eventual implementation

- [ ] Exact delivered candidate and runtime identities recorded.
- [ ] Parser, mapping, conflict, and fail-closed initialization tests pass.
- [ ] Both copy callbacks and VMM allocator obey one immutable pair policy.
- [ ] No forbidden enable, mapping grant, or peer-copy dispatch on any of the eight denied directions.
- [ ] All 12 directed native copy cases pass, including reverse NVLink transfers and relevant buffer families.
- [ ] Whole-scheduler four-GPU layer-like graph passes with native backend assignment and independent output oracle.
- [ ] Actual VMM pool coverage, producer/consumer ordering, reuse, cancellation, and teardown are evidenced.
- [ ] NCCL build remains usable for layer mode; unsupported selective collectives fail before initialization.
- [ ] Legacy behavior without the new key and relevant build/platform guards are checked.
- [ ] Required managed-launcher dependency and documentation match the implementation.
- [ ] Unchanged four-GPU Qwen/MTP baseline/selective/rollback and four-slot results recorded.
- [ ] No correctness/performance gap is hidden by workload downgrade or mock-only PASS.
- [ ] Owner-reviewed acceptance outcome is explicit; issue remains open while required evidence is absent.

Use PASS for an actually satisfied tested scope, FAILED for an observed violation, and PARTIAL for incomplete/undecidable coverage. Do not replace unrelated tool/process status fields with this verdict vocabulary.
