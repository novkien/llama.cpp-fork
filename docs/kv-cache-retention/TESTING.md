# Verification plan and acceptance evidence

Status: all implementation results are **NOT RUN** in the documentation draft. The table specifies required tests, not claimed passes. Use the maintained server pytest harness and disposable files/ports. See [IMPLEMENTATION.md](IMPLEMENTATION.md) for setup.

## 1. Evidence rules

Record exact source/head, binary identity, model digest, backend, context capacity, parallel count, KV types, draft settings, cache budgets, checkpoint settings and fixture identity. Compare before/after using the same fixture and hardware. Preserve stdout/stderr and request counters without publishing private production prompts.

For every completion check HTTP/result status, `prompt_n`, `cache_n`, their sum against actual tokenized input, returned slot/task identity and output correctness. Raw LCP is not actual cache reuse. Planned reuse must match execution or be explicitly downgraded on failure. Do not infer "no prefill" from low wall time or count response text as proof of native cache residency.

For a deterministic model/backend fixture, compare output tokens or logits with an independently cold-evaluated reference where feasible. A valid checkpoint-granularity rollback may process more than one token. The reference for a disk-restored divergent hybrid prompt is the same divergent prompt on a live un-evacuated state, not a universal one-token expectation.

Tests must prove the intended path was exercised through compact cache event tags/counters. A cache miss caused by disabled cache cannot pass a test meant to cover failed restore. Failure injection must target a specific operation, not randomly corrupt unrelated shared memory.

## 2. Deterministic RAM and slot cases

Use stable token arrays where possible. Give A and C a controlled shared prefix P and distinct suffixes. X has negligible overlap. Choose sizes that actually fit the selected fixture and budgets. Test 0/1/2-token boundaries separately from long-context fixtures.

| ID | Fixture / setup | Required observation |
|---|---|---|
| T01 | A -> C -> A-next; A/C share over 50% of A but diverge before A's suffix. | Baseline demonstrates skipped save; candidate captures A before destructive reuse and reuses A's valid prefix on return. |
| T02 | A -> X -> C -> A-next in one slot; A is RAM-only when C borrows P. | A snapshot still exists after C restore; A-next reuses beyond P when its state supports it. |
| T03 | A fixed to slot 0, another slot starts and evacuates 0, A-next fixed to 0. | Prove slot 0 is empty and A is in RAM; restore into 0 rather than cold prefill. |
| T04 | Repeat T03 with slot-prompt-similarity 0 and nonzero. | Both use the same restore policy; no division-by-zero or hidden LRU dependency. |
| T05 | Submit a second request to a confirmed busy explicit slot. | No save/clear/restore of its running state; request waits or cancels according to existing semantics. |
| T06 | Copy checkpoint handles, replace/clear the live checkpoint and evict an entry. | Old snapshot bytes remain unchanged until the last lease is released; no use-after-free. |
| T07 | Inject allocation failure in target allocation, draft allocation, token/media clone, checkpoint metadata and list insertion separately. | No partial published snapshot, crash or successful-save counter; source live state remains valid. |
| T08 | Inject target/draft capture byte-count mismatch. | Typed serialization failure, no indexed unwritten bytes; later unrelated requests remain correct. |
| T09 | RAM A has longer useful prefix but lower normalized keep fraction than resident B; also test equal keep fractions. | Candidate chosen by actual reusable tokens, not the old strict ratio conjunction. |
| T10 | Hybrid/SWA divergence before the first usable checkpoint. | Zero/limited true reuse reported as appropriate; no false hit; borrowing cannot delete the original A snapshot. |
| T11 | Identical prompt, pure extension, shorter prompt, single-token replay and changed suffix. | Raw-prefix and logits-replay semantics differ correctly; pure extension does not snapshot every turn. |
| T12 | Target restore succeeds, draft restore fails. | Source retained; destination metadata not published as valid; safe cold fallback or explicit native error, never mixed-state generation. |
| T13 | Longer token prefix has worse/missing checkpoint coverage than a shorter saved entry. | Deduplication does not delete the only state capable of the shorter rewind. |
| T14 | Byte limit, aggregate token limit, disabled cache, oversized snapshot and all eligible victims pinned. | Bounded memory/termination, correct explicit eviction/failure reasons, no silent unlimited `-cram -1` assumption. |

Do not use `sleep(2)` as the only proof that generation is active. Wait for a server-owned processing/task event or bounded status condition, then release/cancel only that fixture's work.

## 3. SSD cases with RAM unable to mask failures

Each case uses a fresh temporary `--slot-save-path`. The future SSD switch is `--cache-spill-mib`; it is not present in the baseline binary. Set finite RAM small enough to force spill but large enough for one complete snapshot/read staging. Confirm an actual durable commit event and file before stopping the source server.

| ID | Fixture / setup | Required observation |
|---|---|---|
| T15 | Capture A, force A from RAM to SSD with X, stop process, start a new process using the same compatible model/config and directory, send A-next. | New process has no inherited RAM source; disk-read/restore events occur; counters and output match the live-reference behavior. |
| T16 | Repeat T15 with hybrid/SWA, divergent suffix and a compatible draft/MTP configuration. | Target, draft, checkpoint and speculative bytes survive; compare against the same live divergent prompt. |
| T17 | Truncate each section, corrupt checksum/magic/version, forge length/count/position, introduce integer overflow and invalid packed media. | Parser rejects before native state mutation; bad entry cannot erase healthy entries or trigger huge allocations. |
| T18 | Interrupt after temporary creation, mid-write, before rename and after rename; inject short writes, ENOSPC and permission failure. | Only fully committed snapshots enter the catalog. No success acknowledged for incomplete durability; RAM source retained where possible. |
| T19 | Two model/config namespaces share one parent directory; same namespace has two concurrent server processes; include manual slot files and symlinks. | Different models coexist; second same-namespace writer cannot acquire ownership; manual/foreign/symlink-target files are never deleted. |
| T20 | Cancel during SSD read; reuse the slot for another task; complete the old read afterwards. | Epoch/task/slot-generation mismatch discards old result without installing state or generating an extra response. |
| T21 | Inject slow disk while another slot generates. | No inference-thread file write/read wait or global task mutex stall; other slot progresses. GPU-to-host capture latency is measured separately. |
| T22 | Fill the two-job queue, RAM staging budget, disk quota, metadata budget and entry cap. | Counts/bytes remain bounded, no retry spin, catalog-cap misses distinguished from corrupted state, eligible eviction deterministic. |
| T23 | Model reload or shutdown with queued/running reads and writes. | No callback into destroyed contexts/queues, no join under task mutex, no orphan install, committed entries still readable later. |
| T24 | Change weights, draft/mmproj, adapters/scales, KV layout, RoPE settings or state ABI while reusing directory. | Incompatible entries ignored in separate namespace, not restored or deleted as foreign garbage. |

For manual `/slots` regression tests, set RAM cache to zero and restart between phases where the existing API permits it. Otherwise a RAM hit can hide a missing checkpoint appendix. Manual-file compatibility remains separate from automatic spill acceptance.

## 4. Regression surfaces

Run existing `unit/test_completion.py` and `unit/test_slot_save.py`, plus maintained cancellation, multimodal and speculative tests applicable to changed code. Do not replace existing media-aware token serialization with text-only buffers. Test different media with identical placeholder token positions; it must not falsely share a cache entry. Test parent/child completion ownership and cache-disabled behavior.

After adding new tests to existing files, a targeted invocation is:

```bash
# Run from the repository root after the disposable build/venv setup.
export PYTEST_WORKERS=1
unset DEBUG_EXTERNAL
bash tools/server/tests/tests.sh unit/test_completion.py \
  -k 'prompt_cache or cache_retention' -v -x
bash tools/server/tests/tests.sh unit/test_slot_save.py -v -x
```

The new test names should contain `cache_retention`; the selector is a naming instruction, not proof those tests exist yet. Keep expensive hybrid/draft tests marked and explicitly invoked; a skipped hybrid test yields PARTIAL coverage, not PASS for Nex.

Use ASan/UBSan builds for ownership/parser failures where supported. Run affected C++ translation units under GCC/libstdc++ and Clang/libc++ when available. The disk research identified a `file_clock` representation difference; use fixed-width validated representations rather than assuming its native count streams portably.

## 5. Matched performance experiment

Measure four shapes using baseline and candidate: cold A; warmed append-only A; unrelated A/B/A; shared-prefix A/C/A. Use both explicit ID and automatic selection. Add SSD-spilled A after restart as a separate scenario. Warm-up runs and model/token fixtures must be identical; report contention and sample count rather than quoting unmatched medians as a causal speedup.

Required measurements:

- Native processed/cached token counts, valid checkpoint position, output correctness and request success.
- Capture count/bytes/time, restore count/bytes/time, SSD read/write time and total first-token latency.
- Aggregate throughput and latency of other concurrently active slots.
- Accounted RAM entries, staging/pinned bytes, catalog size, disk bytes, queue depth, and peak process RSS.
- Eviction, oversize, incompatible, missing-checkpoint, allocation and I/O failure counts.

A fixture with no destructive change or evacuation must add **zero full-state capture/restore operations after warm-up**. For other cases, show how reduced prefill compares with added copies. Do not impose an invented universal percentage threshold; record the measured tradeoff and whether the intended workload improves without correctness or unrelated-slot regression. Re-run affected measurements after relevant source changes.

## 6. Result record

Use one row per test/scenario:

```text
Case:
Verdict: PASS | PARTIAL | FAILED
Source/head and binary:
Model/draft digests and complete flags:
Fixture/test node:
Baseline result and evidence:
Candidate result and evidence:
Actual cached/processed tokens and selected slot:
Capture/restore/disk timings and bytes:
Output correctness check:
Peak RSS and accounted cache/staging:
What remains untested or changed:
```

PASS means the specific case passed against the delivered candidate. PARTIAL includes unavailable model/backend, unrun disk restart, skipped failure injection or unresolved evidence. FAILED includes incorrect output, lost required branch under ample budget, busy-slot mutation, a false cache hit, unbounded resource growth or a reproducible regression. A documentation-only draft has no native PASS verdict.
