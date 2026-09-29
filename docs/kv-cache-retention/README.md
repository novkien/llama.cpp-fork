# Issue 1: branch-preserving KV cache workspace

Status: **RAM-path implementation and CPU regression evidence are present. Optional SSD spill is not implemented.**

Addresses [novkien/llama.cpp-fork#1](https://github.com/novkien/llama.cpp-fork/issues/1). Historical production evidence: [novkien/llama-proxy#416](https://github.com/novkien/llama-proxy/issues/416).

Prepared on 2026-09-29 against fork `master` at `526c43b8f7dfea9032e9f35e7a1be9183ca7cc20`. This SHA is the source baseline, not a claim about a deployed binary.

## Outcome

When a new prompt replaces a slot's old branch, preserve the old branch's complete reusable state in RAM before destructive changes. Restore it later into either a native-selected slot or an explicitly requested `id_slot`. Optionally spill RAM victims to an SSD directory below the existing **`--slot-save-path`**, including recurrent/SWA checkpoints and draft state.

A physical slot is an execution destination, not a cache key. Multiple saved branches can originate from the same slot. A saved branch is not consumed when another request borrows its prefix.

Retention is bounded by configured resources. Cache eviction, incompatible state and genuinely changed prompts remain possible; this is not unlimited storage or a promise that every request avoids prefill.

## Start here

| File | Purpose |
|---|---|
| [DESIGN.md](DESIGN.md) | Settled behavior, algorithms, state ownership, RAM/SSD contracts, failures and boundaries. |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Junior-oriented implementation order, exact files/symbols, build commands and completion checks. |
| [TESTING.md](TESTING.md) | Deterministic regressions, disk-isolated tests, failure injection and performance evidence. |
| [RESEARCH.md](RESEARCH.md) | Dated upstream PRs, inspected comments, source anchors and reuse decisions. |

Read the current delivery section below before continuing the remaining numbered slices in IMPLEMENTATION. Update the checklist as complete slice acceptance, not just source changes, is established. Do not replace failed behavioral tests with a successful build or a larger cache counter.

## Current RAM delivery

The review of owner commit `8cfa7bb52fe69e904abc53c86b775c01a6ec3b0b` found that the initial two-file patch still consumed snapshots on load, ranked raw LCP rather than valid checkpoints, retained the 25% exclusion, and archived cache-opt-out requests. Optional idle evacuation also cleared state after a failed save.

The repair keeps selection read-only, prepares the cache only after slot availability and family capacity checks, preserves outgoing branches before replacement, restores without consuming snapshots, and shares immutable checkpoint backing. `server_prompt::plan_reuse` is used by both candidate ranking and execution. Exact compatible snapshots, not contained token prefixes, are deduplicated. Capture checks byte counts and protects every allocation; restore prepares metadata before native setters. RAM limits remain in force and actual restoration moves an entry to the most-recently-used end.

`cache_prompt=false` disables incoming reuse and automatic archiving of that request. Infill keeps its existing live-state reuse path without becoming an automatic RAM archive. LoRA identity is checked before candidate selection and old state is saved under its old identity. A failed optional idle save keeps live state; a necessary pressure purge reports any preservation loss.

The RAM source has local CPU build and regression evidence in [TESTING.md](TESTING.md). This is not a claim about Nex/GPU, media, real MTP, disk persistence or production behavior. The requested owner merge/build/live trial remains separate. `--cache-spill-mib` is **not an available flag** in this implementation; no automatic file writing or disk restore was added.

## Implementation checklist

- [ ] S1: RAM regressions reproduce seven failing cases on the owner's initial patch; full hybrid fixture coverage remains.
- [ ] S2: core RAM ownership/capture implemented; exhaustive per-allocation failure injection remains.
- [ ] S3: target rewind planner and pinned/automatic pipeline implemented; real hybrid/draft/media acceptance remains.
- [ ] S4: non-consuming RAM restore, conservative deduplication and budget protection implemented; SSD-pressure integration remains.
- [ ] S5: complete snapshot codec and opt-in namespaced SSD store.
- [ ] S6: bounded disk worker, cancellation, restart and shutdown integration.
- [ ] S7: RAM/disk/hybrid/draft/media regressions and matched performance measurements.
- [ ] S8: implementation documentation, review and owner-selected native/proxy integration.

These are implementation slices, not extra approval stages. The current repair stops at reviewed PR source and isolated checks. It does not perform production restarts, model changes, proxy pinning activation, upstream publication, merge or deployment. Unchecked items preserve the full architecture acceptance rather than hiding unfinished work.

## Workspace safety

Use the existing checkout if appropriate. Inspect its actual remote, branch and modifications before editing; preserve other people's work. Do not reset, stash, clean, force-push or switch a shared checkout merely to follow these instructions. A separate clone is an optional convenience, not a requirement for all contributors.

Names in DESIGN/IMPLEMENTATION describe the complete target architecture; consult the current delivery above and actual source for the implemented RAM subset. SSD flags, I/O tasks and event fields remain **proposed contracts**, not available options. Existing interfaces are identified explicitly. This package was authored with ChatGPT assistance at the fork owner's request; it is not an upstream-maintainer endorsement or a native test report.
