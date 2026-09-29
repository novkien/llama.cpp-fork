# Issue 1: branch-preserving KV cache workspace

Status: **design and implementation instructions only; no runtime fix implemented**.

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

Read DESIGN first, then implement the numbered slices in IMPLEMENTATION. Update the checklist below as real code and tests land on this draft branch. Do not replace failed behavioral tests with a successful build or a larger cache counter.

## Implementation checklist

- [ ] S1: reproduce baseline defects with deterministic fixtures.
- [ ] S2: safe snapshot ownership, complete allocation handling and typed results.
- [ ] S3: pure reuse planner, pinned/automatic pipeline and preserve-before-mutate.
- [ ] S4: non-consuming restore, conservative deduplication and bounded retention.
- [ ] S5: complete snapshot codec and opt-in namespaced SSD store.
- [ ] S6: bounded disk worker, cancellation, restart and shutdown integration.
- [ ] S7: RAM/disk/hybrid/draft/media regressions and matched performance measurements.
- [ ] S8: implementation documentation, review and owner-selected native/proxy integration.

These are implementation slices, not extra approval stages. The present delivery stops at a draft documentation PR. It does not authorize production restarts, model changes, proxy pinning activation, upstream publication, merge or deployment.

## Workspace safety

Use the existing checkout if appropriate. Inspect its actual remote, branch and modifications before editing; preserve other people's work. Do not reset, stash, clean, force-push or switch a shared checkout merely to follow these instructions. A separate clone is an optional convenience, not a requirement for all contributors.

All new CLI names, structs, task types and event fields in this package are **proposed implementation contracts**, not options available in the baseline binary. Existing interfaces are identified explicitly. This package was authored with ChatGPT assistance at the fork owner's request; it is not an upstream-maintainer endorsement or a native test report.
