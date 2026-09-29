# Branch-preserving RAM cache with optional SSD spill

Status: implementation design for [issue 1](https://github.com/novkien/llama.cpp-fork/issues/1), not implemented. Source baseline: `526c43b8f7dfea9032e9f35e7a1be9183ca7cc20`. Evidence and upstream comparisons are in [RESEARCH.md](RESEARCH.md); execution instructions are in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## 1. Requirements and scope

The owner's requested sequence is A(100k) -> B(100k) -> A-next, with A and B allowed to use the same physical slot. A must remain reusable after B replaces the live slot state when the configured cache can retain it. This applies both to automatic selection and explicit `id_slot` requests.

R1. Preserve a reusable outgoing branch before rollback, replacement, idle evacuation or an ordinary idle-slot purge destroys its only copy.

R2. Search the cache independently of the outgoing-state save heuristic. An empty pinned slot and `--slot-prompt-similarity 0` must not bypass lookup. A busy slot must not be mutated.

R3. A borrowed snapshot remains available. Score candidates by actually reusable state, not two independently normalized similarity ratios.

R4. Optional SSD spill lives below the configured `--slot-save-path`, preserves the complete server snapshot and permits reuse of committed entries after process restart.

R5. Bound RAM, pending I/O, metadata and disk usage; preserve normal concurrency, cancellation and correctness. Do not synchronously write SSD files on the inference loop.

R6. Keep normal append-only continuations cheap. Do not capture full KV on every request, force every request through a file or turn this into a proxy-side `/slots` RPC loop.

The current draft contains documentation only. Future implementation includes the RAM path and SSD path described here, but not a CUDA/kernel rewrite, distributed cache, cross-machine portability guarantee, compression, native model recovery, or proxy scheduling changes. `llama-proxy#419` recovery/order work and `#386` admission optimizations are not silently incorporated. Future proxy pinning integration remains separate.

## 2. Existing mechanism and required changes

The inspected `get_available_slot()` gates both save and load with `update_cache`; selected-slot saving normally depends on `f_keep < 0.5` or the LRU path. Pinned empty slots can miss this gate. The caller checks busy state after selection. `server_prompt_cache::load()` restores and then consumes the saved entry. Allocation deduplicates by token-prefix containment, which does not prove checkpoint coverage. These are source findings, not newly observed runtime incidents.

Reuse the existing `server_prompt_cache`, `server_tokens`, `common_prompt_checkpoint` and `llama_state_seq_*_ext` APIs. Do not create a second cache manager. Add a small storage/codec helper for disk operations; it must not own or access live llama contexts.

```text
                       explicit id_slot or automatic choice
                                      |
                              available target slot
                                      |
                         shared, read-only reuse planner
                       /              |                \
                  live state     RAM snapshot      SSD catalog
                       \              |                /
                        preserve outgoing branch if needed
                                      |
                       restore chosen immutable snapshot
                                      |
                       valid rewind + suffix prefill + TG
```

## 3. Ownership and data model

A `server_slot` owns one live sequence. A snapshot describes one reusable prompt branch, not one agent and not one slot. `source_slot_id` is diagnostic provenance only; it is never the lookup key or a reason to restore incompatible data.

Extend the existing cache entry with:

- An internal monotonic snapshot ID, engine epoch, compatibility descriptor and source-slot provenance.
- Exact `server_tokens` including media data, plus a token-position map where needed.
- Full target state, full draft state when present, and current speculative implementation state when required by the active drafter.
- The full checkpoint list: token counts, position ranges, target/draft partial-state bytes and `data_spec`. Persisted task IDs are reset to `-1` on import.
- Source sequence position bounds and memory capability metadata required to plan valid rollback.
- Storage location, byte/token accounting, last-use sequence, and transient pins held by restore/write operations.

Payload bytes become immutable after capture. Copy checkpoint metadata by value but share immutable large target/draft checkpoint buffers, following the ownership idea in upstream #27451. Updating a checkpoint allocates a fresh backing buffer; it must never rewrite a buffer shared by a saved branch. Keep `data_spec` value-owned initially. `server_tokens` needs `clone()`, not a deleted copy constructor or a text-only conversion.

A cache lease keeps an entry alive during planning/restore/spill. Erasing its index entry cannot invalidate a pointer being used by a pending operation. Count pending owned buffers as well as indexed RAM entries; removing an entry from a list is not proof that its memory has been freed.

## 4. One slot-preparation pipeline

### 4.1 Select without mutation

Refactor selection to return a target and a reason only. Preserve existing explicit-ID normalization/validation behavior unless separately changed and tested. For an explicit request, do not silently choose another slot. For auto mode, consider only available slots and retain existing scheduling tie behavior where reuse benefit is equal.

Check busy state before save, clear, restore or checkpoint mutation. Reserve an idle target while preparing an asynchronous disk restore. A reserved target is unavailable to other tasks but is not falsely reported as generating. Parent/child task capacity must be checked before changing any participating slot.

Validate the incoming execution configuration, including LoRA/aLoRA handling, before selecting compatible state. Snapshot the outgoing state under its old compatibility descriptor before applying a configuration change. Do not relabel old KV with the new adapter identity.

Honor `cache_prompt=false`: do not restore a cached candidate or archive that request's resulting state. Preserve a previously cache-eligible outgoing branch when appropriate. Non-completion tasks keep their existing non-cache behavior. Explicit slot erase must not trigger a new snapshot; it is not advertised as a global disk-archive purge.

### 4.2 Plan reuse, once

Extract the existing rollback decision into a pure helper used for live, RAM and disk candidates. Its proposed result is:

```text
cache_reuse_plan
  candidate_id / live-slot identity / version
  raw_lcp_tokens
  reusable_tokens
  checkpoint index, or direct reuse / cold reset
  next token position
  needs_target_restore / needs_draft_restore
  configuration compatibility
```

Use the existing position semantics, not `min(LCP, checkpoint.n_tokens)` as a substitute. The source algorithm uses sequence position minima, SWA coverage, `has_new_tokens`, checkpoint `pos_min`/`pos_max`, media positions and the requirement to evaluate a token for logits. Apply equivalent constraints to target and draft state. When metadata cannot establish a safe rewind, report zero reuse; never claim a speculative hit.

At this baseline `server_tokens::get_common_prefix()` returns the exact common prefix. It does **not** subtract a logits token. Keep raw prefix matching separate from the execution-only replay adjustment.

Rank compatible candidates primarily by `reusable_tokens`. For equal benefit choose live state, then RAM, then disk; use last-use order as a deterministic secondary tie. Do not require both `f_keep` and `f_sim` to increase. Do not restore each candidate merely to evaluate it. A candidate with no useful state must not trigger a costly copy.

Disk candidates have bounded in-memory metadata containing exact token/media identity and checkpoint descriptors. Metadata lookup performs no per-request directory scan or KV-file read. A cold catalog entry not admitted to the bounded index may miss; report that limitation rather than pretending all disk bytes are searchable without memory cost. Block-hash indexing can be added later without changing correctness.

### 4.3 Preserve before the first destructive operation

An append-only compatible request does not need a full snapshot merely because execution replays a final token. A semantic prefix truncation, divergence, configuration replacement or switch to another saved state does require preservation of useful outgoing state unless an equivalent snapshot is already retained.

Use raw prefix and compatibility for this distinction. Preserve before `seq_rm`, chunk shifting, state replacement, checkpoint erasure, `keep_first`, or `prompt_clear`; a hook after any of them is too late. A conservative save is correct for an uncertain semantic divergence; do not guess that different tokenization is harmless.

Capture transaction:

1. Compute full serialized target/draft sizes and accounted tokens/checkpoints.
2. Reserve cache/staging capacity and protect the selected restore candidate from eviction.
3. Allocate every part in an unpublished temporary entry. Catch allocation failure around vectors, token clone, checkpoint metadata and container insertion.
4. Serialize the live state while no decode is using it; require every returned byte count to match. Capture the required speculative implementation state through its existing API.
5. Publish only the completed snapshot. Only now mark the outgoing branch preserved and perform deduplication that removes older equivalents.

Use typed results: `SAVED`, `ALREADY_PRESERVED`, `DISABLED`, `BUDGET_REJECTED`, `ALLOCATION_FAILED`, `SERIALIZATION_FAILED`, `IO_PENDING`, `IO_FAILED`. A duplicate and a failed save are not the same result.

### 4.4 Restore without consuming

Pin the chosen source. Prepare token/checkpoint metadata before mutating live state. Read SSD bytes into bounded ordinary host memory, verify them, then restore target/draft on the inference owner thread. Verify all byte counts and restore required speculative state. Publish slot metadata only when the whole operation succeeds.

Do not clear source byte vectors, move the source prompt away, or erase the source merely because restoration succeeded. Live checkpoint metadata may be pruned without altering the saved copy.

A restore is not hardware-atomic. If target restoration succeeds and draft restoration fails, preserve the source snapshot, clear only the affected destination sequence and its speculative/sampler metadata, and use the existing safe cold-processing path if possible. If safe cold processing cannot proceed, return the existing native error. Never continue with mismatched target/draft state or restart other requests. A failure is not a cache hit.

### 4.5 Execute the verified plan

Recheck task/slot/engine epoch after asynchronous work. Apply the planned rewind and process the remaining prompt. Record actual `cache_n` only from the state accepted by execution. Log planned and actual counts separately so a metadata estimate cannot masquerade as reuse.

## 5. Deduplication, eviction and resource failure

Do not delete A merely because B's tokens contain A as a prefix. For recurrent/SWA models B may lack the checkpoint needed to rewind to A. Initially deduplicate exact compatible snapshots only when the retained entry's position/checkpoint coverage covers the replaced entry. Keep distinct branches until ordinary eviction.

Preserve existing `--cache-ram` byte and token-limit semantics in RAM-only mode, including the fact that `-1` does not disable the separate aggregate token cap in this baseline. Optional per-entry token policy from upstream #28046 is not bundled automatically.

Use deterministic least-recently-used eligible victims, updating recency after real use rather than scanning. Before evicting an unpersisted RAM victim, try SSD spill when enabled. Keep its RAM backing alive until disk commit is acknowledged. An already persisted victim can release RAM immediately, subject to leases. Disk victims are chosen only within this server's owned namespace and never while read-pinned.

When a normal capture cannot reserve RAM but a spill can free it, keep the outgoing slot reserved and unchanged, submit an eligible victim spill, and resume capture after commit releases capacity. Other slots continue. If the queue is full, attach the target to the next capacity-change notification rather than spin or grow the queue. Cancellation releases this wait. A source must not be dropped merely because a healthy, already scheduled spill has not finished.

The retention promise is conditional on resources. If both tiers are full, a snapshot exceeds the finite capture budget, or a save fails, preserve inference correctness and bounded memory: retain existing copies where possible, then use an explicit best-effort cache-loss outcome rather than an infinite wait or a forced server restart. A request may cold-prefill after this outcome. Emit the exact reason and affected snapshot; do not label it a successful save. Optional idle evacuation can be skipped when preservation fails; an unavoidable capacity purge must be logged as loss. This design does not promise zero loss under OOM, disk failure or exhausted quotas.

## 6. SSD contract

### 6.1 Configuration and directory

The existing option is `--slot-save-path PATH`. The new proposed option is **`--cache-spill-mib N`**, with `N=0` by default and `N>0` enabling a finite disk budget. Negative values are rejected. Enablement requires an explicit slot-save path and a finite positive `--cache-ram` budget for capture/read staging; invalid combinations fail startup with an explanation. Setting a slot-save path alone must not enable automatic persistence.

```text
<slot-save-path>/
  existing-user-slot-files                 # never touched by this cache
  prompt-cache-v1/
    <compatibility-digest>/
      <engine-epoch>-<counter>.lpc          # committed complete snapshot
      .<unique-id>.tmp                     # unpublished, owner-created only
```

Use a process-lifetime nonblocking filesystem lock for a single compatibility directory. Another server using that exact namespace disables its SSD tier with a warning and continues RAM-only; it does not steal the lock or delete files. Different model/config namespaces can coexist. This lock protects cache files, not source worktrees or unrelated inference.

### 6.2 Compatibility and trust

Build a canonical descriptor from model shard content digests, draft/mmproj/adapter/control-vector identities and effective scales, native build/state-format version, model architecture, effective context capacities, KV types/layout, SWA/recurrent settings, RoPE/YaRN parameters, and speculative implementation/configuration. Use the existing vendored SHA-256 implementation under `vendor/hash/sha256`; do not use `std::hash` or path/mtime alone as durable identity.

Hash immutable model files once during model initialization for persistent-cache compatibility, not on the request path. This costs startup I/O and must be measured. Detect files changing during fingerprinting and decline persistent reuse. Loaded weights must not be modified in place. Initially require the same tested build/backend/layout; a user's cross-backend success is not a portability contract.

Store the full descriptor and compare it, not only its directory hash. Raw token/media equality still determines prompt matching. Sampling temperature is not a substitute for execution compatibility. No conversation ID is required for the single trusted proxy deployment; a multi-tenant cache/authentication redesign is out of scope.

Cache files contain sensitive derived state and prompt material. Use private directory/file permissions, generated filenames, strict relative paths, no symlink following, and checks on opened file handles. Treat files as untrusted input to the parser even in a private directory. SHA-256 checksums detect corruption; they are not authentication against an attacker who can rewrite the directory.

### 6.3 Complete file format

Use a server-owned versioned envelope around existing serialized sequence bytes; do not change the core llama state format. Explicit little-endian integers and checked lengths describe:

1. Magic `LPCACHE1`, envelope version, byte order, total length, compatibility descriptor and snapshot identity.
2. `server_tokens::serialize()` payload, exact token count, source position bounds and planner capability metadata.
3. Full target state and optional full draft state, each with length and digest.
4. Current speculative implementation bytes and checkpoint count.
5. For every checkpoint: `n_tokens`, `pos_min`, `pos_max`, target/draft partial-state blobs and `data_spec`, each length-delimited. On import, `id_task=-1`.
6. A checksum covering metadata and all payload sections, and an exact end-of-file check.

Reject overflow, oversized counts/blobs, impossible positions, incompatible versions, truncation, checksum mismatch and invalid token/media payload before calling native setters. A malformed checkpoint appendix invalidates the automatic snapshot; never silently restore only its base KV and claim checkpointful reuse.

Keep manual `/slots` save/restore files and their existing API compatible. They are not automatically imported as complete snapshots. Upstream #26004 is a reference for a future manual-file adapter and for checkpoint fields, not permission to call a legacy slot file equivalent to this envelope.

### 6.4 Publication and restart

Write a private temporary file in the same namespace; loop for short writes, flush/sync, close, then atomically rename and sync the directory using platform-appropriate helpers. Publish the catalog entry and release the last RAM-only copy only after durable commit acknowledgement. If commit fails, keep RAM when possible and report failure. Do not replace a valid file with a partially written one.

Do not DMA GPU state directly into a shared file mapping. Capture into ordinary host buffers on the inference owner thread; the storage worker writes those immutable buffers using ordinary file I/O. The ROCm mmap/writeback report is a reason to avoid that path, not proof that all mmap reads are defective.

On startup, scan only the owned namespace, validate bounded metadata and build the index. Never delete a foreign namespace or arbitrary files under slot-save-path. Ignore unfinished temporary files; remove only identifiable stale files belonging to this format and namespace after lock acquisition. Bound indexed metadata to 64 MiB and 4096 entries initially; report entries not indexed. KV payloads are loaded lazily.

This is spill persistence, not write-through persistence of every turn. Only committed SSD entries survive a crash/restart. RAM-only and active states are not promised durable. Tests must force a real spill and confirm commit before testing restart recovery.

## 7. Threading, I/O lifecycle and cancellation

The server context/queue already controls safe decode boundaries. Keep all live KV, checkpoint and speculative-state operations on that owner path. The storage worker receives immutable byte buffers, generated filenames and numeric IDs, never a `llama_context *`, slot pointer or mutable prompt.

Use one disk worker and at most two pending jobs initially. Both queued and in-progress owned bytes count toward the finite RAM/staging budget. A full queue cannot grow unbounded; skip optional idle spill, or let required capture wait on an existing capacity-changing job without growing the queue. Terminal I/O failure or impossible budgets use the documented pressure fallback. Do not block the inference loop waiting for a write.

A disk read reserves only the target task/slot, allowing other slots to decode. Completion posts a typed internal `SERVER_TASK_TYPE_CACHE_IO_DONE` through `server_queue::post`. Process state-changing completion only when the queue is not yielding a decode; otherwise decline it unchanged for later handling. Validate engine epoch, slot generation and task ID before installation. Cancellation releases the reservation/lease; a late I/O result is discarded without generation or success reporting.

Do not hold the task-queue mutex while doing filesystem I/O, joining the worker or invoking sleep callbacks that need queue progress. Shutdown stops new submissions, cancels pending installs, joins the worker, drains results and only then destroys storage/context owners. Normal local-file errors must be handled; portable C++ cannot guarantee a hard deadline for a kernel syscall stuck on a failing filesystem. Do not use network filesystems or claim that a job-count cap can cure kernel hangs.

## 8. Observability and acceptance

Add compact events/counters for capture reason/result, snapshot ID, source/target slot, raw LCP, planned/actual reusable tokens, RAM/disk hit, missing checkpoint, compatibility rejection, copy/read/write milliseconds, bytes, eviction reason and queue/staging use. Do not log prompt text or token arrays by default. IDs must be scoped by engine epoch.

Acceptance requires deterministic branch-loss and pinned-empty regressions; correct output as well as counters; RAM-only, SSD-only-after-restart and mixed-tier evidence; cancellation/failure isolation; and matched cold/warm/alternating performance measurements. Ordinary append continuations must not add full-state save/restore operations. Neither native builds nor these tests have been run for this documentation draft. See [TESTING.md](TESTING.md).
