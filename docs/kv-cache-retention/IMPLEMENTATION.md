# Junior implementation guide

This is an ordered implementation guide, not an implemented patch. Complete the RAM and SSD slices on the draft branch. Keep [DESIGN.md](DESIGN.md) authoritative for behavior and [TESTING.md](TESTING.md) authoritative for evidence. Names introduced below are proposed names.

## 0. Establish a safe local starting point

Run these read-only commands in the checkout you intend to use:

```bash
pwd
git rev-parse --show-toplevel
git remote -v
git branch --show-current
git status --short
git rev-parse HEAD
```

The intended repository is `novkien/llama.cpp-fork`, not `ggml-org/llama.cpp` or `novkien/llama-proxy`. This design's source baseline is `526c43b8f7dfea9032e9f35e7a1be9183ca7cc20`. Newer commits are not automatically wrong: inspect intervening changes to the listed symbols and reconcile them. Never reset to that SHA or switch another person's shared branch to make the commands match.

A fresh clone is an optional way to get this draft without disturbing an existing checkout. Use the head branch shown by the actual PR. Do not push an upstream PR as a side effect of working on this fork.

Read root/scoped AGENTS, CONTRIBUTING, `tools/server/README-dev.md` and `tools/server/tests/README.md`. The current owner request authorizes this fork's design workspace and draft publication; implementation and production activation must be reported separately. Keep complete source files, not placeholder stubs or intentionally failing TODO binaries, in implementation commits.

## 1. Read the source in execution order

Use symbol searches rather than stale line numbers:

```bash
git grep -n -E 'get_available_slot|process_single_task|launch_slot_with_task|try_clear_idle_slots' -- tools/server
git grep -n -E 'prompt_save|prompt_load|server_prompt_cache::|struct server_prompt' -- tools/server
git grep -n -E 'get_common_prefix|keep_first|serialize\(|deserialize\(' -- tools/server/server-common.cpp
git grep -n -E 'common_prompt_checkpoint|data_spec' -- common
git grep -n -E 'slot.save.path|cache.ram|ctx.checkpoints' -- common/arg.cpp common/common.h
git grep -n -E 'cleanup_pending_task|yield_to_queue|on_sleeping_state' -- tools/server/server-queue.*
```

| File/symbol | What to understand before editing |
|---|---|
| `server-context.cpp`: `server_slot::prompt_save/load/clear` | Full target/draft serialization and live-state destruction. |
| `get_available_slot`, `process_single_task` | Selection, update_cache gate, busy checks, parent tasks and task ownership. |
| Prompt-start section of `update_slots()` | Raw matching, chunk shifts, recurrent/SWA checkpoint selection, logits replay and final `cache_n`. |
| `server-task.h/.cpp`: `server_prompt_cache` | Entry allocation, prefix deduplication, consumption on load, byte/token limits. |
| `server-common.cpp`: `server_tokens` | Exact LCP, media-aware token cloning, packed serialization and validation. |
| `common/common.h/.cpp`: checkpoints | Target/draft state and speculative bytes; updates must not mutate shared snapshots. |
| `server-queue.h/.cpp` | `post`, deferred tasks, decode yielding, cancellation and shutdown. |
| `vendor/hash/CMakeLists.txt` | Existing `vendor::hash` target and vendored SHA-256; no new crypto dependency needed. |

Write down one A -> B -> A trace and one pinned-empty trace before modifying their conditions. A blank slot with a matching RAM entry is not a cache miss. A nonzero LCP without a valid recurrent checkpoint is not a reusable-state hit.

## 2. Build a disposable baseline

The following creates only an external build/test environment. It does not launch production inference. It may download Python packages and the maintained test fixtures when tests run.

```bash
ROOT="$(git rev-parse --show-toplevel)"
WORK="$(mktemp -d -t llama-cache-issue1.XXXXXX)"
printf 'Evidence directory: %s\n' "$WORK"
python3 -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
python -m pip install -r "$ROOT/tools/server/tests/requirements.txt"
cmake -S "$ROOT" -B "$WORK/build" \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$WORK/build" --target llama-server -j 2
unset DEBUG_EXTERNAL
export LLAMA_SERVER_BIN_PATH="$WORK/build/bin/llama-server"
export LLAMA_CACHE="$WORK/model-cache"
export PYTEST_WORKERS=1
mkdir -p "$LLAMA_CACHE"
bash "$ROOT/tools/server/tests/tests.sh" \
  unit/test_completion.py -k prompt_cache -v -x
```

The maintained runner changes into its own tests directory. `DEBUG_EXTERNAL` must not point tests at a real service. Use test-helper-owned ports and temporary slot directories. A CPU fixture can prove scheduler/cache behavior; it cannot validate Nex hybrid checkpoints or production GPU performance.

Record baseline commit, flags, model digest, test node, actual counters, output and logs. Rebuild into a different directory for comparisons, or record exactly which rebuilt candidate replaced the baseline binary. Do not call an unrun command PASS.

## S1. Add deterministic failing reproductions

Edit existing `tools/server/tests/unit/test_completion.py` and `test_slot_save.py` where appropriate; extend `tests/utils.py` only for options the new fixtures need. Do not copy an old upstream test helper over the current helper.

Implement T01-T05 from TESTING first. Use explicit token arrays or a tokenizer-derived stable fixture to control the branch split. Avoid random prompt selection and sleeps as the only evidence of busy state. Use `n_predict=0` for pure cache-selection fixtures where supported; multi-turn fixtures must include the actual generated tokens in the next prompt. For the consumption case use A -> unrelated X -> prefix-sharing C -> A-next so that A exists only in RAM when C borrows it. For pinned-empty, prove the requested slot was evacuated before the continuation.

Assert that the tests fail for the intended reason on the baseline and retain their raw counters. A test failing because the model is missing, the port is occupied or the request JSON is wrong is not a reproduced cache defect.

Completion: baseline failures are discriminating; existing prompt-cache tests still have recorded results.

## S2. Make snapshot ownership and capture safe

Change `common/common.h/.cpp` for immutable shared target/draft checkpoint backing buffers, adapting the small ownership pattern in upstream #27451. `reset(size)` must allocate fresh storage before exposing a writable capture buffer. Existing copies must still see the old bytes after reset or clear. Keep checkpoint metadata independent; keep speculative bytes value-owned initially. Search every use of `data_tgt` and `data_dft` and adapt it without dropping bounds checks.

In `server-task.h`, add snapshot metadata, a lease/pin representation and typed capture/restore outcomes. Prefer straightforward structs and RAII ownership. A lease means "this operation holds the bytes alive"; it is not a mutex held across GPU or disk work.

Replace the current alloc-then-publish pattern with reserve -> construct temporary -> fill -> publish. Include token/media clone, checkpoint allocation, container insertion and byte-count verification in error handling. Deduplication removal occurs only after the replacement snapshot is complete. Do not leave an indexed entry with allocated but unwritten bytes after an exception.

Add current target/draft position/capability metadata and an engine epoch. Source-slot ID is provenance. Use an epoch-qualified sequence for snapshot IDs so process restarts cannot overwrite older filenames.

Completion: T06-T08 pass for ownership/allocation; serialization returns precise success/failure; original slot remains untouched during a failed capture.

## S3. Extract the reuse planner and unify slot preparation

In `server-task.h/.cpp`, define the pure `cache_reuse_plan` and candidate descriptor from DESIGN. Move the decision part of the prompt-start rewind logic into a helper. Do not move native state setters into this helper.

Translate the existing position algorithm literally before optimizing it. Inputs include raw LCP, token-to-position conversion, whether new prompt tokens exist, target/draft minimum/maximum positions, SWA coverage, checkpoint descriptors and configuration identity. Output includes the specific usable checkpoint, next position and reusable token count. Preserve media chunk boundaries and the final-logits adjustment. Compare helper results against the old live-state path on fixtures before using it for RAM/disk.

Refactor `get_available_slot()` to selection-only. In `process_single_task`, validate availability/configuration/parent capacity before any cache mutation. Explicit ID requests must never modify busy slots or fall through to another slot. A disk-pending reservation must be distinguishable from idle and generating and included in availability/cancellation logic.

Perform one candidate selection and retain its plan/lease. Remove `update_cache` as the gate for lookup. Do not merely add a zero-denominator guard inside the similarity-enabled block; that still misses similarity=0. Replace the `f_keep && f_sim` comparison with the DESIGN ordering.

Wire execution to the same verified plan. On epoch/version change discard and replan before mutation. Keep the existing errors for invalid request/capacity; this refactor is not authority to reset the engine.

Completion: T01-T05 and T09-T11 pass; no state API call is made by the pure planner or on a busy slot.

## S4. Preserve branches and stop consuming snapshots

Add an `ensure_preserved` call before the first destructive change in ordinary replacement, idle evacuation and idle purge. Audit ALL matching call sites: `prompt_clear`, `seq_rm`, `seq_add`, `keep_first`, checkpoint erase and `llama_state_seq_set_data_ext`. Explicit erase/cancellation/model teardown have different intent; do not accidentally turn an explicit cache erase into another save.

Compute semantic divergence from raw tokens and execution compatibility, not the post-logits `n_past`. A pure extension must not create a full-KV snapshot every turn. A shorter request, a changed suffix or another compatible branch must preserve the useful outgoing branch if not already represented. When preservation cannot succeed, follow the explicit pressure/failure contract and count the loss, not a false successful save.

Change `load` to read immutable source buffers. Clone tokens and copy independent checkpoint metadata into a prepared destination object. Remove source `clear()`, `shrink_to_fit()` and `states.erase()` from successful restore. Do not copy full KV byte vectors a second time just to keep the RAM source. Maintain leases until all setters and metadata installation finish.

Restore errors must not publish a half-restored prompt. Clear only the target sequence and its draft/speculative/sampler state when needed, then safely cold-process or return the existing error. Do not delete a healthy source after an unrelated destination capacity failure.

Use exact-equivalence/coverage checks for deduplication. Maintain deterministic eligible-victim recency and existing RAM-only caps. Test shared-buffer lifetime after eviction; a removed index entry is not necessarily freed memory.

Completion: A survives both destructive slot reuse and another branch's RAM restore. T12-T14 pass, and append-only capture/restore counts remain zero after initial warm-up where no evacuation occurs.

## S5. Implement the codec and opt-in SSD store

Add `tools/server/server-cache-storage.h/.cpp` as a subordinate codec/file-I/O helper, not a second scheduler or cache. Register it in `tools/server/CMakeLists.txt` and use the existing `vendor::hash` target. Keep all `llama_context` calls out of this module.

Implement DESIGN's version-1 envelope using checked little-endian reads/writes, existing `server_tokens::serialize/deserialize/validate`, and complete target/draft/spec/checkpoint sections. Use length-delimited data, never `write(sizeof(struct))`. Validate lengths against actual file size and reserved RAM before allocation. Initial parser limits: at most 1024 checkpoints and at most the configured capture/read budget for a snapshot; unsupported larger snapshots return a reason, not an allocation attempt. Validate position ranges and token counts before narrowing to native integer types.

Use SHA-256 for payload integrity and compatibility namespaces. Link the already vendored implementation; do not invent cryptography or add an HTTP dependency. Include all model shards and execution-affecting inputs in the canonical descriptor. Cache model fingerprints only for the loaded immutable model lifetime. Do not accept path/mtime alone as proof of equal weights.

In `common/common.h` and `common/arg.cpp`, add proposed `cache_spill_mib`, default 0, and `--cache-spill-mib N`. Reject negatives; require a configured `--slot-save-path` and positive finite RAM budget when enabled. Keep RAM-only legacy flags untouched. Update the generated/maintained server option documentation according to its current generation procedure.

Implement the private namespace, exclusive nonblocking namespace lock, generated paths, no-follow file opens, atomic temporary-file publication and exact total-byte accounting. Never delete manual slot files, foreign model directories or symlink targets. Use ordinary buffered writes from host memory, NOT shared mmap as the destination for GPU state capture.

Build the bounded startup metadata index. Do not eagerly read all KV payloads or scan the directory during each request. If metadata exceeds 64 MiB or 4096 entries, keep a deterministic recent subset and expose an index-capacity miss. Do not claim the nonindexed files were corrupt.

Completion: codec tests reject malformed input without calling state setters; T15-T19 establish isolated restart reuse and file safety. The SSD format does not silently upgrade old manual slot files.

## S6. Integrate bounded disk I/O without blocking other slots

Keep one worker and two jobs maximum. Define native storage jobs/results holding immutable buffer leases, numeric IDs, engine epoch and generated file paths. The worker may perform read/write/hash/fsync/rename; it cannot access a live slot or call any llama state function.

Add a typed internal `SERVER_TASK_TYPE_CACHE_IO_DONE` payload in the existing task mechanism. The worker uses the verified `server_queue::post(server_task&&)` entry point. Process the completion on a non-yielding owner iteration; decline it unchanged while decoding. Do not bypass the queue's safety boundary just because disk I/O has finished.

For a read, reserve only the destination and remember task ID, slot generation and engine epoch. Add that pending ownership to cancellation and cleanup paths, including `cleanup_pending_task`. If cancelled, release the reservation; a late result releases bytes and does not resume inference. Prevent parent/child requests from being partially installed.

For a spill, pin the RAM source until commit succeeds. Pending bytes remain accounted; no "evict then write through dangling pointer". After acknowledgement mark the disk copy committed and release RAM only if no live lease needs it. On failure keep RAM where possible, record the specific cause, and avoid retry loops on the same broken path.

Handle queue-full and snapshot-too-large without unbounded allocations or waits. These resource-loss paths must remain distinguishable from branch-loss bugs. Under normal pressure the selected recent outgoing snapshot should replace eligible older entries, spilling them first when possible. A required capture that can fit after a scheduled spill waits with only its source slot reserved and unchanged; wake it on committed capacity change, not a polling loop. Do not discard the outgoing branch merely because the worker has not finished. Do not let a queued optional spill block all incoming requests.

Integrate orderly stop and model epoch changes. Do not join a disk worker while holding the server task mutex or from a callback that prevents its completion processing. Stop submissions and installs before destroying contexts or queues. Only SSD-committed snapshots are crash durable; this version does not flush every active/RAM-only state on shutdown.

Completion: T20-T24 prove cancellation, stale completion rejection, other-slot progress and bounded memory. Document the filesystem/kernel hang limit rather than promising hard cancellation of arbitrary blocking syscalls.

## S7. Finish regression and performance evidence

Run the existing cache, slot save, cancellation and relevant media/speculation tests along with TESTING's deterministic cases. Enable the real hybrid model tests only on isolated hardware/capacity the owner intends for testing. Preserve the exact native model/build/KV/draft configuration in the report.

Measure cold baseline, ordinary warm continuation, unrelated A/B alternation and prefix-sharing A/C alternation. Separate capture, disk write, disk read, restore, prefill and generation. Use counters and deterministic content/logit checks, not latency alone. Inspect peak RSS, accounted cache/staging bytes and other-slot progress under slow disk injection.

A dense toy test is not acceptance for Nex. A successful RAM restore is not acceptance for SSD. A successful build is not acceptance for either. Report each matrix row PASS/PARTIAL/FAILED with evidence and untested boundaries.

## S8. Finish the implementation PR, not production activation

Update this checklist and the runtime README with actual implemented option names, resource behavior and remaining limitations. Remove stale claims if the implementation differs, or revise the design explicitly before changing its contract. Link issue 1 and the precise upstream references used; preserve attribution/licenses for adapted code.

Check the exact diff and run `git diff --check`. Commit only intended paths; no broad staging, force push or unrelated cleanup. Keep the PR draft until its promised implementation evidence exists. Do not mark issue 1 fixed because this design or a compiling patch was merged. Proxy policy integration and native deployment require their own explicit endpoint and runtime verification.
