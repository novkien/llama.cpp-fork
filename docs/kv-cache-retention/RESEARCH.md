# Source and upstream research record

Inspected on **2026-09-29**. PR status is a dated observation, not a promise of merge or future availability. The implementation package is an original integration design informed by these sources; none of the following PRs was cherry-picked, built or benchmarked during this documentation task.

## 1. Fork baseline and evidence boundary

Repository: [novkien/llama.cpp-fork](https://github.com/novkien/llama.cpp-fork). Default branch observed: `master`. Source baseline: [`526c43b8f7dfea9032e9f35e7a1be9183ca7cc20`](https://github.com/novkien/llama.cpp-fork/commit/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20).

[Issue 1](https://github.com/novkien/llama.cpp-fork/issues/1) tracks the native branch-loss and lookup problems from [llama-proxy#416](https://github.com/novkien/llama-proxy/issues/416). Historical native build/revision evidence in that incident is not proof that today's running process uses this fork baseline. No production process was inspected or changed in this task.

The newest owner request adds an implementation-ready design for RAM-first retention and optional SSD spill below the configured slot-save path, and authorizes a full architecture comment plus a draft PR in this fork. It does not request upstream publication or a production rollout.

## 2. Relevant opened PRs inspected

| PR and dated state | Inspected head | What the design takes from it | What it does not establish |
|---|---|---|---|
| [#28992: skipped prompt-cache lookups](https://github.com/ggml-org/llama.cpp/pull/28992), open/unmerged | `94bc965b57b7a1f84eb01e591345226b71e05925` | Independent metadata lookup; pinned-empty denominator guard; two useful regression shapes. | Does not change the strict two-ratio ranking or preserve all outgoing branches. No SSD tier. |
| [#24143: non-consuming load](https://github.com/ggml-org/llama.cpp/pull/24143), open/unmerged; conflicts reported | `6fca89bf228d8f7c7a744629f8f84c1da8a8bcde` | Keep serialized source bytes after restore; A/X/C/A fixture exposes borrowed-entry consumption. | Uses an older prompt/entry layout. Its dense-model report does not validate current Nex/draft/SSD behavior. |
| [#27451: shared checkpoints and OOM handling](https://github.com/ggml-org/llama.cpp/pull/27451), open/unmerged | `5c7018ed7b419197ffda429999a2334cab5df939` | Immutable shared checkpoint backing buffers; catch allocation failure through complete entry construction. | Does not fix ranking, branch preservation or persistence; its validation is not a complete runtime acceptance matrix. |
| [#26004: checkpoint-preserving slot files](https://github.com/ggml-org/llama.cpp/pull/26004), open/unmerged | `06d9d0ff54b586514a59268e2c780abc08473daa` | Checkpoints must travel with saved state; bounded appendix parsing; compare restored divergence with live divergence. | Manual save/restore is not automatic RAM/SSD tiering. Reported cross-machine success is not a portability guarantee. |
| [#28092: persistent prompt cache](https://github.com/ggml-org/llama.cpp/pull/28092), open/unmerged | `f2431a23a2cc059c1d1ed599396aa3b5d6a4342a` | Complete persistent state, compatibility metadata, startup catalog, corruption handling and disk budgets. | The inspected implementation chooses disk versus RAM rather than our RAM-first spill contract. Its mmap write path and namespace deletion require different handling. |

The #28992, #24143, #27451 and #28092 implementation diffs were inspected for the relevant behavior. #26004 was examined through its detailed current implementation/test description and relevant user/author discussion; it must still receive a line-by-line port review before code reuse. Do not merge these PRs as a bundle or assume they apply cleanly to this baseline.

Searches included `cache disk` across open/closed PRs and `"prompt cache" in:title` among opened PRs, plus direct reads of the linked issues/PRs and discussions. This is a bounded relevant search, not a claim that every cache PR has been audited.

## 3. Comments that materially change the architecture

### Disk design is already being discussed upstream

In [#28092, comment 5515802845](https://github.com/ggml-org/llama.cpp/pull/28092#issuecomment-5515802845), maintainer ngxson said on September 2 that a local alternative used chained hashes over fixed token blocks, needed refactoring, and was intended eventually to replace that PR. The maintainer explicitly stopped deeper review of that proposal. This is evidence of a design/refactoring reason for that particular PR's review status, not a hardware prohibition and not proof the unpublished alternative is available today.

Decision: retain a replaceable metadata lookup boundary. Begin with an exact, bounded metadata index; do not invent or depend on an unavailable upstream branch. Avoid scanning or reading all KV files on every request.

### A shared disk directory must not destroy another model's cache

[sammcj's comment 5517314832](https://github.com/ggml-org/llama.cpp/pull/28092#issuecomment-5517314832) reported that mismatched-key cleanup could delete another model's files when router children inherited one cache path. The same report identified a libc++ `file_clock` streaming build failure. [Author response 5518591168](https://github.com/ggml-org/llama.cpp/pull/28092#issuecomment-5518591168) described the numeric cast fix and acknowledged the namespace issue.

Decision: compatibility-specific directories, no foreign-directory deletion, exact opened-file/path checks, and portable validated metadata encodings. Do not carry over a broad startup `remove_files` rule.

### GPU-to-file-mapping capture has a reported kernel-hang risk

[Author comment 5518710877](https://github.com/ggml-org/llama.cpp/pull/28092#issuecomment-5518710877) points to a ROCm report on #26408: GPU state capture directly into shared file-mapped pages could interact badly with writeback. The author said #28092 used the same pattern and had not reproduced it on their CUDA/WSL setup. This is an attributed report, not a universal proven failure across backends.

Decision: ordinary host capture buffers followed by worker-owned file writes. No direct GPU DMA into shared file mappings in this implementation. Snapshot-sized host staging remains a real resource requirement; SSD capacity does not eliminate it.

### RAM can hide a broken file restore

[wagi-sho's comment 5055464245](https://github.com/ggml-org/llama.cpp/pull/26004#issuecomment-5055464245) explains that an apparently warm file restore was actually masked by existing RAM/checkpoint state. The reporter compared a restored divergent prompt with a live reference and checked output content. [terisuke's comment 5077635955](https://github.com/ggml-org/llama.cpp/pull/26004#issuecomment-5077635955) supplied a small cross-backend experiment and noted the mismatch between reported and actual file bytes when checkpoint data was appended.

Decision: new-process SSD tests with a confirmed committed spill, no inherited RAM source, full checkpoint/speculative serialization and exact byte reporting. Do not use another user's timings as this fork's benchmark.

### Non-consuming restore and shared-buffer ownership are separate fixes

#24143's [discussion](https://github.com/ggml-org/llama.cpp/pull/24143#issuecomment-4625637662) contains a bot policy warning and an author AI-disclosure response; no substantive technical rejection was present in the inspected discussion. Its actual diff stops consuming source entries but reflects older structs. Port the behavior and test shape, not the old struct assignments.

#27451's [September 26 comment](https://github.com/ggml-org/llama.cpp/pull/27451#issuecomment-5843549146) discusses recurrent/draft exhaustion over RPC. That is a separate capacity/failure problem, not proof that immutable buffer sharing solves native recovery. Decision: keep target/draft failure injection, but do not import unrelated purge/retry policy or #419 runtime recovery into this feature.

## 4. Historical foundations and nonselected adjacent work

[#16391](https://github.com/ggml-org/llama.cpp/pull/16391), merged October 9, 2025, introduced host-memory prompt caching. [The author's intended workload](https://github.com/ggml-org/llama.cpp/pull/16391#issuecomment-3377934532) included large agent context interleaved with auxiliary calls. This establishes that saving old context for later reuse is an upstream goal, not a novel hardware capability.

[#16560](https://github.com/ggml-org/llama.cpp/pull/16560), merged October 14, 2025, adjusts token limits to a finite memory budget. The baseline still has separate byte/token accounting. Do not infer that `-cram -1` means unlimited token retention.

[#28046](https://github.com/ggml-org/llama.cpp/pull/28046) proposes a per-entry token-limit policy; [#23666](https://github.com/ggml-org/llama.cpp/pull/23666) proposes second-chance eviction. These were located as adjacent options, not selected or fully audited here. The design keeps RAM-only cap semantics and uses a simple eligible-victim policy. Sleep-only retention PRs and recurrent-kernel changes are also outside this branch-replacement design.

## 5. Exact source anchors and terminology corrections

All source links below are pinned to the fork baseline:

- [Slot selection and cache mutation](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/tools/server/server-context.cpp): `get_available_slot`, `process_single_task`, `prompt_save/load/clear`, prompt-start rollback and idle evacuation.
- [RAM alloc/load/update](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/tools/server/server-task.cpp): consume-on-load, ratio comparison, prefix dedup and quota behavior.
- [Token/media handling](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/tools/server/server-common.cpp): exact `get_common_prefix`, packed `serialize`, validation and `clone`.
- [Checkpoint state](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/common/common.h): target, draft and `data_spec` bytes; 64 checkpoints are not 64 complete conversations.
- [Queue and threading](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/tools/server/server-queue.h): `post`, decode yielding, cancellation and mutex-held sleep callbacks.
- [Server development contract](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/tools/server/README-dev.md): filesystem features opt-in; heavy owner-thread work affects other sequences.
- [Test runner](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/tools/server/tests/tests.sh) and [test instructions](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/tools/server/tests/README.md).
- [Existing hash target](https://github.com/novkien/llama.cpp-fork/blob/526c43b8f7dfea9032e9f35e7a1be9183ca7cc20/vendor/hash/CMakeLists.txt): `vendor::hash` already includes SHA-256.

Two discrepancies matter when implementing. First, #28992's prose describes a one-token difference, but the inspected fork's `get_common_prefix()` returns the raw exact prefix; the logits adjustment belongs to execution. Second, #28092's title says `--cache-disk`, while its inspected diff defines `--cache-dir` and `--cache-dir-max`. Our proposed `--cache-spill-mib` is deliberately named and documented as a new fork option using existing `--slot-save-path`, not presented as an existing upstream option.
