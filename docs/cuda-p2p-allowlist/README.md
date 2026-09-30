# Selective CUDA P2P: coder preparation pack

Status: documentation-only draft. No backend feature, executable test, build configuration, workflow, launcher, or production change is included in this PR.

Tracks [llama.cpp-fork #3](https://github.com/novkien/llama.cpp-fork/issues/3). The starting design is the [owner-requested implementation plan](https://github.com/novkien/llama.cpp-fork/issues/3#issuecomment-5904416014). These documents organize that plan for implementation and add an explicitly limited pre-implementation experiment. They do not mark the proposal approved or the feature deployed.

## Intended outcome

Keep all four GPUs in layer-split inference. With the verified process mapping, allow peer transfers only between CUDA0/CUDA1 and CUDA2/CUDA3. Every other distinct local CUDA pair must take an explicit host-memory fallback. Preserve the model, layer distribution, MTP draft placement, context, parallelism, and RPC configuration from the issue.

```text
Target layer groups: CUDA0 -> CUDA1 -> CUDA2 -> CUDA3
Transfer choice:       PEER     HOST     PEER
Reverse transfers:     PEER     HOST     PEER
```

The proposed `GGML_CUDA_P2P_PAIRS=0-1,2-3` option does not exist in the reviewed backend. Merely setting it on that binary proves nothing.

## Read in this order

| Document | What the coder gets |
|---|---|
| [DESIGN.md](DESIGN.md) | Exact policy contract, identity mapping, normal-path clarification, supported scope, and invariants. |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Symbol-level change map, source flow, sensitive code, implementation sequence, and integration dependency. |
| [RESEARCH.md](RESEARCH.md) | Primary-source CUDA/NCCL knowledge and what each source does and does not establish. |
| [PREFLIGHT.md](PREFLIGHT.md) | Experiments to falsify the design before feature implementation, including a native four-GPU seam test. |
| [PREFLIGHT-RESULTS.md](PREFLIGHT-RESULTS.md) | Actual local experiment, transcript, negative controls, and limits. |
| [TESTING.md](TESTING.md) | Candidate regression matrix, real layer-graph validation, Qwen acceptance, and completion checklist. |

## Current evidence

The accessible default branch was rechecked at `526c43b8f7dfea9032e9f35e7a1be9183ca7cc20` on 2026-09-30. This is a source baseline, not the installed host binary. The issue's older `2138591d...` reference previously failed to resolve; do not treat it as a usable implementation base without checking it.

A disposable host-only C++ probe executed five copied source function bodies and two scheduler copy blocks with software test doubles. It passed 1,874 positive scenario combinations and detected the intended incomplete-gate negative controls. It did not compile the whole scheduler, execute a CUDA API, or run a model. Native feasibility remains PARTIAL because this environment has no CUDA driver/device. See the results document before quoting any number.

## Working boundary

Continue development on this draft's branch when implementation is requested. Read the current checkout's applicable instructions and reconcile newer changes before editing. Do not overwrite another coder's work or rewrite a shared branch to match this dated baseline.

Only Markdown files in this folder are introduced. The temporary local probe is evidence gathering, not a shipped implementation. No merge, production restart, BIOS/IOMMU/ACS change, or automatic issue closure is authorized by this docs PR. Link the issue with `Refs #3`, not a closing keyword.

The first implementation can stay small: one policy, filtered peer initialization, filtered VMM access, both copy callbacks gated, and existing host fallback. A new asynchronous staging subsystem is not a prerequisite unless measured performance requires it.
