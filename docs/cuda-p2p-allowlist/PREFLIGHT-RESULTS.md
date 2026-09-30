# Pre-implementation result: host-only source-branch probe

## Verdict

**PASS for the tested control-flow scope; PARTIAL for native feature feasibility.** No CUDA/NVLink/VMM/NCCL or Qwen success is claimed. The feature is not implemented in this PR.

The owner requested a docs-only draft plus any feasible pre-code test in the current environment. A disposable C++ probe was compiled and executed outside the repository. Its final run completed at **2026-09-30T06:48:32Z**.

## Actual environment and access limits

```text
2026-09-30T06:38:12Z
Linux dc64cbaacd3c 6.18.44 #1 SMP Sat Sep 26 20:02:31 UTC 2026 x86_64 GNU/Linux

GPU tools:

GPU device nodes:

CUDA driver library:

Build tools:
cmake version 3.31.6
c++ (Debian 14.2.0-19) 14.2.0
git version 2.47.3

Runtime GPU probe:
find_library(cuda)= None
CUDA_LOAD_ERROR: libcuda.so.1: cannot open shared object file: No such file or directory
```

The container has neither `nvidia-smi`/`nvcc`, NVIDIA device nodes, nor a loadable CUDA driver library. A remote-terminal plugin search returned an available but uninstalled integration; no connected GPU host was established or operated. A full archive download failed in this container, so no full-source CUDA or CPU llama.cpp build was claimed. Source specimens came from successful connected GitHub reads.

## Exactly what executed

The source baseline is `526c43b8f7dfea9032e9f35e7a1be9183ca7cc20`, with the CUDA and generic-backend blob identities in RESEARCH. Five function bodies were copied from those source reads and compiled with minimal test types:

- `ggml_backend_buffer_copy_tensor`;
- `ggml_backend_tensor_copy`;
- `ggml_backend_tensor_copy_async`;
- `ggml_backend_cuda_buffer_cpy_tensor`;
- `ggml_backend_cuda_cpy_tensor_async`.

Two scheduler blocks were compiled in fixture wrappers: ordinary split-input async/fallback and immediate user-input copy. The outer scheduler, graph split/allocation machinery, model layer assignment, real CUDA backend registration, peer initialization, VMM allocator, and NCCL were **not** executed.

The CUDA APIs, memory accessors, physical mapping helpers, streams/events, and proposed policy decorators were software test doubles. Peer API doubles copied bytes with CPU memory operations and recorded the requested endpoints. The decorators hard-coded the proposed two pairs; they did not test a production environment parser. Original callback bodies were not edited to implement a feature.

This is an executable source-specimen experiment, not an independent end-to-end CUDA implementation or a full scheduler test. It answers whether the existing copied branches route to host fallback when both callback seams decline.

## Commands and final result

These commands reproduce execution with the separately archived disposable probe files. The probe is not installed as a repository test target.

```sh
c++ -std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=undefined probe.cpp -o probe
./probe
c++ -std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined probe.cpp -o probe-asan
ASAN_OPTIONS=detect_leaks=1 ./probe-asan
```

Both final builds and executions returned exit code 0. Output was identical; AddressSanitizer/UndefinedBehaviorSanitizer emitted no diagnostics for this harness run. An initial harness build rejected an unused variable and misleading indentation under `-Werror`; those scaffold-only issues were corrected before the reported runs. This is not a full codebase sanitizer result.

```text
HOST-ONLY CONTROL-FLOW PROBE; CUDA API, GPU memory, events, and callback policy are test doubles.
Executed: 5 source function bodies + 2 scheduler copy blocks; not whole ggml scheduler/model.
PASS directional_entrypoint_size_cases=1728 byte_mismatches=0 forbidden_peer_dispatches=0
PASS same_backend_cases=24
PASS physical_alias_cases=48
PASS populated_buffer_view_cases=72
PASS negative_control=async_only forbidden_cases_detected=48/48 (bytes alone still match in doubles)
PASS negative_control=buffer_only forbidden_cases_detected=24/48 (bytes alone still match in doubles)
PASS negative_control=ungated forbidden_cases_detected=48/48 (bytes alone still match in doubles)
TRACE excluded_scheduler: async_decline | backend_sync:1 | event_sync | buffer_decline | D2H:1 | H2D:2
TRACE allowed_scheduler: peer_API:0->1 | record_copy_event | wait_copy_event
TRACE negative_async_only: async_decline | backend_sync:1 | event_sync | peer_API:1->2 | stream_sync
PASS synthetic_layer_handoff_chain=0->1->2->3 dispatch=PEER,HOST,PEER consumer_matches_cpu_oracle=true
PASS synthetic_layer_handoff_chain=3->2->1->0 dispatch=PEER,HOST,PEER consumer_matches_cpu_oracle=true
SUMMARY positive_cases=1874 negative_controls=3 hardware_verdict=PARTIAL reason=no_CUDA_driver_or_GPU
```

## Test population and negative controls

The 1,728 directional combinations are 12 directed physical pairs x 6 entry paths x 8 sizes x 3 patterns/iterations. Sizes are 0, 1, 15, 16, 17, 4096, 65537, and 1048576 bytes. The additional 24 same-backend, 48 physical-alias, 72 populated-buffer-view, and 2 synthetic chain cases give 1,874 positive combinations. These are scenario combinations, not 1,874 independent production regression tests.

All source/destination buffers were CPU allocations. Guard ranges and copied bytes matched. The zero-length cases check branch/no-data behavior in the doubles, not native CUDA zero-copy semantics. View coverage uses valid populated buffer pointers; malformed views are not covered.

For each incomplete-policy negative control, 48 combinations cover the eight denied physical directions through six entry paths:

| Deliberately incomplete seam | Detected forbidden peer-call cases | Meaning |
|---|---:|---|
| Async only | 48/48 | The original buffer callback can re-enter peer copy, including direct synchronous entry paths. |
| Buffer only | 24/48 | Async-success paths bypass the guarded buffer callback. |
| Neither | 48/48 | An ungated peer-call path is not selective. |

The negative controls still copy matching CPU bytes. The detector therefore checks forbidden path selection independently, rather than calling every byte-correct result a success. It does not recreate or measure the host's PCIe corruption.

## Interpretation for the intended layer chain

In the synthetic forward and reverse chains, the same source scheduler blocks selected PEER, HOST, PEER and the final CPU consumer matched an independently transformed expected value. This supports the proposed double-gate design at those branch seams.

It does **not** establish that llama.cpp's full graph allocator will place these operations on four real GPUs, that CUDA events complete in the required order, that NVLink is physically used, or that the actual VMM pool has denied mappings. It also does not prove equality with the legacy implicit staging path's performance.

The strongest valid conclusion is: **the existing generic fallback can be reached without replacing the layer scheduler, but both CUDA copy callbacks must be gated.** Native experiments B/C from PREFLIGHT and candidate tests from TESTING remain required.

## Reproducibility manifest

The original probe files and raw logs are delivered as a separate conversation artifact, not executable additions to this docs PR. The hashes below identify that local artifact content; they are not hashes of complete upstream source files. To recreate the experiment without that artifact, use the five pinned source bodies and two scheduler blocks listed above, explicit doubles, the independent matrix and negative controls described here, and preserve the evidence boundary.

```text
6f1190e459dfa83e655e267f0829992dc1e9880e65466bdeadb8fc0bda94d6da  environment.txt
e58e8d0228b4c3e7038bf77c7a7d2f866c0792d2da98b472de5897d785d0a45a  probe.cpp
9fe0e8380dd9799612afeca7c0e90d867d8963621e23c0c2318c56f0fc131e49  source_excerpts.inc
fc6c755f81d77aaee830a737e08803853ac45aefa907596ef632a16a66c1b18e  scheduler_excerpt.inc
372766983e43a0952fc65a4ec66aa66e5fe42b62371807db08a639aa20d56cf4  scheduler_input_excerpt.inc
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  build.log
1d0565adac46281cc2cc3ed69327465ded24581d1fa85a45965c2b3516ed616d  probe-output.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  build-asan.log
1d0565adac46281cc2cc3ed69327465ded24581d1fa85a45965c2b3516ed616d  probe-asan-output.txt
5ca3fff463f816bef183bd3c2470b596e5b0c2e88d1925008051cbab42161140  run-finished-utc.txt
```

## Unrun work

Native CUDA initialization and directional enablement; real NVLink/PCIe transfers; all actual allocator permissions; whole ggml graph execution; asynchronous race/capture/cancellation behavior; candidate parser and collective rejection; launcher integration; and four-GPU Qwen/MTP performance are unrun. No source patch, deployment, restart, route edit, or host-setting change was made by this experiment.
