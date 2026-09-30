# Deployed four-GPU acceptance, 2026-09-30

## Published build and effective route

PR #2 merged at `7865d7f6b3761cd21b4ab7dbfc91c3d78b83164a`; PR #4 merged at `95e6565970621aa508e80fd805dcf73af413093c`. The native GPU host and RPC guest were built and installed from the latter clean revision. Their manifests have the same source ID, `95e6565970621aa508e80fd805dcf73af413093c:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. Native used CUDA 12.9, `70-real`, NCCL and RPC. The guest used CUDA 13.3, `120-real` and RPC; NCCL was requested but its package was unavailable and was not linked on the guest. Version, library resolution, manifest and RPC-service checks passed. Rollback binaries and the original route were retained.

The launcher/bridge validation dependency is [llama-proxy PR #484](https://github.com/novkien/llama-proxy/pull/484). Each live phase verified the actual managed child's executable, manifest, arguments and filtered CUDA environment. This was not inferred from route metadata alone.

The owner approved the later reduction from context 400000 to 200000 and split adjustment from `28,28,29,15` to `29,29,27,15` to keep pipeline parallelism enabled. All matched trials below use this adjusted profile, four slots, the same target model, q8_0 KV, `-cram -1`, 64 checkpoints/min step 2048 and all four V100-SXM2 16 GB GPUs. Native context aligns to 200192 with unified KV; that is shared capacity, not four times 200192. Pipeline parallelism remained enabled through the 20080-token cache fixtures. Observed free VRAM after long prefill was 1749-2813 MiB per GPU; this does not guarantee every future workload fits.

The effective UUID order was `5ca3,b2e,e149,d4d`, using the full UUIDs in [NATIVE-RESULTS.md](NATIVE-RESULTS.md). This starts on the Gen3 x8 GPU at PCI 08, follows its NV6 peer at PCI 0c, crosses PIX to PCI 0d, and ends at the second NV6 peer at PCI 09. Selective mode used `GGML_CUDA_P2P_PAIRS=0-1,2-3`; global `GGML_CUDA_P2P` was absent. No unsafe PCIe peer condition was repeated.

## Matched Qwen baseline/selective/rollback

Each phase loaded a fresh child. Each ran three sequential requests with the historical deterministic Vietnamese GPU-bandwidth prompt (`temperature=0`, 256 output tokens), followed by four simultaneous labeled requests. All 42 requests completed. The two speculative methods were separate matched comparisons, not changes inside one comparison.

| Speculation / phase | Median sequential decode, tokens/s | Accepted / generated draft per sequential request | Four-request maximum wall time, s |
|---|---:|---:|---:|
| MTP, pairs absent | 44.770 | 159 / 285 | 12.048 |
| MTP, selected pairs | 46.992 | 159 / 285 | 11.708 |
| MTP, pairs absent rollback | 44.408 | 159 / 285 | 12.099 |
| DFlash2, pairs absent | 36.799 | 173 / 563 | 13.213 |
| DFlash2, selected pairs | 36.736 | 173 / 563 | 13.720 |
| DFlash2, pairs absent rollback | 40.720 | 173 / 563 | 14.018 |

Within each speculative method, all nine sequential outputs were exactly equal across phases. MTP used the target's embedded next-token predictor on CUDA3 with `n_max=3`. DFlash2 retained the production external draft on CUDA3 with `n_max=7`. Four simultaneous requests verified serving with four slots; their outputs and draft counts varied across fresh phases, so they do not establish parallel deterministic equivalence.

MTP's selected phase was about 5% faster sequentially and had a modest four-request wall-time improvement. DFlash2 showed no demonstrated general speed benefit: its selected sequential result matched the first baseline, and rollback was faster. The measured variation and simultaneous-request slowdown versus the first baseline are retained. The 126-132 GB/s NVLink copy measurements do not imply a similar inference gain.

The earlier 400000-context DFlash run had another split and disabled pipeline parallelism. It is excluded from this matched comparison.

## RAM retention acceptance and remaining work

Qwen's real hybrid/DFlash2 RAM fixtures passed: A/B exact 20000-token prompts shared about 12000 tokens, a borrower replaced A's slot, and cross-slot A restores reused `cache_n=20015` while processing 65 new tokens. A second restore after erasing the borrower destination proved the RAM entry was not consumed. Four matched warm/cold generated-token comparisons were equal. Warm continuations took 1.036-1.063 s versus about 22.47-22.49 s cold. The opt-out fixture subsequently had `cache_n=0`. Native logs showed checkpoint restores and no capture/restore failure, CUDA error or assertion.

The focused Qwen image append fixture also passed twice across borrowed slots: it reused 1062 cached tokens, cold references reused zero, and warm/cold output tokens matched exactly. Repeating a shorter original image prompt after generation can legitimately be cold when no recurrent checkpoint covers the rewind; the initial fixture incorrectly expected full reuse in that case.

Nex is not accepted for closure of [issue #1](https://github.com/novkien/llama.cpp-fork/issues/1). Its actual 20000-token hybrid/DFlash/RPC fixture restored 20015 tokens and processed 65, but deterministic warm/cold generated outputs diverged. A matched no-RAM live-state control also diverged from its cold reference at token index 4; the RAM and no-RAM warm outputs differed at index 4. This does not isolate the divergence to RAM restoration or prove it harmless. Further discrimination between numerical/layout variation and state corruption is required.

Nex's independent cold-image failure is [issue #6](https://github.com/novkien/llama.cpp-fork/issues/6): the old DFlash GGUF lacks M-RoPE sections, and its one-position validator rejects the gap after skipped image embeddings. No metadata derivative or runtime repair was applied. Optional SSD spill is not implemented or tested.

## Final state and evidence

The persistent Qwen route is `llama.cpp-fork`, context 200000, split `29,29,27,15`, four slots, production DFlash2 and selected pairs `0-1,2-3`. Nex was restored to its original backend/specification. The final Qwen configuration was read back through the revision-checked proxy API; after the owner's explicit request to stop holding the GPU awake, it was not loaded again. All agent-owned idle leases were released, the renewal process stopped, and the block gate was restored to false. The proxy reported `idle_shutdown_active=true`.

The raw live records remain on the GPU host under `/tmp/issue3-live-evidence` and `/tmp/issue1-{qwen,nex}*`; public native harnesses and raw logs are in the owner's `issue3-native-evidence-20260930.tgz`. They are disposable acceptance artifacts, not maintained runtime tests. The published issue/PR comments record deployment and the unresolved cache criteria. Native limits in NATIVE-RESULTS still apply, including untested cancellation, complete copy-slot rotation and platform variants.

Issue #3 is accepted for the implemented four-GPU layer transport scope. Issue #1 remains open for Nex correctness and the separately documented unimplemented SSD scope; a merged RAM PR is not a claim that these passed.
