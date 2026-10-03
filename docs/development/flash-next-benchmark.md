# Flash Next stream benchmark

`scripts/bench_flash_next.py` measures streamed chat requests through an explicitly selected endpoint. It uses the native server's `/apply-template` and `/tokenize` endpoints to prepare an exact prompt length. The runner does not start a server or change its model configuration.

```sh
python3 scripts/bench_flash_next.py \
    --base-url "$BENCH_PROXY_URL" \
    --prepare-url "$BENCH_NATIVE_URL" \
    --model Qwen3.8-Flash-Next \
    --family structured --prompt-tokens 8192 --output-tokens 512 \
    --repetitions 1 --seed 1 \
    --campaign-id flash-next-cold-8k --arm baseline \
    --output-dir /tmp/flash-next-baseline-pair-1
```

Use a new output directory for each invocation. Repeat with `--family prose` and `--family code` for the other fixtures. `--concurrency 2` prepares distinct content for each lane. `--mode cached-prefix` primes each lane before measuring its continuation; cold mode sends `cache_prompt: false` and requires zero cached tokens. A prepared JSON chat body can be supplied with `--request-file` for single-request cold runs. Its model, sampling fields and output limit are preserved; the runner verifies its token count against `--prompt-tokens`.

`--repetitions` counts total requests, not concurrent groups. Use `--concurrency 2 --repetitions 10` for five full two-lane groups; five repetitions would run groups of two, two and one.

For paired A/A or A/B measurements, use the same campaign ID, seed, content family and workload, distinct `--arm` values, and alternate one repetition from each arm. Keep source artifacts, context, layer placement, microbatch, MTP settings and sampling fixed except for the variable under test. Complete model activation before invoking the runner. Profile runs that synchronize CUDA operations must use a separate campaign and cannot establish end-to-end speed gains.

The output contains a manifest, prepared prompts, per-request samples, a CSV summary, timestamped parsed SSE events and byte-for-byte SSE response files with SHA256 hashes. Native properties and proxy metadata are captured before and after collection. `--evidence-manifest` adds immutable host/RPC build and model evidence to the manifest. A changed environment invalidates the campaign and returns a nonzero exit status. Missing or partial evidence produces a screening-only result with `acceptance_eligible: false`. Raw request and response artifacts can contain prompt content; keep campaign artifacts outside the source checkout.

The evidence JSON contains `host` and `rpc` objects, each with `source_id`, `binary_path`, `runtime_library_paths` and a `files` map from artifact paths to bare SHA256 hashes. Include the executable and every backend runtime library in that map, and list the library paths explicitly. Its `models` object contains `target_gguf_shard_count`, `target_gguf_shards` (an array of `path` and `sha256` objects covering every shard) and `draft_gguf_sha256`. Capture these from the actual loaded artifacts; supplying a manifest alone does not verify that those files are running.

PP is newly processed prompt tokens divided by native prompt time. TG uses the native predicted-token count minus the first token, divided by native predicted time. Cache hits and rejected draft tokens do not contribute to those rates. TTFT starts before HTTP submission and ends at the first content or reasoning delta. Wall time includes the complete response transport. Invalid or incomplete responses remain in the artifacts and are excluded from valid-sample summaries.

The runner collects observations; it does not select a winner. Check the recorded token/cache counts and artifact identities, use at least five matched pairs for a claimed improvement, and compute paired confidence intervals separately. Report cold PP, TG, TTFT, wall time and MTP acceptance along with concurrent and cached-prefix cases. Context capacity and VRAM require their own live checks.

Run the focused parser and provenance checks with:

```sh
python3 -m unittest tests.test_bench_flash_next
```
