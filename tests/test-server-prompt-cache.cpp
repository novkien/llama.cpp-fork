#include "server-task.h"

#include <cstdio>
#include <cstdlib>

static int n_checks = 0;

static void check(bool condition, const char * message) {
    ++n_checks;
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static server_prompt prompt(size_t n, llama_token token = 11) {
    server_prompt p;
    p.tokens = server_tokens(llama_tokens(n, token), false);
    return p;
}

static void checkpoint(server_prompt & p, int64_t n, llama_pos lo, llama_pos hi) {
    common_prompt_checkpoint c{};
    c.update_pos(n, lo, hi);
    c.data_tgt.reset(8)[0] = 7;
    p.checkpoints.push_back(std::move(c));
}

static void native_roundtrip(const char * model_path) {
    common_params params;
    params.model.path = model_path;
    params.n_ctx = 2048;
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_parallel = 2;
    params.n_gpu_layers = 0;
    params.kv_unified = true;
    params.warmup = false;
    params.fit_params = false;
    params.cpuparams.n_threads = 2;
    params.cpuparams_batch.n_threads = 2;
    auto init = common_init_from_params(params);
    auto * ctx = init->context();
    check(ctx != nullptr, "initialize native test context");
    const auto decode = [&](const server_prompt & p, int id_slot) {
        llama_batch batch = llama_batch_init(p.n_tokens(), 0, 1);
        for (int i = 0; i < p.n_tokens(); ++i) {
            common_batch_add(batch, p.tokens[i], i, { id_slot }, i + 1 == p.n_tokens());
        }
        const int result = llama_decode(ctx, batch);
        llama_batch_free(batch);
        check(result == 0, "decode native test prompt");
    };
    server_prompt_cache cache(64, 0);
    const std::vector<common_adapter_lora_info> lora;
    auto a = prompt(192, 11);
    decode(a, 0);
    check(cache.save(a, ctx, nullptr, 0, lora, nullptr) == SERVER_PROMPT_SAVED, "capture complete native state");
    const auto * saved = &cache.states.front();
    const auto bytes = saved->data.main;
    check(!bytes.empty() && saved->pos_max == 191, "captured sequence bounds and bytes");
    check(cache.save(a, ctx, nullptr, 0, lora, nullptr) == SERVER_PROMPT_ALREADY_SAVED, "exact duplicate capture");
    auto b = prompt(64, 41);
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    decode(b, 0);
    cache.limit_size = saved->size();
    check(cache.save(b, ctx, nullptr, 0, lora, nullptr, saved) == SERVER_PROMPT_SAVE_FAILED, "protect chosen restore source at budget limit");
    check(cache.states.size() == 1 && saved->data.main == bytes, "failed capture retains source");
    cache.limit_size = 64 * 1024 * 1024;
    check(cache.save(b, ctx, nullptr, 0, lora, nullptr, saved) == SERVER_PROMPT_SAVED, "save replacement branch");
    server_prompt restored;
    check(cache.load(restored, *saved, ctx, nullptr, 1, nullptr), "restore into another native slot");
    check(cache.states.size() == 2 && saved->data.main == bytes, "restore does not consume snapshot");
    check(llama_memory_seq_pos_max(llama_get_memory(ctx), 1) == 191 && restored.n_tokens() == 192, "install consistent native state and tokens");
    check(!cache.load(restored, *saved, ctx, ctx, 1, nullptr), "reject missing draft state before target mutation");
    check(llama_memory_seq_pos_max(llama_get_memory(ctx), 1) == 191, "failed restore leaves native state unchanged before installation");
    a.cache_eligible = false;
    check(cache.save(a, ctx, nullptr, 1, lora, nullptr) == SERVER_PROMPT_NOT_CACHED, "opt-out bypasses archive");
}

int main(int argc, char ** argv) {
    auto p = prompt(1000);
    auto plan = p.plan_reuse(700, 1100, 0, 0);
    check(plan.n_past == 700 && !plan.checkpoint && plan.pos_next == 700, "dense direct reuse");
    check(p.plan_reuse(1000, 1000, 0, 0).usable_tokens(1000) == 999, "logits replay");
    check(p.plan_reuse(0, 1000, -1, 0).n_past == 0, "empty input prefix");
    check(p.plan_reuse(500, 1000, -1, 0).n_past == 0, "no native state");
    check(p.plan_reuse(700, 1100, 999, 0).n_past == 0, "recurrent cache without checkpoint");
    checkpoint(p, 600, 599, 599);
    checkpoint(p, 900, 899, 899);
    plan = p.plan_reuse(700, 1100, 999, 0);
    check(plan.n_past == 600 && plan.pos_next == 600 && plan.checkpoint, "recurrent rollback");
    check(p.plan_reuse(500, 1100, 999, 0).n_past == 0, "divergence before earliest checkpoint");
    check(p.plan_reuse(700, 1100, 300, 128).n_past == 700, "SWA resident coverage");
    check(p.plan_reuse(700, 1100, 600, 128).n_past == 0, "SWA checkpoint does not cover window");

    auto copy = p.clone();
    check(copy.checkpoints.front().data_tgt.data() == p.checkpoints.front().data_tgt.data(), "checkpoint backing shared");
    copy.checkpoints.front().data_tgt.reset(8)[0] = 9;
    check(p.checkpoints.front().data_tgt.data()[0] == 7, "capture does not overwrite old snapshot");
    copy.checkpoints.clear();
    check(p.checkpoints.size() == 2, "live checkpoint pruning is independent");

    server_prompt_cache cache(64, 0);
    const std::vector<common_adapter_lora_info> lora;
    auto resident = prompt(1000);
    resident.tokens.set_token(800, 12);
    // Resident raw LCP is longer but has no usable recurrent checkpoint.
    cache.states.emplace_back();
    auto & useful = cache.states.back();
    useful.prompt = prompt(900);
    useful.prompt.tokens.set_token(700, 12);
    useful.pos_min = 899;
    checkpoint(useful.prompt, 600, 599, 599);
    const auto input = server_tokens(llama_tokens(1000, 11), false);
    check(cache.find_best(resident, input, 999, 0, lora, 1000) == &useful, "rank usable checkpoints, not raw LCP");
    check(cache.find_best(resident, input, 0, 0, lora, 1000) == nullptr, "prefer superior live state");
    auto empty = prompt(0);
    useful.prompt = prompt(1000);
    useful.pos_min = 0;
    const auto short_input = server_tokens(llama_tokens(100, 11), false);
    check(cache.find_best(empty, short_input, -1, 0, lora, 100) == &useful, "no arbitrary 25 percent exclusion");
    useful.lora.emplace_back();
    check(cache.find_best(empty, input, -1, 0, lora, 1000) == nullptr, "adapter mismatch is excluded");
    useful.lora.clear();
    check(cache.find_best(useful.prompt, input, 0, 0, lora, 1000) == nullptr, "resident wins equal benefit");
    check(cache.find_best(empty, input, -1, 0, lora, 0) == nullptr, "zero benefit does not restore");
    if (argc == 2) {
        llama_backend_init();
        native_roundtrip(argv[1]);
        llama_backend_free();
    }
    std::printf("%d prompt-cache checks passed\n", n_checks);
    return 0;
}
