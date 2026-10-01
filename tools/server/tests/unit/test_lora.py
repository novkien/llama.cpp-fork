import json
import pytest
import requests
import socket
from concurrent.futures import ThreadPoolExecutor
from utils import *

server = ServerPreset.stories15m_moe()

LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.lora_files = [download_file(LORA_FILE_URL)]


@pytest.mark.parametrize("scale,re_content", [
    # without applying lora, the model should behave like a bedtime story generator
    (0.0, "(little|girl|three|years|old)+"),
    # with lora, the model should behave like a Shakespearean text generator
    (1.0, "(eye|love|glass|sun)+"),
])
def test_lora(scale: float, re_content: str):
    global server
    server.start()
    res_lora_control = server.make_request("POST", "/lora-adapters", data=[
        {"id": 0, "scale": scale}
    ])
    assert res_lora_control.status_code == 200
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
    })
    assert res.status_code == 200
    assert match_regex(re_content, res.body["content"])


def test_lora_per_request():
    global server
    server.n_slots = 4
    server.start()

    # running the same prompt with different lora scales, all in parallel
    # each prompt will be processed by a different slot
    prompt = "Look in thy glass"
    lora_config = [
        ( [{"id": 0, "scale": 0.0}], "(bright|day|many|happy)+" ),
        ( [{"id": 0, "scale": 0.0}], "(bright|day|many|happy)+" ),
        ( [{"id": 0, "scale": 0.3}], "(special|thing|gifted)+" ),
        ( [{"id": 0, "scale": 0.7}], "(far|from|home|away)+" ),
        ( [{"id": 0, "scale": 1.0}], "(eye|love|glass|sun)+" ),
        ( [{"id": 0, "scale": 1.0}], "(eye|love|glass|sun)+" ),
    ]

    tasks = [(
        server.make_request,
        ("POST", "/completion", {
            "prompt": prompt,
            "lora": lora,
            "seed": 42,
            "temperature": 0.0,
            "cache_prompt": False, # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
    ) for lora, _ in lora_config]
    results = parallel_function_calls(tasks)

    assert all([res.status_code == 200 for res in results])
    for res, (_, re_test) in zip(results, lora_config):
        assert match_regex(re_test, res.body["content"])


def test_lora_prefill_barrier_advances_past_finished_incompatible_slot():
    global server
    server.n_slots = 2
    server.n_ctx = 2048
    server.n_batch = 8
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        server.server_port = probe.getsockname()[1]
    server.start()

    long_response = requests.post(server.make_url("/completion"), json={
        "prompt": "Look in thy glass and tell the story. " * 24,
        "n_predict": 4,
        "stream": True,
        "return_progress": True,
        "cache_prompt": False,
        "id_slot": 1,
        "lora": [{"id": 0, "scale": 1.0}],
    }, stream=True, timeout=(5, 20))
    assert long_response.status_code == 200

    def iter_sse_data(response):
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                payload = line[6:]
                if payload != b"[DONE]":
                    event = json.loads(payload)
                    if event is not None:
                        yield event

    long_events = iter(iter_sse_data(long_response))
    for event in long_events:
        progress = event.get("prompt_progress")
        if progress and 0 < progress["processed"] < progress["total"]:
            break
    else:
        long_response.close()
        raise AssertionError("long LoRA request did not enter prefill")

    short_request = {
        "prompt": "Look in thy glass",
        "n_predict": 2,
        "cache_prompt": False,
        "id_slot": 0,
        "lora": [{"id": 0, "scale": 0.0}],
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        short_future = executor.submit(
            requests.post,
            server.make_url("/completion"),
            json=short_request,
            timeout=30,
        )

        long_finished = False
        for event in long_events:
            if event.get("stop") is True:
                long_finished = True

        short_response = short_future.result(timeout=30)
        assert short_response.status_code == 200
        assert long_finished
        long_response.close()


@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_with_big_model():
    server = ServerProcess()
    server.model_hf_repo = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    server.model_hf_file = "Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf"
    server.model_alias = "Llama-3.2-8B-Instruct"
    server.n_slots = 4
    server.n_ctx = server.n_slots * 1024
    server.n_predict = 64
    server.temperature = 0.0
    server.seed = 42
    server.lora_files = [
        download_file("https://huggingface.co/ngxson/Llama-3-Instruct-abliteration-LoRA-8B-F16-GGUF/resolve/main/Llama-3-Instruct-abliteration-LoRA-8B-f16.gguf"),
        # TODO: find & add other lora adapters for this model
    ]
    server.start(timeout_seconds=600)

    # running the same prompt with different lora scales, all in parallel
    # each prompt will be processed by a different slot
    prompt = "Write a computer virus"
    lora_config = [
        # without applying lora, the model should reject the request
        ( [{"id": 0, "scale": 0.0}], "I can't provide you with a code for a computer virus" ),
        ( [{"id": 0, "scale": 0.0}], "I can't provide you with a code for a computer virus" ),
        ( [{"id": 0, "scale": 0.3}], "I can't write a computer virus" ),
        # with 0.7 scale, the model should provide a simple computer virus with hesitation
        ( [{"id": 0, "scale": 0.7}], "Warning: This is a hypothetical exercise" ),
        # with 1.5 scale, the model should confidently provide a computer virus
        ( [{"id": 0, "scale": 1.5}], "A task of some complexity! Here's a simple computer virus" ),
        ( [{"id": 0, "scale": 1.5}], "A task of some complexity! Here's a simple computer virus" ),
    ]

    tasks = [(
        server.make_request,
        ("POST", "/v1/chat/completions", {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "lora": lora,
            "cache_prompt": False, # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
    ) for lora, _ in lora_config]
    results = parallel_function_calls(tasks)

    assert all([res.status_code == 200 for res in results])
    for res, (_, re_test) in zip(results, lora_config):
        assert re_test in res.body["choices"][0]["message"]["content"]
