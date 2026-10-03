import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Direct execution puts tests/ on sys.path, so add the checkout root before importing scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_flash_next import (  # noqa: E402
    FAMILIES,
    PAD_ATOMS,
    SSEParser,
    _compare_environment,
    _completion_body,
    _campaign_outcome,
    _feed_raw_chunk,
    _fixture_content,
    _load_evidence_manifest,
    _messages,
    _native_props_evidence,
    _proxy_release_evidence,
    _request_evidence,
    _write_raw_sse,
    PromptPreparer,
    fit_prompt,
    summarize_stream,
)


def _timings(*, prompt_ms=10.0, predicted_n=3, predicted_ms=100.0):
    return {
        "prompt_n": 8,
        "cache_n": 0,
        "prompt_ms": prompt_ms,
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
        "draft_n": 2,
        "draft_n_accepted": 1,
    }


def _payload(data):
    return ("data: " + json.dumps(data, separators=(",", ":")) + "\n\n").encode()


def _valid_stream(*, prompt_ms=10.0, predicted_ms=100.0, content_predicted_n=3):
    final = {
        "choices": [],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "timings": _timings(prompt_ms=prompt_ms, predicted_ms=predicted_ms),
    }
    chunks = [
        _payload({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}),
        _payload({
            "choices": [{"delta": {"reasoning_content": "thinking"}, "finish_reason": None}],
            "timings": {
                **_timings(prompt_ms=prompt_ms, predicted_ms=predicted_ms),
                "predicted_n": 1,
                "draft_n": 1,
                "draft_n_accepted": 0,
            },
        }),
        _payload({
            "choices": [{"delta": {"content": "three tokens in one event"}, "finish_reason": None}],
            "timings": {
                **_timings(prompt_ms=prompt_ms, predicted_ms=predicted_ms),
                "predicted_n": content_predicted_n,
            },
        }),
        _payload({"choices": [{"delta": {}, "finish_reason": "length"}]}),
        _payload(final),
        _payload(final),
        b"data: [DONE]\n\n",
    ]
    return chunks


class FakePromptPreparer(PromptPreparer):
    def __init__(self):
        super().__init__("", 0, None, None)

    def count(self, messages: list[dict[str, str]]) -> int:
        return len(re.findall(r"\w+|[^\w\s]", messages[0]["content"]))


def _padding_atom_count(padding):
    counts = {0: 0}
    for offset in range(len(padding)):
        if offset not in counts:
            continue
        for atom in PAD_ATOMS:
            end = offset + len(atom)
            if padding.startswith(atom, offset):
                counts[end] = max(counts.get(end, 0), counts[offset] + 1)
    return counts.get(len(padding), 0)


def _summarize(chunks, *, prompt_ms=10.0, ended_ns=None, incomplete=False):
    parser = SSEParser()
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    now = events[-1].received_ns if events else 1000
    started_ns = events[0].received_ns - 1_000_000 if events else 0
    return summarize_stream(
        events,
        started_ns=started_ns,
        ended_ns=ended_ns or now + 1_000_000,
        prepared_prompt_tokens=8,
        expected_output_tokens=3,
        cache_mode="cold",
        parser_incomplete=incomplete or parser.incomplete,
        parser_error=parser.error,
    ), events, parser


def _complete_evidence_manifest():
    return {
        "host": {
            "source_id": "host-rev",
            "binary_path": "bin/llama-server",
            "runtime_library_paths": ["lib/libggml-cuda.so"],
            "files": {
                "bin/llama-server": "a" * 64,
                "lib/libggml-cuda.so": "b" * 64,
                "lib/libcublas.so": "c" * 64,
            },
        },
        "rpc": {
            "source_id": "rpc-rev",
            "binary_path": "bin/rpc-server",
            "runtime_library_paths": ["lib/libggml-rpc.so"],
            "files": {
                "bin/rpc-server": "d" * 64,
                "lib/libggml-rpc.so": "e" * 64,
            },
        },
        "models": {
            "target_gguf_shard_count": 2,
            "target_gguf_shards": [
                {"path": "Flash-00001-of-00002.gguf", "sha256": "f" * 64},
                {"path": "Flash-00002-of-00002.gguf", "sha256": "1" * 64},
            ],
            "draft_gguf_sha256": "2" * 64,
        },
    }


class FlashBenchStreamTests(unittest.TestCase):
    def test_fit_prompt_preserves_padding_across_iterations(self):
        seed = 731
        preparer = FakePromptPreparer()
        questions = {
            "prose": "Question: summarize the pattern, cite uncertain measurements, and recommend the next inspection.",
            "code": "Question: explain the parsing behavior, identify an edge case, and suggest a focused test.",
            "structured": "Question: summarize the records, separate verified facts from uncertainty, and recommend a check.",
        }
        for family in FAMILIES:
            for lane in (0, 1):
                base = _fixture_content(family, 0, seed, lane)
                target = preparer.count(_messages(base)) + 3
                messages, prepared_tokens = fit_prompt(
                    preparer, family, target, seed=seed, lane=lane,
                )
                content = messages[0]["content"]
                question = questions[family]
                prefix = base[:-len(question)]
                padding = content[len(prefix):-len(question)]

                self.assertEqual(prepared_tokens, target)
                self.assertEqual(preparer.count(messages), target)
                self.assertTrue(content.startswith(prefix))
                self.assertTrue(content.endswith(question))
                self.assertGreaterEqual(_padding_atom_count(padding), 2)

    def test_fragmented_sse_handles_crlf_utf8_and_multiple_delta_tokens(self):
        event = 'data: {"choices":[{"delta":{"reasoning_content":"café"}}]}\r\n\r\n'.encode()
        parser = SSEParser()
        events = []
        for byte in event:
            events.extend(parser.feed(bytes([byte])))
        events.extend(parser.finish())
        self.assertFalse(parser.incomplete)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, '{"choices":[{"delta":{"reasoning_content":"café"}}]}')
        self.assertEqual(events[0].raw.encode(), event)

    def test_raw_chunk_capture_preserves_invalid_utf8_bytes(self):
        chunk = b"data: {\"choices\":[]}\xff\n\n"
        raw_stream = bytearray()
        parser = SSEParser()
        _feed_raw_chunk(raw_stream, parser, chunk)
        self.assertEqual(bytes(raw_stream), chunk)
        self.assertEqual(parser.error, "invalid UTF-8 in SSE stream")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "sample.sse"
            evidence = _write_raw_sse(output, bytes(raw_stream))
            self.assertEqual(output.read_bytes(), chunk)
            self.assertEqual(evidence["raw_sse_bytes"], len(chunk))
            self.assertEqual(evidence["raw_sse_file"], output.as_posix())

    def test_final_usage_duplicate_is_cumulative_not_added_and_role_is_not_ttft(self):
        result, events, _ = _summarize(_valid_stream())
        self.assertTrue(result["valid"], result["invalid_reasons"])
        self.assertEqual(result["predicted_n"], 3)
        self.assertEqual(result["usage_completion_tokens"], 3)
        self.assertEqual(result["sse_events"], 7)
        self.assertEqual(result["reasoning_delta_events"], 1)
        self.assertEqual(result["content_delta_events"], 1)
        self.assertEqual(result["text_delta_events"], 2)
        self.assertEqual(result["draft_n"], 2)
        self.assertEqual(result["draft_n_accepted"], 1)
        self.assertEqual(result["pp_tps"], 800.0)
        self.assertEqual(result["tg_tps"], 20.0)
        role_ns = events[0].received_ns
        first_text_ns = events[1].received_ns
        self.assertEqual(result["ttft_ms"], (first_text_ns - role_ns + 1_000_000) / 1_000_000)

    def test_final_usage_may_advance_after_last_token_timing(self):
        result, _, _ = _summarize(_valid_stream(content_predicted_n=2))
        self.assertTrue(result["valid"], result["invalid_reasons"])
        self.assertEqual(result["predicted_n"], 3)

    def test_conflicting_timing_after_final_usage_invalidates_stream(self):
        chunks = _valid_stream()
        chunks.insert(-1, _payload({
            "choices": [{"delta": {}, "finish_reason": None}],
            "timings": _timings(predicted_n=2),
        }))
        result, _, _ = _summarize(chunks)
        self.assertFalse(result["valid"])
        self.assertIn("native timings changed after final usage", result["invalid_reasons"])

    def test_conflicting_final_usage_duplicate_invalidates_stream(self):
        chunks = _valid_stream()
        duplicate = json.loads(chunks[-2].decode().split("data: ", 1)[1])
        duplicate["usage"]["completion_tokens"] = 2
        duplicate["timings"]["predicted_n"] = 2
        chunks[-2] = _payload(duplicate)
        result, _, _ = _summarize(chunks)
        self.assertFalse(result["valid"])
        self.assertIn("conflicting final usage chunks", result["invalid_reasons"])

    def test_request_file_sampling_is_preserved_without_seed_fallback(self):
        request_file = {
            "model": "flash",
            "messages": [{"role": "user", "content": "fixture"}],
            "max_tokens": 3,
            "temperature": 0.25,
            "cache_prompt": False,
        }
        body = _completion_body(
            "flash", request_file["messages"], 3, None, False, {}, request_file,
        )
        self.assertNotIn("seed", body)
        self.assertEqual(body["temperature"], 0.25)
        self.assertEqual(body["max_tokens"], 3)
        self.assertEqual(body["messages"], request_file["messages"])
        self.assertTrue(body["stream"])
        self.assertTrue(body["timings_per_token"])
        self.assertTrue(body["stream_options"]["include_usage"])
        evidence = _request_evidence(body, {"default_sampling": {"top_p": 0.9}})
        self.assertFalse(evidence["seed_present"])
        self.assertFalse(evidence["deterministic_replay"])
        self.assertIn("seed", evidence["missing_determinism_fields"])

    def test_environment_identity_is_reported_or_unknown(self):
        props = {
            "build_info": "b1-abc",
            "total_slots": 2,
            "default_generation_settings": {"n_ctx": 131072, "params": {"top_p": 0.95}},
        }
        with patch("scripts.bench_flash_next._get_json", return_value=props):
            before = _native_props_evidence("http://native", 1, None)
        self.assertEqual(before["status"], "available")
        self.assertEqual(before["build_info"], "b1-abc")
        self.assertEqual(before["total_slots"], 2)
        self.assertEqual(before["n_ctx_per_slot"], 131072)

        with patch("scripts.bench_flash_next._get_json", side_effect=OSError("unavailable")):
            native_unknown = _native_props_evidence("http://native", 1, None)
            proxy_unknown = _proxy_release_evidence("http://proxy", 1, None)
        self.assertEqual(native_unknown["status"], "unknown")
        self.assertEqual(proxy_unknown["status"], "unknown")
        comparison = _compare_environment(
            {"native_props": native_unknown, "proxy_release": proxy_unknown},
            {"native_props": native_unknown, "proxy_release": proxy_unknown},
        )
        self.assertEqual(comparison["overall"], "unknown")

    def test_optional_evidence_manifest_marks_partial_and_complete_inputs(self):
        complete = _complete_evidence_manifest()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evidence.json"
            path.write_text(json.dumps(complete), encoding="utf-8")
            evidence = _load_evidence_manifest(path)
            self.assertEqual(evidence["status"], "complete")
            self.assertTrue(evidence["acceptance_eligible"])
            self.assertEqual(evidence["recognized_immutable_ids"]["host_source_id"], "host-rev")
            self.assertEqual(evidence["recognized_immutable_ids"]["rpc_source_id"], "rpc-rev")
            partial_manifest = _complete_evidence_manifest()
            partial_manifest["rpc"].pop("runtime_library_paths")
            path.write_text(json.dumps(partial_manifest), encoding="utf-8")
            partial = _load_evidence_manifest(path)
            incomplete_shards = _complete_evidence_manifest()
            incomplete_shards["models"]["target_gguf_shards"].pop()
            path.write_text(json.dumps(incomplete_shards), encoding="utf-8")
            missing_shard = _load_evidence_manifest(path)
        self.assertEqual(partial["status"], "partial")
        self.assertFalse(partial["acceptance_eligible"])
        self.assertIn("rpc.runtime_library_paths", partial["missing_fields"])
        self.assertEqual(missing_shard["status"], "partial")
        self.assertFalse(missing_shard["acceptance_eligible"])
        self.assertIn("models.target_gguf_shard_count does not match target_gguf_shards length",
                      missing_shard["invalid_fields"])
        unknown = _load_evidence_manifest(None)
        self.assertEqual(unknown["status"], "unknown")
        self.assertFalse(unknown["acceptance_eligible"])

    def test_campaign_acceptance_requires_complete_evidence_and_stable_environment(self):
        complete_evidence = {"acceptance_eligible": True}
        unchanged = {"overall": "unchanged"}
        changed = {"overall": "changed"}
        unknown = {"overall": "unknown"}
        self.assertEqual(_campaign_outcome(0, complete_evidence, unchanged), {
            "status": "complete", "acceptance_eligible": True,
            "acceptance_blockers": [], "exit_code": 0,
        })
        self.assertEqual(_campaign_outcome(0, complete_evidence, changed), {
            "status": "invalidated_environment_change", "acceptance_eligible": False,
            "acceptance_blockers": ["pre/post runtime identity is changed"], "exit_code": 1,
        })
        self.assertEqual(_campaign_outcome(0, complete_evidence, unknown), {
            "status": "screening_only", "acceptance_eligible": False,
            "acceptance_blockers": ["pre/post runtime identity is unknown"], "exit_code": 0,
        })
        self.assertEqual(_campaign_outcome(1, complete_evidence, unchanged), {
            "status": "invalid_samples", "acceptance_eligible": False,
            "acceptance_blockers": ["1 measured sample(s) are invalid"], "exit_code": 1,
        })

    def test_environment_change_is_detected(self):
        before = {
            "native_props": {
                "status": "available", "build_info": "b1", "model_alias": "flash",
                "total_slots": 2, "n_ctx_per_slot": 131072,
            },
            "proxy_release": {"status": "available", "fields": {"release": "r1"}},
        }
        after = {
            "native_props": {
                "status": "available", "build_info": "b2", "model_alias": "flash",
                "total_slots": 2, "n_ctx_per_slot": 131072,
            },
            "proxy_release": {"status": "available", "fields": {"release": "r2"}},
        }
        comparison = _compare_environment(before, after)
        self.assertEqual(comparison["overall"], "changed")
        self.assertEqual(comparison["native_props"], "changed")
        self.assertEqual(comparison["proxy_release"], "changed")

    def test_zero_timing_denominator_invalidates_rate(self):
        result, _, _ = _summarize(_valid_stream(prompt_ms=0.0), prompt_ms=0.0)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["pp_tps"])
        self.assertIn("prompt_ms is zero; prefill rate is undefined", result["invalid_reasons"])

        generation, _, _ = _summarize(_valid_stream(predicted_ms=0.0))
        self.assertFalse(generation["valid"])
        self.assertIsNone(generation["tg_tps"])
        self.assertIn("predicted_ms is zero; generation rate is undefined", generation["invalid_reasons"])

    def test_malformed_and_disconnected_streams_are_invalid(self):
        malformed, _, _ = _summarize([b"data: not-json\n\n", b"data: [DONE]\n\n"])
        self.assertFalse(malformed["valid"])
        self.assertTrue(any("malformed SSE JSON" in reason for reason in malformed["invalid_reasons"]))

        parser = SSEParser()
        events = parser.feed(b'data: {"choices":[]')
        events.extend(parser.finish())
        disconnected = summarize_stream(
            events,
            started_ns=0,
            ended_ns=1,
            prepared_prompt_tokens=8,
            expected_output_tokens=3,
            cache_mode="cold",
            transport_error="connection reset",
            parser_incomplete=parser.incomplete,
        )
        self.assertFalse(disconnected["valid"])
        self.assertTrue(any("connection reset" in reason for reason in disconnected["invalid_reasons"]))
        self.assertTrue(any("incomplete SSE" in reason for reason in disconnected["invalid_reasons"]))


if __name__ == "__main__":
    unittest.main()
