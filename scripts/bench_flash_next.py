#!/usr/bin/env python3
"""Measure llama.cpp Flash Next chat streams through an explicit HTTP endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import http.client
import json
import math
import os
import re
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict


DEFAULT_MODEL = "Qwen3.8-Flash-Next"
FAMILIES = ("prose", "code", "structured")
PAD_ATOMS = (
    " a note", " more detail", " the record", " another item", " value",
    " x", " y", " z", " 0", " 1", " .", " ,", " :", " -", " /", " =",
)
TEMPLATE_OPTION_KEYS = (
    "tools", "tool_choice", "response_format", "json_schema", "grammar",
    "parallel_tool_calls", "reasoning_format", "reasoning_effort",
    "add_generation_prompt", "continue_final_message", "reasoning_budget_tokens",
    "thinking_budget_tokens", "reasoning_budget_message",
)
SAMPLING_FIELDS = (
    "seed", "temperature", "dynatemp_range", "dynatemp_exponent", "top_k", "top_p",
    "min_p", "top_n_sigma", "xtc_probability", "xtc_threshold", "typical_p",
    "repeat_last_n", "repeat_penalty", "presence_penalty", "frequency_penalty",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "dry_sequence_breakers", "mirostat", "mirostat_tau", "mirostat_eta",
    "adaptive_target", "adaptive_decay", "samplers", "ignore_eos",
)
DETERMINISM_FIELDS = (
    "temperature", "top_k", "top_p", "min_p", "typical_p", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "samplers", "ignore_eos",
)


@dataclass(frozen=True)
class SSEEvent:
    raw: str
    data: str | None
    event: str | None
    received_ns: int


class PromptLane(TypedDict):
    messages: list[dict[str, str]]
    prepared_tokens: int
    prefix_messages: list[dict[str, str]] | None
    prefix_tokens: int | None
    prefix_sha256: str | None
    prompt_sha256: str


class SSEParser:
    """Frame SSE events while preserving the original text and line endings."""

    def __init__(self) -> None:
        import codecs

        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._text = ""
        self._raw = ""
        self._data: list[str] = []
        self._event: str | None = None
        self.incomplete = False
        self.error: str | None = None

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        try:
            self._text += self._decoder.decode(chunk, final=False)
        except UnicodeDecodeError:
            self.error = "invalid UTF-8 in SSE stream"
            return []
        return self._drain(final=False)

    def finish(self) -> list[SSEEvent]:
        try:
            self._text += self._decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            self.error = "incomplete UTF-8 in SSE stream"
        events = self._drain(final=True)
        if self._text or self._raw:
            self.incomplete = True
        return events

    def _drain(self, *, final: bool) -> list[SSEEvent]:
        events = []
        while self._text:
            lf = self._text.find("\n")
            cr = self._text.find("\r")
            positions = [pos for pos in (lf, cr) if pos >= 0]
            if not positions:
                if not final:
                    break
                line, self._text = self._text, ""
                self._line(line, "", events)
                break
            pos = min(positions)
            delimiter = self._text[pos]
            if delimiter == "\r" and pos + 1 == len(self._text) and not final:
                break
            width = 2 if (
                delimiter == "\r"
                and pos + 1 < len(self._text)
                and self._text[pos + 1] == "\n"
            ) else 1
            line = self._text[:pos]
            ending = self._text[pos:pos + width]
            self._text = self._text[pos + width:]
            self._line(line, ending, events)
        return events

    def _line(self, line: str, ending: str, events: list[SSEEvent]) -> None:
        self._raw += line + ending
        if not line:
            if self._raw:
                events.append(SSEEvent(
                    raw=self._raw,
                    data="\n".join(self._data) if self._data else None,
                    event=self._event,
                    received_ns=time.monotonic_ns(),
                ))
            self._raw = ""
            self._data = []
            self._event = None
            return
        if line.startswith(":"):
            return
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event = value


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _number_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _draft_count(timings: dict[str, Any], name: str) -> int:
    value = _int_value(timings.get(name, 0))
    return value if value is not None else -1


def _timing_signature(timings: dict[str, Any]) -> str:
    return json.dumps(timings, sort_keys=True, separators=(",", ":"))


def _feed_raw_chunk(raw_stream: bytearray, parser: SSEParser, chunk: bytes) -> list[SSEEvent]:
    raw_stream.extend(chunk)
    return parser.feed(chunk)


def _write_raw_sse(path: Path, raw_stream: bytes) -> dict[str, Any]:
    path.write_bytes(raw_stream)
    return {
        "raw_sse_file": path.as_posix(),
        "raw_sse_bytes": len(raw_stream),
        "raw_sse_sha256": hashlib.sha256(raw_stream).hexdigest(),
    }


def summarize_stream(
    events: list[SSEEvent],
    *,
    started_ns: int,
    ended_ns: int,
    prepared_prompt_tokens: int,
    expected_output_tokens: int,
    cache_mode: str,
    http_status: int = 200,
    transport_error: str | None = None,
    parser_incomplete: bool = False,
    parser_error: str | None = None,
) -> dict[str, Any]:
    """Use cumulative native snapshots once; never infer tokens from frame count."""
    invalid: list[str] = []

    def fail(reason: str) -> None:
        if reason not in invalid:
            invalid.append(reason)

    if http_status != 200:
        fail(f"HTTP status {http_status}")
    if transport_error:
        fail(f"transport error: {transport_error}")
    if parser_error:
        fail(parser_error)
    if parser_incomplete:
        fail("incomplete SSE event at end of stream")

    saw_done = False
    finish_reason = None
    first_text_ns = None
    text_delta_events = 0
    reasoning_delta_events = 0
    content_delta_events = 0
    latest_timings: dict[str, Any] | None = None
    previous_counters: dict[str, float] = {}
    final_usage: dict[str, Any] | None = None
    final_usage_timings: dict[str, Any] | None = None
    final_usage_signature: str | None = None
    final_usage_timing_signature: str | None = None
    timings_after_final_usage: list[str] = []
    malformed_events = 0

    for event in events:
        if event.data is None:
            continue
        if event.data.strip() == "[DONE]":
            saw_done = True
            continue
        try:
            payload = json.loads(event.data)
        except (json.JSONDecodeError, TypeError):
            malformed_events += 1
            continue
        if not isinstance(payload, dict):
            malformed_events += 1
            continue
        timings = payload.get("timings")
        choices = payload.get("choices")
        if isinstance(timings, dict) and final_usage is not None:
            timings_after_final_usage.append(_timing_signature(timings))
        if timings is not None:
            if not isinstance(timings, dict):
                fail("malformed native timings object")
            else:
                for name in (
                    "prompt_n", "cache_n", "predicted_n", "draft_n", "draft_n_accepted",
                    "prompt_ms", "predicted_ms",
                ):
                    if name not in timings:
                        continue
                    value = (
                        _int_value(timings[name]) if name.endswith("_n")
                        else _number_value(timings[name])
                    )
                    if value is None:
                        fail(f"invalid native timing counter: {name}")
                    elif name in previous_counters and value < previous_counters[name]:
                        fail(f"native timing counter regressed: {name}")
                    else:
                        previous_counters[name] = value
                latest_timings = timings
        if not isinstance(choices, list):
            fail("malformed choices field")
            continue
        if choices == [] and isinstance(payload.get("usage"), dict):
            usage_signature = _timing_signature(payload["usage"])
            timing_signature = _timing_signature(timings) if isinstance(timings, dict) else None
            if final_usage is not None and (
                usage_signature != final_usage_signature
                or timing_signature != final_usage_timing_signature
            ):
                fail("conflicting final usage chunks")
            final_usage = payload["usage"]
            final_usage_timings = timings if isinstance(timings, dict) else None
            final_usage_signature = usage_signature
            final_usage_timing_signature = timing_signature
        for choice in choices:
            if not isinstance(choice, dict):
                fail("malformed choice")
                continue
            reason = choice.get("finish_reason")
            if isinstance(reason, str):
                finish_reason = reason
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                fail("malformed delta")
                continue
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            has_content = isinstance(content, str) and bool(content)
            has_reasoning = isinstance(reasoning, str) and bool(reasoning)
            if has_content:
                content_delta_events += 1
            if has_reasoning:
                reasoning_delta_events += 1
            if has_content or has_reasoning:
                text_delta_events += 1
                if first_text_ns is None:
                    first_text_ns = event.received_ns

    if malformed_events:
        fail(f"{malformed_events} malformed SSE JSON event(s)")
    if not saw_done:
        fail("stream ended before [DONE]")
    if finish_reason is None:
        fail("final finish_reason missing")
    if final_usage is None:
        fail("final usage chunk missing")
    if final_usage_timings is None:
        fail("final native timings missing")
    if final_usage_timing_signature is not None and any(
        signature != final_usage_timing_signature for signature in timings_after_final_usage
    ):
        fail("native timings changed after final usage")
    if latest_timings is None:
        fail("native timings missing")
    if first_text_ns is None:
        fail("first content or reasoning delta missing")

    timings = final_usage_timings or latest_timings or {}
    prompt_n = _int_value(timings.get("prompt_n"))
    cache_n = _int_value(timings.get("cache_n"))
    predicted_n = _int_value(timings.get("predicted_n"))
    draft_n = _draft_count(timings, "draft_n")
    draft_n_accepted = _draft_count(timings, "draft_n_accepted")
    prompt_ms = _number_value(timings.get("prompt_ms"))
    predicted_ms = _number_value(timings.get("predicted_ms"))
    if prompt_n is None:
        fail("final prompt_n missing or invalid")
    if cache_n is None:
        fail("final cache_n missing or invalid")
    if predicted_n is None:
        fail("final predicted_n missing or invalid")
    if prompt_ms is None:
        fail("final prompt_ms missing or invalid")
    if predicted_ms is None:
        fail("final predicted_ms missing or invalid")
    if draft_n < 0 or draft_n_accepted < 0:
        fail("invalid MTP draft counters")
    if draft_n_accepted > draft_n:
        fail("MTP accepted count exceeds draft count")
    if final_usage_timings is not None and (
        ("draft_n" in final_usage_timings) != ("draft_n_accepted" in final_usage_timings)
    ):
        fail("final native MTP draft counters are incomplete")
    if cache_mode != "any" and draft_n == 0:
        fail("native MTP draft counters are zero")

    usage_prompt = _int_value(final_usage.get("prompt_tokens")) if final_usage else None
    usage_output = _int_value(final_usage.get("completion_tokens")) if final_usage else None
    details = final_usage.get("prompt_tokens_details", {}) if final_usage else {}
    usage_cached = _int_value(details.get("cached_tokens")) if isinstance(details, dict) else None
    if usage_prompt is None:
        fail("final usage prompt_tokens missing or invalid")
    if usage_output is None:
        fail("final usage completion_tokens missing or invalid")
    if usage_cached is None:
        fail("final usage cached_tokens missing or invalid")

    logical_prompt = prompt_n + cache_n if prompt_n is not None and cache_n is not None else None
    if logical_prompt is not None and logical_prompt != prepared_prompt_tokens:
        fail(f"prepared prompt mismatch: native total {logical_prompt}, expected {prepared_prompt_tokens}")
    if logical_prompt is not None and usage_prompt is not None and usage_prompt != logical_prompt:
        fail("native timings and usage prompt counters disagree")
    if cache_n is not None and usage_cached is not None and cache_n != usage_cached:
        fail("native timing and usage cache counters disagree")
    if predicted_n is not None and usage_output is not None and predicted_n != usage_output:
        fail("native timing and usage output counters disagree")
    if predicted_n is not None and predicted_n != expected_output_tokens:
        fail(f"native predicted_n {predicted_n}, expected {expected_output_tokens}")
    if cache_mode == "cold" and cache_n not in (None, 0):
        fail(f"cold request reused {cache_n} prompt tokens")
    if cache_mode == "cached-prefix" and cache_n in (None, 0):
        fail("cached-prefix request reused no prompt tokens")
    if cache_mode != "any":
        if prompt_n == 0:
            fail("prompt_n is zero; prefill rate is undefined")
        if prompt_ms == 0:
            fail("prompt_ms is zero; prefill rate is undefined")
        if predicted_ms == 0:
            fail("predicted_ms is zero; generation rate is undefined")
        if predicted_n is not None and predicted_n <= 1:
            fail("predicted_n has no decode steps after the first token")

    pp_tps = (
        prompt_n * 1000.0 / prompt_ms
        if cache_mode != "any" and prompt_n and prompt_ms else None
    )
    tg_tps = (
        (predicted_n - 1) * 1000.0 / predicted_ms
        if cache_mode != "any" and predicted_n and predicted_n > 1 and predicted_ms else None
    )
    ttft_ms = (first_text_ns - started_ns) / 1_000_000 if first_text_ns is not None else None
    total_ms = (ended_ns - started_ns) / 1_000_000
    return {
        "valid": not invalid,
        "invalid_reasons": invalid,
        "http_status": http_status,
        "prepared_prompt_tokens": prepared_prompt_tokens,
        "prompt_n": prompt_n,
        "cache_n": cache_n,
        "logical_prompt_tokens": logical_prompt,
        "predicted_n": predicted_n,
        "draft_n": draft_n if draft_n >= 0 else None,
        "draft_n_accepted": draft_n_accepted if draft_n_accepted >= 0 else None,
        "draft_acceptance": draft_n_accepted / draft_n if draft_n > 0 and draft_n_accepted >= 0 else None,
        "usage_prompt_tokens": usage_prompt,
        "usage_cached_tokens": usage_cached,
        "usage_completion_tokens": usage_output,
        "finish_reason": finish_reason,
        "sse_events": len(events),
        "text_delta_events": text_delta_events,
        "reasoning_delta_events": reasoning_delta_events,
        "content_delta_events": content_delta_events,
        "prompt_ms": prompt_ms,
        "predicted_ms": predicted_ms,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
    }


def _validated_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise argparse.ArgumentTypeError("URL must use http or https and include a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("URL must not contain credentials, query, or fragment")
    return value.rstrip("/")


def _post_json(url: str, payload: dict[str, Any], timeout: float, api_key: str | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(4 * 1024 * 1024 + 1)
    if len(body) > 4 * 1024 * 1024:
        raise ValueError(f"JSON response from {url} exceeded 4 MiB")
    result = json.loads(body)
    if not isinstance(result, dict):
        raise ValueError(f"JSON response from {url} was not an object")
    return result


def _get_json(url: str, timeout: float, api_key: str | None) -> Any:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(4 * 1024 * 1024 + 1)
    if len(body) > 4 * 1024 * 1024:
        raise ValueError(f"JSON response from {url} exceeded 4 MiB")
    return json.loads(body)


def _native_props_evidence(base_url: str, timeout: float, api_key: str | None) -> dict[str, Any]:
    try:
        props = _get_json(f"{base_url}/props", timeout, api_key)
    except Exception as exc:
        return {"status": "unknown", "reason": f"{type(exc).__name__}: {exc}"}
    if not isinstance(props, dict):
        return {"status": "unknown", "reason": "/props did not return a JSON object"}
    default_settings = props.get("default_generation_settings")
    default_settings = default_settings if isinstance(default_settings, dict) else {}
    params = default_settings.get("params")
    params = params if isinstance(params, dict) else {}
    evidence = {
        "build_info": props.get("build_info"),
        "model_alias": props.get("model_alias"),
        "total_slots": props.get("total_slots"),
        "n_ctx_per_slot": default_settings.get("n_ctx"),
        "default_sampling": {key: params[key] for key in SAMPLING_FIELDS if key in params},
    }
    missing = [key for key in ("build_info", "total_slots", "n_ctx_per_slot") if evidence[key] is None]
    evidence["status"] = "partial" if missing else "available"
    if missing:
        evidence["missing_fields"] = missing
    return evidence


def _release_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    keys = {
        "active_release", "build_info", "commit", "current_release", "release",
        "release_id", "revision", "source_revision", "version",
    }
    found: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key.lower() in keys and isinstance(item, (str, int, float)) and not isinstance(item, bool):
                found[path] = item
            elif isinstance(item, (dict, list)):
                found.update(_release_fields(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_release_fields(item, f"{prefix}[{index}]"))
    return found


def _proxy_release_evidence(base_url: str, timeout: float, api_key: str | None) -> dict[str, Any]:
    try:
        response = _get_json(f"{base_url}/dashboard/api/restart", timeout, api_key)
    except Exception as exc:
        return {"status": "unknown", "reason": f"{type(exc).__name__}: {exc}"}
    fields = _release_fields(response)
    if not fields:
        return {
            "status": "unknown",
            "reason": "restart endpoint did not report a release identity",
        }
    return {"status": "available", "fields": fields}


def _capture_environment(
    base_url: str, prepare_url: str, timeout: float, api_key: str | None,
) -> dict[str, Any]:
    return {
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "native_props": _native_props_evidence(prepare_url, timeout, api_key),
        "proxy_release": _proxy_release_evidence(base_url, timeout, api_key),
    }


def _compare_environment(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    native_before = before.get("native_props", {})
    native_after = after.get("native_props", {})
    native_fields = ("build_info", "model_alias", "total_slots", "n_ctx_per_slot")
    native_differences = {
        key: [native_before.get(key), native_after.get(key)]
        for key in native_fields
        if native_before.get(key) is not None
        and native_after.get(key) is not None
        and native_before.get(key) != native_after.get(key)
    }
    native_compared = any(
        native_before.get(key) is not None and native_after.get(key) is not None
        for key in native_fields
    )
    if native_differences:
        native_status = "changed"
    else:
        native_status = "unchanged" if (
            native_before.get("status") == "available"
            and native_after.get("status") == "available"
        ) else ("partial" if native_compared else "unknown")

    proxy_before = before.get("proxy_release", {})
    proxy_after = after.get("proxy_release", {})
    proxy_fields_before = proxy_before.get("fields")
    proxy_fields_after = proxy_after.get("fields")
    if isinstance(proxy_fields_before, dict) and isinstance(proxy_fields_after, dict):
        proxy_status = "unchanged" if proxy_fields_before == proxy_fields_after else "changed"
    else:
        proxy_status = "unknown"

    statuses = (native_status, proxy_status)
    overall = "changed" if "changed" in statuses else (
        "unchanged" if all(status == "unchanged" for status in statuses) else "unknown"
    )
    return {
        "native_props": native_status,
        "native_differences": native_differences,
        "proxy_release": proxy_status,
        "overall": overall,
    }


def _load_evidence_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "unknown",
            "acceptance_eligible": False,
            "reason": "no evidence manifest supplied",
        }
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "unknown",
            "acceptance_eligible": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(manifest, dict):
        return {
            "status": "unknown",
            "acceptance_eligible": False,
            "reason": "evidence manifest must be a JSON object",
        }

    digest_re = re.compile(r"^[0-9a-fA-F]{64}$")
    missing: list[str] = []
    invalid: list[str] = []

    def require_backend(name: str) -> dict[str, Any]:
        backend = manifest.get(name)
        if not isinstance(backend, dict):
            missing.append(f"{name}")
            return {}
        source_id = backend.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            missing.append(f"{name}.source_id")
        files = backend.get("files")
        if not isinstance(files, dict) or not files:
            missing.append(f"{name}.files")
            files = {}
        for file_path, file_hash in files.items():
            if not isinstance(file_path, str) or not file_path.strip():
                invalid.append(f"{name}.files contains an empty path")
            if not isinstance(file_hash, str) or digest_re.fullmatch(file_hash) is None:
                invalid.append(f"{name}.files[{file_path!r}] must be a bare SHA-256")
        binary_path = backend.get("binary_path")
        if not isinstance(binary_path, str) or not binary_path.strip():
            missing.append(f"{name}.binary_path")
        elif binary_path not in files:
            missing.append(f"{name}.files[{binary_path}]")
        elif re.search(r"\.(?:so(?:\.\d+)*|dll|dylib)$", binary_path, re.IGNORECASE):
            invalid.append(f"{name}.binary_path must identify an executable, not a shared library")
        libraries = backend.get("runtime_library_paths")
        if not isinstance(libraries, list) or not libraries:
            missing.append(f"{name}.runtime_library_paths")
        else:
            for library_path in libraries:
                if not isinstance(library_path, str) or library_path not in files:
                    missing.append(f"{name}.files[{library_path}]")
                elif re.search(r"\.(?:so(?:\.\d+)*|dll|dylib)$", library_path, re.IGNORECASE) is None:
                    invalid.append(f"{name}.runtime_library_paths entry is not a shared library: {library_path}")
        return backend

    host = require_backend("host")
    rpc = require_backend("rpc")
    models = manifest.get("models")
    if not isinstance(models, dict):
        missing.append("models")
        models = {}

    shard_count_value = _int_value(models.get("target_gguf_shard_count"))
    if shard_count_value is None or shard_count_value < 1:
        missing.append("models.target_gguf_shard_count")
        shard_count = 0
    else:
        shard_count = shard_count_value
    shards = models.get("target_gguf_shards")
    if not isinstance(shards, list) or not shards:
        missing.append("models.target_gguf_shards")
        shards = []
    if shard_count and len(shards) != shard_count:
        invalid.append("models.target_gguf_shard_count does not match target_gguf_shards length")
    shard_paths: list[str] = []
    numbered_shards: list[tuple[int, int]] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            invalid.append(f"models.target_gguf_shards[{index}] must be an object")
            continue
        shard_path = shard.get("path")
        shard_hash = shard.get("sha256")
        if not isinstance(shard_path, str) or not shard_path.lower().endswith(".gguf"):
            invalid.append(f"models.target_gguf_shards[{index}].path must end in .gguf")
            continue
        if shard_path in shard_paths:
            invalid.append(f"duplicate target GGUF shard path: {shard_path}")
        shard_paths.append(shard_path)
        if not isinstance(shard_hash, str) or digest_re.fullmatch(shard_hash) is None:
            invalid.append(f"models.target_gguf_shards[{index}].sha256 must be a bare SHA-256")
        match = re.search(r"(\d+)-of-(\d+)\.gguf$", shard_path, re.IGNORECASE)
        if match:
            numbered_shards.append((int(match.group(1)), int(match.group(2))))
    if shard_count > 1:
        if len(numbered_shards) != shard_count:
            invalid.append("multi-shard target filenames must declare each index with NN-of-NN.gguf")
        elif (
            {total for _, total in numbered_shards} != {shard_count}
            or {index for index, _ in numbered_shards} != set(range(1, shard_count + 1))
        ):
            invalid.append("target GGUF shard indices are incomplete or inconsistent")
    elif numbered_shards and numbered_shards != [(1, 1)]:
        invalid.append("single target GGUF shard must declare 01-of-01 when numbered")

    draft_hash = models.get("draft_gguf_sha256")
    if not isinstance(draft_hash, str) or digest_re.fullmatch(draft_hash) is None:
        invalid.append("models.draft_gguf_sha256 must be a bare SHA-256")

    identities = {
        "host_source_id": host.get("source_id"),
        "rpc_source_id": rpc.get("source_id"),
        "target_gguf_shards": shards,
        "draft_gguf_sha256": draft_hash,
    }
    complete = not missing and not invalid
    return {
        "status": "complete" if complete else "partial",
        "acceptance_eligible": complete,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "recognized_immutable_ids": identities,
        "missing_fields": missing,
        "invalid_fields": invalid,
        "manifest": manifest,
    }


def _campaign_outcome(
    invalid_samples: int,
    evidence: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    environment_status = environment.get("overall", "unknown")
    blockers = []
    if evidence.get("acceptance_eligible") is not True:
        blockers.append("complete immutable host/RPC/model evidence is required")
    if environment_status != "unchanged":
        blockers.append(f"pre/post runtime identity is {environment_status}")
    if invalid_samples:
        blockers.append(f"{invalid_samples} measured sample(s) are invalid")
    if environment_status == "changed":
        return {
            "status": "invalidated_environment_change",
            "acceptance_eligible": False,
            "acceptance_blockers": blockers,
            "exit_code": 1,
        }
    if invalid_samples:
        return {
            "status": "invalid_samples",
            "acceptance_eligible": False,
            "acceptance_blockers": blockers,
            "exit_code": 1,
        }
    eligible = evidence.get("acceptance_eligible") is True and environment_status == "unchanged"
    if not eligible:
        return {
            "status": "screening_only",
            "acceptance_eligible": False,
            "acceptance_blockers": blockers,
            "exit_code": 0,
        }
    return {
        "status": "complete",
        "acceptance_eligible": True,
        "acceptance_blockers": [],
        "exit_code": 0,
    }


def _sampling_evidence(body: dict[str, Any], native_props: dict[str, Any]) -> dict[str, Any]:
    sent = {key: body[key] for key in SAMPLING_FIELDS if key in body}
    defaults = native_props.get("default_sampling", {})
    defaults = defaults if isinstance(defaults, dict) else {}
    seed_present = _int_value(body.get("seed")) is not None
    effective = {key: body[key] if key in body else defaults[key]
                 for key in DETERMINISM_FIELDS if key in body or key in defaults}
    unresolved = [key for key in DETERMINISM_FIELDS if key not in effective]
    if not seed_present:
        unresolved.insert(0, "seed")
    return {
        "sent": sent,
        "seed_present": seed_present,
        "missing_determinism_fields": unresolved,
        "resolved_from_native_defaults": {
            key: defaults[key] for key in DETERMINISM_FIELDS if key not in body and key in defaults
        },
        "deterministic_replay": seed_present and not unresolved,
    }


def _request_evidence(body: dict[str, Any], native_props: dict[str, Any]) -> dict[str, Any]:
    sampling = _sampling_evidence(body, native_props)
    return {
        "model": body.get("model"),
        "sampling_fields_sent": sampling["sent"],
        "seed_present": sampling["seed_present"],
        "missing_determinism_fields": sampling["missing_determinism_fields"],
        "resolved_from_native_defaults": sampling["resolved_from_native_defaults"],
        "deterministic_replay": sampling["deterministic_replay"],
        "max_tokens": body.get("max_tokens"),
        "n_predict": body.get("n_predict"),
        "cache_prompt": body.get("cache_prompt"),
        "stream": body.get("stream"),
        "timings_per_token": body.get("timings_per_token"),
        "stream_options": body.get("stream_options"),
        "chat_template_kwargs": body.get("chat_template_kwargs"),
    }


class PromptPreparer:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        api_key: str | None,
        template_kwargs: dict[str, Any] | None,
        template_options: dict[str, Any] | None = None,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.api_key = api_key
        self.template_kwargs = template_kwargs
        self.template_options = template_options or {}
        self.cache: dict[str, int] = {}

    def count(self, messages: list[dict[str, str]]) -> int:
        key = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        if key in self.cache:
            return self.cache[key]
        template_body = {"messages": messages, **self.template_options}
        if self.template_kwargs is not None:
            template_body["chat_template_kwargs"] = self.template_kwargs
        rendered = _post_json(
            f"{self.base_url}/apply-template", template_body, self.timeout, self.api_key,
        )
        prompt = rendered.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("/apply-template response did not contain a string prompt")
        tokenized = _post_json(
            f"{self.base_url}/tokenize",
            {"content": prompt, "add_special": True, "parse_special": True},
            self.timeout,
            self.api_key,
        )
        tokens = tokenized.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("/tokenize response did not contain a token array")
        self.cache[key] = len(tokens)
        return len(tokens)


def _messages(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def _fixture_line(family: str, index: int, rng: random.Random) -> str:
    station = rng.choice(("north", "south", "east", "west", "central"))
    status = rng.choice(("verified", "pending review", "within range", "flagged"))
    reading = f"{rng.uniform(-18, 92):.2f}"
    if family == "prose":
        templates = (
            "At {station} station, sample {index:04d} measured {reading} units and was marked {status}.",
            "The {station} notebook says sample {index:04d} measured {reading} units; its status is {status}.",
            "During cycle {index:04d}, the {station} team logged {reading} units and noted {status}.",
        )
        return rng.choice(templates).format(
            station=station, index=index, reading=reading, status=status,
        )
    if family == "code":
        field = f"reading_{index:04d}"
        return (
            f"def parse_{station}_{index:04d}(record):\n"
            f"    raw = record.get({field!r})\n"
            "    value = float(raw) if raw is not None else None\n"
            f"    return {{'station': {station!r}, 'value': value, 'status': {status!r}}}"
        )
    return json.dumps({
        "record": f"R{index:06d}",
        "station": station,
        "reading": float(reading),
        "unit": rng.choice(("C", "kPa", "ppm")),
        "status": status,
        "tags": rng.sample(("daily", "sensor", "calibrated", "manual", "northbound"), 2),
    }, sort_keys=True, separators=(",", ":"))


def _fixture_content(family: str, rows: int, seed: int, lane: int, padding: str = "") -> str:
    rng = random.Random(seed + lane * 100_003)
    header = f"Benchmark fixture lane {lane:02d}, seed {seed:08d}."
    lines = [_fixture_line(family, index, rng) for index in range(rows)]
    questions = {
        "prose": "Question: summarize the pattern, cite uncertain measurements, and recommend the next inspection.",
        "code": "Question: explain the parsing behavior, identify an edge case, and suggest a focused test.",
        "structured": "Question: summarize the records, separate verified facts from uncertainty, and recommend a check.",
    }
    return header + "\n" + "\n".join(lines) + "\n\n" + padding + questions[family]


def fit_prompt(
    preparer: PromptPreparer,
    family: str,
    target: int,
    *,
    seed: int,
    lane: int,
) -> tuple[list[dict[str, str]], int]:
    def count_rows(rows: int, padding: str = "") -> int:
        return preparer.count(_messages(_fixture_content(family, rows, seed, lane, padding)))

    empty_count = count_rows(0)
    if empty_count >= target:
        raise ValueError(f"chat template alone uses {empty_count} tokens, target is {target}")
    low, high = 0, 1
    high_count = count_rows(high)
    while high_count < target:
        low, high = high, high * 2
        if high > target * 4:
            raise ValueError("could not build prompt within the bounded fixture size")
        high_count = count_rows(high)
    while high - low > 1:
        middle = (low + high) // 2
        if count_rows(middle) <= target:
            low = middle
        else:
            high = middle
    padding = ""
    content = _fixture_content(family, low, seed, lane)
    current = count_rows(low)
    if current == target:
        return _messages(content), current

    for _ in range(128):
        best_content = None
        best_count = current
        for atom in PAD_ATOMS:
            candidate_padding = padding + atom
            candidate = _fixture_content(family, low, seed, lane, candidate_padding)
            measured = preparer.count(_messages(candidate))
            if measured == target:
                return _messages(candidate), measured
            if current < measured <= target and measured > best_count:
                best_content, best_count = candidate, measured
        if best_content is None:
            break
        content, current = best_content, best_count
        padding = content[len(_fixture_content(family, low, seed, lane)):]
    raise ValueError(f"fixture could not reach exactly {target} tokens (nearest lower count {current})")


def _cached_prefix(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    content = messages[0]["content"]
    cutoff = content.rfind("\n", 0, int(len(content) * 0.75))
    if cutoff <= 0:
        raise ValueError("prompt is too short to make a cached prefix")
    prefix_messages = _messages(content[:cutoff + 1])
    return prefix_messages, len(prefix_messages[0]["content"])


def _completion_body(
    model: str,
    messages: list[dict[str, str]],
    output_tokens: int,
    seed: Any,
    cache_prompt: bool,
    template_kwargs: dict[str, Any],
    request_template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = dict(request_template or {})
    if request_template is None:
        body.update({"model": model, "messages": messages})
    if request_template is None or "chat_template_kwargs" in request_template:
        body["chat_template_kwargs"] = template_kwargs
    body["stream"] = True
    stream_options = body.get("stream_options")
    if not isinstance(stream_options, dict):
        stream_options = {}
    body["stream_options"] = {**stream_options, "include_usage": True}
    body["timings_per_token"] = True
    if request_template is None:
        if seed is None:
            raise ValueError("generated fixtures require an explicit seed")
        body.update({
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
            "ignore_eos": True,
            "cache_prompt": cache_prompt,
        })
    elif body.get("cache_prompt") is not cache_prompt:
        raise ValueError("request-file cache_prompt must explicitly match the selected benchmark mode")
    return body


def _run_request(
    base_url: str,
    body: dict[str, Any],
    *,
    timeout: float,
    api_key: str | None,
    sample_id: str,
    stage: str,
    expected_prompt_tokens: int,
    expected_output_tokens: int,
    cache_mode: str,
    barrier: threading.Barrier | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    if barrier:
        barrier.wait()
    started_ns = time.monotonic_ns()
    parser = SSEParser()
    events: list[SSEEvent] = []
    raw_stream = bytearray()
    status = 0
    transport_error = None
    error_body = None
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for chunk in response:
                events.extend(_feed_raw_chunk(raw_stream, parser, chunk))
            events.extend(parser.finish())
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            error_body = exc.read(16 * 1024).decode("utf-8", errors="replace")
        except OSError:
            error_body = None
        transport_error = f"HTTP error response: {exc.reason}"
        events.extend(parser.finish())
    except (http.client.HTTPException, OSError, TimeoutError, ValueError) as exc:
        transport_error = f"{type(exc).__name__}: {exc}"
        events.extend(parser.finish())
    ended_ns = time.monotonic_ns()
    metrics = summarize_stream(
        events,
        started_ns=started_ns,
        ended_ns=ended_ns,
        prepared_prompt_tokens=expected_prompt_tokens,
        expected_output_tokens=expected_output_tokens,
        cache_mode=cache_mode,
        http_status=status,
        transport_error=transport_error,
        parser_incomplete=parser.incomplete,
        parser_error=parser.error,
    )
    record = {
        "sample_id": sample_id,
        "stage": stage,
        "started_ns": started_ns,
        "ended_ns": ended_ns,
        **metrics,
    }
    if error_body:
        record["error_body"] = error_body
    raw_events = [
        {
            "sample_id": sample_id,
            "stage": stage,
            "sequence": index,
            "monotonic_ns": event.received_ns,
            "elapsed_ms": (event.received_ns - started_ns) / 1_000_000,
            "event": event.event,
            "data": event.data,
            "raw_sse": event.raw,
        }
        for index, event in enumerate(events)
    ]
    return record, raw_events, bytes(raw_stream)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, type=_validated_base_url,
                        help="explicit chat-completions endpoint base URL; no production default")
    parser.add_argument("--prepare-url", required=True, type=_validated_base_url,
                        help="explicit native server base URL exposing /apply-template and /tokenize")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--family", choices=FAMILIES, default="prose")
    parser.add_argument("--request-file", type=Path,
                        help="use a prepared chat body instead of generating fixture content")
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--mode", choices=("cold", "cached-prefix"), default="cold")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--campaign-id", help="stable identifier shared by paired A/A or A/B runs")
    parser.add_argument("--arm", help="pair label such as baseline or candidate")
    parser.add_argument("--evidence-manifest", type=Path,
                        help="optional immutable host/RPC source and model-hash manifest JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chat-template-kwargs", default='{"enable_thinking":true}')
    parser.add_argument("--api-key-env", help="environment variable with a bearer token")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prompt_tokens < 1 or (args.output_tokens is not None and args.output_tokens < 1) or args.repetitions < 1:
        raise SystemExit("prompt, output, and repetition counts must be positive")
    if args.concurrency < 1 or args.concurrency > 16:
        raise SystemExit("concurrency must be between 1 and 16")
    if args.timeout <= 0:
        raise SystemExit("timeout must be positive")
    request_template = None
    if args.request_file:
        try:
            request_template = json.loads(args.request_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"could not read request file: {exc}") from exc
        if not isinstance(request_template, dict) or not isinstance(request_template.get("messages"), list):
            raise SystemExit("request file must be a JSON object with a messages array")
        if args.concurrency > 1 or args.mode == "cached-prefix":
            raise SystemExit("--request-file supports cold, single-request runs; use generated fixtures otherwise")
        if not isinstance(request_template.get("model"), str) or not request_template["model"]:
            raise SystemExit("request file must contain the model value it will send")
        if "cache_prompt" not in request_template or request_template["cache_prompt"] is not False:
            raise SystemExit("cold request files must explicitly set cache_prompt to false")
        if args.seed is not None:
            raise SystemExit("--seed cannot override a request file; preserve its exact sampling fields")
    try:
        template_kwargs = (
            request_template.get("chat_template_kwargs", {})
            if request_template is not None
            else json.loads(args.chat_template_kwargs)
        )
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --chat-template-kwargs JSON: {exc}") from exc
    if not isinstance(template_kwargs, dict):
        raise SystemExit("chat_template_kwargs must be a JSON object")
    template_kwargs_present = request_template is None or "chat_template_kwargs" in request_template
    if request_template is not None and not template_kwargs_present:
        template_kwargs = {}
    template_options = {}
    if request_template is not None:
        template_options = {
            key: request_template[key]
            for key in TEMPLATE_OPTION_KEYS
            if key in request_template
        }
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise SystemExit(f"environment variable {args.api_key_env} is not set")

    model = request_template["model"] if request_template is not None else args.model
    if request_template is not None:
        native_output_limit = _int_value(
            request_template.get("n_predict", request_template.get("max_tokens"))
        )
        if native_output_limit is None or native_output_limit < 1:
            raise SystemExit("request file must contain a positive max_tokens or n_predict limit")
        if args.output_tokens is not None and args.output_tokens != native_output_limit:
            raise SystemExit("--output-tokens must match the request-file output limit")
        output_tokens = native_output_limit
        seed = request_template.get("seed")
    else:
        output_tokens = args.output_tokens if args.output_tokens is not None else 512
        seed = args.seed if args.seed is not None else 1
    campaign_id = args.campaign_id or str(uuid.uuid4())

    output_dir = args.output_dir or Path(
        "bench_flash_next_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "benchmark": "llama.cpp Flash Next proxy stream",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": args.label,
        "campaign_id": campaign_id,
        "arm": args.arm,
        "base_url": args.base_url,
        "prepare_url": args.prepare_url,
        "model": model,
        "family": args.family,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": output_tokens,
        "repetitions": args.repetitions,
        "concurrency": args.concurrency,
        "mode": args.mode,
        "seed": seed,
        "chat_template_kwargs": template_kwargs,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "running",
        "acceptance_eligible": False,
        "acceptance_blockers": ["campaign has not been finalized"],
    }
    immutable_evidence = _load_evidence_manifest(args.evidence_manifest)
    manifest["immutable_artifact_evidence"] = immutable_evidence
    if args.request_file:
        manifest["request_file_sha256"] = hashlib.sha256(args.request_file.read_bytes()).hexdigest()
        (output_dir / "request-template.json").write_text(
            json.dumps(request_template, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
    environment_before = _capture_environment(
        args.base_url, args.prepare_url, args.timeout, api_key,
    )
    manifest["environment_before"] = environment_before
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    preparer = PromptPreparer(
        args.prepare_url,
        args.timeout,
        api_key,
        template_kwargs if template_kwargs_present else None,
        template_options,
    )
    try:
        lanes: list[PromptLane] = []
        for lane in range(args.concurrency):
            if request_template is not None:
                messages = request_template["messages"]
                prepared_tokens = preparer.count(messages)
                if prepared_tokens != args.prompt_tokens:
                    raise ValueError(
                        f"request file prepares to {prepared_tokens} tokens, expected {args.prompt_tokens}"
                    )
            else:
                messages, prepared_tokens = fit_prompt(
                    preparer, args.family, args.prompt_tokens, seed=seed, lane=lane,
                )
            prefix_messages = None
            prefix_tokens = None
            prefix_hash = None
            if args.mode == "cached-prefix":
                prefix_messages, _ = _cached_prefix(messages)
                prefix_tokens = preparer.count(prefix_messages)
                if prefix_tokens <= 0 or prefix_tokens >= prepared_tokens:
                    raise ValueError("cached prefix must contain between 1 and prompt target tokens")
                prefix_hash = hashlib.sha256(
                    json.dumps(prefix_messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
            content_hash = hashlib.sha256(
                json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            lanes.append({
                "messages": messages,
                "prepared_tokens": prepared_tokens,
                "prefix_messages": prefix_messages,
                "prefix_tokens": prefix_tokens,
                "prefix_sha256": prefix_hash,
                "prompt_sha256": content_hash,
            })
    except Exception as exc:
        (output_dir / "preparation-error.json").write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(f"prompt preparation failed; details saved in {output_dir}: {exc}") from exc

    prompt_file = {
        "family": args.family,
        "target_tokens": args.prompt_tokens,
        "lanes": [],
    }
    for lane_index, lane_data in enumerate(lanes):
        lane_prompt = {
            "lane": lane_index,
            "prepared_tokens": lane_data["prepared_tokens"],
            "prompt_sha256": lane_data["prompt_sha256"],
            "messages": lane_data["messages"],
        }
        if lane_data["prefix_messages"] is not None:
            lane_prompt["cached_prefix"] = {
                "prepared_tokens": lane_data["prefix_tokens"],
                "prompt_sha256": lane_data["prefix_sha256"],
                "messages": lane_data["prefix_messages"],
            }
        prompt_file["lanes"].append(lane_prompt)
    (output_dir / "prompts.json").write_text(
        json.dumps(prompt_file, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    manifest.update({
        "prepared_prompt_tokens": args.prompt_tokens,
        "lane_prompt_sha256": [lane["prompt_sha256"] for lane in lanes],
    })
    sample_body = _completion_body(
        model,
        lanes[0]["messages"],
        output_tokens,
        seed,
        args.mode == "cached-prefix",
        template_kwargs,
        request_template,
    )
    manifest["request_evidence"] = _request_evidence(
        sample_body, environment_before["native_props"],
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    sample_path = output_dir / "samples.jsonl"
    events_path = output_dir / "events.jsonl"
    raw_dir = output_dir / "raw_sse"
    raw_dir.mkdir()
    sample_path.touch()
    events_path.touch()

    def save_result(record: dict[str, Any], events: list[dict[str, Any]], raw_bytes: bytes) -> None:
        event_metadata = {
            key: record.get(key)
            for key in (
                "campaign_id", "arm", "label", "family", "mode", "repetition", "lane",
                "prompt_sha256",
            )
        }
        for event in events:
            event.update(event_metadata)
        raw_name = f"{record['sample_id']}-{record['stage']}.sse"
        raw_path = raw_dir / raw_name
        raw_evidence = _write_raw_sse(raw_path, raw_bytes)
        raw_evidence["raw_sse_file"] = f"raw_sse/{raw_name}"
        record.update(raw_evidence)
        with sample_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            output.flush()
        if events:
            with events_path.open("a", encoding="utf-8") as output:
                for event in events:
                    output.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
                output.flush()

    summary_columns = (
        "sample_id", "stage", "campaign_id", "arm", "label", "family", "mode",
        "repetition", "lane", "prompt_sha256", "concurrency",
        "valid", "invalid_reasons", "prepared_prompt_tokens", "prompt_n", "cache_n",
        "logical_prompt_tokens", "predicted_n", "draft_n", "draft_n_accepted",
        "draft_acceptance", "text_delta_events", "reasoning_delta_events", "content_delta_events",
        "prompt_ms", "predicted_ms", "pp_tps", "tg_tps", "ttft_ms", "total_ms", "finish_reason",
    )
    summary_path = output_dir / "summary.csv"
    import csv
    summary_output = summary_path.open("w", encoding="utf-8", newline="")
    summary_writer = csv.DictWriter(summary_output, fieldnames=summary_columns, extrasaction="ignore")
    summary_writer.writeheader()

    def save_summary(record: dict[str, Any]) -> None:
        row = dict(record)
        row["invalid_reasons"] = "; ".join(row.get("invalid_reasons", []))
        summary_writer.writerow(row)
        summary_output.flush()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        repetition = 0
        while repetition < args.repetitions:
            group_size = min(args.concurrency, args.repetitions - repetition)
            sample_ids = [str(uuid.uuid4()) for _ in range(group_size)]
            if args.mode == "cached-prefix":
                warm_barrier = threading.Barrier(group_size) if group_size > 1 else None
                warm_futures = {}
                for index, sample_id in enumerate(sample_ids):
                    lane_data = lanes[index]
                    repetition_id = repetition + index
                    prefix_messages = lane_data["prefix_messages"]
                    prefix_tokens = lane_data["prefix_tokens"]
                    if prefix_messages is None or prefix_tokens is None:
                        raise RuntimeError("cached-prefix prompt was not prepared")
                    body = _completion_body(
                        model, prefix_messages, 1, seed,
                        True, template_kwargs,
                    )
                    future = executor.submit(
                        _run_request,
                        args.base_url,
                        body,
                        timeout=args.timeout,
                        api_key=api_key,
                        sample_id=sample_id,
                        stage="warmup",
                        expected_prompt_tokens=prefix_tokens,
                        expected_output_tokens=1,
                        cache_mode="any",
                        barrier=warm_barrier,
                    )
                    warm_futures[future] = (sample_id, repetition_id, index, body)
                for future in concurrent.futures.as_completed(warm_futures):
                    record, events, raw_bytes = future.result()
                    sample_id, repetition_id, lane, body = warm_futures[future]
                    record.update({
                        "sample_id": sample_id,
                        "campaign_id": campaign_id,
                        "arm": args.arm,
                        "label": args.label,
                        "family": args.family,
                        "mode": args.mode,
                        "repetition": repetition_id,
                        "lane": lane,
                        "prompt_sha256": lanes[lane]["prefix_sha256"],
                        "concurrency": group_size,
                        "request_evidence": _request_evidence(body, environment_before["native_props"]),
                    })
                    save_result(record, events, raw_bytes)
                    save_summary(record)

            barrier = threading.Barrier(group_size) if group_size > 1 else None
            futures = {}
            for index, sample_id in enumerate(sample_ids):
                lane_data = lanes[index]
                repetition_id = repetition + index
                body = _completion_body(
                    model,
                    lane_data["messages"],
                    output_tokens,
                    seed,
                    args.mode == "cached-prefix",
                    template_kwargs,
                    request_template,
                )
                future = executor.submit(
                    _run_request,
                    args.base_url,
                    body,
                    timeout=args.timeout,
                    api_key=api_key,
                    sample_id=sample_id,
                    stage="sample",
                    expected_prompt_tokens=lane_data["prepared_tokens"],
                    expected_output_tokens=output_tokens,
                    cache_mode=args.mode,
                    barrier=barrier,
                )
                futures[future] = (sample_id, repetition_id, index, body)
            for future in concurrent.futures.as_completed(futures):
                record, events, raw_bytes = future.result()
                sample_id, repetition_id, lane, body = futures[future]
                record.update({
                    "sample_id": sample_id,
                    "campaign_id": campaign_id,
                    "arm": args.arm,
                    "label": args.label,
                    "family": args.family,
                    "mode": args.mode,
                    "repetition": repetition_id,
                    "lane": lane,
                    "prompt_sha256": lanes[lane]["prompt_sha256"],
                    "concurrency": group_size,
                    "request_evidence": _request_evidence(body, environment_before["native_props"]),
                })
                save_result(record, events, raw_bytes)
                save_summary(record)
            repetition += group_size
    finally:
        executor.shutdown(wait=True)
        summary_output.close()
    all_records = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    invalid_samples = [record for record in all_records if record.get("stage") == "sample" and not record.get("valid")]
    environment_after = _capture_environment(
        args.base_url, args.prepare_url, args.timeout, api_key,
    )
    environment_comparison = _compare_environment(environment_before, environment_after)
    outcome = _campaign_outcome(len(invalid_samples), immutable_evidence, environment_comparison)
    manifest.update({
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": outcome["status"],
        "acceptance_eligible": outcome["acceptance_eligible"],
        "acceptance_blockers": outcome["acceptance_blockers"],
        "request_count": len(all_records),
        "invalid_samples": len(invalid_samples),
        "environment_after": environment_after,
        "environment_comparison": environment_comparison,
    })
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(
        f"Wrote {len(all_records)} requests ({len(invalid_samples)} invalid measured samples, "
        f"acceptance_eligible={outcome['acceptance_eligible']}, status={outcome['status']}) to {output_dir}\n"
    )
    return outcome["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
