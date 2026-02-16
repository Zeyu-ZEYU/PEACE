from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, Iterable, List

from .types import RequestResult


def summarize(results: List[RequestResult]) -> Dict[str, float]:
    if not results:
        return {}

    n = len(results)
    avg_ttft = sum(r.ttft_s for r in results) / n
    avg_tbt = sum(r.tbt_s for r in results) / n
    avg_rt = sum(r.response_time_s for r in results) / n

    # Throughput: tokens / wall-time window (prefill and decode separately).
    prefill_tokens = sum(r.req.input_len for r in results)
    decode_tokens = sum(r.req.output_len for r in results)

    prefill_start = min(r.prefill_start_s for r in results)
    prefill_end = max(r.prefill_end_s for r in results)
    decode_start = min(r.decode_start_s for r in results)
    decode_end = max(r.decode_end_s for r in results)

    prefill_window = max(1e-9, prefill_end - prefill_start)
    decode_window = max(1e-9, decode_end - decode_start)

    prefill_tps = prefill_tokens / prefill_window
    decode_tps = decode_tokens / decode_window

    return {
        "n_requests": float(n),
        "avg_ttft_s": avg_ttft,
        "avg_tbt_s": avg_tbt,
        "avg_response_time_s": avg_rt,
        "prefill_throughput_tps": prefill_tps,
        "decode_throughput_tps": decode_tps,
        "prefill_window_s": prefill_window,
        "decode_window_s": decode_window,
        "trace_makespan_s": max(r.decode_end_s for r in results) - min(r.req.arrival_s for r in results),
    }


def save_results_json(results: List[RequestResult], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)


def save_summary_json(summary: Dict[str, float], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
