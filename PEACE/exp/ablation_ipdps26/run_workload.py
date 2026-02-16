#!/usr/bin/env python3
"""
Trace replay driver for PEACE ablation experiments (real inference, not simulation).

It reads a prepared JSONL trace (see trace_utils.py), synthesizes prompts with
target token lengths, sends streaming requests to an OpenAI-compatible endpoint,
and logs per-request metrics.

Outputs:
- records.jsonl: one JSON object per request
- summary.json  : aggregated metrics

The script is robust to missing server-side instrumentation. If the server does
not provide explicit queueing-delay fields, the runner falls back to TTFT.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from tqdm import tqdm

from metrics import read_jsonl, summarize_records
from prompt_builder import PromptBuilder, TokenizerNotAvailable
from sse_client import stream_openai_request


@dataclass(frozen=True)
class PreparedRequest:
    request_id: int
    arrival_time_s: float
    prompt_tokens: int
    max_new_tokens: int
    request_type: str  # short|long


def _load_prepared_trace(path: str, max_requests: Optional[int] = None) -> List[PreparedRequest]:
    reqs: List[PreparedRequest] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_requests is not None and len(reqs) >= int(max_requests):
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            reqs.append(
                PreparedRequest(
                    request_id=int(obj["request_id"]),
                    arrival_time_s=float(obj["arrival_time_s"]),
                    prompt_tokens=int(obj["prompt_tokens"]),
                    max_new_tokens=int(obj["max_new_tokens"]),
                    request_type=str(obj["request_type"]),
                )
            )
    return reqs


def _build_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if endpoint == "chat":
        return f"{base}/v1/chat/completions"
    if endpoint == "completion":
        return f"{base}/v1/completions"
    raise ValueError(f"Unsupported endpoint: {endpoint}")


def _build_payload(endpoint: str, model: str, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
    if endpoint == "chat":
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_new_tokens),
            "temperature": 0.0,
            "stream": True,
        }
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_new_tokens),
        "temperature": 0.0,
        "stream": True,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a prepared trace against an OpenAI-compatible server.")
    p.add_argument("--base-url", type=str, required=True, help="Server base URL, e.g., http://127.0.0.1:8000")
    p.add_argument("--api-key", type=str, default="EMPTY", help="API key for Authorization header.")
    p.add_argument("--endpoint", type=str, choices=["chat", "completion"], default="chat")
    p.add_argument("--model", type=str, required=True, help="Model name/path to pass to the server.")
    p.add_argument("--trace", type=str, required=True, help="Prepared trace JSONL (from trace_utils.py).")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for records.jsonl and summary.json")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--arrival-scale", type=float, default=1.0, help="Scale factor for replay speed (1.0 = real time).")
    p.add_argument("--max-requests", type=int, default=None, help="Limit number of requests replayed.")
    p.add_argument("--request-timeout-s", type=float, default=600.0)
    p.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name/path (default: --model).")
    p.add_argument("--prompt-prefix", type=str, default="", help="Optional prompt prefix.")
    p.add_argument("--prompt-suffix", type=str, default="", help="Optional prompt suffix.")
    p.add_argument("--bucket-size", type=int, default=1, help="Round prompt_tokens to multiples of this size.")
    return p.parse_args()


async def _worker(
    worker_id: int,
    q: asyncio.Queue,
    session: aiohttp.ClientSession,
    url: str,
    headers: Dict[str, str],
    model: str,
    endpoint: str,
    prompt_builder: PromptBuilder,
    bucket_size: int,
    arrival_scale: float,
    request_timeout_s: float,
    t0_wall: float,
    out_fp,
    pbar: tqdm,
):
    while True:
        item = await q.get()
        if item is None:
            q.task_done()
            break
        req: PreparedRequest = item

        # Sleep until the intended arrival time (scaled).
        target = t0_wall + (req.arrival_time_s / max(arrival_scale, 1e-9))
        now = time.perf_counter()
        if target > now:
            await asyncio.sleep(target - now)

        # This is the time we actually send the request (relative to experiment start).
        send_wall = time.perf_counter()
        send_rel = send_wall - t0_wall

        # Prompt synthesis (token-accurate under tokenizer).
        prompt_tokens = int(req.prompt_tokens)
        if bucket_size > 1:
            prompt_tokens = int(round(prompt_tokens / bucket_size) * bucket_size)
            prompt_tokens = max(prompt_tokens, 1)

        prompt_text = prompt_builder.build(prompt_tokens)

        payload = _build_payload(endpoint=endpoint, model=model, prompt=prompt_text, max_new_tokens=req.max_new_tokens)

        m = await stream_openai_request(
            session=session,
            url=url,
            headers=headers,
            payload=payload,
            request_timeout_s=request_timeout_s,
        )

        finish_rel = time.perf_counter() - t0_wall

        # Extract queue_delay_s if server provided it in peace_metrics.
        queue_delay_s = None
        if isinstance(m.peace_metrics, dict):
            if "queue_delay_ms" in m.peace_metrics:
                try:
                    queue_delay_s = float(m.peace_metrics["queue_delay_ms"]) / 1000.0
                except Exception:
                    queue_delay_s = None

        rec = {
            "request_id": req.request_id,
            "request_type": req.request_type,
            "arrival_time_s": float(req.arrival_time_s),
            "prompt_tokens": int(req.prompt_tokens),
            "max_new_tokens": int(req.max_new_tokens),
            "start_time_s": float(send_rel),
            "finish_time_s": float(finish_rel),
            "http_status": int(m.http_status),
            "error": m.error,
            "ttft_s": m.ttft_s,
            "latency_s": m.latency_s,
            "generated_tokens": m.generated_tokens,
            "queue_delay_s": queue_delay_s,
            "peace_metrics": m.peace_metrics,
        }

        out_fp.write(json.dumps(rec) + "\n")
        out_fp.flush()

        pbar.update(1)
        q.task_done()


async def run() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = _build_url(args.base_url, args.endpoint)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    tokenizer_name = args.tokenizer or args.model
    try:
        prompt_builder = PromptBuilder(
            tokenizer_name_or_path=tokenizer_name,
            prefix=args.prompt_prefix,
            suffix=args.prompt_suffix,
        )
    except TokenizerNotAvailable as e:
        raise SystemExit(str(e))

    trace = _load_prepared_trace(args.trace, max_requests=args.max_requests)

    # Experiment reference start time (wall clock, monotonic).
    t0_wall = time.perf_counter()

    records_path = out_dir / "records.jsonl"
    summary_path = out_dir / "summary.json"

    connector = aiohttp.TCPConnector(limit=0)  # no global connector limit; we control concurrency.
    async with aiohttp.ClientSession(connector=connector) as session:
        q: asyncio.Queue = asyncio.Queue()
        for r in trace:
            q.put_nowait(r)
        for _ in range(int(args.concurrency)):
            q.put_nowait(None)

        with open(records_path, "w", encoding="utf-8") as out_fp, tqdm(total=len(trace), desc="Requests") as pbar:
            workers = [
                asyncio.create_task(
                    _worker(
                        worker_id=i,
                        q=q,
                        session=session,
                        url=url,
                        headers=headers,
                        model=args.model,
                        endpoint=args.endpoint,
                        prompt_builder=prompt_builder,
                        bucket_size=int(args.bucket_size),
                        arrival_scale=float(args.arrival_scale),
                        request_timeout_s=float(args.request_timeout_s),
                        t0_wall=t0_wall,
                        out_fp=out_fp,
                        pbar=pbar,
                    )
                )
                for i in range(int(args.concurrency))
            ]

            await q.join()
            for w in workers:
                await w

    # Summarize results.
    records = read_jsonl(str(records_path))
    summary = summarize_records(records)
    summary["meta"] = {
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "model": args.model,
        "trace": os.path.abspath(args.trace),
        "concurrency": int(args.concurrency),
        "arrival_scale": float(args.arrival_scale),
        "bucket_size": int(args.bucket_size),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {records_path}")
    print(f"Wrote: {summary_path}")


def main() -> int:
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
