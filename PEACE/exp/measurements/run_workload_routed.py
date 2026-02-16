#!/usr/bin/env python3
"""Replay a prepared Azure trace against one or two OpenAI-compatible endpoints.

This runner is designed for the PEACE paper's *Motivation* measurements.
Unlike the simulator, it performs **real** inference by sending HTTP requests.

Key feature vs. the ablation runner:
- Optional **routing** of requests by type (short/long) to different base URLs.
  This is needed for the paper's **Reservation** baseline, which runs two
  dedicated serving pools.

It produces the same output format as `ablation_ipdps26/run_workload.py`:
- records.jsonl : per-request metrics (client-observed + optional server metrics)
- summary.json  : aggregated metrics (percentiles, throughput, etc.)

All code and comments are English-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from tqdm import tqdm


# Reuse prompt construction + SSE streaming client + metrics from the ablation harness.
# This keeps experiment outputs consistent across PEACE paper scripts.
EXP_DIR = Path(__file__).resolve().parents[1]
ABLATION_DIR = EXP_DIR / "ablation_ipdps26"
if not ABLATION_DIR.exists():
    raise RuntimeError(f"Expected ablation harness at: {ABLATION_DIR}")

sys.path.insert(0, str(ABLATION_DIR))

from prompt_builder import PromptBuilder, TokenizerNotAvailable  # type: ignore
from sse_client import StreamMetrics, stream_openai_request  # type: ignore
from metrics import summarize_records  # type: ignore


PERCENTILES = [1, 25, 50, 75, 99]


@dataclass(frozen=True)
class PreparedRequest:
    request_id: int
    arrival_time_s: float
    prompt_tokens: int
    max_new_tokens: int
    request_type: str  # "short" | "long"


def _build_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if endpoint == "chat":
        return f"{base}/v1/chat/completions"
    if endpoint == "completion":
        return f"{base}/v1/completions"
    raise ValueError(f"Unknown endpoint: {endpoint}")


def _parse_kv_list(items: Optional[List[str]], parse_values: bool) -> Dict[str, Any]:
    """Parse repeated KEY=VALUE arguments into a dict.

    Args:
        items: List like ["k1=v1", "k2=v2"].
        parse_values: If True, attempt to parse VALUE as JSON / int / float / bool.
                      If False, keep VALUE as a raw string (useful for headers).
    """
    out: Dict[str, Any] = {}
    if not items:
        return out

    for it in items:
        if "=" not in it:
            raise ValueError(f"Expected KEY=VALUE, got: {it}")
        k, v = it.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Empty key in: {it}")

        if not parse_values:
            out[k] = v
            continue

        # Try JSON first (allows lists/dicts/strings with quotes).
        try:
            out[k] = json.loads(v)
            continue
        except Exception:
            pass

        # Basic scalar parsing.
        if re.fullmatch(r"-?\d+", v):
            out[k] = int(v)
            continue
        if re.fullmatch(r"-?\d+\.\d+", v):
            out[k] = float(v)
            continue
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
            continue

        out[k] = v

    return out


def _load_trace(path: str, max_requests: Optional[int], bucket_size: int) -> List[PreparedRequest]:
    reqs: List[PreparedRequest] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_requests is not None and i >= int(max_requests):
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt_tokens = int(obj["prompt_tokens"])
            if bucket_size > 1:
                prompt_tokens = int(round(prompt_tokens / bucket_size) * bucket_size)
            reqs.append(
                PreparedRequest(
                    request_id=int(obj.get("request_id", i)),
                    arrival_time_s=float(obj["arrival_time_s"]),
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=int(obj.get("max_new_tokens", 0)),
                    request_type=str(obj.get("request_type", "short")),
                )
            )
    return reqs


def _extract_queue_delay_s(metrics: StreamMetrics) -> Optional[float]:
    """Best-effort extraction of queueing delay from server-side metrics."""
    pm = metrics.peace_metrics
    if not isinstance(pm, dict):
        return None

    # Common key variants.
    for k in ("queue_delay_ms", "queue_time_ms", "queue_ms"):
        v = pm.get(k)
        if isinstance(v, (int, float)):
            return float(v) / 1000.0

    v = pm.get("queue_delay_s")
    if isinstance(v, (int, float)):
        return float(v)

    return None


async def _worker(
    wid: int,
    session: aiohttp.ClientSession,
    prompt_builder: PromptBuilder,
    endpoint: str,
    api_key: str,
    model_short: str,
    model_long: str,
    url_short: str,
    url_long: str,
    headers_base: Dict[str, str],
    headers_short: Dict[str, str],
    headers_long: Dict[str, str],
    payload_base: Dict[str, Any],
    payload_short: Dict[str, Any],
    payload_long: Dict[str, Any],
    request_timeout_s: float,
    arrival_scale: float,
    t0: float,
    queue: "asyncio.Queue[PreparedRequest]",
    out_records: List[Dict[str, Any]],
    pbar: tqdm,
) -> None:
    while True:
        try:
            req = queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        # Schedule request based on arrival time.
        send_at = float(req.arrival_time_s) / max(float(arrival_scale), 1e-9)
        now = time.time() - t0
        if send_at > now:
            await asyncio.sleep(send_at - now)

        # Routing by request_type.
        is_long = str(req.request_type).lower() == "long"
        url = url_long if is_long else url_short
        model = model_long if is_long else model_short

        headers = dict(headers_base)
        headers.update(headers_long if is_long else headers_short)

        # Build prompt text with the right token length.
        try:
            prompt_text = prompt_builder.build(int(req.prompt_tokens))
        except Exception as e:
            out_records.append(
                {
                    "request_id": int(req.request_id),
                    "request_type": str(req.request_type),
                    "arrival_time_s": float(req.arrival_time_s) / max(float(arrival_scale), 1e-9),
                    "prompt_tokens": int(req.prompt_tokens),
                    "max_new_tokens": int(req.max_new_tokens),
                    "error": f"prompt_build_error: {e}",
                }
            )
            pbar.update(1)
            queue.task_done()
            continue

        payload = dict(payload_base)
        payload.update(payload_long if is_long else payload_short)

        if endpoint == "chat":
            payload["model"] = model
            payload["messages"] = [{"role": "user", "content": prompt_text}]
            payload["max_tokens"] = int(req.max_new_tokens)
            payload.setdefault("temperature", 0.0)
            payload.setdefault("stream", True)
        else:
            payload["model"] = model
            payload["prompt"] = prompt_text
            payload["max_tokens"] = int(req.max_new_tokens)
            payload.setdefault("temperature", 0.0)
            payload.setdefault("stream", True)

        start_time_s = time.time() - t0

        record: Dict[str, Any] = {
            "request_id": int(req.request_id),
            "request_type": str(req.request_type),
            "arrival_time_s": float(req.arrival_time_s) / max(float(arrival_scale), 1e-9),
            "prompt_tokens": int(req.prompt_tokens),
            "max_new_tokens": int(req.max_new_tokens),
            "start_time_s": float(start_time_s),
            "endpoint_url": str(url),
            "model": str(model),
        }

        try:
            metrics = await stream_openai_request(
                session=session,
                url=url,
                headers=headers,
                payload=payload,
                request_timeout_s=float(request_timeout_s),
            )

            finish_time_s = time.time() - t0

            record.update(
                {
                    "finish_time_s": float(finish_time_s),
                    "ttft_s": float(metrics.ttft_s) if metrics.ttft_s is not None else None,
                    "latency_s": float(metrics.latency_s) if metrics.latency_s is not None else None,
                    "generated_tokens": int(metrics.generated_tokens) if metrics.generated_tokens is not None else None,
                    "http_status": int(metrics.http_status) if metrics.http_status is not None else None,
                    "error": metrics.error,
                    "peace_metrics": metrics.peace_metrics,
                }
            )

            qd = _extract_queue_delay_s(metrics)
            if qd is not None:
                record["queue_delay_s"] = float(qd)

        except Exception as e:
            finish_time_s = time.time() - t0
            record.update(
                {
                    "finish_time_s": float(finish_time_s),
                    "error": f"request_error: {e}",
                }
            )

        out_records.append(record)
        pbar.update(1)
        queue.task_done()


async def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide routing.
    if args.base_url is not None:
        url_short = _build_url(args.base_url, args.endpoint)
        url_long = url_short
        model_short = args.model
        model_long = args.model
    else:
        if args.base_url_short is None or args.base_url_long is None:
            raise SystemExit("Routing mode requires both --base-url-short and --base-url-long")
        url_short = _build_url(args.base_url_short, args.endpoint)
        url_long = _build_url(args.base_url_long, args.endpoint)
        model_short = args.model_short or args.model
        model_long = args.model_long or args.model

    # Parse extra headers/payloads.
    headers_base = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    headers_base.update(_parse_kv_list(args.extra_header, parse_values=False))
    headers_short = _parse_kv_list(args.extra_header_short, parse_values=False)
    headers_long = _parse_kv_list(args.extra_header_long, parse_values=False)

    payload_base = _parse_kv_list(args.extra_payload, parse_values=True)
    payload_short = _parse_kv_list(args.extra_payload_short, parse_values=True)
    payload_long = _parse_kv_list(args.extra_payload_long, parse_values=True)

    # Load trace.
    trace = _load_trace(args.trace, max_requests=args.max_requests, bucket_size=int(args.bucket_size))
    if not trace:
        raise SystemExit("Prepared trace is empty.")

    # Prompt builder.
    tokenizer_name = args.tokenizer or args.model
    try:
        prompt_builder = PromptBuilder(
            tokenizer_name_or_path=tokenizer_name,
            prefix=args.prompt_prefix or "",
            suffix=args.prompt_suffix or "",
        )
    except TokenizerNotAvailable as e:
        raise SystemExit(str(e))

    # Async run.
    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0)

    records: List[Dict[str, Any]] = []
    q: asyncio.Queue[PreparedRequest] = asyncio.Queue()
    for r in trace:
        q.put_nowait(r)

    t0 = time.time()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        pbar = tqdm(total=len(trace), desc="requests", unit="req")
        workers = []
        for wid in range(int(args.concurrency)):
            workers.append(
                asyncio.create_task(
                    _worker(
                        wid=wid,
                        session=session,
                        prompt_builder=prompt_builder,
                        endpoint=args.endpoint,
                        api_key=args.api_key,
                        model_short=model_short,
                        model_long=model_long,
                        url_short=url_short,
                        url_long=url_long,
                        headers_base=headers_base,
                        headers_short=headers_short,
                        headers_long=headers_long,
                        payload_base=payload_base,
                        payload_short=payload_short,
                        payload_long=payload_long,
                        request_timeout_s=float(args.request_timeout_s),
                        arrival_scale=float(args.arrival_scale),
                        t0=t0,
                        queue=q,
                        out_records=records,
                        pbar=pbar,
                    )
                )
            )

        await asyncio.gather(*workers)
        pbar.close()

    # Write per-request records.
    rec_path = out_dir / "records.jsonl"
    with open(rec_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Aggregate summary.
    summary = summarize_records(records)
    summary.update(
        {
            "endpoint": args.endpoint,
            "trace": os.path.abspath(args.trace),
            "base_url": args.base_url,
            "base_url_short": args.base_url_short,
            "base_url_long": args.base_url_long,
            "model": args.model,
            "model_short": model_short,
            "model_long": model_long,
            "concurrency": int(args.concurrency),
            "arrival_scale": float(args.arrival_scale),
            "max_requests": int(args.max_requests) if args.max_requests is not None else None,
            "request_timeout_s": float(args.request_timeout_s),
            "bucket_size": int(args.bucket_size),
        }
    )

    sum_path = out_dir / "summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {rec_path}")
    print(f"Wrote: {sum_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a prepared trace against OpenAI-compatible endpoint(s).")

    # Endpoint selection.
    p.add_argument("--endpoint", choices=["chat", "completion"], default="chat")

    # Base URL(s): single-endpoint OR routed endpoints.
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--base-url", type=str, default=None, help="Single endpoint base URL, e.g., http://127.0.0.1:8000")
    g.add_argument(
        "--route-by-type",
        action="store_true",
        help="Enable routing: short requests -> --base-url-short, long requests -> --base-url-long",
    )

    p.add_argument("--base-url-short", type=str, default=None, help="Base URL for short requests (routing mode).")
    p.add_argument("--base-url-long", type=str, default=None, help="Base URL for long requests (routing mode).")

    p.add_argument("--api-key", type=str, default="EMPTY")

    p.add_argument("--model", type=str, required=True, help="Model name/path (OpenAI field: model).")
    p.add_argument("--model-short", type=str, default=None, help="Optional model name for short endpoint.")
    p.add_argument("--model-long", type=str, default=None, help="Optional model name for long endpoint.")

    p.add_argument("--trace", type=str, required=True, help="Prepared JSONL trace.")
    p.add_argument("--out-dir", type=str, required=True)

    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--arrival-scale", type=float, default=1.0)
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--request-timeout-s", type=float, default=600.0)

    p.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name/path (default: --model).")
    p.add_argument("--prompt-prefix", type=str, default="")
    p.add_argument("--prompt-suffix", type=str, default="")
    p.add_argument("--bucket-size", type=int, default=1)

    # Extra knobs for compatibility with different servers.
    p.add_argument("--extra-header", type=str, nargs="*", default=None, help="Extra headers as KEY=VALUE.")
    p.add_argument(
        "--extra-header-short",
        type=str,
        nargs="*",
        default=None,
        help="Extra headers for short requests as KEY=VALUE (routing mode).",
    )
    p.add_argument(
        "--extra-header-long",
        type=str,
        nargs="*",
        default=None,
        help="Extra headers for long requests as KEY=VALUE (routing mode).",
    )

    p.add_argument("--extra-payload", type=str, nargs="*", default=None, help="Extra payload fields as KEY=VALUE.")
    p.add_argument(
        "--extra-payload-short",
        type=str,
        nargs="*",
        default=None,
        help="Extra payload fields for short requests as KEY=VALUE (routing mode).",
    )
    p.add_argument(
        "--extra-payload-long",
        type=str,
        nargs="*",
        default=None,
        help="Extra payload fields for long requests as KEY=VALUE (routing mode).",
    )

    args = p.parse_args()

    # Normalize routing flag: if --route-by-type is used, require both URLs.
    if args.route_by_type:
        if args.base_url_short is None or args.base_url_long is None:
            raise SystemExit("--route-by-type requires --base-url-short and --base-url-long")
        args.base_url = None

    return args


def main() -> int:
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
