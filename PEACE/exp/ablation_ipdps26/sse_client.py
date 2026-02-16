"""
Minimal OpenAI-compatible streaming client with TTFT measurement.

This client is designed to work with OpenAI-style SSE responses used by vLLM
OpenAI server and other compatible stacks.

It measures:
- TTFT (time to first token chunk)
- Total latency
- Best-effort generated token count

If the server embeds extra instrumentation (e.g., peace_metrics), we preserve it.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import aiohttp


@dataclass
class StreamMetrics:
    """Per-request client-side streaming metrics."""
    http_status: int
    error: Optional[str]
    ttft_s: Optional[float]
    latency_s: Optional[float]
    generated_tokens: Optional[int]
    peace_metrics: Optional[Dict[str, Any]]


def _extract_generated_tokens_from_final(payload: Dict[str, Any]) -> Optional[int]:
    usage = payload.get("usage")
    if isinstance(usage, dict):
        # OpenAI format: usage.completion_tokens
        for key in ("completion_tokens", "generated_tokens"):
            if key in usage:
                try:
                    return int(usage[key])
                except Exception:
                    return None
    return None


def _merge_peace_metrics(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if k not in dst:
            dst[k] = v


async def stream_openai_request(
    session: aiohttp.ClientSession,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    request_timeout_s: float = 600.0,
) -> StreamMetrics:
    """
    Send a streaming request and measure TTFT/latency.

    The response is expected to be Server-Sent Events (SSE) with lines like:
      data: {...json...}
      data: [DONE]
    """
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    peace_metrics: Dict[str, Any] = {}
    generated_tokens: Optional[int] = None

    timeout = aiohttp.ClientTimeout(total=request_timeout_s)

    try:
        async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
            status = resp.status
            if status >= 400:
                text = await resp.text()
                return StreamMetrics(
                    http_status=status,
                    error=f"HTTP {status}: {text[:500]}",
                    ttft_s=None,
                    latency_s=time.perf_counter() - t0,
                    generated_tokens=None,
                    peace_metrics=None,
                )

            # SSE stream parsing.
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue

                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break

                try:
                    evt = json.loads(data)
                except Exception:
                    # Ignore malformed chunks.
                    continue

                # Capture TTFT at the first meaningful chunk.
                if ttft is None:
                    # Some servers send an initial chunk with empty delta; try to detect actual token output.
                    has_text = False
                    for choice in evt.get("choices", []) or []:
                        if "text" in choice and choice.get("text"):
                            has_text = True
                        delta = choice.get("delta") or {}
                        if isinstance(delta, dict) and delta.get("content"):
                            has_text = True
                    if has_text:
                        ttft = time.perf_counter() - t0

                # Preserve optional PEACE-side instrumentation.
                pm = evt.get("peace_metrics")
                if isinstance(pm, dict):
                    _merge_peace_metrics(peace_metrics, pm)

                # Some servers include usage at the end even in stream mode.
                gt = _extract_generated_tokens_from_final(evt)
                if gt is not None:
                    generated_tokens = gt

            latency = time.perf_counter() - t0
            return StreamMetrics(
                http_status=status,
                error=None,
                ttft_s=ttft,
                latency_s=latency,
                generated_tokens=generated_tokens,
                peace_metrics=peace_metrics or None,
            )

    except asyncio.TimeoutError:
        return StreamMetrics(
            http_status=0,
            error=f"Timeout after {request_timeout_s}s",
            ttft_s=None,
            latency_s=time.perf_counter() - t0,
            generated_tokens=None,
            peace_metrics=None,
        )
    except Exception as e:
        return StreamMetrics(
            http_status=0,
            error=str(e),
            ttft_s=None,
            latency_s=time.perf_counter() - t0,
            generated_tokens=None,
            peace_metrics=None,
        )
