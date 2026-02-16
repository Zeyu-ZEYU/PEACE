#!/usr/bin/env python3
"""PEACE experiment helper (real inference, not simulation).

This script is a convenience wrapper for running **PEACE-only** experiments in
the IPDPS'26 camera-ready paper.

What it does:

1) Prepares a JSONL replay trace from the Azure CSV trace
   (including the paper’s long-input resampling: 100K–500K tokens).
2) Optionally starts an OpenAI-compatible PEACE server command.
3) Replays the trace against the server and writes client-observed metrics.

Under the hood, the implementation reuses the runnable ablation harness in
`PEACE/exp/ablation_ipdps26/` (trace preparation + OpenAI streaming client).

Important notes:

- This repository does not ship a full PEACE-patched serving stack.
  You must provide a server command that implements PEACE and exposes an
  OpenAI-compatible endpoint.
- The workload driver will use server-reported queueing delay if it is
  available as `peace_metrics.queue_delay_ms`; otherwise it falls back to TTFT.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from peace_defaults import DEFAULT_LONG_RANGE, DEFAULT_SHORT_THRESHOLD, PAPER_MODELS


def _repo_paths() -> Tuple[Path, Path]:
    """Return (exp_dir, ablation_dir)."""
    exp_dir = Path(__file__).resolve().parent
    ablation_dir = exp_dir / "ablation_ipdps26"
    if not ablation_dir.exists():
        raise FileNotFoundError(f"Expected ablation harness at: {ablation_dir}")
    return exp_dir, ablation_dir


def _print_paper_defaults() -> None:
    print("Paper-aligned defaults:")
    print(f"  short_threshold: {DEFAULT_SHORT_THRESHOLD} tokens")
    print(f"  long_range     : {DEFAULT_LONG_RANGE[0]}–{DEFAULT_LONG_RANGE[1]} tokens")
    print("  models:")
    for k in sorted(PAPER_MODELS.keys()):
        m = PAPER_MODELS[k]
        print(
            f"    - {m.key}: {m.display_name} | TP={m.tensor_parallel_size} | "
            f"short-decode replicas={m.short_decode_replicas}"
        )


def _parse_kv_list(kv_list: Sequence[str]) -> Dict[str, str]:
    """Parse KEY=VALUE pairs into a dict."""
    out: Dict[str, str] = {}
    for item in kv_list:
        if "=" not in item:
            raise ValueError(f"Invalid --env item (expected KEY=VALUE): {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid --env item (empty key): {item}")
        out[k] = v
    return out


def _wait_for_ready(base_url: str, timeout_s: float = 300.0, interval_s: float = 1.0) -> bool:
    """Best-effort readiness check.

    We try a small set of common health endpoints used by OpenAI-compatible
    servers (including vLLM- and LoongServe-style deployments).
    """
    base = base_url.rstrip("/")
    candidates = [
        f"{base}/health",
        f"{base}/healthz",
        f"{base}/v1/models",
    ]

    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        for url in candidates:
            try:
                req = urllib.request.Request(url=url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if 200 <= int(getattr(resp, "status", 0) or 0) < 300:
                        return True
            except urllib.error.HTTPError as e:
                # /v1/models might return 401 if auth is enabled. Treat that
                # as "server is up".
                if int(getattr(e, "code", 0) or 0) in (401, 403):
                    return True
            except Exception:
                pass
        time.sleep(float(interval_s))
    return False


def _run_subprocess(cmd: List[str], cwd: Optional[str] = None) -> int:
    """Run a subprocess and return its exit code."""
    proc = subprocess.run(cmd, cwd=cwd)
    return int(proc.returncode)


def _prepare_trace(args: argparse.Namespace) -> int:
    _exp_dir, ablation_dir = _repo_paths()
    trace_utils = ablation_dir / "trace_utils.py"
    cmd = [
        sys.executable,
        str(trace_utils),
        "prepare",
        "--input-csv",
        args.input_csv,
        "--output-jsonl",
        args.output_jsonl,
        "--short-threshold",
        str(args.short_threshold),
        "--long-range",
        str(args.long_range[0]),
        str(args.long_range[1]),
        "--seed",
        str(args.seed),
    ]
    if args.max_requests is not None:
        cmd += ["--max-requests", str(args.max_requests)]

    print("[run_peace] Preparing trace:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    return _run_subprocess(cmd)


def _run_workload(args: argparse.Namespace) -> int:
    _exp_dir, ablation_dir = _repo_paths()
    run_workload = ablation_dir / "run_workload.py"
    cmd = [
        sys.executable,
        str(run_workload),
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--trace",
        args.trace,
        "--out-dir",
        args.out_dir,
        "--concurrency",
        str(args.concurrency),
        "--arrival-scale",
        str(args.arrival_scale),
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--bucket-size",
        str(args.bucket_size),
    ]
    if args.max_requests is not None:
        cmd += ["--max-requests", str(args.max_requests)]
    if args.tokenizer is not None:
        cmd += ["--tokenizer", args.tokenizer]
    if args.prompt_prefix:
        cmd += ["--prompt-prefix", args.prompt_prefix]
    if args.prompt_suffix:
        cmd += ["--prompt-suffix", args.prompt_suffix]

    print("[run_peace] Replaying trace:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    return _run_subprocess(cmd)


def _start_server_blocking(args: argparse.Namespace) -> int:
    """Start a server command and block (Ctrl+C to stop)."""
    env = os.environ.copy()
    env.update(_parse_kv_list(args.env or []))

    log_fp = None
    stdout = None
    stderr = None
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(args.log_file, "ab")
        stdout = log_fp
        stderr = log_fp

    cmd_list = shlex.split(args.cmd)
    print("[run_peace] Starting server (blocking):")
    print(f"  cmd: {cmd_list}")
    if args.env:
        for kv in args.env:
            print(f"  env: {kv}")

    proc = subprocess.Popen(cmd_list, env=env, cwd=args.cwd, stdout=stdout, stderr=stderr)

    def _handle(_sig, _frame):
        try:
            proc.terminate()
        except Exception:
            pass

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    rc = 0
    try:
        rc = int(proc.wait())
    finally:
        if log_fp is not None:
            log_fp.close()
    return rc


def _end_to_end(args: argparse.Namespace) -> int:
    """Start server → wait → run workload → stop server."""
    env = os.environ.copy()
    env.update(_parse_kv_list(args.env or []))

    cmd_list = shlex.split(args.cmd)

    log_fp = None
    stdout = None
    stderr = None
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(args.log_file, "ab")
        stdout = log_fp
        stderr = log_fp

    print("[run_peace] Starting server (background):")
    print(f"  cmd: {cmd_list}")
    if args.env:
        for kv in args.env:
            print(f"  env: {kv}")

    proc = subprocess.Popen(cmd_list, env=env, cwd=args.cwd, stdout=stdout, stderr=stderr)

    try:
        if args.wait_ready:
            print(f"[run_peace] Waiting for server readiness at {args.base_url} ...")
            ok = _wait_for_ready(args.base_url, timeout_s=args.ready_timeout_s)
            if not ok:
                print("[run_peace] ERROR: server did not become ready in time.")
                return 2
            print("[run_peace] Server is ready.")
        else:
            print("[run_peace] Skipping readiness check (--no-wait-ready).")

        # Run the workload replay.
        rc = _run_workload(args)
        return int(rc)
    finally:
        # Best-effort shutdown.
        try:
            proc.terminate()
        except Exception:
            pass

        try:
            proc.wait(timeout=float(args.kill_timeout_s))
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        if log_fp is not None:
            log_fp.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PEACE experiment helper (real inference).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_defaults = sub.add_parser("paper-defaults", help="Print paper-aligned defaults.")
    p_defaults.set_defaults(_handler=lambda a: (_print_paper_defaults() or 0))

    p_prep = sub.add_parser("prepare-trace", help="Prepare Azure CSV into JSONL (paper-style).")
    p_prep.add_argument("--input-csv", type=str, required=True)
    p_prep.add_argument("--output-jsonl", type=str, required=True)
    p_prep.add_argument("--short-threshold", type=int, default=DEFAULT_SHORT_THRESHOLD)
    p_prep.add_argument(
        "--long-range",
        type=int,
        nargs=2,
        default=[DEFAULT_LONG_RANGE[0], DEFAULT_LONG_RANGE[1]],
        metavar=("MIN", "MAX"),
    )
    p_prep.add_argument("--max-requests", type=int, default=None)
    p_prep.add_argument("--seed", type=int, default=0)
    p_prep.set_defaults(_handler=_prepare_trace)

    # Workload replay args (shared by run-workload and end-to-end).
    def add_replay_args(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--base-url", type=str, required=True, help="e.g., http://127.0.0.1:8000")
        pp.add_argument("--api-key", type=str, default="EMPTY")
        pp.add_argument("--endpoint", type=str, choices=["chat", "completion"], default="chat")
        pp.add_argument("--model", type=str, required=True)
        pp.add_argument("--trace", type=str, required=True, help="Prepared JSONL trace.")
        pp.add_argument("--out-dir", type=str, required=True)
        pp.add_argument("--concurrency", type=int, default=64)
        pp.add_argument("--arrival-scale", type=float, default=1.0)
        pp.add_argument("--max-requests", type=int, default=None)
        pp.add_argument("--request-timeout-s", type=float, default=600.0)
        pp.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name/path (default: --model).")
        pp.add_argument("--prompt-prefix", type=str, default="")
        pp.add_argument("--prompt-suffix", type=str, default="")
        pp.add_argument("--bucket-size", type=int, default=1)

    p_run = sub.add_parser("run-workload", help="Replay a prepared trace against a running PEACE server.")
    add_replay_args(p_run)
    p_run.set_defaults(_handler=_run_workload)

    p_srv = sub.add_parser("start-server", help="Start a PEACE server command (blocking).")
    p_srv.add_argument("--cmd", type=str, required=True, help="Server launch command as a quoted string.")
    p_srv.add_argument("--log-file", type=str, default=None)
    p_srv.add_argument("--cwd", type=str, default=None)
    p_srv.add_argument("--env", type=str, nargs="*", default=None, help="Extra env vars KEY=VALUE")
    p_srv.set_defaults(_handler=_start_server_blocking)

    p_e2e = sub.add_parser("end-to-end", help="Start server → run workload → stop server.")
    p_e2e.add_argument("--cmd", type=str, required=True, help="Server launch command as a quoted string.")
    p_e2e.add_argument("--log-file", type=str, default=None)
    p_e2e.add_argument("--cwd", type=str, default=None)
    p_e2e.add_argument("--env", type=str, nargs="*", default=None, help="Extra env vars KEY=VALUE")
    p_e2e.add_argument("--wait-ready", action="store_true", default=True)
    p_e2e.add_argument("--no-wait-ready", dest="wait_ready", action="store_false")
    p_e2e.add_argument("--ready-timeout-s", type=float, default=300.0)
    p_e2e.add_argument("--kill-timeout-s", type=float, default=10.0)
    add_replay_args(p_e2e)
    p_e2e.set_defaults(_handler=_end_to_end)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    handler = getattr(args, "_handler", None)
    if handler is None:
        raise RuntimeError("No handler registered for subcommand")
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
