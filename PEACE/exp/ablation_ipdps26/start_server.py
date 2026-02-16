#!/usr/bin/env python3
"""
Start an inference server process with the environment variables corresponding
to a PEACE ablation variant.

This script is intentionally generic: you provide the actual server command via
--cmd. For example, you can point it to a PEACE-patched vLLM OpenAI server.

Example:
  python start_server.py \
    --variant peace \
    --cmd "python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model /path/to/model" \
    --log-file ./logs/peace.log
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from variants import get_variant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch a server with PEACE ablation env vars.")
    p.add_argument("--variant", type=str, required=True, help="Variant key: peace|pe|dis|col|fsp")
    p.add_argument("--cmd", type=str, required=True, help="Server launch command (quoted string).")
    p.add_argument("--log-file", type=str, default=None, help="Optional path to redirect stdout/stderr.")
    p.add_argument("--cwd", type=str, default=None, help="Optional working directory for the server process.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    variant = get_variant(args.variant)

    env = os.environ.copy()
    env.update(variant.env)

    log_fp: Optional[object] = None
    stdout = None
    stderr = None
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(args.log_file, "a", encoding="utf-8")
        stdout = log_fp
        stderr = log_fp

    print(f"[start_server] Variant: {variant.name} ({args.variant})", flush=True)
    if variant.env:
        for k, v in variant.env.items():
            print(f"[start_server]   ENV {k}={v}", flush=True)
    else:
        print("[start_server]   (no variant env vars)", flush=True)

    cmd_list = shlex.split(args.cmd)
    print(f"[start_server] Command: {cmd_list}", flush=True)

    proc = subprocess.Popen(
        cmd_list,
        env=env,
        cwd=args.cwd,
        stdout=stdout,
        stderr=stderr,
        text=True if not args.log_file else False,
    )

    print(f"[start_server] PID={proc.pid}. Press Ctrl+C to stop.", flush=True)

    def _handle_sigint(_sig, _frame):
        print("[start_server] Received interrupt. Terminating server...", flush=True)
        proc.terminate()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    try:
        rc = proc.wait()
    finally:
        if log_fp:
            log_fp.close()

    print(f"[start_server] Server exited with code {rc}.", flush=True)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
