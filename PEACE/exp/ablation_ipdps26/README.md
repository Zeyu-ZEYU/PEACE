# IPDPS'26 PEACE Ablation Study (Runnable Experiments)

This folder contains **runnable ablation-study experiment code** aligned with the
IPDPS'26 camera-ready paper (`PEACE.zip` → `paper_ipdps26_camera_ready.tex`).

The ablation study in the paper evaluates the contribution of four PEACE design
components by testing PEACE variants:

- **PEACE**: full system
- **PEACE/PE**: PEACE **without Preemption**
- **PEACE/Dis**: PEACE **without short-request decode disaggregation**
- **PEACE/CoL**: PEACE **without prefill–decode colocation concurrency**
- **PEACE/FSP**: PEACE **without Fast-SP** (uses ring attention for long-prefill)

> Important: This is **NOT a simulator**. The workload driver sends **real**
> requests to an **OpenAI-compatible inference server** (e.g., a PEACE-patched
> vLLM deployment) and measures client-observed metrics, optionally consuming
> server-provided instrumentation if available.

---

## What you get

- `trace_utils.py`  
  Prepare a workload trace from Azure LLM Inference Trace CSV.
  It also supports the paper's long-input resampling (e.g., 100K–500K tokens).

- `run_workload.py`  
  Replays a trace against an OpenAI-compatible endpoint (streaming),
  records per-request metrics to `records.jsonl`, and writes `summary.json`.

- `start_server.py`  
  Convenience wrapper to launch a server command with the **variant-specific**
  environment variables (you can replace the command with your own).

- `analyze_ablation.py`  
  Reads multiple `summary.json` files, produces plots and a CSV table.

All code and comments are **English-only** as requested.

---

## 0) Requirements

- Python 3.9+ recommended
- A running OpenAI-compatible server that can serve your model(s)
  (e.g., vLLM OpenAI server).

Install Python deps:

```bash
cd PEACE/ablation_ipdps26
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1) Prepare the Azure trace

The repo already contains instructions for downloading the dataset:
`PEACE/trace/AzureLLMInferenceDataset2024.md`.

After downloading a CSV (e.g., `AzureLLMInferenceTrace_conv_1week.csv`),
convert it into a compact JSONL trace used by the runner:

```bash
python trace_utils.py prepare   --input-csv /path/to/AzureLLMInferenceTrace_conv_1week.csv   --output-jsonl ./traces/azure_prepared.jsonl   --short-threshold 4096   --long-range 100000 500000   --max-requests 20000
```

Notes:
- Requests with `ContextTokens < 4096` are treated as **short**.
- Requests with `ContextTokens >= 4096` are treated as **long** and are
  linearly rescaled into the configured long range (default 100K–500K)
  to mimic long-input workloads as described in the paper.

---

## 2) Start the server for each ablation variant

This folder does not vendor a full PEACE-patched vLLM fork.
Instead, it provides a **standardized set of environment variables** that your
server can read to enable/disable PEACE components.

Variant → environment variables:

- PEACE: *(none)*
- PEACE/PE: `PEACE_DISABLE_PREEMPTION=1`
- PEACE/Dis: `PEACE_DISABLE_DISAGGREGATION=1`
- PEACE/CoL: `PEACE_DISABLE_COLOCATION=1`
- PEACE/FSP: `PEACE_DISABLE_FAST_SP=1`

### Option A: Use `start_server.py`

You provide the server command as a string. Example for vLLM OpenAI server:

```bash
python start_server.py   --variant peace   --cmd "python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model /path/to/model"   --log-file ./logs/peace.log
```

For an ablation variant (example: PEACE/FSP):

```bash
python start_server.py   --variant fsp   --cmd "python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model /path/to/model"   --log-file ./logs/fsp.log
```

### Option B: Start your server manually

Just export the env vars before launching the server:

```bash
export PEACE_DISABLE_FAST_SP=1
# ... then run your server command ...
```

---

## 3) Replay the workload trace (real experiments)

Run the workload driver against your server:

```bash
python run_workload.py   --base-url http://127.0.0.1:8000   --api-key EMPTY   --endpoint chat   --model /path/to/model-or-name   --trace ./traces/azure_prepared.jsonl   --out-dir ./results/peace   --concurrency 64   --arrival-scale 1.0
```

Run the ablation variants by changing `--out-dir` (and restarting the server
with the corresponding env vars). Example:

- `./results/peace`
- `./results/pe`
- `./results/dis`
- `./results/col`
- `./results/fsp`

### About queueing delay

The paper reports **queueing delay** (arrival → start of execution).
Client-only measurements generally cannot perfectly separate queueing from prefill.

This runner supports two modes:
1. If your server returns a field like `peace_metrics.queue_delay_ms`
   in the JSON stream or final response, the runner will use it directly.
2. Otherwise, it uses **TTFT** (time-to-first-token) as a conservative proxy.

Either way, all raw measurements are stored in `records.jsonl` for auditing.

---

## 4) Analyze results and generate plots

After collecting results for all variants:

```bash
python analyze_ablation.py   --results-root ./results   --out-dir ./figures
```

Outputs:
- `figures/queue_delay_percentiles.png`
- `figures/short_throughput_rps.png`
- `figures/long_jct_avg.png`
- `figures/ablation_summary.csv`

If you have a baseline “short-only” run for normalization, pass it as:

```bash
python analyze_ablation.py   --results-root ./results   --baseline-summary ./results/short_only_baseline/summary.json   --out-dir ./figures
```

---

## Repro tips

- Keep `--arrival-scale` fixed across variants.
- Pin the same model weights and server config.
- Use the same prepared trace file.

---

## Disclaimer

This folder provides the **experiment harness** needed to run the ablation study.
The actual behavior of `/PE`, `/Dis`, `/CoL`, `/FSP` depends on the serving stack
implementing the corresponding PEACE mechanisms and reading the env vars above.
