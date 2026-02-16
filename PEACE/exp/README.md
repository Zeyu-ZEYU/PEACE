# PEACE Experiments (IPDPS'26 Camera-Ready)

This folder contains **experiment drivers and scripts** for reproducing the
results described in the PEACE IPDPS'26 camera-ready paper
(`PEACE.zip` → `paper_ipdps26_camera_ready.tex`).

The focus here is **runnable experiments (real inference)** rather than
simulation:

- The workload drivers replay a trace against an **OpenAI-compatible inference
  server** (e.g., a PEACE-patched vLLM deployment) and measure client-observed
  metrics.
- A separate folder provides the **scalability simulator** used for the paper’s
  large-scale (10K-GPU) study.

> Note: This repository does **not** vendor a complete PEACE-patched vLLM fork.
> You must run a serving stack that implements PEACE’s mechanisms (preemption,
> disaggregation, prefill–decode co-location, fast-SP) and exposes an
> OpenAI-compatible endpoint.


## Directory overview

- `run_peace.py`  
  A **PEACE-only** helper CLI to (1) prepare the Azure trace, (2) start a PEACE
  server command, and (3) replay the trace.

- `measurements/`  
  Runnable measurement experiments for the paper's **Motivation** section (request-length
  distributions, FIFO/Reservation/Priority limitations, preemption pressure).

- `ablation_ipdps26/`  
  Runnable ablation-study harness aligned with the paper’s ablations:
  **/PE**, **/Dis**, **/CoL**, **/FSP**.

- `scalability_simulation/`  
  Event-driven simulator used for the scalability study beyond the physical
  4-node cluster.

- `comparison_methods/`  
  Vendored codebases used as baselines in the paper:
  **LoongServe (FIFO)**, **FasterTransformer (Reservation)**,
  **Past-Future (Priority)**.


## Paper-aligned defaults

These defaults match the paper’s evaluation setup:

- **Trace**: Azure LLM inference trace (May 2024 sample), using recorded
  arrival times and the original output-length distribution.
- **Short vs. long**: prompt length < **4K** → short; ≥ **4K** → long.
- **Long-input resampling**: long prompts are rescaled into **100K–500K**
  tokens (while preserving the long-input shape), to reflect long-input
  workloads described in the paper.
- **Models / TP size** (paper Table “Model size and TP size”):
  - Mistral-v0.3 7B: TP=1
  - Phi-3 14B: TP=1
  - Yi 34B: TP=4
  - Llama-3.1 70B: TP=4
- **Dedicated short decode replicas** (paper “Short request decode”):
  - Mistral-v0.3 7B: 4
  - Phi-3 14B: 4
  - Yi 34B: 1
  - Llama-3.1 70B: 1


## 0) Python environment

The experiment drivers are pure Python. Create a virtualenv and install deps:

```bash
cd PEACE/exp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## 1) Download the Azure trace

See `PEACE/trace/AzureLLMInferenceDataset2024.md` for dataset links.
You typically download one of:

- `AzureLLMInferenceTrace_conv_1week.csv`
- `AzureLLMInferenceTrace_code_1week.csv`


## 2) Prepare a paper-style trace JSONL

Convert the raw CSV into a compact JSONL trace used by the workload runner.
This step also applies the paper’s **long-input resampling**.

```bash
python run_peace.py prepare-trace \
  --input-csv /path/to/AzureLLMInferenceTrace_conv_1week.csv \
  --output-jsonl ./traces/azure_prepared_100k_500k.jsonl \
  --short-threshold 4096 \
  --long-range 100000 500000 \
  --max-requests 20000
```


## 3) Start a PEACE server

You must run an **OpenAI-compatible** server that implements PEACE.

If you have a PEACE-patched vLLM OpenAI server, an example command looks like:

```bash
python run_peace.py start-server \
  --cmd "python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model /path/to/model" \
  --log-file ./logs/peace_server.log
```

If your deployment is multi-node (paper testbed: 4× p4de.24xlarge), start your
Ray cluster and then launch the PEACE server with your cluster-specific flags.


## 4) Replay the trace (real inference)

In a separate terminal (or after starting the server externally), replay the
prepared trace:

```bash
python run_peace.py run-workload \
  --base-url http://127.0.0.1:8000 \
  --api-key EMPTY \
  --endpoint chat \
  --model /path/to/model-or-name \
  --trace ./traces/azure_prepared_100k_500k.jsonl \
  --out-dir ./results/peace \
  --concurrency 64 \
  --arrival-scale 1.0
```

Outputs:

- `results/peace/records.jsonl` (per-request metrics)
- `results/peace/summary.json` (aggregated metrics)


## 5) One-command end-to-end run (optional)

If you want the script to start the server, wait for health, run the workload,
and then stop the server:

```bash
python run_peace.py end-to-end \
  --cmd "python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model /path/to/model" \
  --base-url http://127.0.0.1:8000 \
  --model /path/to/model-or-name \
  --trace ./traces/azure_prepared_100k_500k.jsonl \
  --out-dir ./results/peace \
  --concurrency 64 \
  --arrival-scale 1.0
```


## Notes on queueing delay

The paper reports **queueing delay** (arrival → start of execution).
Client-only measurements cannot always separate queueing from compute.

The workload runner supports two modes:

1. If the server returns `peace_metrics.queue_delay_ms` in streamed events or
   the final response, it will be used directly.
2. Otherwise, the runner falls back to **TTFT** (time-to-first-token) as a
   conservative proxy.

All raw records are stored in `records.jsonl` for auditing.
