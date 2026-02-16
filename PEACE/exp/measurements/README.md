# Motivation Measurements (IPDPS'26 Camera-Ready)

This directory contains **runnable (real inference) experiment code** for the
**Motivation** section of the PEACE paper:

- `PEACE.zip` → `paper_ipdps26_camera_ready.tex` → `\section{Motivation}`

These experiments are **not simulations**. They replay an Azure trace against
one or more **OpenAI-compatible** serving endpoints and compute the metrics
used in the Motivation argumentation.

> Important: This repository does **not** vendor a complete PEACE-patched vLLM
> fork. You must run a serving stack that implements PEACE’s mechanisms and
> exposes an OpenAI-compatible endpoint.
>
> For baselines (FIFO / Reservation / Priority), see:
> `PEACE/exp/comparison_methods/README.md`.


## What is included

### A) Request length distributions
- **Input** length distribution (ContextTokens)
- **Output** length distribution (GeneratedTokens)

Script:
- `plot_trace_distribution.py`

### B) FIFO head-of-line blocking (Motivation Fig. `q_delay_fifo`, `throughput_fifo`)
Compare short-request **queueing delay** and **throughput**:
- FIFO **with** long requests
- FIFO **without** long requests (short-only trace)

Scripts:
- `trace_ops.py` (filter short-only trace)
- `run_workload_routed.py` (single endpoint mode)
- `analyze_motivation.py fifo`

### C) Reservation underutilization (Motivation Table `gpu_idle_rate` + Fig. `q_delay_resv`, `throughput_resv`)
Compare FIFO vs Reservation:
- short-request queueing delay + throughput
- GPU idle rate (measured via utilization sampling)

Scripts:
- `run_workload_routed.py` (supports **routing** short/long to different endpoints)
- `gpu_idle_monitor.py`
- `analyze_motivation.py reservation` and `analyze_motivation.py gpu-idle`

### D) Priority starvation (Motivation Table `starvation_rate`)
Compute long-request starvation rate under Priority scheduling.

Scripts:
- `run_workload_routed.py`
- `analyze_motivation.py priority`

### E) Long-prefill preemption pressure (Motivation Table `num_preempt`)
Extract the **total number of long-prefill preemptions**.

In the paper, this table corresponds to the **ring-attention** (no fast-SP)
setting, i.e., the **/FSP** variant.

Scripts:
- `run_workload_routed.py`
- `analyze_motivation.py preemptions`


## 0) Python environment

Reuse the experiment environment from `PEACE/exp`:

```bash
cd PEACE/exp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## 1) Download the Azure trace

See `PEACE/trace/AzureLLMInferenceDataset2024.md`.
You typically download one of:

- `AzureLLMInferenceTrace_conv_1week.csv`
- `AzureLLMInferenceTrace_code_1week.csv`


## 2) Plot request-length distributions (Motivation Fig. distribution)

```bash
cd PEACE/exp
python measurements/plot_trace_distribution.py \
  --input-csv /path/to/AzureLLMInferenceTrace_conv_1week.csv \
  --out-dir ./motivation_figs/distribution
```

Outputs:
- `motivation_figs/distribution/distribution_input.pdf`
- `motivation_figs/distribution/distribution_output.pdf`


## 3) Prepare a paper-style trace JSONL (for real inference replay)

This applies the paper defaults:
- short threshold: 4096
- long resampling range: 100K–500K

```bash
cd PEACE/exp
python measurements/trace_ops.py prepare \
  --input-csv /path/to/AzureLLMInferenceTrace_conv_1week.csv \
  --output-jsonl ./traces/azure_prepared_100k_500k.jsonl \
  --short-threshold 4096 \
  --long-range 100000 500000 \
  --max-requests 20000
```


## 4) Generate a short-only trace (for FIFO “w/o long”)

```bash
cd PEACE/exp
python measurements/trace_ops.py filter \
  --input-jsonl ./traces/azure_prepared_100k_500k.jsonl \
  --output-jsonl ./traces/azure_short_only.jsonl \
  --keep-type short
```


## 5) FIFO head-of-line blocking experiment

### 5.1 Start the FIFO server (LoongServe)

Follow:
- `PEACE/exp/comparison_methods/README.md` → **LoongServe (FIFO)**

### 5.2 Run “with long” vs “without long”

Recommended result layout (expected by `analyze_motivation.py`):

```
results_motivation/
  fifo/
    mistral7b/
      with_long/
      without_long/
```

Example commands:

```bash
cd PEACE/exp
RESULTS=./results_motivation

# with long requests
python measurements/run_workload_routed.py \
  --base-url http://127.0.0.1:8000 \
  --api-key EMPTY \
  --endpoint chat \
  --model /path/to/model-or-name \
  --trace ./traces/azure_prepared_100k_500k.jsonl \
  --out-dir ${RESULTS}/fifo/mistral7b/with_long \
  --concurrency 64 \
  --arrival-scale 1.0

# without long requests (short-only)
python measurements/run_workload_routed.py \
  --base-url http://127.0.0.1:8000 \
  --api-key EMPTY \
  --endpoint chat \
  --model /path/to/model-or-name \
  --trace ./traces/azure_short_only.jsonl \
  --out-dir ${RESULTS}/fifo/mistral7b/without_long \
  --concurrency 64 \
  --arrival-scale 1.0
```

Repeat the same layout for other paper models:
- `phi3_14b`, `yi34b`, `llama31_70b`

### 5.3 Analyze & plot

```bash
cd PEACE/exp
python measurements/analyze_motivation.py fifo \
  --results-root ./results_motivation \
  --out-dir ./results_motivation/figs
```

Outputs:
- `figs/q_delay_fifo.pdf`
- `figs/throughput_fifo.pdf`


## 6) Reservation underutilization experiment

### 6.1 Start the Reservation baseline servers (FasterTransformer)

Follow:
- `PEACE/exp/comparison_methods/README.md` → **FasterTransformer (Reservation)**

You will run **two endpoints**:
- a short-request endpoint (smaller context)
- a long-request endpoint (supports 100K–500K tokens)

### 6.2 Run FIFO reference (full trace)

```bash
cd PEACE/exp
RESULTS=./results_motivation

python measurements/run_workload_routed.py \
  --base-url http://127.0.0.1:8000 \
  --api-key EMPTY \
  --endpoint chat \
  --model /path/to/model-or-name \
  --trace ./traces/azure_prepared_100k_500k.jsonl \
  --out-dir ${RESULTS}/reservation/mistral7b/fifo \
  --concurrency 64 \
  --arrival-scale 1.0
```

### 6.3 Run Reservation (routing short/long)

```bash
cd PEACE/exp
RESULTS=./results_motivation

python measurements/run_workload_routed.py \
  --route-by-type \
  --base-url-short http://127.0.0.1:8001 \
  --base-url-long  http://127.0.0.1:8002 \
  --api-key EMPTY \
  --endpoint chat \
  --model /path/to/model-or-name \
  --trace ./traces/azure_prepared_100k_500k.jsonl \
  --out-dir ${RESULTS}/reservation/mistral7b/reservation \
  --concurrency 64 \
  --arrival-scale 1.0
```

### 6.4 GPU idle-rate monitoring (Table `gpu_idle_rate`)

Run the monitor on **each server node** while the workload is running.

Example (on each node):

```bash
cd PEACE/exp
python measurements/gpu_idle_monitor.py \
  --output ./results_motivation/gpu_idle/fifo/mistral7b/node0.json
```

Stop it (Ctrl+C) after the workload finishes.
Repeat for `reservation/` as well:

```bash
python measurements/gpu_idle_monitor.py \
  --output ./results_motivation/gpu_idle/reservation/mistral7b/node0.json
```

### 6.5 Analyze & plot

```bash
cd PEACE/exp
python measurements/analyze_motivation.py reservation \
  --results-root ./results_motivation \
  --out-dir ./results_motivation/figs

python measurements/analyze_motivation.py gpu-idle \
  --results-root ./results_motivation \
  --out-dir ./results_motivation/tables
```

Outputs:
- `figs/q_delay_resv.pdf`
- `figs/throughput_resv.pdf`
- `tables/gpu_idle_rate.csv`


## 7) Priority starvation experiment

### 7.1 Start the Priority server (Past-Future)

Follow:
- `PEACE/exp/comparison_methods/README.md` → **Past-Future (Priority)**

### 7.2 Run the workload

Because long requests may be starved indefinitely, use a finite timeout.

```bash
cd PEACE/exp
RESULTS=./results_motivation

python measurements/run_workload_routed.py \
  --base-url http://127.0.0.1:8010 \
  --api-key EMPTY \
  --endpoint chat \
  --model /path/to/model-or-name \
  --trace ./traces/azure_prepared_100k_500k.jsonl \
  --out-dir ${RESULTS}/priority/mistral7b \
  --concurrency 64 \
  --arrival-scale 1.0 \
  --request-timeout-s 300
```

### 7.3 Analyze starvation

```bash
cd PEACE/exp
python measurements/analyze_motivation.py priority \
  --results-root ./results_motivation \
  --out-dir ./results_motivation/tables
```

Output:
- `tables/starvation_rate.csv`


## 8) Preemption count experiment (Motivation Table `num_preempt`)

This table corresponds to **/FSP** (fast-SP disabled), i.e., ring attention.

### 8.1 Start a PEACE server with fast-SP disabled

Set the same environment variable used by the ablation harness:

```bash
export PEACE_DISABLE_FAST_SP=1
```

Then start your PEACE server (OpenAI-compatible). Example:

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model /path/to/model
```

### 8.2 Run the workload

```bash
cd PEACE/exp
RESULTS=./results_motivation

python measurements/run_workload_routed.py \
  --base-url http://127.0.0.1:8000 \
  --api-key EMPTY \
  --endpoint chat \
  --model /path/to/model-or-name \
  --trace ./traces/azure_prepared_100k_500k.jsonl \
  --out-dir ${RESULTS}/preemptions/mistral7b/fsp \
  --concurrency 64 \
  --arrival-scale 1.0
```

### 8.3 Extract the table

```bash
cd PEACE/exp
python measurements/analyze_motivation.py preemptions \
  --results-root ./results_motivation \
  --out-dir ./results_motivation/tables \
  --variant fsp
```

Output:
- `tables/preemptions_total.csv`


## 9) One-shot analysis

If you have populated the recommended `results_motivation/` layout, run:

```bash
cd PEACE/exp
python measurements/analyze_motivation.py all \
  --results-root ./results_motivation \
  --out-dir ./results_motivation/figs
```

(For CSV tables, `gpu-idle`, `priority`, and `preemptions` also write under the
same `--out-dir`.)


## Notes on queueing delay

The paper reports **queueing delay** (arrival → start of execution).

Client-only measurements cannot always separate queueing from compute.
The runner uses:

1) `peace_metrics.queue_delay_ms` if the server provides it, or
2) falls back to `ttft_s` (time-to-first-token) as a conservative proxy.

All raw records are written to `records.jsonl` for auditing.
