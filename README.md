# PEACE: Preemptive and Efficient Cluster Scheduling for LLM Inference with Mixed Prompts

This repository contains the code and experiment drivers for the **PEACE**
research system (IPDPS'26 camera-ready).

PEACE targets **multi-tenant LLM inference** with **mixed short and long
prompts** (e.g., long-context 100K–500K tokens). It combines:

- **Prefill preemption** to protect short requests from head-of-line blocking.
- **Disaggregation** of short-request decode.
- **Prefill–decode co-location** to reduce KV movement.
- **Fast-SP** (fast sequence parallelism) to reduce long-prefill runtime and
  mitigate preemption pressure.


## Repository structure

The PEACE-specific code lives under `./PEACE`:

- `PEACE/exp/`
  **Runnable paper experiments (real inference)** and analysis scripts.
  - `run_peace.py`: helper CLI to prepare traces, start a server, replay a trace
  - `measurements/`: Motivation section measurements (FIFO/Reservation/Priority)
  - `ablation_ipdps26/`: ablation study (**/PE**, **/Dis**, **/CoL**, **/FSP**)
  - `comparison_methods/`: baseline codebases (LoongServe/FasterTransformer/Past-Future)
  - `scalability_simulation/`: event-driven simulator for the 10K-GPU study

- `PEACE/trace/`
  Trace documentation (Azure LLM inference trace download pointers).

- `PEACE/kernels/` and `PEACE/fast_sp/`
  Implementation components used by PEACE's **fast-SP** path (CUDA extensions,
  Megatron-based sequence-parallel components).

- `PEACE/measurement/`
  Legacy measurement prototypes (simulation-style). For **paper reproduction**,
  prefer the runnable scripts under `PEACE/exp/`.

The rest of the repository is based on a Megatron-LM codebase that PEACE
leverages for sequence-parallel and long-context components.


## Quick start (run a trace against a PEACE server)

> The experiment drivers assume an **OpenAI-compatible** server endpoint.
> This repository does **not** vendor a complete PEACE-patched vLLM fork.

### 0) Python environment

```bash
cd PEACE/exp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Download the Azure trace

See `PEACE/trace/AzureLLMInferenceDataset2024.md`.

### 2) Prepare a paper-style trace JSONL

This step preserves arrival times and output-length distribution, and rescales
long requests into the paper's **100K–500K** token range.

```bash
cd PEACE/exp
python run_peace.py prepare-trace \
  --input-csv /path/to/AzureLLMInferenceTrace_conv_1week.csv \
  --output-jsonl ./traces/azure_prepared_100k_500k.jsonl \
  --short-threshold 4096 \
  --long-range 100000 500000 \
  --max-requests 20000
```

### 3) Start your PEACE server

Example (single node):

```bash
python run_peace.py start-server \
  --cmd "python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model /path/to/model" \
  --log-file ./logs/peace_server.log
```

### 4) Replay the trace

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
- `results/peace/summary.json` (aggregated percentiles, throughput, JCT, etc.)


## Reproducing paper experiments

The paper's evaluation section is `\section{Performance Evaluation}` in
`paper_ipdps26_camera_ready.tex`. The code below is organized to match those
subsections.

### 1) Motivation measurements (Section “Motivation”)

**Code:** `PEACE/exp/measurements/`

This reproduces the Motivation evidence:

- Request length distributions
- FIFO head-of-line blocking (with vs without long requests)
- Reservation underutilization (GPU idle rate) + short performance impact
- Priority starvation (long requests)
- Long-prefill preemption pressure (Table `num_preempt`)

Follow the step-by-step guide:

- `PEACE/exp/measurements/README.md`

### 2) Overall performance (Section “Overall Performance”)

The paper compares **FIFO**, **Reservation**, **Priority**, and **PEACE** on:

- Short-request queueing delay percentiles
- Short-request throughput
- Long-request average JCT

**How to run:**

1. Prepare a trace JSONL with `run_peace.py prepare-trace`.
2. Start each serving system (PEACE + baselines).
3. Replay the same prepared trace against each system.

For baselines, use the instructions under:

- `PEACE/exp/comparison_methods/README.md`

For replaying the trace you can use:

- `PEACE/exp/run_peace.py run-workload` (single endpoint)
- `PEACE/exp/measurements/run_workload_routed.py` (supports Reservation routing)

Each run produces a `summary.json` containing:
- `short_queue_delay_s` percentiles
- `short_throughput_rps`
- `long_jct_s` (mean, p50, p99, etc.)

These fields are sufficient to recreate the figures.

### 3) Ablation study (Section “Ablation Study”)

**Code:** `PEACE/exp/ablation_ipdps26/`

Ablations match the camera-ready paper:

- **/PE**   (disable preemption)
- **/Dis**  (disable disaggregation)
- **/CoL**  (disable prefill–decode co-location)
- **/FSP**  (disable fast-SP; fall back to ring attention)

Follow:

- `PEACE/exp/ablation_ipdps26/README.md`

The ablation harness includes an analysis helper:

- `PEACE/exp/ablation_ipdps26/analyze_ablation.py`

### 4) Time overhead (Section “Time Overhead”)

The paper reports overhead sources such as:

- KV migration overhead
- Preemption overhead
- fast-SP overhead vs ring attention

These overheads require **server-side instrumentation** (e.g., logging internal
scheduler timestamps and migration bytes). The client replay drivers still
produce end-to-end metrics in `records.jsonl` / `summary.json`.

If your serving stack exposes additional timing fields (e.g., via
`peace_metrics` in SSE events), they will be persisted in `records.jsonl` and
can be post-processed.

### 5) Scalability test (Section “Scalability Test”)

**Code:** `PEACE/exp/scalability_simulation/`

The paper's 10K-GPU study is reproduced via an **event-driven simulator**.
See:

- `PEACE/exp/scalability_simulation/README.md`


## Baselines (comparison methods)

Baseline implementations used in the paper are vendored under:

- `PEACE/exp/comparison_methods/`

They include:

- **LoongServe** (FIFO scheduling)
- **FasterTransformer** (Reservation scheduling)
- **Past-Future** (Priority scheduling)

Each baseline has its own build/start instructions. A consolidated guide is in:

- `PEACE/exp/comparison_methods/README.md`


## Building PEACE fast-SP CUDA extensions (optional)

If you are working on fast-SP kernels, `PEACE/kernels/` includes CUDA
extensions with a `setup.py`.

Example:

```bash
cd PEACE/kernels
python setup.py install
```

This requires a working CUDA toolchain and PyTorch with CUDA enabled.


## Notes on queueing delay

The paper reports **queueing delay** (arrival → start of execution).
Client-only measurements cannot always separate queueing from compute.

The replay drivers support two modes:

1. If the server returns `peace_metrics.queue_delay_ms` in streamed events or
   the final response, it will be used.
2. Otherwise, the runner falls back to **TTFT** (time-to-first-token) as a
   conservative proxy.

All raw per-request records are stored in `records.jsonl` for auditing.


## License

See `LICENSE`.
