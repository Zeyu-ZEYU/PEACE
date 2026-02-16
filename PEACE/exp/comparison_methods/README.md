# Comparison Methods (Baselines Used in the PEACE Paper)

This folder vendors the three baseline systems used in the PEACE IPDPS'26
camera-ready evaluation (`paper_ipdps26_camera_ready.tex`).

Mapping to the paper’s comparison methods:

- **FIFO** → `LoongServe/` (LoongServe scheduler baseline)
- **Reservation** → `FasterTransformer/` (GPU reservation for long requests)
- **Priority** → `Past-Future/` (Past-Future scheduler)

> Important: Each baseline is a full external codebase with its own build and
> runtime requirements. The most reliable instructions are the upstream
> READMEs shipped inside each subdirectory.


## 1) LoongServe (FIFO)

Location: `LoongServe/`

Where to start:

- `LoongServe/README.md`
- `LoongServe/docs/artifact-eval/README.md` (detailed artifact instructions)

Quick hint (server entry point):

- `python -m loongserve.longserve_server.api_server ...`

LoongServe also exposes an OpenAI-compatible endpoint:

- `POST /v1/chat/completions`


## 2) FasterTransformer (Reservation)

Location: `FasterTransformer/`

Paper definition of “Reservation”:

- Pre-allocate enough GPUs (and memory) to serve **500K-token** long-input
  requests (typically **4–6 replicas**, depending on model size).
- Dedicate those GPUs to long requests (100K–500K input tokens).
- Reserve the remaining GPUs exclusively for all other short requests.

Where to start:

- `FasterTransformer/README.md`

Operationally, this baseline is typically implemented as **two deployments**:

1. A *long-request* server running on the reserved “long” GPUs.
2. A *short-request* server running on the remaining GPUs.

Your client/workload driver must then route requests to the correct endpoint
based on input length.


## 3) Past-Future (Priority)

Location: `Past-Future/`

Where to start:

- `Past-Future/README.md`
- LightLLM docs referenced by that README

Quick hint (server entry point):

- `python -m lightllm.server.api_server ...`

Paper definition of “Priority”:

- Requests with **100K–500K** input tokens are assigned **low priority**.
- All other short requests are assigned **high priority**.

How priority is expressed depends on the Past-Future/LightLLM version you run
(e.g., request fields, headers, or scheduler configuration). Please follow the
baseline’s documentation and ensure the same policy is applied during
evaluation.
