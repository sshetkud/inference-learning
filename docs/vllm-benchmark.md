# What vLLM benchmark does

How **`vllm bench serve`** works in the Kimi-K3 lab — load generation, metrics, flags, and
how it differs from live server logs.

Related docs: [vllm-server.md](vllm-server.md),
[vllm-engine-log-metrics.md](vllm-engine-log-metrics.md),
[vllm-grafana-architecture.md](vllm-grafana-architecture.md),
[fixed-vs-random-dataset.md](fixed-vs-random-dataset.md).

---

## Short answer

In your lab, **`vllm bench serve`** is a **load generator + metrics collector**. It talks to an
**already running** vLLM server over HTTP — it does **not** start the model or GPUs.

```
vllm serve          ← server (loads Kimi-K3, listens :8000)
     ↑
     │  many HTTP requests
     │
vllm bench serve    ← client (sends prompts, measures results)
```

See [vllm-server.md](vllm-server.md) for what the server side does.

---

## What it actually does (step by step)

### 1. Builds a workload

Example: random prompts with fixed length:

- `--random-input-len 8192` (ISL)
- `--random-output-len 1024` (OSL)
- `--num-prompts 1280` total requests
- `--max-concurrency 32` (at most 32 in flight at once)

### 2. Fires requests at the server

Uses `/v1/completions` by default. Sends requests **async**, keeping concurrency at the limit
until all prompts finish.

### 3. Times each request

| Metric | Meaning |
|--------|---------|
| **TTFT** | Time to first token (prefill + scheduling) |
| **TPOT** | Time per output token (decode) |
| **ITL** | Inter-token latency |
| **E2E** | End-to-end latency (start → last token) |

### 4. Aggregates throughput

- **Output token throughput** (tok/s)
- **Total token throughput** (input + output)
- **Request throughput** (req/s)

### 5. Saves results

Prints a summary to stdout. With **`--save-result`**, writes JSON to NFS — Grafana reads these
as `vllm_k3_bench_*` metrics.

---

## Example (Kimi-K3 lab)

```bash
vllm bench serve --model /model_weights \
  --dataset-name random \
  --ignore-eos --temperature 0 \
  --max-concurrency 32 \
  --num-prompts 320 \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --result-filename kimi_k3_vllm_serving_c32.json
```

**Typical sweep:** concurrency **16 → 32 → 64 → 128**, one run per level (sbatch scripts or
MCP `vllm_benchmark` tool).

Inside the container:

```bash
sudo docker exec kimi_k3 bash -lc 'cd /tmp && vllm bench serve ...'
```

---

## Bench vs server logs

| | **vLLM server logs** (`Engine 000: Avg prompt throughput…`) | **`vllm bench serve`** |
|---|---|---|
| **Role** | Live health / phase view | Formal performance test |
| **Granularity** | Engine-wide, ~10s windows | Per-request + percentiles |
| **Good for** | "Prefill or decode right now?" | TTFT, TPOT, E2E, tok/s at concurrency C |
| **When** | Always while serving | After `/health` or `/v1/models` is ready |

Server logs may show ~4096 prompt tok/s and ~739 aggregate generation tok/s — bench tells you
**mean/p99 TTFT and TPOT** for that workload. See
[vllm-engine-log-metrics.md](vllm-engine-log-metrics.md).

---

## Important flags (lab)

| Flag | Why |
|------|-----|
| `--dataset-name random` | Synthetic fixed-length prompts (repeatable load) |
| `--ignore-eos` | Forces full OSL even if model emits EOS early |
| `--temperature 0` | Deterministic generation |
| `--max-concurrency` | How hard you push the scheduler / KV cache |
| `--random-range-ratio 0` | Fixed ISL/OSL (no length jitter) |
| `--base-url http://localhost:8000` | When bench runs outside the server process |
| `--percentile-metrics ttft,tpot,itl,e2el` | Latency percentiles in summary + JSON |
| `--save-result` | Write JSON for Grafana / offline analysis |

---

## What it does not do

- Does **not** load weights or allocate GPU memory (server must already be up)
- Does **not** replace Slurm — usually runs inside the same Docker container as `vllm serve`
  or via `docker exec kimi_k3`
- Does **not** test RCCL/network directly — if multi-node TP is broken, you see bad
  TTFT/throughput, but bench itself is just HTTP + timing

---

## Lab pipeline

```
Slurm → docker → vllm serve (TP=8)
                      ↓ ready
              vllm bench serve (c=16,32,64,128)
                      ↓
         JSON → NFS → Grafana benchmark dashboard
```

Result paths (typical):

```
/mnt/dcgpuval/afde/sshetkud/kimi-k3-bench/kimi_k3_vllm_serving_c32.json.<job_id>
```

Grafana **Benchmark** dashboard: [vllm-grafana-architecture.md](vllm-grafana-architecture.md).

---

## References

- [vLLM bench serve CLI](https://docs.vllm.ai/en/stable/cli/bench/serve.html)
- [vllm-engine-log-metrics.md](vllm-engine-log-metrics.md) — prefill/decode in server logs vs bench metrics
