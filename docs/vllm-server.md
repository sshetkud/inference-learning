# What the vLLM server does

How **`vllm serve`** works in the Kimi-K3 lab — components, request flow, endpoints, and
what it is *not* (Slurm, Ray, RCCL, training).

Related docs: [vllm-benchmark.md](vllm-benchmark.md),
[vllm-engine-log-metrics.md](vllm-engine-log-metrics.md),
[vllm-grafana-architecture.md](vllm-grafana-architecture.md),
[multinode-vllm-ray.md](multinode-vllm-ray.md).

---

## Short answer

The vLLM server is a **long-running inference process** that:

1. **Loads** a large language model onto GPU(s)
2. **Accepts HTTP requests** (OpenAI-compatible API on port **8000**)
3. **Schedules** many concurrent requests onto the GPUs
4. **Runs inference** in two phases: **prefill** (read the prompt) then **decode** (generate tokens)
5. **Streams or returns** the generated text

In your lab, that's the **`kimi_k3` Docker container** on the compute node, started by Slurm.

---

## What you start

When you run something like:

```bash
vllm serve /model_weights \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 --port 8000
```

vLLM starts several cooperating pieces:

```
Client (curl, bench, Grafana MCP)
        ↓ HTTP :8000
   ┌─────────────────┐
   │   APIServer     │  ← OpenAI API (/v1/chat/completions, /v1/models)
   │   (FastAPI)     │  ← /health, /metrics
   └────────┬────────┘
            ↓
   ┌─────────────────┐
   │ Engine /        │  ← scheduler, batching, KV cache
   │ EngineCore      │  ← actual model forward passes on GPU
   └────────┬────────┘
            ↓
      GPU 0–7 (TP=8 on one MI355X node)
```

| Piece | Job |
|-------|-----|
| **APIServer** | HTTP front door; parses requests, returns responses; prints those **Engine 000** log lines every ~10s |
| **Engine / EngineCore** | Owns the model weights, GPU memory, scheduler, and inference loop |
| **Scheduler** | Decides which requests run together, who prefills vs decodes, queue (`Running` / `Waiting`) |
| **KV cache** | Stores attention state per request so decode can generate one token at a time efficiently |

Multi-node (Ray) adds worker processes on other nodes, but there is still **one API entry point**
on the head (port 8000). See [multinode-vllm-ray.md](multinode-vllm-ray.md).

---

## What it does for each request

For a chat/completion request:

```
1. Receive prompt tokens
2. PREFILL  — process all input tokens in parallel → fill KV cache
3. DECODE   — generate output tokens one step at a time (autoregressive)
4. Return   — stream or batch JSON response
```

That's why server logs oscillate:

- **High prompt throughput (~4096 tok/s)** → many requests prefilling long inputs
- **High generation throughput (~700 tok/s), prompt = 0** → decode-only window for in-flight requests

See [vllm-engine-log-metrics.md](vllm-engine-log-metrics.md).

---

## What it exposes (endpoints you use)

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Is the engine ready? |
| `GET /v1/models` | Model list (exporter uses this for "server up") |
| `POST /v1/chat/completions` | Chat inference (Kimi-K3) |
| `POST /v1/completions` | Text completion |
| `GET /metrics` | Prometheus metrics (throughput, latency histograms, queue depth) |

The Grafana stack on the jump host polls `:8000/metrics` via `srun` into the compute node —
see [vllm-grafana-architecture.md](vllm-grafana-architecture.md).

---

## What it is not

| Not vLLM server | Actually |
|-----------------|----------|
| **Slurm** | Allocates the node/GPUs; vLLM runs *inside* the job |
| **Ray** | Only needed when one engine spans multiple nodes (TP/PP + Ray backend) |
| **RCCL** | GPU-to-GPU comms during TP; vLLM uses it, but vLLM ≠ RCCL |
| **nginx / load balancer** | Distributes across **replicas**; each replica is its own vLLM server |
| **Training** | Inference only — load weights, serve tokens |

---

## Kimi-K3 single-node pattern (lab)

```
Slurm job → Docker kimi_k3 → vllm serve (TP=8, mp backend)
                                    ↓
                            OpenAI API :8000
                                    ↓
              bench / MCP vllm_chat / clients hit the node
```

One node, one engine, 8 GPUs sharded with **tensor parallelism** — no Ray.

To measure performance after the server is ready, use [vllm-benchmark.md](vllm-benchmark.md).

---

## References

- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)
- [vLLM metrics](https://docs.vllm.ai/en/stable/serving/metrics.html)
