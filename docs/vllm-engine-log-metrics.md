# vLLM engine log metrics — reading the APIServer throughput lines

What the periodic **Engine 000** lines from vLLM's APIServer mean, how to tell
**prefill** from **decode**, and what to infer from a real Kimi-K3 serving log.

Related docs: [vllm-grafana-architecture.md](vllm-grafana-architecture.md),
[log-file-locations.md](log-file-locations.md),
[fixed-vs-random-dataset.md](fixed-vs-random-dataset.md).

---

## Example log lines

These lines come from vLLM's API server (`loggers.py`) and print roughly every
**10 seconds** — a rolling snapshot of engine activity:

```
(APIServer pid=2036) INFO 08-21 19:45:28 [loggers.py:310] Engine 000: Avg prompt throughput: 4096.0 tokens/s, Avg generation throughput: 13.5 tokens/s, Running: 30 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.3%, Prefix cache hit rate: 0.0%
(APIServer pid=2036) INFO 08-21 19:46:28 [loggers.py:310] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 739.2 tokens/s, Running: 32 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.4%, Prefix cache hit rate: 0.0%
```

The most important thing in this pattern is the **transition between prefill and decode**.

---

## Field reference

| Field | Meaning |
|-------|---------|
| **Avg prompt throughput** | Rate at which vLLM processes **input/prompt tokens** (prefill phase) |
| **Avg generation throughput** | Rate at which vLLM **generates output tokens** (decode phase) |
| **Running** | Requests currently being processed by the engine |
| **Waiting** | Requests queued in the scheduler, not yet running |
| **GPU KV cache usage** | Percent of **allocated** KV-cache capacity in use |
| **Prefix cache hit rate** | Percent of prompt tokens reused from the prefix cache |

**Engine 000** — index of the vLLM engine instance (single-engine deployments always show `000`).

**10-second window** — each line averages metrics over the interval since the previous log line,
not the lifetime of the server.

---

## Prefill vs decode — two very different phases

The same workload produces very different numbers depending on whether the engine is
**ingesting prompts** or **generating tokens**.

### Phase 1 — prefill-heavy

```
19:45:28  Prompt: ~4096 tok/s   Generation: ~13.5 tok/s   Running: 30
19:45:38  Prompt: ~4096 tok/s   Generation: ~13.5 tok/s   Running: 30
```

vLLM is processing incoming prompts. For a benchmark with ~4096 input tokens per request:

```
             Input
              ↓
        4096 tokens
              ↓
           PREFILL        ← compute-heavy, many tokens in parallel
              ↓
        KV cache created
              ↓
          DECODING
```

Prefill is **compute-heavy** and can process many prompt tokens in parallel — hence
**~4096 prompt tok/s** while generation is still low.

### Phase 2 — transition

```
19:46:18  Prompt: ~4096 tok/s   Generation: ~242 tok/s    Running: 32
```

Some requests finish prefill and start decoding; aggregate generation throughput rises.

### Phase 3 — decode-heavy

```
19:46:28  Prompt: 0 tok/s       Generation: ~739 tok/s    Running: 32
19:46:38  Prompt: 0 tok/s       Generation: ~733 tok/s    Running: 32
19:46:48  Prompt: 0 tok/s       Generation: ~730 tok/s    Running: 32
```

No new prompts in this window; the engine is generating output for requests already running.

```
Prompt = 0, Generation = ~739 tok/s
  → "Not processing new input; generating output for in-flight requests."
```

This is **normal** for batched serving workloads.

---

## Why generation jumps from ~13.5 → ~739 tokens/s

**Critical:** `Avg generation throughput` is **engine-wide aggregate**, not per-request speed.

```
32 running requests
       ↓
Each generates tokens (autoregressive, one token step at a time per request)
       ↓
Aggregate ≈ 739 tokens/sec across all requests
```

Rough sanity check:

```
739 / 32 ≈ 23 tok/s per request (average)
```

Individual requests differ in state (some still prefilling, some decoding). The metric is a
**windowed aggregate** — do **not** treat **13.5 tok/s** as TPOT for a single request.

| Misread | Correct read |
|---------|----------------|
| "Each request generates 13.5 tok/s" | 13.5 is total engine generation during a prefill-heavy window |
| "739 tok/s = one request's speed" | 739 is sum of output across ~32 concurrent requests |

For per-request **TPOT / ITL**, use `vllm bench serve` percentiles or Prometheus histograms —
not these log lines alone.

---

## Why prompt throughput cycles: 4096 → 0 → 4096

The workload moves in **batches**:

```
             Batch arrives
                  ↓
       ┌──────────────────┐
       │     PREFILL      │  ~4096 prompt tok/s
       └────────┬─────────┘
                ↓
       ┌──────────────────┐
       │      DECODE      │  ~700 generation tok/s, 0 prompt tok/s
       └────────┬─────────┘
                ↓
          Requests finish
                ↓
       New batch arrives → PREFILL again
```

Observed pattern in the sample log:

| Time | Prompt tok/s | Generation tok/s | Interpretation |
|------|--------------|------------------|----------------|
| 19:45:28 | ~4096 | ~13.5 | Prefill-heavy |
| 19:46:28 | 0 | ~739 | Decode-heavy |
| 19:47:08 | ~2458 | ~42 | Mixed / batch turnover |
| 19:47:18 | ~4096 | ~13.5 | New prefill wave |

Continuous benchmark traffic (e.g. `--max-concurrency 32`, random 8192-token inputs) produces
this oscillation.

---

## Running vs Waiting

```
Running: 30 → 31 → 32
Waiting: 0
```

| Signal | Meaning |
|--------|---------|
| **Running ~30–32, Waiting 0** | Scheduler is keeping concurrency full; no backlog |
| **Waiting > 0** | Requests queued — may need more capacity or lower arrival rate |
| **Running flat, Waiting rising** | Engine saturated |

If the benchmark uses `--max-concurrency 32`, **Running: 32, Waiting: 0** is expected steady state.

---

## GPU KV cache usage (~1.2–1.4%)

```
GPU KV cache usage: 1.3%
```

Very low — you are **not** near KV-cache capacity limits.

| KV usage | Typical implication |
|----------|---------------------|
| **< 10%** | Plenty of headroom for longer contexts or more concurrent seqs |
| **70–90%** | Approaching capacity; may block new requests or trigger preemption |
| **> 95%** | KV cache is likely a bottleneck |

From this log alone: **KV-cache capacity is not the bottleneck.**

Context length, concurrency, and `--gpu-memory-utilization` set the ceiling; at 1.3% usage
the limiter is elsewhere (prefill compute, decode batching, network if multi-node, etc.).

---

## Prefix cache hit rate (0.0%)

```
Prefix cache hit rate: 0.0%
```

vLLM is **not** reusing previously computed prompt prefixes.

Expected when:

- Every request has a **unique/random** prompt (common in `--dataset-name random` benchmarks)
- Prefix caching is disabled (`--no-enable-prefix-caching`)
- Shared prefixes are too short vs total prompt length

Would be **non-zero** if many requests share the same system prompt / document prefix and
prefix caching is enabled.

For random-input benchmarks, **0% is normal**.

---

## What this specific log tells us

Inferred workload shape:

```
              ~4096 input tokens (per request)
                     │
                     ↓
                 PREFILL  (~4096 prompt tok/s)
                     │
                     ↓
                KV cache
                     │
                     ↓
                  DECODE  (~700 aggregate generation tok/s)
                     │
                     ↓
              requests finish → next batch
```

**Key observations:**

1. **KV cache not under pressure** — 1.2–1.4% usage
2. **Scheduler not backing up** — `Waiting: 0`
3. **Clear prefill/decode phases** — high prompt tok/s, then zero prompt + high generation
4. **~700 tok/s generation is aggregate** — divide by `Running` for a rough per-request hint only
5. **Prefix caching not helping** — 0% hit rate (likely random prompts)

---

## Don't use these logs alone for benchmarking

These lines are useful for **live health checks** and **phase identification**. For formal
benchmarks, collect per-request metrics:

| Metric | What it measures |
|--------|------------------|
| **TTFT** | Time to first output token (prefill + scheduling) |
| **TPOT / ITL** | Time per output token (decode) |
| **E2E latency** | Request start → last token |
| **Request throughput** | Completed requests / second |
| **Output token throughput** | Total generated tokens / second (bench summary) |

Example with vLLM bench:

```bash
vllm bench serve --model /model_weights \
  --base-url http://localhost:8000 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --dataset-name random \
  --max-concurrency 32 \
  --num-prompts 320 \
  --random-input-len 8192 --random-output-len 1024
```

Then you can explain **4096 prompt tok/s → 0 prompt → ~730 generation tok/s** and pinpoint
whether the bottleneck is prefill, decode, scheduling, GPU compute, or (multi-node) RCCL/network.

---

## Quick grep while a job is running

On the jump host / node:

```bash
# Last throughput line from container logs
sudo docker logs --tail 120 kimi_k3 2>&1 | tr '\r' '\n' \
  | grep -E 'Avg (prompt|generation) throughput|Running:' | tail -5
```

See also [log-file-locations.md](log-file-locations.md) for NFS paths to `vllm_serve_<JID>.log`.

---

## References

- [vLLM metrics / observability](https://docs.vllm.ai/en/stable/serving/metrics.html)
- [vLLM bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve.html)
- Lab: [vllm-grafana-architecture.md](vllm-grafana-architecture.md) — Prometheus/Grafana view of the same signals
