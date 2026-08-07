# Disaggregated prefill / decode

A zoom-in on splitting the two inference phases onto **separate GPU pools** so each can be scheduled,
scaled, and tuned independently — and how the KV cache is handed off between them (where MORI / KV
transfer comes in).

---

## 1. Why disaggregate at all

Prefill and decode have opposite resource profiles, and colocating them on the same GPUs forces a bad
compromise:

| | Prefill | Decode |
|---|---|---|
| Work | process the whole prompt at once | 1 token/step per request |
| Bottleneck | **compute** (matrix cores) | **memory bandwidth** (KV reads) |
| Latency metric | **TTFT** (time to first token) | **TPOT** (time per output token) |
| Batch behavior | bursty, big | steady, many concurrent streams |

In a **colocated** server, continuous batching mixes both. A long prefill can **head-of-line block**
decode steps of other requests → TPOT jitter; conversely reserving capacity for decode wastes matrix
cores during prefill. You can't independently tune batch policy, parallelism, or replica count.

**Disaggregation** puts prefill on one pool and decode on another:

```
request ─▶ [ PREFILL pool ]  ──ship KV──▶  [ DECODE pool ]  ─▶ stream tokens
           compute-bound                    bandwidth-bound
           optimize TTFT                     optimize TPOT
```

---

## 2. The lifecycle of one request

```
1. Router sends prompt to a PREFILL worker.
2. Prefill worker runs the full forward over the prompt:
     - computes logits for the first output token
     - populates the KV cache for every prompt token
3. KV handoff: the prompt's KV cache (all layers) is transferred to a DECODE worker.
4. Decode worker resumes generation from token 1, reading the received KV cache,
   emitting one token per step until EOS / max tokens.
```

The **KV handoff (step 3)** is the crux: it's a large, layer-by-layer tensor transfer that must be
fast enough not to erase the benefit of splitting.

---

## 3. The KV transfer (where it gets hard)

Per request you move `num_layers × 2 × seqlen × kv_dim` worth of cache. Techniques:

- **Layer-by-layer / pipelined transfer**: start shipping layer 0's KV while prefill is still
  computing later layers — overlap transfer with compute so handoff latency is largely hidden.
- **Fast transport**: intra-node XGMI / Infinity Fabric; inter-node RDMA (RoCE/IB). AMD uses the same
  **MORI** primitives that handle MoE dispatch to also move KV between pools.
- **Quantized KV in flight** (FP8) to cut wire bytes.
- **Layout awareness**: if decode consumes the shuffled KV layout, either ship shuffled or shuffle on
  arrival — avoid a double conversion.
- **Paged, not contiguous**: KV lives in paged blocks; the transfer moves block lists + a remapping so
  the decode worker's block manager can adopt them.

If the transfer is slower than a prefill step saved, disaggregation loses — so the KV path is the
make-or-break engineering piece.

---

## 4. Independent scaling & tuning (the payoff)

Once split, each pool is tuned on its own axis:

| Knob | Prefill pool | Decode pool |
|---|---|---|
| Replica count | scale to hit TTFT SLO under prompt burstiness | scale to hold TPOT across concurrent streams |
| Parallelism | TP/EP sized for compute throughput | TP/DP sized for bandwidth + KV capacity |
| Batching | large batches, chunked prefill | many-stream continuous decode |
| Attention backend | prefill/extend kernels matter most | assembly decode kernel matters most |

You can even run **different GPU SKUs** per pool (e.g. compute-dense parts for prefill, bandwidth-rich
parts for decode).

---

## 5. Trade-offs / when NOT to

- Adds a **network hop + KV transfer** to every request → only wins when prefill/decode interference
  or independent scaling outweighs that cost (typically high-concurrency, long-context serving).
- More moving parts: a router, KV transfer service, and failure handling if a decode worker dies
  mid-stream.
- For small deployments or short outputs, **colocated continuous batching** (with the 3-path attention
  routing) is simpler and often enough.

---

## 6. How it composes with the rest of the stack

- **vLLM** provides the scheduler, paged KV manager, and (increasingly) the disaggregation
  orchestration / connector interfaces.
- **ATOM/AITER** provide the phase-specialized kernels each pool leans on (prefill flash-attn vs
  assembly decode / MLA decode).
- **MORI** provides the KV transport (and MoE dispatch if the model is MoE), across nodes.
- Pairs naturally with the [3-path attention routing](rocm-aiter-fa-3path.md) (each pool mostly runs
  one path) and [full-graph decode capture](cuda-graph-capture.md) (the decode pool is steady-state,
  ideal for full HIP-graph replay).

## Sources

- [Single Node and Distributed Inference on MI355X (AMD)](https://www.amd.com/en/developer/resources/technical-articles/2026/distributed-inference-performance-on-instinct-mi355x-gpu.html)
- [Scaling AI Inference on MI355X (ROCm blogs)](https://rocm.blogs.amd.com/artificial-intelligence/scaling-ai-inference/README.html)
- [vLLM disaggregated prefill docs](https://docs.vllm.ai/en/latest/features/disagg_prefill.html) · [ATOM](https://github.com/ROCm/ATOM)
