# CUDA-graph capture — `FULL_AND_PIECEWISE` on vLLM/ROCm

A zoom-in on how vLLM (and ATOM) use HIP/CUDA graphs to erase per-kernel launch overhead during
decode, and what the `FULL_AND_PIECEWISE` compilation mode actually does. On ROCm these are **HIP
graphs**, but the vLLM config names keep the `cuda_graph` spelling.

---

## 1. Why graphs matter for decode

Decode issues **one token at a time**, so each step launches dozens–hundreds of tiny kernels
(projections, RoPE, attention, norms, MoE). At small batch sizes each kernel runs for microseconds,
so **CPU launch overhead dominates** — the GPU sits idle between launches waiting on the host.

A **graph** records the entire sequence of kernel launches once, then **replays** it with a single
host call. The CPU submits one graph instead of hundreds of launches → the GPU stays fed. This only
works when the shape/control-flow is static across steps, which is true for decode (fixed batch of
1-token requests) but **not** for prefill (variable prompt lengths).

---

## 2. The two capture strategies

### Full graphs (capture everything)
Record the **entire model forward** — every layer, attention included — as one graph per captured
batch size. Lowest overhead at replay, but:
- Requires the whole forward to be shape-static and capture-safe (no data-dependent host branches,
  no dynamic allocation, no CPU sync).
- Attention with variable context lengths is the hard part — needs persistent metadata buffers
  (block tables, seqlens) that the graph reads by pointer, updated in place before replay.

### Piecewise graphs (capture the safe regions)
Capture the **graph-safe sub-regions** (the dense GEMM/norm/RoPE stretches between attention ops)
as separate graphs, and run the **dynamic parts (attention) eagerly** in between. Compromise:
- Removes most launch overhead (the long GEMM chains are graphed).
- Keeps flexibility for attention shapes that can't be safely captured.
- vLLM's `torch.compile` piecewise backend splits the fx graph at attention boundaries and captures
  each dense piece.

---

## 3. `FULL_AND_PIECEWISE` — use both

`FULL_AND_PIECEWISE` is a compilation/cudagraph mode that keeps **both** captures available and picks
per step:

| Situation | What runs |
|---|---|
| Pure decode, batch size matches a captured bucket | **full graph** — single replay, minimum overhead |
| Mixed / prefill / uncaptured shape | **piecewise graphs** for dense regions + eager attention |

So you get the best-case full-graph replay when the batch is "nice" (steady-state decode), and a safe
piecewise fallback otherwise — no crash on shapes that can't be fully captured.

Related knobs:
- `cudagraph_capture_sizes` — the batch-size buckets to capture (decode is padded up to the nearest
  captured size).
- Warmup pass captures each bucket once at startup (adds startup time, saves per-step time forever).

---

## 4. What makes attention capture-safe (the MLA/persistent-buffer trick)

For attention to live inside a full graph, its metadata must be at **fixed addresses** the graph can
replay against, updated in place each step:

- **Persistent metadata buffers**: block tables, `seqlens`, slot mappings live in pre-allocated
  tensors; the runner writes new values into them before `graph.replay()` — the captured kernels read
  the same pointers.
- **Fixed max shapes**: buffers sized to the max captured batch × max blocks; unused entries masked.
- This is exactly why the AITER MLA path advertises "persistent metadata buffers for CUDA-graph
  `FULL_AND_PIECEWISE`" — its assembly decode kernel is written to consume these fixed buffers so the
  whole DeepSeek/Kimi decode forward can be captured as a full graph.

---

## 5. Interaction with `torch.compile`

vLLM layers this on `torch.compile`:

```
torch.compile (fx graph)
   └─ split at attention ops  → piecewise regions
        └─ each dense region: Inductor-compiled + HIP-graph captured
   └─ if full-graph-safe: capture the whole forward per batch bucket
runtime: pick full replay vs piecewise+eager per step (FULL_AND_PIECEWISE)
```

Net effect on decode: launch overhead collapses from hundreds of host submits to ~one, which is a
large TPOT win at low/medium concurrency where the GPU would otherwise starve.

## 6. Quick mental model of the modes

| Mode | Capture | Use when |
|---|---|---|
| `NONE` | nothing (eager) | debugging, max flexibility |
| `PIECEWISE` | dense regions only | attention shapes vary a lot |
| `FULL` | entire forward | fully static decode only |
| `FULL_AND_PIECEWISE` | both, pick per step | production default — full when possible, safe fallback otherwise |

## Sources

- [vLLM `torch.compile` / cudagraph design docs](https://docs.vllm.ai/en/latest/design/compilation.html)
- [Beyond Porting: vLLM on AMD ROCm](https://vllm.ai/blog/2026-02-27-rocm-attention-backend)
- [HIP graph API (ROCm)](https://rocm.docs.amd.com/) · [ATOM](https://github.com/ROCm/ATOM)
