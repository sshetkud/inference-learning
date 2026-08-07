# `ROCM_AITER_FA` — the MHA 3-path routing backend

A zoom-in on the recommended AMD MHA attention backend in vLLM (auto-selected when
`VLLM_ROCM_USE_AITER=1`). Instead of one unified kernel, it routes each token type in a mixed
continuous-batching step to a specialized kernel — the direct answer to the "every batch is a mixed
workload" problem.

---

## 1. The problem it solves

One decoding step contains three token populations with opposite hardware needs:

| Phase | Tokens | Bottleneck | Ideal kernel |
|---|---|---|---|
| Prefill | fresh prompt, thousands at once | compute-bound (matrix cores) | big-tile flash-attn |
| Extend | new tokens for a partly-cached request (chunked prefill, prefix reuse, multi-turn) | hybrid | chunked ctx-attn + merge |
| Decode | 1 token/req, reads whole KV cache | memory-bandwidth-bound | assembly paged-attn |

A single kernel tuned for one phase penalizes the others. `ROCM_AITER_FA` splits the batch and sends
each segment to a kernel built for it.

---

## 2. Batch reordering (the enabler)

The model runner reorders every mixed batch so tokens of the same type are contiguous in memory:

```
incoming (arbitrary order):  [P E D P D E D ...]
reordered:                   [ D D D | E E | P P ]
                               decode  extend  prefill
```

- Controlled by `reorder_batch_threshold=1`: a request with `> 1` query token this step is prefill/extend;
  exactly `1` query token is decode.
- Contiguity means each kernel does one coalesced pass over its segment — **no redundant KV fetches**
  and no per-token branching inside the kernel.
- The metadata builder produces per-segment cumulative seqlen arrays (`cu_seqlens_q`, `cu_seqlens_k`)
  and block tables so each kernel gets exactly the slice it owns.

---

## 3. The three paths

### Path 1 — Prefill: `flash_attn_varlen_func`
- Variable-length flash attention over the packed prompt tokens (AITER MHA → CK or assembly).
- Causal masking; large tiles keep CDNA matrix cores busy.
- No KV cache reads for the prompt's own tokens (they're computed in-place), only writes to cache.

### Path 2 — Extend: chunked context attention + LSE merge
- For requests whose KV cache already has context (100K+), the context is split into chunks (~32K).
- `cp_mha_gather_cache` (Triton) gathers cached K/V **out of the shuffled layout into standard layout**
  because AITER's long-context MHA kernel can't read the shuffled form.
- Attention is computed per chunk; each chunk emits a partial output + **log-sum-exp (LSE)**.
- **LSE merge** combines partial outputs: `out = Σ softmax-weighted chunk outputs`, rescaled by each
  chunk's LSE so the global softmax is exact — chunking never corrupts the result numerically.
- The fresh (new) tokens attend to both the merged context and each other (causal).

### Path 3 — Decode: `pa_fwd_asm` (hand-tuned assembly paged attention)
- 1 query token per request; reads the whole paged KV cache for that request.
- Runs **directly on the shuffled KV layout** — zero conversion overhead:

```
k_cache: [num_blocks, num_heads, head_dim // x, block_size, x]
v_cache: [num_blocks, num_heads, block_size // x, head_dim, x]
```

- Streaming softmax in registers; the assembly schedule saturates HBM3 bandwidth (decode is
  bandwidth-bound, so the goal is "never stall the load engine").

---

## 4. The shuffled KV cache (why layout is a first-class lever)

A custom `reshape_and_cache_flush` writes new K/V into the cache **already shuffled**, so the
common-case decode path pays zero swizzle cost → ~15–20% decode throughput from layout alone.
The trade-off lands entirely on the extend path, which must gather back to standard layout
(`cp_mha_gather_cache`) — an intentional bet that decode steps vastly outnumber extend steps.

---

## 5. End-to-end per step

```
1. runner reorders batch → [decode | extend | prefill], builds per-segment metadata
2. reshape_and_cache_flush: write new K/V into shuffled paged cache
3. decode  segment → pa_fwd_asm            (shuffled layout, bandwidth-bound)
4. extend  segment → gather + chunk attn + LSE merge  (numerically exact long ctx)
5. prefill segment → flash_attn_varlen_func (compute-bound, big tiles)
6. concatenate outputs back into original request order → next layer
```

## 6. Where it sits among ROCm MHA backends

| Backend | Paths | Notes |
|---|---|---|
| `TRITON_ATTN` | unified | baseline, Radeon support |
| `ROCM_ATTN` | 2-path (legacy) | older split, Radeon support |
| `ROCM_AITER_UNIFIED_ATTN` | unified AITER | single-kernel AITER path |
| **`ROCM_AITER_FA`** | **3-path** | **recommended; auto-selected with AITER** |

Measured: `ROCM_AITER_FA` vs legacy `ROCM_ATTN` ≈ **2.7–4.4× TPS** (model/HW/concurrency dependent;
~3.6× on Qwen3-235B MHA, MI355X, TP8, 64 concurrency). MLA models use the sibling `ROCM_AITER_MLA`
backend instead.

## Sources

- [Beyond Porting: vLLM on AMD ROCm (attention backends)](https://vllm.ai/blog/2026-02-27-rocm-attention-backend)
- [AITER](https://github.com/ROCm/aiter) · [ATOM](https://github.com/ROCm/ATOM)
