# `mla_decode_fwd` — the absorbed-MLA assembly decode kernel

A zoom-in on the single kernel that dominates decode throughput for DeepSeek / Kimi–class
models on AMD Instinct (CDNA 3/4). This is the "absorbed" MLA decode path used by both
`ROCM_AITER_MLA` and `ROCM_AITER_TRITON_MLA` (they share this kernel; they differ only in prefill).

---

## 1. What MLA is (and why it exists)

Standard Multi-Head Attention (MHA) caches a full **K** and **V** per head per token. For a model
with `n_h` heads × `d_h` head-dim, the KV cache per token is `2 · n_h · d_h` values — for large
models that's ~8K+ values/token and it's the #1 memory consumer during decode.

**Multi-head Latent Attention (MLA)** caches a single **low-rank latent** `c_KV` per token instead
of per-head K and V:

```
c_KV  = W_DKV · h        # compressed latent, dim d_c  (e.g. 512)
k_rope = RoPE(W_KR · h)  # a small decoupled RoPE key,  dim d_r (e.g. 64)
```

So the cache stores `[c_KV (512) ‖ k_rope (64)] = 576` values/token — regardless of head count.
For DeepSeek-V3 that's roughly a **14× smaller KV cache** than the equivalent MHA. Smaller cache =
fewer bytes read per decode step = higher token/s on a bandwidth-bound phase.

---

## 2. The "absorption" trick (the math)

Naively you'd decompress the latent back into per-head K and V every step:

```
K_i = W_UK_i · c_KV        # up-project to head i's key
V_i = W_UV_i · c_KV        # up-project to head i's value
```

That reintroduces the big per-head tensors you were trying to avoid. **Matrix absorption** folds
those up-projections into the neighboring projection matrices so you never materialize per-head K/V:

For the **score** `q_iᵀ K_i`:

```
q_iᵀ K_i = q_iᵀ (W_UK_i · c_KV) = (W_UK_iᵀ · q_i)ᵀ · c_KV
                                  └────── absorb into the Q projection ──────┘
```

Define `q̃_i = W_UK_iᵀ · q_i`. Then the score is just `q̃_iᵀ · c_KV` — a dot product against the
**shared 512-dim latent**, computed once per token and reused by all heads.

For the **output** `Σ_j a_ij V_i(j)`:

```
out_i = Σ_j a_ij (W_UV_i · c_KV(j)) = W_UV_i · (Σ_j a_ij c_KV(j))
                                       └── attend in latent space, up-project ONCE at the end ──┘
```

So the whole attention runs in the compressed 512-dim latent space, and `W_UV_i` (and the
`k_rope` term, kept separate because RoPE is position-dependent and can't be absorbed) are applied
outside the inner loop. Net effect:

- **Reads** the 576-dim latent from HBM (small, coalesced).
- **No** per-head K/V materialization → far less HBM traffic and no big intermediate writes.
- More FLOPs get done per byte read → the kernel becomes arithmetic-efficient on a bandwidth-bound
  problem, which is exactly what you want for decode.

> Prefill/extend do **not** absorb — with many query tokens the up-projected form is cheaper and more
> numerically convenient, so those phases run the non-absorbed MHA-style kernels. Absorption is a
> **decode-only** win (1 query token, huge cache).

---

## 3. Why it's hand-written assembly

Triton/CK can express this, but the decode inner loop is tiny and latency-sensitive, so the
overheads matter. The AITER assembly kernel (`mla_decode_fwd` / `pa_fwd_asm` family) hard-codes:

- **VGPR/LDS allocation** and the exact `global_load_dwordx4` schedule to keep HBM3 lanes saturated
  (decode is bandwidth-bound, so the game is "never stall the load engine").
- **MFMA (matrix core) instruction selection** for the `q̃ᵀ · c_KV` and `a · c_KV` products, packed to
  the native FP16/BF16/FP8 MFMA shapes on CDNA.
- **Paged KV gather** directly in the shuffled cache layout (see below) with no format conversion.
- **Softmax in registers** with online/streaming max + LSE, so no round-trip to HBM for scores.
- **Persistent metadata buffers** so it composes with CUDA-graph capture (`FULL_AND_PIECEWISE`) —
  the block tables / seqlen pointers live in fixed buffers the graph can replay.

### The shuffled paged-KV layout it consumes

The latent cache is stored pre-shuffled to match CDNA vector-load granularity:

```
kv_cache: [num_blocks, (d_c + d_r) // x, block_size, x]     # x = vector width (e.g. 8 for fp16)
```

so each wavefront's `global_load` pulls a contiguous, aligned chunk — zero swizzle in the kernel.
This is where a big chunk of the decode speedup comes from, on top of the algorithmic MLA win.

---

## 4. Data flow per decode step (one new token)

```
for each request in the (reordered) decode segment:
  1. q      = W_Q · h                         # this token's query, all heads
  2. q_nope, q_rope = split(q)
     q̃      = W_UKᵀ · q_nope                   # absorb up-proj into query  (per head)
     q_rope = RoPE(q_rope, pos)
  3. append c_KV(t), k_rope(t) to paged cache  # reshape_and_cache into shuffled layout
  4. scores = q̃ · c_KV[:t]  +  q_rope · k_rope[:t]      # dot vs 512-dim latent + 64-dim rope
  5. a      = softmax(scores)                  # streaming max + LSE, in registers
  6. o_lat  = a · c_KV[:t]                      # attend in latent space (512-dim)
  7. out    = W_UV · o_lat                      # up-project ONCE, per head
```

Steps 4–6 are the assembly hot loop; steps 1–2 and 7 are fused GEMMs around it.

---

## 5. Why this drives throughput

Decode iterations dominate wall-clock (e.g. OSL=1K ⇒ ~1000 decode steps but 1 prefill). TPOT is
almost entirely this kernel. Measured: `ROCM_AITER_MLA` vs `TRITON_MLA` ≈ **1.5× TPS** on
DeepSeek-R1 (MI355X, TP8, 64 concurrency). The two AITER MLA backends tie on decode (shared kernel)
and differ only on TTFT via their prefill choice (assembly MHA vs Triton MHA).

## Sources

- [Beyond Porting: vLLM on AMD ROCm (attention backends)](https://vllm.ai/blog/2026-02-27-rocm-attention-backend)
- [DeepSeek-V2/V3 technical reports — MLA formulation](https://github.com/deepseek-ai/DeepSeek-V3)
- [AITER library](https://github.com/ROCm/aiter) · [ATOM](https://github.com/ROCm/ATOM)
