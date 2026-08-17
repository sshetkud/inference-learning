# RoPE Fusion

**RoPE fusion** = fusing the Rotary Position Embedding computation into a single GPU kernel (or into an adjacent kernel like the QK projection / attention), instead of running it as several separate elementwise operations.

---

## Background: what RoPE does

**RoPE (Rotary Position Embedding)** encodes token position by *rotating* pairs of dimensions in the query (Q) and key (K) vectors by an angle proportional to the position. For position `m` and dimension pair `i`:

```
[ q'_2i   ]   [ cos(m*theta_i)  -sin(m*theta_i) ] [ q_2i   ]
[ q'_2i+1 ] = [ sin(m*theta_i)   cos(m*theta_i) ] [ q_2i+1 ]
```

So each Q and K vector gets an elementwise multiply by `cos`/`sin` tables plus a "rotate-half" shuffle.

---

## The problem RoPE fusion solves

Done naively, RoPE is a sequence of small ops per layer:

```
gather cos/sin  ->  rotate_half(q)  ->  q*cos + rotate_half(q)*sin   (same for k)
```

Each is a separate kernel launch that reads Q/K from HBM, does a tiny bit of math, and writes back. RoPE is **memory-bandwidth bound**, so those extra round-trips to HBM and kernel-launch overheads dominate — the arithmetic itself is trivial.

---

## What fusion does

**RoPE fusion** collapses those steps into one kernel:

- Load Q (and K) into registers / shared memory **once**.
- Compute or look up `cos`/`sin` inline.
- Apply the rotation and write back **once**.

Two common levels:

1. **Standalone fused RoPE kernel** — the rotate-half + cos/sin multiply for Q and K in a single kernel (e.g. Apex/Megatron `fused_rope`, or a Triton kernel).
2. **Fused into a bigger kernel** — RoPE folded into the attention prologue or the QKV projection epilogue, so positions are applied without ever materializing rotated Q/K in HBM (common in FlashAttention variants and inference engines like vLLM / AITER kernels).

---

## Unfused vs. fused (conceptual)

**Unfused (multiple HBM round-trips):**

```
q      = load(Q)                 # HBM read
rot    = rotate_half(q)          # HBM read + write
q_out  = q*cos + rot*sin         # HBM read x2 + write
store(q_out)                     # HBM write
# repeat for K
```

**Fused (single pass):**

```
one kernel:
  q = load(Q)                    # HBM read (once)
  cos, sin = lookup(pos)         # inline / cached
  q_out = q*cos + rotate_half(q)*sin
  store(q_out)                   # HBM write (once)
  # K handled in same kernel
```

---

## Why it matters

- **Fewer HBM round-trips** -> higher effective bandwidth utilization.
- **Fewer kernel launches** -> less overhead, especially at small batch / decode where launch cost is significant.
- **Lower latency & better TTFT / decode throughput**, and lower peak memory (no intermediate rotated tensors).

---

## How to enable it

In frameworks you'll see this as a flag, e.g. Megatron-LM's `--apply-rope-fusion` / `apply_rope_fusion: true`, which swaps the Python elementwise RoPE for a single fused CUDA/HIP kernel. Transformer Engine and inference engines (vLLM, AITER) ship their own fused RoPE kernels that are used automatically in the attention path.
