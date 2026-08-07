# AMD ATOM vs vLLM — Architecture & Kernels

How AMD's **ATOM** inference engine and **vLLM** differ in architecture and at the kernel
layer, on AMD Instinct GPUs (MI300X / MI325X / MI355X, CDNA 3 / gfx942–gfx950).

> Key framing: ATOM and vLLM are **not** competitors doing the same job. vLLM is the
> orchestration layer; ATOM is the AMD-native execution layer; **AITER** provides the kernels.
> ATOM can run standalone or plug into vLLM/SGLang.

---

## 1. The core distinction

- **vLLM** = general-purpose, cross-vendor **serving framework** (orchestration layer).
- **ATOM** = purpose-built **AMD execution engine** (compute layer), minimalist / AITER-centric,
  runnable standalone or as a vLLM/SGLang plugin backend.

### Three-layer separation of concerns (ATOM-as-vLLM-plugin)

| Layer | Owner | Responsibility |
|---|---|---|
| **vLLM** | vLLM community | Request scheduling, KV-cache/PagedAttention management, continuous batching, OpenAI-compatible API, broad model coverage |
| **ATOM plugin** | AMD | Platform registration, AMD-optimized model implementations, attention-backend routing, kernel-level tuning |
| **AITER** | AMD | Low-level GPU kernels — fused MoE, fused MLA / flash-attention, quantized GEMM, RoPE fusion |

### Architecture differences

| Aspect | vLLM (native ROCm) | AMD ATOM |
|---|---|---|
| Design goal | Portable, cross-vendor production serving | Minimal, AMD-Instinct-native "to the metal" path |
| Design style | Broad, feature-rich framework | Minimalist/modular (nano-vLLM-inspired), abstraction stripped, centered on AITER routing |
| Deployment | The framework itself | Dual-mode: standalone server **or** vLLM/SGLang plugin backend |
| Multi-GPU/node comms | vLLM + generic collectives | **MORI** comms, optimized for MoE dispatch / expert aggregation / KV traffic |
| Model integration | N/A | Registers via vLLM `entry_points`; patches `MLAAttention.forward_impl` at import; wraps models via `ATOMModelBase` |
| New-HW feature access | Waits for upstream integration | Immediate (e.g. FP4/MXFP4 on MI355X); ATOM is AMD's fast "sandbox", then upstreams to vLLM |

---

## 2. The root problem: every batch is a *mixed* workload

In continuous batching, one forward step contains three token types with opposite hardware needs:

| Phase | What it does | Bottleneck | Wants |
|---|---|---|---|
| **Prefill** | New prompt, thousands of tokens at once | Compute-bound (matrix cores) | Large tiles, max ALU utilization |
| **Extend** | New tokens for a request whose KV cache is partly built (chunked prefill, prefix reuse, multi-turn) | Hybrid | Attend to cached context + fresh tokens |
| **Decode** | 1 output token at a time; loads whole KV cache | Memory-bandwidth-bound | Coalesced access, minimal cache fetches |

A single kernel tuned for prefill leaves decode performance on the table, and vice-versa. This is
the crux of the AITER-vs-generic-kernel difference.

---

## 3. MHA path — `ROCM_AITER_FA` explicit 3-path routing

Instead of one unified kernel, AMD routes each token type to a specialized kernel:

| Path | Kernel | Why |
|---|---|---|
| Prefill | `flash_attn_varlen_func` (AITER MHA → CK or assembly) | CDNA matrix cores for compute-heavy work |
| Extend | chunked attention + `cp_mha_gather_cache` (Triton gather) + LSE merge | 100K+ contexts in ~32K chunks, numerically stable |
| Decode | `pa_fwd_asm` (hand-tuned assembly) | Saturate HBM3 bandwidth |

Two orchestration tricks make it work:

- **Batch reordering**: the model runner reorders every mixed batch to `[decode : extend : prefill]`
  (via `reorder_batch_threshold=1`) so each kernel operates on contiguous tokens — no redundant KV fetches.
- **LSE merging**: each context chunk emits an output + log-sum-exp; merging by LSE lets high-attention
  chunks dominate correctly, so chunking doesn't corrupt softmax.

### The shuffled KV-cache layout (a big, underappreciated lever)

AITER stores the KV cache pre-shuffled to match CDNA memory access:

```
k_cache: [num_blocks, num_heads, head_dim // x, block_size, x]
v_cache: [num_blocks, num_heads, block_size // x, head_dim, x]
```

A custom `reshape_and_cache_flush` keeps the cache always in this layout, so the decode kernel
(`pa_fwd_asm`) runs with **zero layout-conversion overhead** → ~15–20% decode throughput just from
layout. Trade-off: the extend path must `cp_mha_gather_cache` back to standard layout, since AITER's
long-context MHA kernel can't consume the shuffled form.

---

## 4. MLA path (DeepSeek / Kimi) — absorbed vs non-absorbed

MLA compresses the KV cache to **576 dims** (vs ~8K for standard MHA) — a ~14× memory reduction that
changes the optimization strategy:

| Phase | Recipe | Kernel |
|---|---|---|
| Prefill / Extend | **Non-absorbed** — attention on the *uncompressed* representation with standard MHA kernels | AITER MHA or AITER Triton MHA |
| Decode | **Absorbed** — operate directly on the compressed 576-dim latent space | **`mla_decode_fwd` (assembly)** |

The two AITER MLA backends (`ROCM_AITER_MLA`, `ROCM_AITER_TRITON_MLA`) differ **only** in the prefill
kernel; they **share the same assembly decode kernel**, which is where ~all the gain is. Since TPOT is
decode-dominated (e.g. 1K decode iters for OSL=1K), optimizing decode drives throughput.

### vLLM ROCm attention backends (reference)

| Category | Backend | Notes |
|---|---|---|
| MHA | `TRITON_ATTN` | Baseline, Radeon support |
| MHA | `ROCM_AITER_UNIFIED_ATTN` | AITER unified single-kernel path |
| MHA | `ROCM_ATTN` | Legacy 2-path, Radeon support |
| MHA | `ROCM_AITER_FA` | Recommended, auto-selected with AITER (3-path routing) |
| MLA | `TRITON_MLA` | Baseline |
| MLA | `ROCM_AITER_MLA` | Recommended, auto-selected; assembly decode + AITER assembly prefill |
| MLA | `ROCM_AITER_TRITON_MLA` | Alternative; assembly decode + Triton MHA prefill |

Enable via `export VLLM_ROCM_USE_AITER=1` and let vLLM auto-select (`ROCM_AITER_FA` for MHA,
`ROCM_AITER_MLA` for MLA).

### Measured deltas (TP8, ISL=10K / OSL=1K)

| Comparison | Result |
|---|---|
| `ROCM_AITER_MLA` vs `TRITON_MLA` — DeepSeek-R1, MI355X, 64 conc | **1.52× TPS** |
| `ROCM_AITER_FA` vs legacy `ROCM_ATTN` — Qwen3-235B MHA, MI355X, 64 conc | **~3.6× TPS** |

On MI355X (gfx950) `ROCM_AITER_MLA` also wins TTFT because its prefill uses the AITER assembly MHA path.

---

## 5. What ATOM adds *on top of* the AITER backends

vLLM-native ROCm already calls AITER (`VLLM_ROCM_USE_AITER=1`). ATOM goes further by owning the whole
model-execution path:

| Mechanism | What ATOM does |
|---|---|
| `ATOMPlatform.get_attn_backend_cls()` | Forces every attention layer to `AiterBackend` (MHA) or `AiterMLABackend` (MLA) — no generic fallback |
| Import-time patch | Replaces vLLM's `MLAAttention.forward_impl` with ATOM's fused implementation |
| `AiterMLABackend` extras | Fused **QK-RoPE-cache-update** in one op, batched **FP4/FP8 GEMM** for V-projection, persistent metadata buffers for CUDA-graph |
| `ATOMModelBase` | Native ATOM model classes (not vLLM's), AMD-specific compile policies, `load_model_in_plugin_mode()` for AMD quant formats (MXFP4) |
| MORI | AMD comms library for MoE dispatch / expert aggregation / cross-node KV traffic |

Layering: **vLLM keeps orchestration** (scheduling, KV allocation, sampling, API) → **ATOM replaces
the model + attention forward path** → **AITER provides the kernels**. This is why ATOM can edge out
even AITER-enabled vLLM: it also fuses the surrounding ops (RoPE / cache / GEMM) and uses native-format
weights, not just the attention kernel.

---

## 6. ATOM standalone vs plugin mode

| | Standalone ATOM | ATOM-as-vLLM-plugin |
|---|---|---|
| Scheduler / batching / API | ATOM's own (minimalist, nano-vLLM style) | vLLM's mature scheduler + OpenAI API |
| Best for | Peak MI355X perf on MoE/MLA, research, newest FP4 | Production — keep vLLM ops/features, get AMD kernels transparently |
| Activation | Run ATOM server directly | `pip install` alongside vLLM; auto-registers via `entry_points`; `ATOM_DISABLE_VLLM_PLUGIN=1` disables |
| Feature maturity | Fewer production features | Full vLLM feature set (prefix cache, MTP, CUDA-graph FULL_AND_PIECEWISE) |

Intended lifecycle: new HW/kernels land in **ATOM first** (fast sandbox) → validated → **upstreamed
into vLLM's native ROCm backend** for the whole community.

---

## Sources

- [vLLM-ATOM: Unlocking Native AMD Performance in the vLLM Ecosystem (ROCm blogs)](https://rocm.blogs.amd.com/software-tools-optimization/vllm-atom/README.html)
- [Beyond Porting: How vLLM Orchestrates High-Performance Inference on AMD ROCm (vLLM blog)](https://vllm.ai/blog/2026-02-27-rocm-attention-backend)
- [Single Node and Distributed Inference Performance on AMD Instinct MI355X GPU (AMD)](https://www.amd.com/en/developer/resources/technical-articles/2026/distributed-inference-performance-on-instinct-mi355x-gpu.html)
- [Scaling AI Inference on MI355X (ROCm blogs)](https://rocm.blogs.amd.com/artificial-intelligence/scaling-ai-inference/README.html)
- [ATOM repository](https://github.com/ROCm/ATOM) · [AITER library](https://github.com/ROCm/aiter)
