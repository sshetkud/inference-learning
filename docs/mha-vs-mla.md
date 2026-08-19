# MHA vs MLA

**MHA (Multi-Head Attention)** is the original Transformer attention. **MLA (Multi-head Latent Attention)**, introduced by DeepSeek-V2, is a redesign whose goal is to shrink the **KV cache** — the dominant memory cost of LLM inference — while keeping full-quality attention.

**One-line difference:** MHA stores a full key and value vector *for every head* at every token. MLA stores a single small **latent vector** per token and reconstructs per-head keys/values on the fly — cutting KV-cache memory by roughly an order of magnitude with no quality loss.

> This is the conceptual companion to [`mla_decode_fwd` — the absorbed-MLA decode kernel](mla-decode-kernel.md), which covers how MLA is actually executed at decode time.

---

## Why this matters: the KV cache

During autoregressive decoding, each new token attends to *all* previous tokens. To avoid recomputing them, the keys (K) and values (V) of past tokens are cached in GPU memory. This **KV cache** grows linearly with sequence length and batch size, and for long contexts it — not the model weights — becomes the memory bottleneck that caps batch size and throughput.

```
KV cache bytes = 2 * n_layers * (KV elements per token per layer) * bytes_per_elem
```

Every attention variant below is essentially a different answer to: *how few KV elements can we store per token per layer?*

---

## The attention family (context)

MHA and MLA sit at opposite ends of a lineage of KV-cache optimizations. MQA and GQA reduce the **number** of KV heads; MLA instead reduces the **dimensionality** of what is stored.

| Variant | KV sharing strategy | Effect on KV cache |
|---|---|---|
| **MHA** | Full: every head has its own K and V | Baseline (largest cache) |
| **MQA** | All heads share ONE K and one V | ~n_heads× smaller, some quality loss |
| **GQA** | Heads grouped; each group shares K/V | Tunable middle ground (today's default) |
| **MLA** | Store a small latent; reconstruct K/V per head | ~order-of-magnitude smaller, no quality loss |

---

## MHA — Multi-Head Attention (the baseline)

The input hidden state `h_t` (dim `d_model`) is projected into `h` independent heads. Each head gets its own query, key, and value:

```
q_i = W_q^i * h_t ,  k_i = W_k^i * h_t ,  v_i = W_v^i * h_t      (i = 1..h)
head_i = softmax(q_i * K_i^T / sqrt(d_head)) * V_i
out    = W_o * concat(head_1, ..., head_h)
```

**Key properties**

- Every head has a **distinct** K and V — maximum expressiveness.
- KV cache stores `2 * h * d_head` elements per token per layer — the largest of any variant.
- RoPE (rotary position embedding) is applied directly to each `q_i`, `k_i`.

**The cost.** For a model with 128 heads × 128 head-dim, MHA caches `2 * 128 * 128 = 32,768` elements per token per layer. Over 60+ layers and long contexts this is enormous — which is exactly what MQA, GQA, and MLA attack.

---

## MLA — Multi-head Latent Attention (DeepSeek-V2 / V3)

MLA keeps **multi-head** attention math, but instead of caching full per-head K and V, it caches one small **low-rank latent** vector `c_KV_t` per token. Per-head keys and values are re-expanded from it via learned up-projections.

### 1. Low-rank KV compression (the cached part)

```
c_KV_t = W_DKV * h_t          -> compressed latent, dim d_c (e.g. 512); THIS is what's cached
k_i^C  = W_UK^i * c_KV_t       -> per-head key   reconstructed at compute time
v_i^C  = W_UV^i * c_KV_t       -> per-head value reconstructed at compute time
```

### 2. Decoupled RoPE key (the clever bit)

RoPE is position-dependent, so it can't be absorbed into the up-projection matrices (that would break the matrix-absorption math that lets MLA skip re-expansion during decode). MLA splits the key into two parts: a **compressed content** part (no RoPE) and a small **decoupled RoPE** part `k_t^R` of dim `d_r` (e.g. 64) that is shared across heads and also cached.

```
Cached per token per layer:  c_KV_t (dim d_c)  +  k_t^R (dim d_r)  =  d_c + d_r elements
                             e.g. 512 + 64 = 576
```

### Why quality is preserved

Unlike MQA/GQA which throw away heads, MLA still gives every head a distinct K and V — they're just reconstructed from a shared latent. Queries are also low-rank compressed to cut activation memory during training. DeepSeek reports MLA matches or beats full MHA quality while using a fraction of the KV cache.

---

## Head-to-head

| Dimension | MHA | MLA |
|---|---|---|
| What's cached | Full K and V per head | One small latent (+ tiny RoPE key) |
| KV elems / token / layer | `2 * h * d_head` | `d_c + d_r` (≈ 576) |
| Per-head distinct K/V | Yes (stored) | Yes (reconstructed) |
| KV cache size | Largest | ~2–15% of MHA |
| Quality | Full | Matches / exceeds MHA |
| RoPE handling | Applied to all q, k | Decoupled RoPE key, shared across heads |
| Extra compute | None | Up-projection to rebuild K/V |
| Training memory | Baseline | Lower (queries also compressed) |
| Complexity | Simple | More moving parts (2 latents + decoupled RoPE) |

*Legend:* `d_head` = per-head dimension · `h` = number of heads · `d_c` = compressed latent dim · `d_r` = decoupled RoPE dim. Sizes shown are DeepSeek-V2 illustrative values.

---

## KV cache: concrete numbers

Illustrative, using DeepSeek-V2-scale settings (`h = 128`, `d_head = 128`, `d_c = 512`, `d_r = 64`), counting KV elements stored per token per layer:

| Variant | KV elems / token / layer | Relative to MHA |
|---|---:|---:|
| MHA (128 KV heads) | 32,768 | 1.0× |
| GQA (8 KV groups) | 2,048 | 0.06× |
| **MLA** | **576** | **0.018×** |

That is a **≈ 57×** reduction vs MHA. Actual cache bytes = elements × 2 (FP16/BF16) × `n_layers` × `seq_len` × `batch`.

---

## Where each is used

| Attention | Representative models |
|---|---|
| MHA | Original Transformer, GPT-2/3, LLaMA-1, BERT |
| MQA | PaLM, Falcon |
| GQA | LLaMA-2 70B, LLaMA-3, Mistral, Qwen2 — the modern default |
| MLA | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1 |

---

## Trade-offs

**MLA wins**

- Massive KV-cache reduction → larger batches, longer context, higher throughput.
- No quality sacrifice (unlike MQA/GQA which drop heads).
- Lower training activation memory (compressed queries).

**MLA costs**

- More complex: two latent projections + a decoupled RoPE path.
- Extra up-projection compute to rebuild K/V each step.
- Kernel/serving support is newer — needs MLA-aware attention kernels (e.g. the absorbed-MLA decode kernel in ATOM/AITER, vLLM, SGLang).
