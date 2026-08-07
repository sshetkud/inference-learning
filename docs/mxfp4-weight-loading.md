# MXFP4 weight loading in ATOM

A zoom-in on how ATOM loads and executes **MXFP4** (OCP Microscaling FP4) weights natively on
MI355X (CDNA 4 / gfx950), via `load_model_in_plugin_mode()` and the AITER quantized-GEMM path.

---

## 1. What MXFP4 actually is

MXFP4 is the **OCP Microscaling** 4-bit floating format. It is *block* floating point:

```
A block = 32 consecutive elements sharing ONE scale.
  - each element:  E2M1  (4 bits: 1 sign, 2 exponent, 1 mantissa)  → 16 representable values
  - block scale:   E8M0  (8 bits, a power-of-two exponent, shared by all 32 elements)
```

So a 32-element block costs `32 × 4 bits (data) + 8 bits (scale) = 136 bits` ≈ **4.25 bits/weight**.
E2M1 can represent `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` times the block scale. The shared E8M0
scale slides that tiny grid to wherever each block's magnitudes live — that's how 4 bits stays usable.

Contrast:
- **FP8 (E4M3/E5M2)**: per-tensor or per-row scale, 8 bits/element — safer, larger.
- **MXFP4**: per-32-block scale, ~4.25 bits/element — ~2× smaller weights, needs hardware/kernels that
  understand the micro-scale.

MI355X has **native FP4/MXFP4 matrix-core support**, so the GEMM consumes packed FP4 directly rather
than dequantizing to FP16 first.

---

## 2. On-disk layout of an MXFP4 checkpoint

A quantized linear layer typically stores two tensors instead of one weight:

```
weight_packed : uint8[out, in/2]      # two FP4 nibbles packed per byte
weight_scale  : uint8[out, in/32]     # one E8M0 scale per 32-element block  (a.k.a. "scale" / "block_scale")
```

Sometimes a `weight_scale_2` / global scale and per-layer metadata accompany it (format tag, block
size, whether activations are also quantized). The loader must know the **packing order** (which
nibble is element 0) and the **block axis** (scales run along the reduction/`in` dimension).

---

## 3. `load_model_in_plugin_mode()` — what it does

When ATOM runs as a vLLM plugin, vLLM builds the model skeleton but ATOM takes over weight
loading so it can honor AMD-native formats. Conceptually:

```python
def load_model_in_plugin_mode(model, checkpoint):
    for name, module in model.named_modules():
        if is_quantized_linear(module):                     # MXFP4 / FP8 linear
            packed = checkpoint[name + ".weight_packed"]     # uint8 FP4 pairs
            scale  = checkpoint[name + ".weight_scale"]      # E8M0 per-32-block

            # 1) keep FP4 PACKED — do NOT upcast to fp16 (that would kill the memory win)
            # 2) reshuffle to the AITER GEMM's expected tile/interleave layout
            packed = aiter.shuffle_mxfp4_weight(packed)
            scale  = aiter.shuffle_scale(scale)

            module.register_quant_weight(packed, scale, fmt="mxfp4", block=32)
        else:
            standard_load(module, checkpoint[name])          # norms, embeddings, router in bf16/fp16
```

Key behaviors:

- **No dequant on load.** Weights stay 4-bit in HBM; only the small block scales ride alongside. This
  is the whole point — 4.25 bits/weight resident, so a 4-bit model fits where fp16 wouldn't.
- **Layout shuffle.** Like the KV cache, the packed weights + scales are pre-shuffled to the exact
  interleave the AITER FP4 MFMA path wants, so the GEMM does zero repacking at runtime.
- **Selective precision.** Only the big GEMMs (attention/MLP/expert projections) are FP4; embeddings,
  layernorms, router gates, and often the `lm_head` stay in bf16/fp16 for accuracy.
- **Format detection.** ATOM reads the checkpoint's quant metadata (e.g. Quark/`quantization_config`)
  to pick MXFP4 vs FP8 vs mixed per layer.

---

## 4. The GEMM dequant path (how FP4 gets used)

At compute time the AITER quantized GEMM fuses dequant into the matrix-core pipeline:

```
for each output tile:
   load packed FP4 weights  (uint8) ─┐
   load E8M0 block scales   ─────────┤→ on-chip: expand nibble→E2M1 value, apply block scale
   load activations (bf16 or fp8) ───┘
   MFMA accumulate in fp32
   (optionally) requantize activation to fp8/fp4 for the next layer
```

- On MI355X the **matrix cores natively accept FP4 operands**, so "expand nibble + apply scale"
  happens inline against the shared E8M0 scale per 32-block — no full-precision weight tensor is ever
  materialized in HBM or LDS.
- Accumulation is **fp32** for numerical safety; the scale is applied per block during accumulation.
- Because scales are per-32-block, the kernel indexes `scale[k // 32]` as it strides the reduction
  dim `k` — cheap, and the shuffle in step 3 makes those loads contiguous.

---

## 5. Why ATOM does this before vLLM does

- New AMD number formats (FP4/MXFP4) land in **ATOM first** — it's the fast sandbox where AMD can wire
  format + loader + AITER kernel together without waiting on upstream vLLM's generic quant framework.
- It delivers the concrete wins MXFP4 promises: **~2× smaller weights** (fit bigger models / more KV
  cache headroom) and **higher matrix-core throughput** on gfx950, at accuracy close to FP8 for many
  models when embeddings/norms/router stay high-precision.
- Once proven, the format support gets **upstreamed** into vLLM's ROCm quantization path.

## Sources

- [OCP Microscaling (MX) Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Single Node and Distributed Inference on MI355X (AMD)](https://www.amd.com/en/developer/resources/technical-articles/2026/distributed-inference-performance-on-instinct-mi355x-gpu.html)
- [AMD Quark quantization](https://quark.docs.amd.com/) · [AITER](https://github.com/ROCm/aiter) · [ATOM](https://github.com/ROCm/ATOM)
