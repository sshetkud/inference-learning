# LoRA vs QLoRA

**LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)** are parameter-efficient fine-tuning (PEFT) methods: adapt a large pretrained model to a task without updating all of its weights.

**One-line difference:** LoRA freezes a full-precision base model and trains tiny low-rank adapters. QLoRA does the same, but keeps the **base weights in 4-bit** so much larger models fit in the same VRAM.

> Training technique, not an inference kernel — but the adapters you produce are what serving stacks (vLLM, SGLang, ATOM) load at runtime, often merged into the base weights so there is no extra latency.

---

## Why this matters: full fine-tuning is expensive

A full fine-tune of a model with parameters \(W\) stores:

- a copy of \(W\) (forward)
- gradients \(\nabla W\)
- optimizer state (Adam ≈ 2× params)

So peak training memory is roughly **~12–16+ bytes per parameter** in FP16/BF16, before activations. For a 70B model that is tens to hundreds of GB — often more than a single GPU (or even a small node) can hold.

PEFT answers: *freeze \(W\), train only a small add-on*. Memory and compute then scale with the adapters, not the full model.

---

## LoRA — Low-Rank Adaptation

Instead of updating the full weight matrix \(W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\), LoRA freezes \(W\) and learns two small matrices \(A\) and \(B\):

```
W' = W + (α / r) · B A

  A : r × d_in     (often initialized Gaussian)
  B : d_out × r    (often initialized zero → ΔW starts at 0)
  r ≪ min(d_in, d_out)   # rank, typically 8–64
  α                   # scaling hyperparameter (often ≈ 2r)
```

Forward pass:

```
y = W x + (α / r) · B (A x)
```

Only \(A\) and \(B\) receive gradients. Parameter count for one adapted linear layer:

```
trainable = r · (d_in + d_out)     ≪    d_in · d_out
```

### Where adapters are attached

Common targets (Hugging Face PEFT naming):

| Target set | Modules | Trade-off |
|---|---|---|
| Minimal | `q_proj`, `v_proj` | Fewest params, often enough |
| Attention | `q/k/v/o_proj` | Stronger capacity |
| All-linear | Attention + MLP (`gate/up/down`) | Best quality, more VRAM / slower |

### Inference: merge or keep separate

At serve time you can:

1. **Merge** — bake \(BA\) into \(W\) once → same latency as the base model (preferred for a single adapter).
2. **Keep adapters** — load base once, swap/stack adapters per request (multi-LoRA / LoRA servers).

---

## QLoRA — Quantized LoRA

QLoRA = **4-bit quantized base model** + **LoRA adapters in higher precision** (FP16/BF16).

Typical stack (Dettmers et al.):

1. Load base weights in **4-bit NF4** (NormalFloat4; tuned for weight distributions that look roughly Gaussian).
2. Optional **double quantization** of the quantization constants themselves (further VRAM savings).
3. Compute in a **paged / higher-precision** path (dequant on the fly for the matmul).
4. Train **only** the LoRA adapters (not the 4-bit base).

```
Base W  : stored 4-bit (NF4) in HBM
Adapters: A, B in FP16/BF16 — these are what you optimize
Forward : dequant(W_4bit) · x  +  (α/r) · B (A x)
```

**Why it works:** the huge tensor stays 4-bit resident; only a thin high-precision compute view is used per op. Adapters remain full precision so gradients are stable.

---

## Head-to-head

| Dimension | LoRA | QLoRA |
|---|---|---|
| Base weights in HBM | FP16 / BF16 | **4-bit NF4** (typical) |
| Trainable params | Adapters only | Adapters only |
| Peak VRAM | Much lower than full FT | **Much lower than LoRA** |
| Training speed | Faster than QLoRA (no dequant) | Slower (dequant overhead) |
| Quality | Strong PEFT baseline | Usually close to LoRA; can trail slightly on hard tasks |
| Best when | Mid-size models / enough VRAM | Very large models / tight VRAM |
| Serving after train | Merge adapters → full-precision (or keep LoRA) | Often dequantize + merge, or serve quantized base + adapters |

---

## Practical knobs

| Knob | Typical values | Effect |
|---|---|---|
| **Rank \(r\)** | 8, 16, 32, 64 | Capacity vs cost; raise if underfitting |
| **Alpha \(\alpha\)** | often \(2r\) | Scales adapter contribution |
| **Adapter dropout** | 0–0.1 | Light regularization |
| **Target modules** | q/v → all-linear | Quality vs trainable size |
| **Learning rate** | often higher than full FT | Adapters are small; 1e-4–2e-4 common starting point |
| **QLoRA bits / type** | 4-bit NF4 | Main VRAM win; FP4/MXFP4 stacks are a related but different path |

**Rule of thumb:** enough VRAM → **LoRA**; need to fit a bigger model → **QLoRA**.

---

## Memory intuition (order of magnitude)

Illustrative, one GPU, same model / sequence length (not a benchmark — just the shape of the difference):

| Method | Dominant resident cost |
|---|---|
| Full fine-tune | Weights + grads + optimizer ≈ many × params |
| LoRA | Full-precision weights + small adapters + their grads/opt |
| QLoRA | **4-bit weights** + small adapters + their grads/opt |

Activations still grow with batch and sequence length; PEFT does not remove that — it mainly shrinks the **parameter** side of the bill.

---

## Relation to inference on AMD / ROCm

- Training with LoRA/QLoRA is usually done via **PEFT + Transformers** (and a bitsandbytes-like 4-bit path where available on ROCm).
- For **serving**, adapters are typically:
  - **merged** into a deployable checkpoint, then loaded like any other model in vLLM / SGLang / ATOM, or
  - kept as **multi-LoRA** adapters if the serving engine supports dynamic LoRA.
- Separately, **inference quantization** (FP8, MXFP4, etc. — see [MXFP4 weight loading in ATOM](mxfp4-weight-loading.md)) is about *serving* weight format, not the QLoRA training recipe — though both exploit low-bit weights for memory.

---

## When to use which

**Prefer LoRA when**

- The base model already fits in FP16/BF16 with room for adapters + activations.
- You want simpler tooling and faster train steps.
- Quality is the priority and VRAM is not the limiter.

**Prefer QLoRA when**

- You are fine-tuning a model that barely (or doesn’t) fit in full precision.
- You accept slightly slower steps for a large VRAM cut.
- You will merge / export afterward for a full-precision or separately quantized serve path.

**Prefer full fine-tune when**

- You need maximum adaptation (domain shift, continued pretraining style) and have the cluster budget.
- PEFT underfits even at high rank / all-linear targets.

---

## Sources

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (4-bit NF4 path used by classic QLoRA)
