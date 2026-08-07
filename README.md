# inference-learning

Notes and deep-dives on LLM inference engines, kernels, and serving on AMD Instinct GPUs.

## Contents

- [AMD ATOM vs vLLM — architecture & kernels](docs/atom-vs-vllm.md) — how AMD's ATOM engine and vLLM differ in architecture and the kernel layer, and how they combine (ATOM as a vLLM plugin backend on top of AITER).

## TL;DR — ATOM vs vLLM

ATOM and vLLM are **not** competing engines doing the same job; they live at different layers and are often used together.

- **vLLM** = general-purpose, cross-vendor **serving framework** (orchestration: scheduling, KV-cache/PagedAttention, continuous batching, OpenAI-compatible API).
- **ATOM** = purpose-built **AMD execution engine** (minimalist, AITER-centric) that runs standalone **or** as a vLLM/SGLang plugin backend.

When combined, a clean 3-layer separation of concerns applies:

| Layer | Owner | Responsibility |
|---|---|---|
| **vLLM** | vLLM community | Request scheduling, KV-cache management, continuous batching, OpenAI-compatible API |
| **ATOM plugin** | AMD | Platform registration, AMD-optimized model implementations, attention-backend routing, kernel tuning |
| **AITER** | AMD | Low-level GPU kernels — fused MoE, fused MLA / flash-attention, quantized GEMM, RoPE fusion |

See the [full write-up](docs/atom-vs-vllm.md) for the kernel-level details.

## Sources

- [vLLM-ATOM: Unlocking Native AMD Performance in the vLLM Ecosystem (ROCm blogs)](https://rocm.blogs.amd.com/software-tools-optimization/vllm-atom/README.html)
- [Beyond Porting: How vLLM Orchestrates High-Performance Inference on AMD ROCm (vLLM blog)](https://vllm.ai/blog/2026-02-27-rocm-attention-backend)
- [Single Node and Distributed Inference Performance on AMD Instinct MI355X GPU (AMD)](https://www.amd.com/en/developer/resources/technical-articles/2026/distributed-inference-performance-on-instinct-mi355x-gpu.html)
- [ATOM repository](https://github.com/ROCm/ATOM) · [AITER library](https://github.com/ROCm/aiter)
