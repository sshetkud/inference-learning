# MORI — the multi-node MoE collective

A zoom-in on the communication library AMD uses to make **Mixture-of-Experts** models scale across
many GPUs and nodes. MoE turns a dense FFN into a routed, all-to-all problem; MORI is the transport
that makes that routing fast on Instinct fabrics (Infinity Fabric intra-node, RDMA/RoCE inter-node).

---

## 1. Why MoE needs a special collective

In a dense model every token flows through the same FFN — no cross-GPU shuffling. In an MoE layer:

```
for each token:
   scores   = router(token)            # gate over E experts
   experts  = top_k(scores)            # e.g. top-2 of 256 experts
   token → sent to the GPUs that own those experts
```

With **Expert Parallelism (EP)** the experts are sharded across GPUs, so tokens must be **physically
moved** to wherever their chosen experts live, processed, then moved **back**. That's two irregular
all-to-all exchanges per MoE layer:

- **Dispatch** (a.k.a. *scatter*): route each token to its top-k experts' GPUs.
- **Combine** (a.k.a. *gather/reduce*): send expert outputs back and weight-sum them per token.

The traffic is **data-dependent and unbalanced** (routing is dynamic), latency-sensitive
(it's on the critical path of every layer), and, at scale, crosses node boundaries. Generic
`all_to_all` collectives handle this poorly. MORI is purpose-built for it — the AMD analog to
NVIDIA's DeepEP.

---

## 2. What MORI provides

| Capability | What it does |
|---|---|
| **EP dispatch/combine kernels** | Fused token permute → transfer → un-permute; overlaps packing with the network send |
| **Intra- vs inter-node paths** | Uses Infinity Fabric / XGMI within a node, RDMA (RoCE/IB) across nodes, with different tuned kernels for each |
| **Low-latency vs high-throughput modes** | A latency-optimized path for small decode batches (few tokens, must be quick) and a throughput path for large prefill batches |
| **Compute/comm overlap** | Streams dispatch of layer *L+1* tokens while experts of layer *L* still compute (hides network behind GEMM) |
| **Quantized transport** | Can send activations in FP8/FP4 to cut wire bytes, dequant on arrival |
| **KV / general transfers** | Also used for cross-node KV traffic in disaggregated prefill/decode setups |

---

## 3. Dispatch → expert compute → combine (the loop)

```
                     ┌──────────────── one MoE layer, EP over N GPUs ────────────────┐
 tokens (local)      router picks top-k experts per token
      │                        │
      ▼                        ▼
 [1] build permutation:  group local tokens by destination GPU/expert
 [2] DISPATCH (all-to-all): send token groups to expert-owner GPUs   ← MORI
      │                        (intra-node XGMI / inter-node RDMA)
      ▼
 [3] expert GEMM: each GPU runs its local experts on the tokens it received
      │                        (AITER fused-MoE grouped GEMM, FP8/FP4)
      ▼
 [4] COMBINE (all-to-all): send outputs back to each token's origin GPU  ← MORI
      │
      ▼
 [5] weighted sum: reduce top-k expert outputs per token using router weights
                     └───────────────────────────────────────────────────────────────┘
```

Key implementation points:

- **Token permutation** ([1]) and its inverse ([5]) are fused into the transfer kernels so tokens
  arrive already grouped per-expert — the expert GEMM sees contiguous input, no gather inside GEMM.
- **Payload sizing**: MORI packs variable per-expert token counts (dynamic routing) into transfers
  and carries the counts as metadata so the receiver knows how to slice.
- **Overlap**: dispatch of the next micro-batch/layer is issued on a separate stream so the RDMA
  latency is hidden behind expert compute — critical because inter-node hops are ~µs-scale.

---

## 4. Prefill vs decode behavior

| | Prefill (many tokens) | Decode (1 token/req) |
|---|---|---|
| Batch to route | thousands of tokens | a handful |
| MORI mode | throughput path — big, bandwidth-bound all-to-all | low-latency path — tiny messages, latency-bound |
| Dominant cost | wire bandwidth | round-trip latency + kernel launch |
| Optimization | quantized (FP8/FP4) payloads, large transfers | fused small-message kernels, persistent buffers, graph capture |

This split mirrors the attention story: the same MoE layer needs *different* comm strategies for the
compute-bound vs latency-bound regimes, which is why a MoE-aware library beats a one-size `all_to_all`.

---

## 5. How it fits with ATOM / vLLM

- vLLM/ATOM own scheduling and the EP layout (which experts live where).
- **AITER** provides the local fused-MoE grouped GEMM (the expert compute in step [3]).
- **MORI** provides steps [2] and [4] — the dispatch/combine transport — plus the KV transfers in
  disaggregated serving.
- ATOM standalone leans on MORI for its multi-node MoE story; the same primitives are being upstreamed
  so vLLM's ROCm EP path benefits too.

## Sources

- [Single Node and Distributed Inference on MI355X (AMD)](https://www.amd.com/en/developer/resources/technical-articles/2026/distributed-inference-performance-on-instinct-mi355x-gpu.html)
- [Scaling AI Inference on MI355X (ROCm blogs)](https://rocm.blogs.amd.com/artificial-intelligence/scaling-ai-inference/README.html)
- [MORI / ROCm communication libraries](https://github.com/ROCm) · [DeepEP (NVIDIA analog, for contrast)](https://github.com/deepseek-ai/DeepEP)
