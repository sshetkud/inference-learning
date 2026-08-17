# AMD MORI — Deep Dive

**MORI** (Modular RDMA Interface) is AMD's open-source, GPU-centric communication library built for **large-scale Mixture-of-Experts (MoE) inference and training on ROCm/Instinct GPUs**. Its job is to make the **expert-parallel all-to-all** (the "dispatch" and "combine" steps of MoE) fast and to hide network latency by letting the **GPU itself initiate and drive RDMA transfers** over RoCE/InfiniBand, instead of bouncing through the CPU.

Think of it as AMD's counterpart to NVIDIA's DeepEP — a purpose-built EP (Expert Parallel) communication layer, rather than a general collective library like RCCL/NCCL.

---

## 1. Why MORI exists: the MoE all-to-all problem

In a dense transformer, every token goes through the same FFN, so communication is regular (all-reduce / all-gather over TP/DP). In an **MoE** layer:

- A router picks **top-k experts** (e.g. 8 of 256) for each token.
- Experts are **sharded across GPUs/nodes** (Expert Parallelism).
- So each token must be **sent to the GPU(s) that host its chosen experts**, processed, then **sent back**.

This produces two irregular, data-dependent all-to-all exchanges per MoE layer:

```
DISPATCH:  tokens  ->  scatter to the GPUs owning their top-k experts
           (expert FFN runs locally)
COMBINE:   results ->  gather back to the token's original GPU, weighted-sum
```

Characteristics that make it hard:

- **Irregular / dynamic sizes** — routing is data-dependent, so per-peer message sizes change every step.
- **Small messages, many peers** — latency-bound, not bandwidth-bound.
- **Two exchanges per layer × dozens of layers** — communication can dominate MoE latency, especially in **decode** (batch of 1-token steps).

A generic all-to-all (RCCL/NCCL) is CPU-launched, synchronous at kernel boundaries, and not specialized for this token-routing pattern — so it leaves a lot of performance on the table.

---

## 2. Core idea: GPU-initiated communication

MORI's key lever is **GPU-initiated (device-side) RDMA**, analogous to NVIDIA's IBGDA / NVSHMEM model:

- The **GPU kernel posts RDMA work directly** to the NIC (writes doorbells / work queue entries) — the CPU is not in the critical path per message.
- Communication is expressed as **one-sided put/get into remote GPU memory** (SHMEM-style), rather than matched send/recv collectives.
- This lets MORI **overlap routing, data movement, and expert compute**, and issue thousands of tiny transfers without per-message CPU launch overhead.

Result: lower latency and better overlap for the small, irregular MoE messages.

---

## 3. Architecture / layers

MORI is typically described as a few cooperating layers:

1. **Transport / RDMA core ("Modular RDMA Interface")**
   - Thin abstraction over RDMA verbs (RoCEv2 / InfiniBand) plus intranode transports.
   - Provides symmetric memory regions (SHMEM-like) that both local and remote GPUs can address.
   - Handles queue-pair setup, memory registration, doorbell ringing from the device.

2. **EP dispatch/combine engine**
   - The MoE-specific layer: takes the router's expert assignments and performs the **dispatch** (scatter tokens to experts) and **combine** (gather + weighted reduce).
   - Packs/unpacks tokens, computes per-peer offsets, and drives the one-sided transfers.

3. **Kernels**
   - Fused GPU kernels for token permutation/packing, index computation, and the put/get issue path, so routing metadata and data movement stay on-GPU.

4. **Topology-aware path selection**
   - **Intranode**: peer GPUs reached over **XGMI / Infinity Fabric** (or PCIe) with direct GPU-to-GPU copies.
   - **Internode**: reached over **RoCE/IB** via the NIC using GPU-initiated RDMA.
   - MORI picks the right path per destination so intranode traffic never touches the NIC.

---

## 4. Dispatch / combine in more detail

```
             ┌─────────── per MoE layer ───────────┐

 router logits ─► top-k expert ids + weights (on GPU)
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
   build send layout               (symmetric buffers
   (counts/offsets per peer)        pre-registered)
        │
        ▼
   DISPATCH  ── GPU-initiated put ──►  remote expert GPUs
        │                                   │
        │                             expert FFN (grouped GEMM)
        │                                   │
   COMBINE   ◄── GPU-initiated put ──   results back
        │
        ▼
   weighted sum by router weights  ─► MoE layer output
```

Key optimizations that libraries in this class use (and MORI targets):

- **Overlap** dispatch/combine transfers with expert GEMMs.
- **Warp-specialized issue** — some warps compute routing/packing while others post RDMA.
- **Separate low-latency (decode) vs high-throughput (prefill/training) paths** — decode favors tiny latency-optimized transfers; prefill/training favors bandwidth-optimized bulk moves.
- **Zero-copy** into symmetric buffers to avoid staging copies.

---

## 5. How it fits in the software stack

```
  MoE model (SGLang / vLLM / training framework)
                │  expert-parallel dispatch/combine calls
                ▼
             MORI (EP engine + RDMA core + kernels)
                │
     ┌──────────┴───────────┐
     ▼                      ▼
  XGMI / Infinity Fabric   RoCEv2 / InfiniBand (Pensando/other NICs)
  (intranode GPU-GPU)      (internode, GPU-initiated RDMA)
                │
                ▼
            ROCm / HIP runtime, Instinct GPUs (e.g. MI300X/MI355X)
```

- It sits **below** the serving/training framework and **beside** RCCL: RCCL still handles dense TP/DP collectives, while MORI handles the **EP all-to-all**.
- Exposes host APIs to set up symmetric memory and device-side APIs/kernels to issue transfers.

---

## 6. MORI vs. RCCL vs. DeepEP

| | RCCL/NCCL | DeepEP (NVIDIA-side) | **MORI (AMD)** |
|---|---|---|---|
| Purpose | General collectives (all-reduce, all-gather, all-to-all) | MoE expert-parallel dispatch/combine | MoE expert-parallel dispatch/combine + RDMA core |
| Launch model | CPU-launched collectives | GPU-initiated (IBGDA/NVSHMEM) | **GPU-initiated RDMA** |
| Message pattern | Regular, symmetric | Irregular token routing | Irregular token routing |
| Best for | Dense TP/DP | MoE on NVIDIA | **MoE on ROCm/Instinct** |
| Transport | RoCE/IB/XGMI/NVLink | RoCE/IB + NVLink | RoCE/IB + XGMI/Infinity Fabric |

---

## 7. Why it matters

- **MoE is communication-bound**, especially at decode and at large expert counts / many nodes. Speeding up dispatch/combine directly improves **tokens/sec and TTFT/ITL**.
- **GPU-initiated RDMA** removes CPU launch overhead and enables fine-grained **compute/communication overlap**.
- **Topology awareness** keeps intranode traffic on XGMI and only uses the NIC when truly crossing nodes — important on 8-rail RoCE nodes where each rail matters (a single dead RoCE rail can bottleneck EP all-to-all).
- It gives AMD an **open, ROCm-native EP stack** so large MoE models (DeepSeek-style, Llama-4-style) run efficiently on Instinct without depending on NVIDIA-only libraries.

---

## 8. One-line summary

> **MORI is AMD's ROCm-native, GPU-initiated RDMA library that accelerates the MoE expert-parallel all-to-all (dispatch/combine), overlapping token routing, network transfer, and expert compute across XGMI (intranode) and RoCE/IB (internode).**
