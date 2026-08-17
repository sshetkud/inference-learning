# Fixed vs. Random Dataset (8192 in + 1024 out): What's the Difference and Why It Matters

## TL;DR

The main difference is whether the 8,192 input tokens are **the same every request** (fixed) or **different/random every request** (random).

Both configurations have the **same nominal token counts** per request:

| Metric | Value |
|---|---|
| Input tokens | 8,192 |
| Output tokens | 1,024 |
| **Total** | **9,216 tokens/request** |

But they can behave very differently in inference benchmarking, because the fixed-vs-random choice decides whether the engine can reuse prefill work via **prefix caching**.

---

## 1. Fixed: 8,192 input + 1,024 output

Every request uses essentially the **same prompt**:

```
Request 1 -> Prompt A (8192) -> 1024 output
Request 2 -> Prompt A (8192) -> 1024 output
Request 3 -> Prompt A (8192) -> 1024 output
...
```

If the inference engine supports **prefix caching**, the shared prompt can potentially be reused. After the first request:

```
Prompt processing (prefill)
        |
        v
     KV Cache
        |
        v
Request 2/3/4/... reuse it (skip most/all prefill)
```

This can dramatically reduce prefill work and **TTFT (time to first token)**.

> **Caveat:** Fixed prompts can produce *artificially good* performance when prefix caching is enabled — you're measuring cache hits, not real prefill throughput.

---

## 2. Random dataset: 8,192 input + 1,024 output

Every request has a **different** 8,192-token prompt:

```
Request 1 -> Prompt A (8192)
Request 2 -> Prompt B (8192)
Request 3 -> Prompt C (8192)
Request 4 -> Prompt D (8192)
...
```

There is little/no opportunity to reuse the prompt's KV cache. Every request must perform the full:

```
8192-token PREFILL
        +
1024-token DECODE
```

This is generally a **more realistic stress test** for workloads where users send different prompts.

---

## Performance Difference

| | Fixed 8192 + 1024 | Random 8192 + 1024 |
|---|---|---|
| Input tokens | 8,192 | 8,192 |
| Output tokens | 1,024 | 1,024 |
| Total | 9,216 | 9,216 |
| Prompt identical? | Yes | No |
| Prefix cache reuse | High (after 1st request) | ~None |
| Prefill work | Amortized across requests | Full, every request |
| TTFT | Low (cache hits) | Higher (real prefill) |
| Realism | Optimistic / best-case | Realistic / worst-case |
| Good for measuring | Decode throughput, cache efficiency | End-to-end prefill + decode under load |

---

## Why It Matters

- **Apples-to-apples comparisons:** Only compare fixed-vs-fixed or random-vs-random. Mixing them makes one engine look faster purely because of caching.
- **Prefix-caching sensitivity:** If a benchmark uses fixed prompts, verify whether prefix caching is on. If it is, the numbers reflect cache hits, not sustained prefill capacity.
- **Capacity planning:** Random datasets better predict real production behavior (diverse user prompts) and stress the prefill path, memory bandwidth, and scheduler.
- **Regression testing:** Fixed prompts give lower variance and are useful for detecting decode-path regressions in isolation.

**Rule of thumb:** Use **random** to size for production and expose prefill bottlenecks; use **fixed** to isolate decode performance and validate prefix-caching gains.
