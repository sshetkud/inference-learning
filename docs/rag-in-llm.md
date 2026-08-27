# RAG in LLM serving

**Retrieval-Augmented Generation (RAG)** connects an LLM to an external knowledge store so answers are
grounded in retrieved evidence instead of only model weights. This doc covers both halves: the
**retrieval pipeline and its parameters**, and — more importantly for a serving lab — **what RAG
actually does inside the model and to your GPU**.

Related docs: [lora-vs-qlora.md](lora-vs-qlora.md) (knowledge vs behavior),
[mha-vs-mla.md](mha-vs-mla.md) (KV-cache pressure),
[vllm-benchmark.md](vllm-benchmark.md) (TTFT/TPOT under long prompts),
[disaggregated-prefill-decode.md](disaggregated-prefill-decode.md) (prefill-heavy workloads).

**One-line idea:** *retrieve relevant chunks → put them in the prompt → generate grounded, cited output.*

---

## 1. Why RAG exists

| Problem | How RAG helps |
|---|---|
| Hallucination on facts, dates, product details | Condition generation on retrieved text |
| Knowledge cutoff | Update the corpus, no retraining |
| Private / domain data | Data stays in your index, not in weights |
| Traceability | Return the passages actually used |
| Cost | Usually far cheaper than continued pretraining |

RAG is **not** a replacement for fine-tuning:

- **Fine-tuning (LoRA/QLoRA)** changes *behavior* — style, format, tool-use habits.
- **RAG** injects *knowledge* — facts that change, or that you can't put in weights.

Most real systems use both. See [lora-vs-qlora.md](lora-vs-qlora.md).

---

## 2. What RAG changes mathematically

A plain LLM models `p_θ(y | x)`. RAG conditions on retrieved evidence `C = {c_1 … c_k}`:

```
p(y | x) = Σ_c   p_η(c | x)   ·   p_θ(y | x, c)
                 └ retriever ┘     └ generator ┘
```

In practice the sum is truncated to top-`k` and, for prompt-based RAG, collapsed entirely: all `k`
chunks are concatenated into one prompt and run in a single forward pass.

The knowledge lives in **activations produced from the prompt**, not in the weights — which is why
reindexing updates answers instantly.

Two marginalization modes from the original RAG paper:

| Mode | Formulation | Behavior |
|---|---|---|
| **RAG-Sequence** | `p(y\|x) ≈ Σ_c p(c\|x) p(y\|x,c)` | One document conditions the whole answer |
| **RAG-Token** | `p(y\|x) = Π_i Σ_c p(c\|x) p(y_i\|x,c,y_<i)` | Different tokens can draw on different docs |

---

## 3. Pipeline overview

```
User query
   │
   ▼
Query processing      (rewrite / expand / HyDE / multi-query / routing)
   │
   ▼
Retriever             (sparse / dense / hybrid)  ──►  Index (vector + BM25 + metadata)
   │
   ▼
Reranker (optional)   (cross-encoder / LLM judge)
   │
   ▼
Context packing       (top-k, token budget, dedupe, ordering)
   │
   ▼
Prompt assembly       (system + instructions + chunks + query)
   │
   ▼
Generator LLM         (vLLM / SGLang / ATOM)
   │
   ▼
Post-process          (citations, abstain if low confidence, verification)
```

---

## 4. Indexing (offline)

### 4.1 Corpus preparation

- Clean HTML/PDF/markdown, strip nav and boilerplate
- Normalize encoding, detect language
- Attach metadata: `source`, `url`, `title`, `section`, `updated_at`, `acl`, `doc_type`, `version`

Metadata is not decoration — it is what lets retrieval enforce tenancy and recency:

```
similarity(query, chunk)  AND  product = X  AND  updated_at > T  AND  acl ⊇ user
```

### 4.2 Chunking methods

| Method | Idea | Pros | Cons |
|---|---|---|---|
| **Fixed-size** | N tokens with overlap | Simple, predictable | Cuts mid-idea |
| **Recursive** | Split by `\n\n` → `\n` → sentence → char | Respects structure | Needs good separators |
| **Semantic** | Split where embedding similarity drops | Topic-coherent | Costly, threshold-sensitive |
| **Document-structure** | Headers, sections, code blocks, tables | Best for manuals/code | Format-specific |
| **Parent–child (small-to-big)** | Retrieve small chunk, expand to parent section | Precise hit + rich context | More plumbing |
| **Proposition / atomic** | One fact per chunk | High precision | Index size grows |
| **Late chunking** | Embed long context, then derive chunk vectors | Better global context | Needs model support |

**Chunking parameters**

| Parameter | Typical | Meaning |
|---|---|---|
| `chunk_size` | 256–1024 tokens | Chunk length |
| `chunk_overlap` | 10–20% of size | Continuity across boundaries |
| `min_chunk_size` | 50–100 tokens | Drop fragments |
| `separators` | `["\n\n","\n"," ",""]` | Recursive split priority |
| `respect_headers` | bool | Keep sections intact |
| `parent_window` | 1–3 sections | Expansion for small-to-big |

### 4.3 Embeddings (dense retrieval)

Map text to a vector; score with cosine or dot product (identical when normalized).

| Choice | Notes |
|---|---|
| **Bi-encoder** | Query and doc embedded separately → scalable ANN search |
| **Cross-encoder** | Query+doc scored jointly → accurate but slow; use as **reranker** |
| **Sparse-learned** (SPLADE) | Learned sparse vectors; hybrid-friendly |
| **Matryoshka / truncatable** | Trade dimension for storage |

**Embedding parameters**

| Parameter | Meaning |
|---|---|
| `model_name` | e.g. `bge`, `e5`, `gte`, `voyage`, `text-embedding-3` |
| `dimension` | 384–3072 typical |
| `normalize` | L2-normalize for cosine |
| `max_seq_length` | Truncation limit — watch long chunks |
| `pooling` | `cls` / `mean` / `last_token` |
| `instruction_prefix` | e.g. E5 `query:` / `passage:` — **must match training recipe** |

> A very common silent bug: using the wrong (or no) instruction prefix. Retrieval quality drops
> sharply and looks like a "bad embedding model."

### 4.4 Sparse / lexical index

BM25 remains strong for exact tokens — error codes, part numbers, flags, identifiers.

| Parameter | Typical | Role |
|---|---|---|
| `k1` | ~1.2 | Term-frequency saturation |
| `b` | ~0.75 | Length normalization |
| analyzer | language-specific | Stemming, stopwords |

### 4.5 Vector index (ANN)

| Index | When |
|---|---|
| **Flat** | Small corpus, exact search |
| **HNSW** | Default high-recall ANN |
| **IVF / IVF-PQ** | Large corpora, memory-constrained |
| **DiskANN / tiered** | Billion-scale |

| Parameter | Effect |
|---|---|
| `M` | Graph degree → recall vs memory |
| `efConstruction` | Build quality/time |
| `efSearch` | Query recall vs latency |
| `nprobe` (IVF) | Clusters probed |
| `metric` | `cosine` / `ip` / `l2` |

---

## 5. Retrieval at query time

### 5.1 Hybrid retrieval

Production systems rarely use dense alone. Blend dense and sparse:

```
score = α · s_dense + (1 − α) · s_sparse
```

or fuse by rank with **RRF (Reciprocal Rank Fusion)**:

```
RRF(d) = Σ_r  1 / (k_rrf + rank_r(d))
```

| Parameter | Typical | Meaning |
|---|---|---|
| `α` | 0.3–0.7 dense | Dense vs sparse blend |
| `k_rrf` | 60 | RRF smoothing constant |
| `top_k_dense` | 20–100 | Candidates before fusion |
| `top_k_sparse` | 20–100 | Candidates before fusion |
| `final_k` | 5–20 | After fusion / before rerank |

### 5.2 Query transformation

| Method | What it does |
|---|---|
| **Query rewrite** | LLM clarifies/expands the question |
| **Multi-query** | Generate N paraphrases, retrieve and union |
| **HyDE** | LLM drafts a hypothetical answer; embed *that* |
| **Step-back** | Ask a broader question first |
| **Decomposition** | Split a complex question into sub-questions |
| **Routing** | Pick the right index/collection/tool |
| **Self-query** | Extract structured metadata filters from natural language |

### 5.3 Reranking

Retrieve a wide net (`N` ≈ 50), then rescore with a stronger model and keep `k` ≈ 5.

| Reranker | Notes |
|---|---|
| Cross-encoder (`bge-reranker`) | Best quality/cost at mid `N` |
| ColBERT / late interaction | Strong recall, heavier |
| LLM-as-reranker | Flexible, expensive |
| Metadata boost | Recency, title match, ACL |

Parameters: `top_n_in`, `top_k_out`, `score_threshold`, batch size.

---

## 6. How retrieved text physically enters the model

For prompt-based RAG (nearly all production systems), chunks become **ordinary tokens**:

```
[system][instructions][chunk 1][chunk 2] … [chunk k][question] │ [answer …]
└──────────────────────── prefill ────────────────────────────┘ └─ decode ─┘
```

Three consequences that matter:

- **Self-attention is the fusion mechanism.** There is no special retrieval pathway — answer tokens
  attend to chunk tokens through the same QKᵀ softmax as anything else. Grounding is an *emergent*
  behavior of instruction tuning, not an architectural guarantee.
- **Chunks attend to each other.** With causal masking, chunk 3 sees chunks 1–2. A wrong early chunk
  can bias how later ones are interpreted.
- **Position is real.** Retrieved passages occupy actual positions in the RoPE/ALiBi space, which
  produces the "lost in the middle" effect below.

### 6.1 Fusion architectures

| Architecture | Where fusion happens | Cost in `k` | Notes |
|---|---|---|---|
| **In-context / prompt RAG** | Input tokens, self-attention | `O((k·L)²)` | Default; works with any API model |
| **Fusion-in-Decoder (FiD)** | Encode passages independently, decoder cross-attends | `O(k·L²)` | Passages can't see each other; scales to many |
| **RETRO** | Chunked cross-attention layers | Retrieval every ~64 tokens | Retrieval baked into architecture |
| **kNN-LM** | Output distribution | Datastore lookup per token | `p = λ·p_kNN + (1−λ)·p_LM` |
| **KV-cache injection** | Precomputed KV for hot docs | Skips prefill | Cache-augmented generation |

FiD is the instructive contrast: because passages are encoded separately, cost grows **linearly** in
`k` rather than quadratically — which is why FiD historically scaled to ~100 passages while prompt
RAG struggles past 10–20.

### 6.2 Lost in the middle

Recall of retrieved facts is empirically **U-shaped** in position: best at the start and end of the
context, worst in the middle.

Practical rules:

- Place the highest-scoring chunk **first or last**, not buried at rank 5 of 10.
- More context is **not** monotonically better — irrelevant chunks measurably degrade accuracy.
- Long-context models reduce but do not remove this; effective context < advertised context.

### 6.3 Token budgeting

```
T_system + k · T_chunk + T_query + T_answer  ≤  T_window
```

Reserve the answer budget **first**, then let `k` be the free variable.

**Packing parameters**

| Parameter | Meaning |
|---|---|
| `max_context_tokens` | Budget for retrieved text |
| `top_k` | Chunks actually placed in the prompt |
| `min_score` | Drop weak hits |
| `mmr_lambda` | Relevance vs diversity |
| `ordering` | Relevance-first vs document order |
| `citation_format` | `[n]`, footnotes, JSON |

**MMR (Maximal Marginal Relevance)** trades relevance against redundancy:

```
MMR = λ · sim(q, d) − (1 − λ) · max_{d' ∈ S} sim(d, d')
```

---

## 7. Serving economics — RAG is prefill-heavy

This is where RAG collides with the lab's serving work.

A normal chat turn might be ~50 input tokens. A RAG turn is **2k–8k input tokens** with a similar
output length. That flips the workload shape:

| Phase | Non-RAG chat | RAG |
|---|---|---|
| **Prefill** | Small | **Dominant** — compute-bound GEMMs |
| **Decode** | Dominant | Unchanged — memory-bandwidth-bound |
| **KV cache / request** | Small | Large, grows with `k · chunk_size` |

Implications:

- **TTFT is the metric that suffers.** In [vllm-benchmark.md](vllm-benchmark.md) terms, RAG stresses
  `mean_ttft_ms` / `p99_ttft_ms`; TPOT barely moves. A RAG workload looks like a **high-ISL** sweep
  (e.g. `--random-input-len 8192`), which is exactly the shape the Kimi-K3 sweeps already exercise.
- **KV pressure caps concurrency.** Long prompts consume KV blocks, so max in-flight requests drops —
  the same trade-off visible sweeping concurrency 16 → 128.
- **Prefix caching is the biggest single win.** A fixed system prompt (and any repeatedly retrieved
  document) can be KV-cached across requests, turning repeated prefill into a cache hit.
- **Chunked prefill** stops long RAG prefills from head-of-line blocking other requests' decode.
- **Low-KV attention helps disproportionately.** Since KV cache is the binding constraint, MLA's
  ~57× KV reduction ([mha-vs-mla.md](mha-vs-mla.md)) directly buys more concurrent RAG requests.
- **Prefill/decode disaggregation** maps naturally onto RAG, since the two phases have very different
  profiles — see [disaggregated-prefill-decode.md](disaggregated-prefill-decode.md).

---

## 8. Generation parameters for grounded answers

| Parameter | RAG setting | Why |
|---|---|---|
| `temperature` | 0.0–0.2 | Factual extraction, not creativity |
| `top_p` | 0.9–1.0 | Low temperature already narrows |
| `max_tokens` | Bounded | Long answers drift off-evidence |
| `repetition_penalty` | ~1.0 | Penalties suppress verbatim quotes/citations |
| `stop` | Section markers | Stop the model continuing the context pattern |
| `logprobs` | Enabled | Confidence signal for abstention |
| `seed` | Fixed | Reproducible evaluation |

**Prompt pattern**

```
System: Answer ONLY using the context below. If the context is insufficient,
        say you don't know. Cite sources as [n].

Context:
[1] <chunk …>
[2] <chunk …>

Question: <user query>
Answer:
```

**Abstention gating.** Check the max rerank score *before* calling the LLM; if it is below threshold,
skip generation entirely and return "no supporting documents." This removes a whole class of
hallucination and saves the most expensive step.

---

## 9. Advanced RAG architectures

| Variant | Idea |
|---|---|
| **Naive RAG** | Retrieve once → generate |
| **Advanced RAG** | Rewrite + hybrid + rerank + careful packing |
| **Agentic RAG** | Agent decides when and what to retrieve, loops |
| **Corrective RAG (CRAG)** | Grade retrieved docs; fall back to web search |
| **Self-RAG** | Model emits retrieve/critique reflection tokens |
| **GraphRAG** | Entity/relation graph; retrieve subgraphs |
| **RAPTOR** | Tree of clustered summaries; retrieve at multiple abstraction levels |
| **Adaptive RAG** | Route easy questions to no-retrieval, hard ones to multi-hop |
| **Multimodal RAG** | Images/tables in the index, VLM generator |
| **Long-context RAG** | Fewer, larger chunks into 128k–1M windows |

**Multi-hop retrieval** for questions needing chained facts: retrieve → form an intermediate query →
retrieve again → synthesize. Parameters: `max_hops`, stop on confidence or no-new-documents.

---

## 10. Training-time RAG (retrieval-aware models)

| Model | Retriever trained? | Key idea |
|---|---|---|
| **REALM** | Yes, end-to-end | Latent retrieval during masked-LM pretraining |
| **RAG** (Lewis 2020) | Query encoder only | Marginalize over top-`k` docs (sequence/token modes) |
| **FiD** | No (frozen DPR) | Encode passages separately, fuse in decoder |
| **Atlas** | Yes | Few-shot retrieval-augmented training |
| **RETRO** | Frozen | Chunked cross-attention over a huge datastore |
| **Self-RAG** | — | Reflection tokens control retrieval and critique |

The core difficulty in end-to-end RAG is that top-`k` retrieval is **discrete and
non-differentiable**. Approaches treat documents as latent variables and backprop through
`p_η(c | x)`, usually with a periodically refreshed (stale) index.

---

## 11. Evaluation

| Family | Metrics | Question answered |
|---|---|---|
| **Retrieval** | Recall@k, nDCG, MRR, hit rate | Did we fetch the right documents? |
| **Faithfulness** | Groundedness, hallucination rate | Is the answer supported by context? |
| **Answer quality** | EM, F1, LLM-as-judge | Is it correct vs gold? |
| **Citation** | Citation precision/recall | Are the references real and used? |
| **Ops** | TTFT p50/p99, cost/query, index freshness | Is it production-viable? |

Evaluate the stages **separately**. If Recall@50 is low, no amount of prompt tuning will help; if
Recall is high but faithfulness is low, the problem is packing/prompt/model, not the retriever.

---

## 12. Failure modes

| Symptom | Mechanism | Fix |
|---|---|---|
| Right docs indexed, wrong answer | Chunking / embedding mismatch | Re-chunk, fix instruction prefixes, add rerank |
| Misses IDs, codes, flags | Dense-only retrieval | Add BM25 (hybrid) |
| Contradictory answers | Conflicting or stale chunks | Dedupe, version filters, prefer recency |
| Model ignores context | Parametric prior dominates | Explicit grounding instruction, lower temperature |
| Model over-trusts a bad chunk | No verification | Rerank + faithfulness check |
| Accuracy drops as `k` grows | Distraction / lost-in-middle | Fewer, better chunks; reorder to edges |
| Slow TTFT | Huge prompts, cross-encoder on large `N` | Prefix caching, chunked prefill, rerank fewer |
| Stale answers | Index not refreshed | Incremental indexing, TTL |
| Cross-tenant leakage | Missing ACL filter | Enforce metadata filters at retrieval |

---

## 13. Starting defaults

| Stage | Start here | Tune toward |
|---|---|---|
| `chunk_size` | 512 tokens, overlap 64 | Smaller for precision, larger for narrative |
| Dense `top_k` | 20–50 | Increase if recall is low |
| Hybrid | RRF, or α ≈ 0.5 | Weight BM25 up for jargon/IDs |
| Rerank | 50 in → 5 out | Increase input if Recall@50 is weak |
| Chunks in prompt | 4–8 | Until distraction or token limit |
| `temperature` | 0.0–0.2 | Raise only for creative synthesis |
| `efSearch` (HNSW) | 64–200 | Up for recall, down for latency |
| Abstention | Threshold on rerank score | Calibrate on an eval set |

---

## 14. Choosing between RAG, fine-tuning, and long context

| Approach | Best for |
|---|---|
| **RAG** | Changing facts, large corpora, citations, multi-tenancy |
| **Fine-tune (LoRA/QLoRA)** | Style, format, domain language, tool-use behavior |
| **Long context** | Whole-document reasoning when per-query corpus is small |
| **Combined** | Fine-tune for behavior + RAG for knowledge |

---

## 15. Summary

```
retrieval quality   →  what the model CAN know
prompt & position   →  what the model DOES attend to
decoding params     →  how faithfully it reports it
prefill & KV budget →  how fast, and how many concurrently
```

RAG **quality** is usually bottlenecked by retrieval and reranking. RAG **latency** is bottlenecked by
prefill and KV cache. Those are different problems, and conflating them is the most common reason RAG
systems get tuned in the wrong place.

---

## References

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020)
- Guu et al., *REALM: Retrieval-Augmented Language Model Pre-Training* (2020)
- Izacard & Grave, *Fusion-in-Decoder* (2021)
- Borgeaud et al., *Improving Language Models by Retrieving from Trillions of Tokens* (RETRO, 2021)
- Khandelwal et al., *Generalization through Memorization: Nearest Neighbor Language Models* (kNN-LM, 2020)
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023)
- Asai et al., *Self-RAG* (2023)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching.html)
