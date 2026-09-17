# EcoPrompt — Sample Test Queries & Expected Behaviour

Run these queries in order against a freshly started EcoPrompt instance
(cache cleared, threshold = 0.85).

---

## Test Set A — Semantic Cache Demonstration

### Query 1 (first query — cache miss)
**Input:**
> What is the capital city of France?

**Expected behaviour:**
- Cache miss (FAISS index is empty)
- Compression: query is already concise → stages = "none (already concise)"
- LLM called → response stored in cache
- Metrics after: total=1, hits=0, misses=1, hit_rate=0%

---

### Query 2 (near-duplicate — should hit cache)
**Input:**
> What is France's capital city?

**Expected behaviour:**
- Embedding similarity to Query 1 ≈ **0.91–0.96** (above 0.85 threshold)
- ✅ **CACHE HIT** — cached response returned immediately, no LLM call
- Dashboard: water_saved += 0.5 ml, energy_saved += 0.003 kWh
- Metrics after: total=2, hits=1, misses=1, hit_rate=50%

---

### Query 3 (different topic — cache miss + compression triggered)
**Input:**
> Hi there, I was wondering if you could please help me understand,
> in order to make use of machine learning effectively, what are the
> main types of machine learning algorithms? Thanks in advance!

**Expected behaviour:**
- Cache miss (no semantically similar entries)
- **Compression applied:**
  - Before: ~35 tokens
  - Filler removal strips: "Hi there,", "I was wondering if you could please help me understand,"
  - Verbose substitution: "in order to" → "to", "make use of" → "use"
  - Trailing filler removal: "Thanks in advance!"
  - After: ~16 tokens
  - **~19 tokens saved (~54% reduction)**
- Compressed query sent to LLM:
  > "To use machine learning effectively, what are the main types of machine learning algorithms?"
- Response stored in cache
- Metrics: total=3, hits=1, misses=2, tokens_saved_compression=~19

---

### Query 4 (paraphrase of Query 3 — should hit cache)
**Input:**
> Can you explain the different categories of machine learning?

**Expected behaviour:**
- Embedding similarity to Query 3's original ≈ **0.86–0.93**
- ✅ **CACHE HIT** — returns cached response for Query 3
- No LLM call, no compression needed
- Metrics: total=4, hits=2, misses=2, hit_rate=50%

---

## Test Set B — Batch Nudge Demonstration

### Query 5 (non-urgent — enters batch queue)
**Input:**
> When you can, summarise the key differences between SQL and NoSQL databases.

**Expected behaviour:**
- Non-urgency detection: keyword **"when you can"** + **"summarise"** detected
- Query added to batch queue (pending count = 1)
- UI shows: _"Query added to the non-urgent queue (1 pending)."_
- No LLM call yet

---

### Query 6 (non-urgent — triggers batch nudge)
**Input (with "non-urgent" checkbox ticked):**
> What are the best practices for REST API design?

**Expected behaviour:**
- User manually flagged as non-urgent
- Query added to batch queue (pending count = 2)
- **Batch nudge banner appears:**
  > ♻️ You have **2 non-urgent queries** pending. Combining them into one request saves ~1 LLM call(s), reducing water and energy use.
- Combined prompt preview shown:
  ```
  Please answer each of the following questions in order,
  clearly labelling your answer with the corresponding number:

  1. When you can, summarise the key differences between SQL and NoSQL databases.
  2. What are the best practices for REST API design?
  ```

**If user clicks "Accept — send as batch":**
- 1 LLM call made instead of 2 → saves ~50% of compute for these queries
- `queries_batched` counter increments by 2
- Response contains numbered answers to both questions

**If user clicks "Dismiss — send individually":**
- 2 separate LLM calls made
- `queries_individual` counter increments by 2

---

## Summary Table

| # | Query | Module Triggered | Outcome |
|---|-------|-----------------|---------|
| 1 | "What is the capital city of France?" | Cache (miss) | LLM called; stored in cache |
| 2 | "What is France's capital city?" | Cache (hit, ~0.93 similarity) | Cached response returned; 0.5 ml water saved |
| 3 | "Hi there, I was wondering if you could…ML algorithms…Thanks!" | Compression + Cache (miss) | ~19 tokens saved; LLM called |
| 4 | "Can you explain the different categories of machine learning?" | Cache (hit, ~0.89 similarity) | Cached response returned; 0.5 ml water saved |
| 5 | "When you can, summarise SQL vs NoSQL…" | Batch Nudge (queued) | Deferred; no LLM call yet |
| 6 | "What are the best practices for REST API design?" (non-urgent) | Batch Nudge (nudge shown) | User prompted to batch; 1 call saved if accepted |

**After running all 6 queries (batch accepted):**
- Total queries: 6
- Cache hits: 2 (33%)
- Tokens saved via compression: ~19
- Queries batched: 2
- Estimated water saved: **1.0 ml** (from 2 cache hits)
- Estimated energy saved: **0.006 kWh**
