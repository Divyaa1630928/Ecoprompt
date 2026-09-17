# EcoPrompt — Architecture & Flow

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                               │
│              Streamlit Web App  (app.py)                            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ user types query + presses Send
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1 — Non-Urgency Detection          (batch_nudge.py)           │
│                                                                     │
│  Does the query contain keywords like "later", "no rush",           │
│  "summarise", "when you can"?  OR did the user tick the             │
│  "non-urgent" checkbox?                                             │
│                                                                     │
│  YES → Add to pending queue → evaluate queue size                  │
│         If ≥ 2 pending queries within batch window:                 │
│           Show Batch Nudge Banner                                   │
│             ├─ User accepts → combine into one prompt ──────────────┐
│             └─ User dismisses → flush individually (each enters     │
│                                  pipeline separately)               │
│  NO  → proceed immediately ─────────────────────────────────────────┤
└────────────────────────────────────────────────────────┬────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2 — Semantic Cache Lookup          (semantic_cache.py)        │
│                                                                     │
│  • Encode query → 384-dim embedding (all-MiniLM-L6-v2)             │
│  • Search FAISS flat inner-product index for nearest neighbour      │
│  • Compute cosine similarity                                        │
│                                                                     │
│  similarity ≥ threshold (default 0.85)?                             │
│                                                                     │
│  YES → ✅ CACHE HIT                                                 │
│         Return cached response immediately                          │
│         Increment cache_hits counter                                │
│         Log: water_saved += 0.5 ml, energy_saved += 0.003 kWh      │
│         ──────────────────────────────────────────────► RESPONSE   │
│                                                                     │
│  NO  → ⚡ CACHE MISS — continue to Step 3                          │
└────────────────────────────────────────────────────────┬────────────┘
                                                         │ (cache miss)
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3 — Prompt Compression             (prompt_compressor.py)     │
│                                                                     │
│  Four-stage pipeline applied to the raw query:                      │
│                                                                     │
│  Stage 1 — Filler Phrase Removal                                    │
│    Strips: "Can you please", "I was wondering if", "Thanks in       │
│    advance", "Just a quick question", etc.                          │
│                                                                     │
│  Stage 2 — Whitespace Normalisation                                 │
│    Collapses multiple spaces / blank lines.                         │
│                                                                     │
│  Stage 3 — Redundant Sentence Deduplication                         │
│    Removes sentences with TF-IDF cosine similarity > 0.92 to a     │
│    preceding sentence in the same prompt.                           │
│                                                                     │
│  Stage 4 — Verbose Qualifier Substitution                           │
│    "in order to" → "to"   |   "due to the fact that" → "because"   │
│    "has the ability to" → "can"   |   "make use of" → "use"  etc.  │
│                                                                     │
│  Output: compressed_query  +  token_delta (before vs after)        │
└────────────────────────────────────────────────────────┬────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4 — LLM Inference                  (llm_client.py)            │
│                                                                     │
│  • Send compressed_query to OpenAI API (or demo stub)               │
│  • Receive response text                                            │
└────────────────────────────────────────────────────────┬────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5 — Cache Store                    (semantic_cache.py)        │
│                                                                     │
│  • Encode ORIGINAL query (not compressed) → embedding               │
│  • Store {embedding, response, timestamp} in FAISS + entry list     │
│  • Entry expires after TTL (default 24 h)                           │
└────────────────────────────────────────────────────────┬────────────┘
                                                         │
                                                         ▼
                                               ► RESPONSE displayed
                                                 Metrics dashboard updated
```

## File Structure

```
ecoprompt/
├── app.py               ← Streamlit entry point (run this)
├── semantic_cache.py    ← Module 1: Semantic caching (FAISS + MiniLM)
├── prompt_compressor.py ← Module 2: Rule-based prompt compression
├── batch_nudge.py       ← Module 3: Non-urgent batch nudge logic
├── llm_client.py        ← LLM wrapper (OpenAI or demo mode)
├── requirements.txt     ← Python dependencies
└── .env.example         ← Configuration template
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| all-MiniLM-L6-v2 for embeddings | 22 MB, MIT licence, runs on CPU, 384-dim vectors provide strong semantic quality |
| FAISS flat inner-product index | Simple, exact-search, no approximate-NN trade-off needed at individual-user scale |
| Cosine similarity (via L2-norm + inner product) | Scale-invariant, ideal for sentence embeddings |
| Rule-based compression (no LLM needed) | Zero extra inference cost; deterministic and auditable by user |
| Streamlit for UI | Single-file deployment, real-time state, runs locally with one command |
| Demo mode (no API key needed) | Lets the project run and demonstrate all features offline |
