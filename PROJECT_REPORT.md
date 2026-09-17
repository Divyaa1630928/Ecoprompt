# EcoPrompt — Project Report
### 1M1B–IBM SkillsBuild AI for Sustainability Virtual Internship
**Internship Track:** AI for Sustainability | **Collaboration:** IBM SkillsBuild & AICTE

---

## Slide 1 — Problem Statement

### The Hidden Environmental Cost of Everyday AI

Every time a user submits a query to a large language model (LLM), real
physical resources are consumed — water to cool the data centre hardware, and
electricity to power the inference compute.

**Published estimates (illustrative approximations):**
- Microsoft's 2023 Sustainability Report indicates ChatGPT-class queries
  consume approximately **0.5 ml of water per query** on average.
- Google's 2023 Environmental Report discloses data-centre Power Usage
  Effectiveness (PUE) and water usage intensity values that imply
  **~0.003 kWh of electricity per mid-size LLM inference call**.
- Global LLM query volumes are in the billions per month and growing rapidly.

**The gap in existing solutions:**
Efforts to reduce this footprint today operate almost entirely at the
infrastructure level — more efficient cooling, renewable energy procurement,
better hardware. Individual users have no tools or feedback to understand or
reduce the environmental cost of _how they use AI_.

**Our problem statement:**
> How might we use AI to reduce redundant and inefficient AI queries at the
> individual user level, so that the water and energy footprint of everyday
> generative AI usage can become genuinely lower?

---

## Slide 2 — Target Users

### Who Benefits from EcoPrompt?

| User Type | Pain Point | How EcoPrompt Helps |
|---|---|---|
| **Individual knowledge workers** | Repeatedly ask similar questions to AI assistants | Semantic cache returns instant answers; no LLM call needed |
| **Students & researchers** | Verbose, exploratory queries consume excess tokens | Prompt compressor reduces query length by 20–60% |
| **Developers integrating LLM APIs** | Every API call costs money and energy | Cache hit rate reduces API call volume |
| **Organisations with AI usage policies** | Hard to enforce sustainable AI usage | EcoPrompt dashboard provides per-user metrics |
| **AI-aware sustainability advocates** | Want visibility into the eco-cost of AI tools | Impact dashboard surfaces water/energy savings in real time |

**Primary persona:** A university student or professional who uses an AI
assistant daily for research, summarisation, or code help and wants to make
their AI usage more sustainable without changing their workflow.

---

## Slide 3 — The Solution: EcoPrompt

### Three Integrated Optimisation Modules

**Module 1 — Semantic Caching**
Uses a lightweight sentence embedding model (all-MiniLM-L6-v2, 22 MB, runs
on CPU) to convert each query into a dense vector. Before forwarding a query
to the LLM, the system checks whether a semantically similar query was already
answered recently (configurable similarity threshold, default 0.85). If a
match is found, the cached response is returned instantly — **zero LLM calls,
zero water, zero energy consumed for that query**.

*Responsible AI note:* The similarity threshold is user-configurable and the
matched query is always shown, allowing users to verify the cached answer is
appropriate. Cached entries expire after 24 hours (TTL) to avoid serving stale
information.

**Module 2 — Prompt Compression**
A four-stage rule-based pipeline trims filler phrases ("Can you please…",
"Thanks in advance"), normalises whitespace, removes near-duplicate sentences
within a prompt, and substitutes verbose qualifiers ("in order to" → "to",
"due to the fact that" → "because"). Compression is applied only on cache
misses — i.e., only when a real LLM call is unavoidable. **Token savings of
20–60% have been observed on naturally verbose user queries.**

*Responsible AI note:* The compressor is conservative by design — it never
removes named entities, numbers, or domain-specific terms. The compressed
version is always displayed alongside the original for user verification.

**Module 3 — Batch / Defer Nudge**
Detects queries flagged as non-urgent (via keywords or user toggle) and holds
them in a time-windowed queue. When 2 or more non-urgent queries accumulate
within the window (default: 60 seconds), the user is prompted to combine them
into a single numbered-list request. **One combined LLM call replaces N
separate calls**, directly multiplying the energy/water saving.

---

## Slide 4 — AI's Role in EcoPrompt

### AI Used to Reduce AI's Own Footprint

| AI Component | Purpose | Model / Technique |
|---|---|---|
| Sentence Embedding | Convert query text to semantic vectors for similarity search | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector Similarity Search | Find nearest cached query above threshold | FAISS flat inner-product index (cosine similarity) |
| Token Counting | Accurately measure query length in LLM token units | `tiktoken` (cl100k_base BPE, same as GPT-3.5/4) |
| TF-IDF Cosine Similarity | Detect duplicate sentences inside a single prompt | Pure Python (no model needed) |
| LLM (wrapped) | Answer queries that are genuine cache misses | OpenAI API (pluggable; any OpenAI-compatible endpoint) |

**Key insight:** The embedding model (22 MB, CPU-only) costs orders of
magnitude less energy than a full LLM inference call. The upfront cost of
running MiniLM on every query is paid back as soon as one cache hit occurs.

---

## Slide 5 — Expected Impact

### Quantified & Projected Savings

**Per-user scenario (moderate usage: 20 queries/day, 30% cache hit rate):**

| Metric | Daily | Monthly | Annually |
|---|---|---|---|
| Queries avoided by cache | 6 | 180 | 2,190 |
| Water saved (est.) | 3 ml | 90 ml | 1,095 ml (~1.1 L) |
| Energy saved (est.) | 0.018 kWh | 0.54 kWh | 6.57 kWh |
| Tokens saved (compression, 25% avg) | ~150 | ~4,500 | ~54,750 |

**At organisational scale (1,000 users × above scenario):**
- ~1,095 litres of water saved per year
- ~6,570 kWh of electricity saved per year (equivalent to powering ~2 average
  EU households for a month)

*These are illustrative projections. Actual savings depend on query diversity,
cache hit rate, and the energy/water intensity of the specific LLM provider.*

**SDG Impact Mapping:**

| SDG | How EcoPrompt Contributes |
|---|---|
| SDG 6 — Clean Water & Sanitation | Reduces data-centre water consumption per AI interaction |
| SDG 12 — Responsible Consumption & Production | Eliminates redundant compute through reuse and compression |
| SDG 13 — Climate Action | Lowers energy-related CO₂ emissions from LLM inference |

---

## Slide 6 — Responsible AI Considerations

### Balancing Efficiency with Accuracy and Transparency

**1. Cache Similarity Threshold Trade-off**

| Threshold | Effect |
|---|---|
| Too low (e.g. 0.60) | High hit rate but risk of returning a subtly wrong cached answer to a different question |
| Too high (e.g. 0.99) | Near-perfect accuracy but very few cache hits — minimal eco-savings |
| Recommended (0.85) | Balances accuracy and efficiency for typical factual and how-to queries |

Mitigation: The matched cached query is always shown to the user. The
threshold is user-adjustable in the dashboard.

**2. Prompt Compression Safety**

The compressor removes only patterns empirically identified as semantically
redundant across natural-language prompts. Validation steps:
- Named entities, numbers, and technical terms are never removed.
- The user can inspect a side-by-side before/after view and token count
  before sending.
- Compression is not applied to cached-hit responses (no user-visible change).

**3. Water / Energy Estimate Provenance**

All savings figures in the dashboard are:
- Derived from third-party published sustainability reports
  (Google 2023, Microsoft 2023), not independently measured.
- Per-call approximations; actual values vary by model, hardware, and
  data-centre efficiency.
- Clearly labelled as estimates throughout the UI.

EcoPrompt does **not** claim to provide exact environmental accounting. Its
purpose is to make the eco-cost of AI usage _visible and reducible_ at the
individual level.

---

## Slide 7 — Getting Started

### Running the Prototype Locally

```bash
# 1. Install dependencies (Python 3.9+)
pip install -r ecoprompt/requirements.txt

# 2. Configure (optional — leave blank for demo mode)
cp ecoprompt/.env.example ecoprompt/.env
# Edit .env and add OPENAI_API_KEY if you have one

# 3. Launch the dashboard
streamlit run ecoprompt/app.py
```

The app runs fully offline in demo mode — no API key required to demonstrate
all three optimisation modules and the impact dashboard.

### Repository Structure
```
ecoprompt/
├── app.py               ← Streamlit UI (entry point)
├── semantic_cache.py    ← Module 1: Semantic caching
├── prompt_compressor.py ← Module 2: Prompt compression
├── batch_nudge.py       ← Module 3: Batch nudge
├── llm_client.py        ← LLM adapter (OpenAI / demo)
├── ARCHITECTURE.md      ← System flow diagram
├── TEST_QUERIES.md      ← Sample test scenarios
├── PROJECT_REPORT.md    ← This document
└── requirements.txt
```

---

*Report prepared for the 1M1B–IBM SkillsBuild AI for Sustainability Virtual Internship.*
*All water and energy figures are approximations for educational purposes only.*
