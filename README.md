# 🌿 EcoPrompt — AI Query Optimizer for Sustainability

> **Reduce the water & energy footprint of your everyday AI usage — before a query even reaches the model.**

Built for the **1M1B × IBM SkillsBuild AI for Sustainability Virtual Internship** (in collaboration with IBM SkillsBuild & AICTE).

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![Gemini](https://img.shields.io/badge/Gemini-3.6--flash-green)
![SDG](https://img.shields.io/badge/SDG-6%20%7C%2012%20%7C%2013-yellowgreen)

---

## 🌍 The Problem

Every AI query consumes real physical resources:
- ~**0.5 ml of water** per query for data centre cooling *(Microsoft 2023 Sustainability Report)*
- ~**0.003 kWh of energy** per inference call

With billions of AI queries made every day, the environmental cost is significant — and growing. Most solutions operate at the **infrastructure level** (better cooling, renewable energy). **EcoPrompt shifts this responsibility to the individual user**, optimizing queries *before* they reach the AI model.

---

## ✨ Features

### 🧠 1. Semantic Caching
- Converts each query into a dense embedding vector using `all-MiniLM-L6-v2` (22 MB, runs on CPU)
- Stores query+response pairs in a local FAISS vector index
- Before sending any query to the LLM, checks for semantically similar past queries (configurable threshold, default 0.85)
- **Cache hit → returns cached response instantly. Zero LLM calls. Zero water. Zero energy.**
- 24-hour TTL ensures answers stay fresh and accurate

### ✂️ 2. Prompt Compression
- Automatically trims redundant wording before sending to the LLM
- Removes filler phrases (*"Can you please…"*, *"I was wondering if…"*, *"Thanks in advance"*)
- Substitutes verbose qualifiers (*"in order to" → "to"*, *"due to the fact that" → "because"*)
- Removes near-duplicate sentences within the same prompt
- Shows **before vs. after token count** so you can verify nothing important was removed

### ♻️ 3. Batch / Defer Nudge
- Detects non-urgent queries via keywords (*"later"*, *"no rush"*, *"when you can"*, *"summarise"*)
- Queues them and nudges you to combine multiple queries into **one single request**
- Saves N−1 LLM calls per batch

### 📊 4. Impact Dashboard
Real-time tracking of:
- Total queries & cache hit rate
- Tokens saved via compression
- Queries batched vs individual
- **Estimated water & energy saved** (based on Microsoft/Google 2023 sustainability reports)

---

## 🌐 SDG Alignment

| SDG | How EcoPrompt Contributes |
|---|---|
| 💧 **SDG 6** — Clean Water & Sanitation | Reduces data-centre water consumption per AI interaction |
| ♻️ **SDG 12** — Responsible Consumption | Eliminates redundant compute through cache reuse and compression |
| 🌡️ **SDG 13** — Climate Action | Lowers energy-related CO₂ emissions from LLM inference |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- A Gemini API key — free at [aistudio.google.com](https://aistudio.google.com)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Divyaa1630928/Ecoprompt.git
cd Ecoprompt

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file
# Create a file named .env with the following content:
# GEMINI_API_KEY=your_key_here
# GEMINI_MODEL=gemini-3.6-flash

# 4. Run the app
streamlit run app.py
```

> 💡 **No API key?** The app runs in **Demo Mode** automatically — all three modules and the dashboard work fully offline without any API key.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[Non-urgency check] ──► batch queue ──► Batch Nudge Banner
    │
    ▼
[Semantic Cache Lookup]
    │
    ├── HIT (similarity ≥ 0.85) ──► Return cached response
    │                                (0 LLM calls, water & energy saved ✅)
    │
    └── MISS ──► [Prompt Compressor] ──► trim tokens
                        │
                        ▼
                 [LLM — Gemini API]
                        │
                        ▼
                 [Cache Store] ──► save embedding + response
                        │
                        ▼
                   Response displayed
                   Metrics updated
```

---

## 📁 Project Structure

```
Ecoprompt/
├── app.py                  ← Streamlit dashboard (entry point)
├── semantic_cache.py       ← FAISS-backed semantic cache with TTL
├── prompt_compressor.py    ← Rule-based prompt compression
├── batch_nudge.py          ← Non-urgent batch nudge logic
├── llm_client.py           ← Gemini / OpenAI / demo adapter
├── requirements.txt        ← Python dependencies
├── ARCHITECTURE.md         ← Detailed system flow diagram
├── TEST_QUERIES.md         ← Sample test scenarios with expected outputs
└── PROJECT_REPORT.md       ← Full internship project report
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Embeddings | `sentence-transformers` — all-MiniLM-L6-v2 |
| Vector Search | FAISS (flat inner-product index) |
| Token Counting | `tiktoken` (cl100k_base BPE) |
| LLM | Google Gemini via `google-genai` SDK |
| Config | `python-dotenv` |

---

## 🔬 Responsible AI

- **Cache threshold is user-adjustable** (0.50–1.00) — users control the accuracy vs. savings trade-off
- **Compression is conservative** — never removes named entities, numbers, or domain-specific terms; before/after always visible
- **All savings figures are clearly labelled estimates** derived from Google & Microsoft 2023 sustainability disclosures, not exact measurements

---


