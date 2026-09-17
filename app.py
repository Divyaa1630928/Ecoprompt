"""
ecoprompt/app.py
──────────────────────────────────────────────────────────────────────────────
EcoPrompt — AI Query Optimizer for Sustainability
Streamlit dashboard integrating:
  • Semantic Caching Module
  • Prompt Compression Module
  • Batch / Defer Nudge Module
  • Impact Metrics Dashboard

Run with:
    streamlit run ecoprompt/app.py
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from batch_nudge import BatchNudgeManager
from llm_client import call_llm, is_demo_mode
from prompt_compressor import PromptCompressor
from semantic_cache import SemanticCache, WATER_PER_SAVED_CALL_ML, ENERGY_PER_SAVED_CALL_KWH

# Load .env from the ecoprompt folder regardless of launch directory
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoPrompt",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #15803d; }
  .metric-label { font-size: 0.85rem; color: #4b5563; margin-top: 4px; }
  .badge-hit   { background:#dcfce7; color:#166534; padding:2px 8px;
                  border-radius:9999px; font-size:0.8rem; }
  .badge-miss  { background:#fef9c3; color:#854d0e; padding:2px 8px;
                  border-radius:9999px; font-size:0.8rem; }
  .badge-batch { background:#dbeafe; color:#1e40af; padding:2px 8px;
                  border-radius:9999px; font-size:0.8rem; }
  .responsible-box {
    background:#fafafa; border:1px solid #e5e7eb; border-radius:8px;
    padding:16px; font-size:0.88rem; color:#374151;
  }
</style>
""", unsafe_allow_html=True)


# ── session-state initialisation ───────────────────────────────────────────
def _init_state() -> None:
    """Initialise all session state objects on first load."""
    if "cache" not in st.session_state:
        c = SemanticCache(
            similarity_threshold=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.85)),
            ttl_seconds=float(os.getenv("CACHE_TTL_SECONDS", 86400)),
        )
        # Load persisted cache from disk if available
        if hasattr(c, "load"):
            c.load()
        st.session_state["cache"] = c

    defaults = {
        "compressor": PromptCompressor(),
        "nudge_mgr":  BatchNudgeManager(
                          window_seconds=float(os.getenv("BATCH_WINDOW_SECONDS", 60)),
                      ),
        "history":    [],
        "pending_accept": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

cache:    SemanticCache      = st.session_state["cache"]
comp:     PromptCompressor   = st.session_state["compressor"]
nudge_mgr: BatchNudgeManager = st.session_state["nudge_mgr"]


# ── sidebar settings ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌿 EcoPrompt")
    st.caption("AI Query Optimizer for Sustainability")

    if is_demo_mode():
        st.warning("**Demo Mode** — no API key detected.  "
                   "Simulated LLM responses will be used.",
                   icon="🔑")

    st.divider()
    st.subheader("⚙️ Settings")

    new_threshold = st.slider(
        "Cache similarity threshold",
        min_value=0.50, max_value=1.00, step=0.01,
        value=cache.similarity_threshold,
        help="Higher = stricter match required before serving a cached answer. "
             "Lower = more cache hits but higher risk of serving a subtly wrong answer.",
    )
    if new_threshold != cache.similarity_threshold:
        cache.similarity_threshold = new_threshold

    new_window = st.slider(
        "Batch window (seconds)",
        min_value=10, max_value=300, step=10,
        value=int(nudge_mgr.window_seconds),
        help="Non-urgent queries that arrive within this window are eligible for batching.",
    )
    if new_window != nudge_mgr.window_seconds:
        nudge_mgr.window_seconds = float(new_window)

    if st.button("🗑️ Clear cache & reset metrics", use_container_width=True):
        cache.clear()
        comp.total_tokens_saved = 0
        nudge_mgr.total_nudges_shown = 0
        nudge_mgr.queries_batched = 0
        nudge_mgr.queries_individual = 0
        st.session_state["history"] = []
        st.success("All data cleared.")

    st.divider()
    st.subheader("🔬 Responsible AI")
    st.markdown("""
<div class="responsible-box">

**Cache threshold trade-off**  
A threshold that is too loose (e.g. 0.60) risks serving a cached answer
to a query that is only superficially similar, potentially giving incorrect
information.  A threshold that is too strict (e.g. 0.99) will rarely reuse
cached answers, reducing eco-savings.  The default (0.85) balances accuracy
and efficiency.

**Prompt compression safety**  
The compressor only removes statistically redundant patterns (filler
phrases, verbose synonyms, duplicate sentences).  It never removes named
entities, numbers, or domain-specific terms.  Users can inspect the
before/after diff before sending.

**Water/energy estimates**  
Figures are approximations derived from Google's 2023 Environmental Report
and Microsoft's 2023 Sustainability Report.  Actual consumption varies by
model size, hardware, and data-centre efficiency. These numbers are
illustrative only and should not be cited as exact measurements.
</div>
""", unsafe_allow_html=True)


# ── main layout ────────────────────────────────────────────────────────────
st.title("🌿 EcoPrompt — AI Query Optimizer")
st.caption(
    "Reduce the water & energy footprint of your AI queries through semantic "
    "caching, prompt compression, and intelligent batching."
)

tab_query, tab_dashboard, tab_log = st.tabs(
    ["💬 Query", "📊 Impact Dashboard", "📋 Activity Log"]
)


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — QUERY
# ══════════════════════════════════════════════════════════════════════════
with tab_query:

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Submit a query")
        user_query = st.text_area(
            "Your query",
            height=140,
            placeholder="Type your question here…  (include words like 'later' or "
                        "'no rush' to trigger the batch nudge)",
        )
        non_urgent_flag = st.checkbox(
            "Mark as non-urgent (eligible for batching)",
            help="Tick this to add the query to the batch queue even if it doesn't "
                 "contain non-urgency keywords.",
        )
        send_btn = st.button("🚀 Send Query", type="primary", use_container_width=True)

    # ── query processing function (defined before the nudge banner uses it) ─
    def _process_query(query: str, is_batch: bool = False) -> None:
        """Run a query through the full EcoPrompt pipeline."""
        if not query.strip():
            return

        event: dict = {
            "timestamp": time.strftime("%H:%M:%S"),
            "query":     query,
            "is_batch":  is_batch,
        }

        # ── Step 1: Semantic cache lookup ──────────────────────────────────
        cache_result = cache.query(query)

        if cache_result.hit:
            event.update({
                "type":       "cache_hit",
                "similarity": round(cache_result.similarity, 4),
                "matched":    cache_result.matched_query,
                "response":   cache_result.response,
                "tokens_saved_compression": 0,
            })
            st.session_state["history"].append(event)
            st.session_state["last_event"] = event
            return

        # ── Step 2: Prompt compression (cache miss only) ───────────────────
        comp_result = comp.compress(query)
        prompt_to_send = comp_result.compressed

        # ── Step 3: Call LLM ───────────────────────────────────────────────
        response = call_llm(prompt_to_send)

        # ── Step 4: Store in cache ─────────────────────────────────────────
        cache.store(query, response)

        event.update({
            "type":                     "cache_miss",
            "original_tokens":          comp_result.original_tokens,
            "compressed_tokens":        comp_result.compressed_tokens,
            "tokens_saved_compression": comp_result.tokens_saved,
            "compression_ratio":        comp_result.compression_ratio,
            "stages":                   comp_result.stages_applied,
            "compressed_query":         comp_result.compressed,
            "response":                 response,
            "similarity":               0.0,
        })
        st.session_state["history"].append(event)
        st.session_state["last_event"] = event
        if hasattr(cache, "save"):
            cache.save()   # persist cache to disk after every new entry


    # ── batch nudge banner (shown between input and results) ───────────────
    nudge_decision = nudge_mgr.evaluate()
    if nudge_decision.should_nudge:
        st.divider()
        st.markdown("### ♻️ Batch Opportunity Detected!")
        st.info(nudge_decision.message)

        with st.expander("📋 Pending queries in batch queue", expanded=True):
            for i, q in enumerate(nudge_decision.pending_queries, 1):
                st.markdown(f"**{i}.** {q}")
            st.code(nudge_decision.combined_prompt, language="text")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Accept — send as batch", use_container_width=True, type="primary"):
                combined = nudge_mgr.accept_batch()
                _process_query(combined, is_batch=True)
                st.rerun()
        with c2:
            if st.button("❌ Dismiss — send individually", use_container_width=True):
                individual = nudge_mgr.dismiss_batch()
                for q in individual:
                    _process_query(q, is_batch=False)
                st.rerun()

    # ── handle send button ─────────────────────────────────────────────────
    if send_btn and user_query.strip():
        is_non_urgent = nudge_mgr.is_non_urgent(user_query) or non_urgent_flag

        if is_non_urgent:
            nudge_mgr.add_to_queue(user_query, manually_flagged=non_urgent_flag)
            st.info(
                f"Query added to the non-urgent queue "
                f"({nudge_mgr.queue_size} pending).  "
                f"It will be suggested for batching once {nudge_mgr.min_batch_size} "
                f"or more queries accumulate."
            )
        else:
            _process_query(user_query, is_batch=False)

        st.rerun()

    # ── display last result ────────────────────────────────────────────────
    with col_result:
        st.subheader("Result")
        last = st.session_state.get("last_event")

        if last is None:
            st.markdown("_Submit a query to see results here._")
        elif last["type"] == "cache_hit":
            st.markdown(
                '<span class="badge-hit">✅ CACHE HIT</span>',
                unsafe_allow_html=True,
            )
            st.metric("Similarity to cached query", f"{last['similarity']:.2%}")
            st.markdown(f"**Matched cached query:**  _{last['matched']}_")
            st.markdown("**Response (from cache):**")
            st.success(last["response"])
            st.caption("🌍 No LLM call made — water & energy saved!")
        else:
            st.markdown(
                '<span class="badge-miss">⚡ CACHE MISS — LLM called</span>',
                unsafe_allow_html=True,
            )
            # Compression details
            if last["tokens_saved_compression"] > 0:
                st.markdown("**Prompt compression applied:**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Original tokens", last["original_tokens"])
                c2.metric("Compressed tokens", last["compressed_tokens"])
                c3.metric("Tokens saved", last["tokens_saved_compression"],
                          delta=f"-{last['compression_ratio']:.0%}")
                st.markdown(f"_Stages: {', '.join(last['stages'])}_")
                with st.expander("See compressed query"):
                    st.code(last["compressed_query"], language="text")
            else:
                st.caption("_Query was already concise — no compression needed._")

            st.markdown("**Response:**")
            st.info(last["response"])

            if is_demo_mode():
                st.caption("🔑 Demo mode — configure OPENAI_API_KEY for real responses.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — IMPACT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.subheader("📊 Real-Time Impact Metrics")

    # ── top metric cards ───────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{cache.total_queries}</div>
          <div class="metric-label">Total Queries</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        hit_pct = f"{cache.hit_rate:.0%}"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{hit_pct}</div>
          <div class="metric-label">Cache Hit Rate<br>({cache.cache_hits} hits)</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{comp.total_tokens_saved}</div>
          <div class="metric-label">Tokens Saved<br>(compression)</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{nudge_mgr.queries_batched}</div>
          <div class="metric-label">Queries Batched<br>({nudge_mgr.queries_individual} individual)</div>
        </div>""", unsafe_allow_html=True)

    with m5:
        water = f"{cache.water_saved_ml:.1f} ml"
        energy = f"{cache.energy_saved_kwh * 1000:.2f} Wh"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{water}</div>
          <div class="metric-label">Est. Water Saved*<br>{energy} energy saved*</div>
        </div>""", unsafe_allow_html=True)

    st.caption(
        "\\* Water savings estimated at 0.5 ml per avoided LLM call (Microsoft 2023 "
        "Sustainability Report approximation). Energy estimated at 0.003 kWh per "
        "avoided call. These are illustrative figures only."
    )

    st.divider()

    # ── query type breakdown bar ───────────────────────────────────────────
    st.subheader("Query Breakdown")
    col_a, col_b = st.columns(2)

    with col_a:
        total = cache.total_queries
        if total > 0:
            hit_frac  = cache.cache_hits / total
            miss_frac = cache.cache_misses / total
            st.markdown(f"""
            <div style="background:#e5e7eb;border-radius:9999px;height:22px;overflow:hidden;">
              <div style="width:{hit_frac*100:.1f}%;background:#16a34a;height:100%;
                          display:inline-block;"></div><div
                   style="width:{miss_frac*100:.1f}%;background:#ca8a04;height:100%;
                          display:inline-block;"></div>
            </div>
            <div style="margin-top:6px;font-size:0.82rem;">
              <span style="color:#16a34a">■</span> Cache hits ({cache.cache_hits})&nbsp;&nbsp;
              <span style="color:#ca8a04">■</span> Cache misses ({cache.cache_misses})
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No queries yet.")

    with col_b:
        batched = nudge_mgr.queries_batched
        individual = nudge_mgr.queries_individual
        total_sent = batched + individual
        if total_sent > 0:
            b_frac = batched / total_sent
            i_frac = individual / total_sent
            st.markdown(f"""
            <div style="background:#e5e7eb;border-radius:9999px;height:22px;overflow:hidden;">
              <div style="width:{b_frac*100:.1f}%;background:#2563eb;height:100%;
                          display:inline-block;"></div><div
                   style="width:{i_frac*100:.1f}%;background:#9ca3af;height:100%;
                          display:inline-block;"></div>
            </div>
            <div style="margin-top:6px;font-size:0.82rem;">
              <span style="color:#2563eb">■</span> Batched ({batched})&nbsp;&nbsp;
              <span style="color:#9ca3af">■</span> Individual ({individual})
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No batch data yet.")

    st.divider()

    # ── cache entries table ────────────────────────────────────────────────
    st.subheader(f"Cache Contents ({cache.cache_size} entries)")
    if cache.cache_size == 0:
        st.info("The semantic cache is empty.  Submit a query to populate it.")
    else:
        rows = []
        for entry in cache._entries:
            age_min = (time.time() - entry.created_at) / 60
            rows.append({
                "Query": entry.query[:80] + ("…" if len(entry.query) > 80 else ""),
                "Response preview": entry.response[:60] + "…",
                "Age (min)": f"{age_min:.1f}",
            })
        st.dataframe(rows, use_container_width=True)

    st.divider()

    # ── SDG alignment callout ──────────────────────────────────────────────
    st.subheader("🌐 SDG Alignment")
    sdg_cols = st.columns(3)
    sdg_data = [
        ("💧 SDG 6", "Clean Water & Sanitation",
         "Reducing AI water consumption through semantic caching and query batching."),
        ("♻️ SDG 12", "Responsible Consumption",
         "Avoiding redundant compute through cache reuse and prompt compression."),
        ("🌡️ SDG 13", "Climate Action",
         "Lowering energy-related carbon emissions from LLM inference."),
    ]
    for col, (icon_title, subtitle, body) in zip(sdg_cols, sdg_data):
        with col:
            st.markdown(f"**{icon_title} — {subtitle}**")
            st.markdown(body)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — ACTIVITY LOG
# ══════════════════════════════════════════════════════════════════════════
with tab_log:
    st.subheader("📋 Activity Log")
    history: list[dict] = st.session_state["history"]

    if not history:
        st.info("No activity yet.  Submit a query to see the log.")
    else:
        for i, event in enumerate(reversed(history), 1):
            badge = (
                '<span class="badge-hit">CACHE HIT</span>'   if event["type"] == "cache_hit"
                else '<span class="badge-batch">BATCHED</span>'  if event.get("is_batch")
                else '<span class="badge-miss">LLM CALL</span>'
            )
            with st.expander(
                f"[{event['timestamp']}]  {event['query'][:60]}{'…' if len(event['query'])>60 else ''}",
                expanded=(i == 1),
            ):
                st.markdown(badge, unsafe_allow_html=True)
                st.markdown(f"**Full query:** {event['query']}")
                if event["type"] == "cache_hit":
                    st.markdown(f"**Similarity:** {event['similarity']:.4f}")
                    st.markdown(f"**Matched:** _{event['matched']}_")
                else:
                    saved = event.get("tokens_saved_compression", 0)
                    if saved > 0:
                        st.markdown(
                            f"**Compression:** {event['original_tokens']} → "
                            f"{event['compressed_tokens']} tokens "
                            f"(saved {saved})"
                        )
                st.markdown(f"**Response:** {event['response']}")
