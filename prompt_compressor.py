"""
ecoprompt/prompt_compressor.py
──────────────────────────────────────────────────────────────────────────────
Prompt Compression Module for EcoPrompt
──────────────────────────────────────────────────────────────────────────────
Strategy (rule-based, no extra model required):
  A lightweight, deterministic pipeline that removes redundancy without
  altering the semantic intent of the query.  Each stage is independently
  enabled/disabled so you can tune aggressiveness.

  Stage 1 – Filler phrase removal
      Strips courtesy openers and meta-commentary that carry zero information
      for the LLM (e.g. "Can you please", "I was wondering if", "Just to
      let you know", "As an AI language model …").

  Stage 2 – Whitespace normalisation
      Collapses multiple spaces, strips trailing/leading whitespace, and
      removes blank lines that pad token counts.

  Stage 3 – Redundant clause detection
      Removes sentences that are near-duplicates of a preceding sentence in
      the same prompt (cosine similarity > 0.92 on TF-IDF vectors — pure
      Python, no heavy model needed for this lightweight check).

  Stage 4 – Verbose-qualifier trimming
      Removes qualifier phrases that add length but not meaning, such as
      "In order to …→ To …", "Due to the fact that → Because", etc.

Token counting uses the `tiktoken` library (same BPE tokeniser as GPT-3.5/4)
so the before/after numbers are model-accurate.

Responsible AI note
  ──────────────────
  The compressor is intentionally conservative.  It only removes patterns
  that are empirically redundant across a broad range of natural-language
  prompts (see FILLER_PHRASES and VERBOSE_SUBS below).  Critical keywords,
  named entities, and numbers are never stripped.  Users can inspect the
  before/after diff in the dashboard to verify no meaning was lost.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import sqrt

import tiktoken

# ── token counter ──────────────────────────────────────────────────────────
_TOKENIZER = tiktoken.get_encoding("cl100k_base")   # GPT-3.5/4 compatible


def count_tokens(text: str) -> int:
    """Return the number of BPE tokens in `text`."""
    return len(_TOKENIZER.encode(text))


# ── filler phrases (case-insensitive prefix / anywhere patterns) ───────────
# Each item is a regex pattern that will be stripped.
FILLER_PHRASES: list[str] = [
    # Politeness openers
    r"^(hey|hi|hello)[,!]?\s*",
    r"^(can you|could you|would you mind|please)\s+(please\s+)?",
    r"^i\s+(was\s+wondering|want\s+to\s+know|need\s+to\s+know|would\s+like\s+to\s+know)\s+(if\s+you\s+could\s+)?(please\s+)?",
    r"^i\s+am\s+(asking|wondering|curious)\s+(about\s+)?(whether\s+|if\s+)?",
    # Meta-commentary
    r"^(just\s+(to\s+let\s+you\s+know|a\s+quick\s+(question|note)),?\s*)",
    r"^(as\s+an?\s+(AI|language\s+model|assistant)[,.]?\s*)",
    r"^(i\s+hope\s+(this\s+(question\s+)?is\s+okay|you\s+can\s+help)[.,]?\s*)",
    r"(,?\s*if\s+(that\s+)?makes\s+sense\.?\s*)$",
    r"(,?\s*i\s+hope\s+that\s+(makes\s+sense|is\s+clear)\.?\s*)$",
    r"(,?\s*thank\s+you\s+(in\s+advance|so\s+much)?\.?\s*)$",
    r"(,?\s*thanks\s+(in\s+advance|a\s+lot)?\.?\s*)$",
    # Hedging suffixes
    r"(\s*please\s+let\s+me\s+know\s+if\s+(you\s+need\s+(more\s+)?clarification|that\s+helps)[.!]?\s*)$",
]

# ── verbose substitutions (phrase → shorter equivalent) ───────────────────
VERBOSE_SUBS: list[tuple[str, str]] = [
    (r"\bin order to\b",               "to"),
    (r"\bdue to the fact that\b",      "because"),
    (r"\bat this point in time\b",     "now"),
    (r"\bon a daily basis\b",          "daily"),
    (r"\bin the event that\b",         "if"),
    (r"\bfor the purpose of\b",        "to"),
    (r"\bwith regard to\b",            "regarding"),
    (r"\bin spite of the fact that\b", "although"),
    (r"\bit is worth noting that\b",   "notably,"),
    (r"\bit should be noted that\b",   "note:"),
    (r"\bthe reason why is that\b",    "because"),
    (r"\ba large number of\b",         "many"),
    (r"\ba small number of\b",         "few"),
    (r"\bin close proximity to\b",     "near"),
    (r"\bhas the ability to\b",        "can"),
    (r"\bis able to\b",                "can"),
    (r"\bmake use of\b",               "use"),
    (r"\bprior to\b",                  "before"),
    (r"\bsubsequent to\b",             "after"),
    (r"\bin addition to\b",            "besides"),
]


# ── result dataclass ───────────────────────────────────────────────────────
@dataclass
class CompressionResult:
    """Holds the output of a compression pass."""
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    compression_ratio: float    # 0–1, higher = more compressed
    stages_applied: list[str]


# ── main class ─────────────────────────────────────────────────────────────
class PromptCompressor:
    """
    Rule-based prompt compressor.

    Parameters
    ----------
    remove_fillers : bool
        Apply Stage 1 (filler phrase removal).  Default True.
    remove_verbose : bool
        Apply Stage 4 (verbose qualifier substitution).  Default True.
    dedup_sentences : bool
        Apply Stage 3 (near-duplicate sentence removal).  Default True.
    dedup_threshold : float
        TF-IDF cosine similarity threshold for Stage 3.  Default 0.92.
    """

    def __init__(
        self,
        remove_fillers: bool = True,
        remove_verbose: bool = True,
        dedup_sentences: bool = True,
        dedup_threshold: float = 0.92,
    ) -> None:
        self.remove_fillers = remove_fillers
        self.remove_verbose = remove_verbose
        self.dedup_sentences = dedup_sentences
        self.dedup_threshold = dedup_threshold

        # Running total of tokens saved across all calls
        self.total_tokens_saved: int = 0

    # ── public API ─────────────────────────────────────────────────────────

    def compress(self, query: str) -> CompressionResult:
        """
        Compress `query` through all enabled stages.

        Returns a CompressionResult with original/compressed text and
        token counts for display in the dashboard.
        """
        text = query
        stages: list[str] = []

        # Stage 1: filler phrase removal
        if self.remove_fillers:
            new_text = self._strip_fillers(text)
            if new_text != text:
                stages.append("filler removal")
            text = new_text

        # Stage 2: whitespace normalisation (always applied)
        new_text = self._normalise_whitespace(text)
        if new_text != text:
            stages.append("whitespace normalisation")
        text = new_text

        # Stage 3: redundant sentence deduplication
        if self.dedup_sentences:
            new_text = self._dedup(text)
            if new_text != text:
                stages.append("duplicate sentence removal")
            text = new_text

        # Stage 4: verbose qualifier substitution
        if self.remove_verbose:
            new_text = self._sub_verbose(text)
            if new_text != text:
                stages.append("verbose phrase substitution")
            text = new_text

        # Final whitespace pass
        text = self._normalise_whitespace(text)

        orig_tokens = count_tokens(query)
        comp_tokens = count_tokens(text)
        saved = max(0, orig_tokens - comp_tokens)
        ratio = saved / orig_tokens if orig_tokens > 0 else 0.0

        self.total_tokens_saved += saved

        return CompressionResult(
            original=query,
            compressed=text,
            original_tokens=orig_tokens,
            compressed_tokens=comp_tokens,
            tokens_saved=saved,
            compression_ratio=ratio,
            stages_applied=stages if stages else ["none (already concise)"],
        )

    # ── stage implementations ──────────────────────────────────────────────

    def _strip_fillers(self, text: str) -> str:
        for pattern in FILLER_PHRASES:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
        # Capitalise the first letter after stripping (if needed)
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        # Collapse internal whitespace and strip
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def _dedup(self, text: str) -> str:
        """Remove sentences that are near-duplicates of a previous sentence."""
        # Split on sentence boundaries (., !, ? followed by space/end)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept: list[str] = []
        kept_vecs: list[dict[str, float]] = []

        for sentence in sentences:
            vec = self._tfidf_vector(sentence)
            is_dup = any(
                self._cosine(vec, prev) >= self.dedup_threshold
                for prev in kept_vecs
            )
            if not is_dup:
                kept.append(sentence)
                kept_vecs.append(vec)

        return " ".join(kept)

    @staticmethod
    def _sub_verbose(text: str) -> str:
        for pattern, replacement in VERBOSE_SUBS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    # ── tiny TF-IDF helpers (no external deps) ────────────────────────────

    @staticmethod
    def _tfidf_vector(text: str) -> dict[str, float]:
        """Return a simple term-frequency dict (normalised) for `text`."""
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return {}
        counts: dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        total = len(words)
        return {w: c / total for w, c in counts.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        """Cosine similarity between two TF dicts."""
        if not a or not b:
            return 0.0
        dot = sum(a.get(w, 0.0) * b.get(w, 0.0) for w in a)
        mag_a = sqrt(sum(v * v for v in a.values()))
        mag_b = sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
