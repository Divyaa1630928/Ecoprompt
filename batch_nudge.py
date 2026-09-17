"""
ecoprompt/batch_nudge.py
──────────────────────────────────────────────────────────────────────────────
Batch / Defer Nudge Module for EcoPrompt
──────────────────────────────────────────────────────────────────────────────
Purpose:
  Reduce the number of separate LLM inference calls by identifying queries
  that the user does NOT need answered immediately and encouraging them to
  combine multiple such queries into a single, consolidated request.

How it works:
  1. NON-URGENCY DETECTION
     A query is classified as non-urgent if:
       (a) It contains one or more keyword signals from NON_URGENT_SIGNALS
           (e.g. "later", "when you can", "no rush", "summarise", "batch"),
     OR
       (b) The user manually toggles the "non-urgent" flag in the UI.

  2. PENDING QUEUE
     Non-urgent queries are placed in a time-windowed pending queue rather
     than being sent immediately.

  3. BATCH NUDGE
     Whenever ≥ 2 queries are pending within the configured `window_seconds`
     (default 60 s), the system surfaces a nudge:
       "You have N pending queries.  Combine them into one request to save
        water and energy!"
     The user can then either:
       (a) Accept the nudge → queries are merged into one combined prompt.
       (b) Dismiss → all pending queries are flushed individually.

  4. COMBINED PROMPT GENERATION
     Combines pending queries using a numbered-list template so the LLM can
     answer them in a single call.

  5. METRICS
     Tracks:
       – total_nudges_shown : how many batch opportunities were surfaced
       – queries_batched    : how many queries were sent in batch mode
       – queries_individual : how many were sent individually
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

# ── non-urgency keyword signals ────────────────────────────────────────────
NON_URGENT_SIGNALS: list[str] = [
    "later",
    "when you can",
    "no rush",
    "not urgent",
    "non-urgent",
    "whenever",
    "at your convenience",
    "summarize",
    "summarise",
    "batch",
    "compile",
    "when you get a chance",
    "sometime",
    "eventually",
    "background",
    "low priority",
]


# ── data structures ────────────────────────────────────────────────────────
@dataclass
class PendingQuery:
    """A query waiting in the batch queue."""
    query: str
    arrived_at: float = field(default_factory=time.time)
    manually_flagged: bool = False   # True if user explicitly marked non-urgent

    def age_seconds(self) -> float:
        return time.time() - self.arrived_at


@dataclass
class NudgeDecision:
    """
    Returned by BatchNudgeManager.evaluate().

    Attributes
    ----------
    should_nudge : bool
        True if the user should be shown the batch suggestion.
    pending_count : int
        Number of queries currently in the pending queue.
    pending_queries : list[str]
        The actual query texts waiting.
    message : str
        Human-readable nudge message to display in the UI.
    combined_prompt : str
        Ready-to-send combined prompt (use this if user accepts nudge).
    """
    should_nudge: bool
    pending_count: int
    pending_queries: list[str]
    message: str
    combined_prompt: str


# ── main class ─────────────────────────────────────────────────────────────
class BatchNudgeManager:
    """
    Manages the non-urgent query queue and batch nudge logic.

    Parameters
    ----------
    window_seconds : float
        Queries that arrive within this window are eligible for batching.
        Default: 60 seconds.
    min_batch_size : int
        Minimum number of pending queries before a nudge is surfaced.
        Default: 2.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        min_batch_size: int = 2,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_batch_size = min_batch_size

        self._queue: list[PendingQuery] = []

        # ── metrics ───────────────────────────────────────────────────────
        self.total_nudges_shown: int = 0
        self.queries_batched: int = 0
        self.queries_individual: int = 0

    # ── public API ─────────────────────────────────────────────────────────

    def is_non_urgent(self, query: str) -> bool:
        """
        Return True if the query contains non-urgency signals.

        This check is case-insensitive and looks for whole-phrase matches.
        """
        lower = query.lower()
        return any(signal in lower for signal in NON_URGENT_SIGNALS)

    def add_to_queue(self, query: str, manually_flagged: bool = False) -> None:
        """
        Add a non-urgent query to the pending queue.

        Call this when:
          – is_non_urgent() returns True, OR
          – the user manually toggles the non-urgent flag in the UI.
        """
        self._queue.append(PendingQuery(query=query, manually_flagged=manually_flagged))

    def evaluate(self) -> NudgeDecision:
        """
        Evaluate the current queue state and decide whether to nudge.

        Prunes queries that have aged beyond window_seconds before deciding.
        Returns a NudgeDecision.
        """
        # Remove queries that have aged out of the window
        self._prune_old()

        pending_texts = [pq.query for pq in self._queue]
        count = len(pending_texts)

        if count >= self.min_batch_size:
            combined = self._build_combined_prompt(pending_texts)
            self.total_nudges_shown += 1
            return NudgeDecision(
                should_nudge=True,
                pending_count=count,
                pending_queries=pending_texts,
                message=(
                    f"♻️  You have **{count} non-urgent queries** pending.  "
                    f"Combining them into a single request saves ~{count - 1} "
                    f"LLM call(s), reducing water and energy use."
                ),
                combined_prompt=combined,
            )

        return NudgeDecision(
            should_nudge=False,
            pending_count=count,
            pending_queries=pending_texts,
            message="",
            combined_prompt="",
        )

    def accept_batch(self) -> str:
        """
        User accepted the nudge.  Clear the queue and return the combined prompt.
        Increments `queries_batched` by the number of queries merged.
        """
        count = len(self._queue)
        combined = self._build_combined_prompt([pq.query for pq in self._queue])
        self.queries_batched += count
        self._queue.clear()
        return combined

    def dismiss_batch(self) -> list[str]:
        """
        User dismissed the nudge.  Flush all pending queries individually.
        Increments `queries_individual` and clears the queue.
        """
        queries = [pq.query for pq in self._queue]
        self.queries_individual += len(queries)
        self._queue.clear()
        return queries

    def flush_single(self, query: str) -> None:
        """
        Mark a single query as sent individually (used when queue has <2 items).
        """
        self.queries_individual += 1
        # Remove from queue if present
        self._queue = [pq for pq in self._queue if pq.query != query]

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_combined_prompt(queries: list[str]) -> str:
        """
        Format multiple queries into a single numbered-list prompt.

        The template instructs the LLM to answer each sub-query in order,
        which keeps the single response well-structured.
        """
        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))
        return (
            "Please answer each of the following questions in order, "
            "clearly labelling your answer with the corresponding number:\n\n"
            + numbered
        )

    def _prune_old(self) -> None:
        """Remove queue entries older than window_seconds."""
        self._queue = [
            pq for pq in self._queue
            if pq.age_seconds() <= self.window_seconds
        ]
