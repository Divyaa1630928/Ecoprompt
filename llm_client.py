"""
ecoprompt/llm_client.py
──────────────────────────────────────────────────────────────────────────────
Thin LLM wrapper for EcoPrompt.

Priority order (first key found wins):
  1. GEMINI_API_KEY  → Google Gemini (gemini-2.0-flash by default)
  2. OPENAI_API_KEY  → OpenAI (gpt-3.5-turbo by default)
  3. No key found    → DEMO MODE (no network, no cost)
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Always load .env from the same folder as this file, regardless of where
# the app was launched from.
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo").strip()

# ── demo responses (used when no API key is configured) ────────────────────
_DEMO_RESPONSES = [
    "This is a simulated LLM response (demo mode — no API key configured). "
    "In a live deployment this would be a real answer from the language model.",
    "Demo mode active. EcoPrompt intercepted this query and would normally "
    "forward it to the LLM. Configure GEMINI_API_KEY in .env to enable live responses.",
    "Simulated answer: The EcoPrompt system is working correctly. "
    "Add your Gemini API key to .env to get real responses.",
]
_demo_counter = 0


def call_llm(prompt: str) -> str:
    """
    Send `prompt` to the LLM and return the text response.

    Uses Gemini if GEMINI_API_KEY is set, otherwise OpenAI,
    otherwise falls back to demo mode.
    """
    global _demo_counter

    # ── Gemini (google-genai SDK only) ─────────────────────────────────────
    if _GEMINI_KEY:
        try:
            import warnings
            import google.genai as genai                     # type: ignore
            client = genai.Client(api_key=_GEMINI_KEY)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                response = client.models.generate_content(
                    model=_GEMINI_MODEL,
                    contents=prompt,
                )
            return response.text
        except Exception as exc:                             # noqa: BLE001
            return f"[Gemini error: {exc}]"

    # ── OpenAI fallback ────────────────────────────────────────────────────
    if _OPENAI_KEY:
        try:
            from openai import OpenAI                        # type: ignore
            client = OpenAI(api_key=_OPENAI_KEY)
            completion = client.chat.completions.create(
                model=_OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:                             # noqa: BLE001
            return f"[OpenAI error: {exc}]"

    # ── Demo mode ──────────────────────────────────────────────────────────
    response = _DEMO_RESPONSES[_demo_counter % len(_DEMO_RESPONSES)]
    _demo_counter += 1
    return response


def is_demo_mode() -> bool:
    """Return True if running without any real API key."""
    return not bool(_GEMINI_KEY or _OPENAI_KEY)
