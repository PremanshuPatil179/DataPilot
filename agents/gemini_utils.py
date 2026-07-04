"""
Shared Gemini helpers for DataPilot AI.
"""

from __future__ import annotations

import os


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
LEGACY_MODEL_ALIASES = {
    "gemini-1.5-flash": DEFAULT_GEMINI_MODEL,
}


def resolve_gemini_api_key(api_key: str | None = None) -> str:
    """Resolve the Gemini API key from arguments, env vars, or Streamlit secrets."""
    if api_key and api_key.strip():
        return api_key.strip()

    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        import streamlit as st
    except Exception:
        return ""

    try:
        secrets = getattr(st, "secrets", None)
        if secrets and "GEMINI_API_KEY" in secrets:
            return str(secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        return ""

    return ""


def resolve_gemini_model(model: str | None = None) -> str:
    """Normalize deprecated Gemini model names to a supported default."""
    selected = (model or os.getenv("GEMINI_MODEL", "") or DEFAULT_GEMINI_MODEL).strip()
    return LEGACY_MODEL_ALIASES.get(selected, selected or DEFAULT_GEMINI_MODEL)
