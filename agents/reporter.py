"""
agents/reporter.py - Reporter Agent for DataPilot AI

Generates a human-readable, structured summary report of all
data-cleaning transformations performed by the CleanerAgent.
It also optionally asks Gemini to produce a natural-language narrative.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from google import genai
from dotenv import load_dotenv

from .gemini_utils import resolve_gemini_api_key, resolve_gemini_model

load_dotenv()


@dataclass
class CleaningReport:
    """Final report produced by ReporterAgent."""

    generated_at: str = ""
    original_shape: tuple = (0, 0)
    cleaned_shape: tuple = (0, 0)
    rows_removed: int = 0
    columns_removed: int = 0
    transformations: list[dict] = field(default_factory=list)
    narrative: str = ""          # LLM-generated prose summary
    markdown_report: str = ""    # Full markdown text
    warnings: list[str] = field(default_factory=list)


class ReporterAgent:
    """
    Agent that compiles a transformation report and optionally
    generates an AI narrative summarising what was done and why.

    Args:
        api_key: Gemini API key.  Falls back to GEMINI_API_KEY env var.
        model:   Gemini model identifier.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = resolve_gemini_api_key(api_key)
        self.model_name = resolve_gemini_model(model or self.DEFAULT_MODEL)
        self._llm_available = bool(key)
        self._llm_error: str | None = None
        if self._llm_available:
            try:
                self.client = genai.Client(api_key=key)
            except Exception as exc:
                self._llm_available = False
                self.client = None
                self._llm_error = str(exc)
        else:
            self.client = None

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def run(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        transformations: list[dict],
        llm_plan: str = "",
    ) -> CleaningReport:
        """
        Build the final cleaning report.

        Args:
            original_df:     DataFrame before cleaning
            cleaned_df:      DataFrame after cleaning
            transformations: Ordered list of transformation log dicts
            llm_plan:        Raw LLM JSON plan (for audit trail)

        Returns:
            CleaningReport populated with stats, markdown, and narrative
        """
        report = CleaningReport()
        report.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report.original_shape = original_df.shape
        report.cleaned_shape = cleaned_df.shape
        report.rows_removed = original_df.shape[0] - cleaned_df.shape[0]
        report.columns_removed = original_df.shape[1] - cleaned_df.shape[1]
        report.transformations = transformations

        # Generate LLM narrative if possible
        if self._llm_available and transformations:
            report.narrative = self._generate_narrative(
                original_df, cleaned_df, transformations
            )
        else:
            if self._llm_error:
                report.warnings.append(f"Gemini setup failed: {self._llm_error}")
            report.narrative = self._default_narrative(report)

        # Build full markdown report
        report.markdown_report = self._build_markdown(report, llm_plan)

        return report

    # ──────────────────────────────────────────────────────────────────────
    # Narrative generation
    # ──────────────────────────────────────────────────────────────────────

    def _generate_narrative(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        transformations: list[dict],
    ) -> str:
        """Ask Gemini to write a plain-English summary of the cleaning steps."""
        steps_text = "\n".join(
            f"  {i + 1}. [{t['action']}] {t['detail']}"
            for i, t in enumerate(transformations)
        )
        prompt = f"""
You are a professional data analyst. Write a concise, clear summary (3-5 sentences)
describing the data cleaning process below. Focus on what was done and why it improves
data quality. Use plain language — no bullet points, no headings.

Original dataset: {original_df.shape[0]:,} rows × {original_df.shape[1]} columns
Cleaned dataset:  {cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]} columns

Cleaning steps performed:
{steps_text}
"""
        try:
            if not self.client:
                raise RuntimeError("Gemini client is not available.")

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Gemini returned an empty response.")
            return text.strip()
        except Exception as exc:
            self._llm_error = str(exc)
            return self._default_narrative(
                CleaningReport(
                    original_shape=original_df.shape,
                    cleaned_shape=cleaned_df.shape,
                    transformations=transformations,
                    warnings=[f"Gemini narrative failed: {exc}"],
                )
            )

    def _default_narrative(self, report: "CleaningReport") -> str:
        """Produce a basic narrative without LLM."""
        n = len(report.transformations)
        return (
            f"The DataPilot AI cleaning pipeline performed {n} transformation(s) on "
            f"the dataset, reducing it from "
            f"{report.original_shape[0]:,} × {report.original_shape[1]} to "
            f"{report.cleaned_shape[0]:,} × {report.cleaned_shape[1]}. "
            f"{report.rows_removed:,} row(s) and "
            f"{report.columns_removed} column(s) were removed during the process."
        )

    # ──────────────────────────────────────────────────────────────────────
    # Markdown report builder
    # ──────────────────────────────────────────────────────────────────────

    def _build_markdown(self, report: "CleaningReport", llm_plan: str) -> str:
        """Assemble the full markdown report string."""
        action_icons = {
            "drop_duplicates": "🔁",
            "drop_column": "🗑️",
            "fill_missing": "🩹",
            "drop_missing_rows": "✂️",
            "convert_dtype": "🔄",
            "strip_whitespace": "🧹",
            "rename_to_snake_case": "✏️",
        }

        rows = []
        for i, t in enumerate(report.transformations, 1):
            icon = action_icons.get(t["action"], "⚙️")
            col_tag = f"`{t['column']}`" if t.get("column") else "—"
            rows.append(f"| {i} | {icon} `{t['action']}` | {col_tag} | {t['detail']} |")

        table = (
            "| # | Action | Column | Detail |\n"
            "|---|--------|--------|--------|\n"
            + "\n".join(rows)
            if rows
            else "_No transformations were applied._"
        )

        md = f"""# 📋 DataPilot AI — Cleaning Report

**Generated:** {report.generated_at}

---

## 📊 Dataset Overview

| Metric | Before | After |
|--------|--------|-------|
| Rows | {report.original_shape[0]:,} | {report.cleaned_shape[0]:,} |
| Columns | {report.original_shape[1]} | {report.cleaned_shape[1]} |
| Rows Removed | — | {report.rows_removed:,} |
| Columns Removed | — | {report.columns_removed} |

---

## 🤖 AI Narrative Summary

{report.narrative}

---

## 🔧 Transformation Log

{table}

---

## 🧠 LLM Cleaning Plan (Audit Trail)

```json
{llm_plan if llm_plan else "N/A — default cleaning was used."}
```

---
*Report generated by DataPilot AI*
"""
        return md
