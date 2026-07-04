"""
agents/cleaner.py - Cleaner Agent for DataPilot AI

Uses Google Gemini (via google-generativeai) to:
  1. Reason about the best cleaning strategy for this specific dataset.
  2. Return a structured JSON action plan.
  3. Execute each action using Pandas, logging every transformation.

Cleaning capabilities
---------------------
- Drop duplicate rows
- Fill or drop missing values (per-column strategy: mean / median / mode / drop)
- Fix data types (numeric coercion, datetime parsing)
- Strip leading/trailing whitespace from string columns
- Rename columns to snake_case
- Drop columns with excessive missingness
"""

import json
import re
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from google import genai
from dotenv import load_dotenv

from .gemini_utils import resolve_gemini_api_key, resolve_gemini_model

load_dotenv()


@dataclass
class CleaningResult:
    """Structured output from CleanerAgent.run()."""

    cleaned_df: pd.DataFrame | None = None
    transformations: list[dict] = field(default_factory=list)   # ordered log
    llm_plan: str = ""          # raw LLM response (for transparency)
    errors: list[str] = field(default_factory=list)
    success: bool = False


class CleanerAgent:
    """
    LLM-powered data cleaning agent.

    The agent asks Gemini to propose a JSON cleaning plan based on the
    dataset profile, then executes each step with Pandas.

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

    def run(self, df: pd.DataFrame, inspection_issues: list[str]) -> CleaningResult:
        """
        Clean *df* guided by LLM reasoning.

        Args:
            df:                Raw DataFrame
            inspection_issues: Issue strings from InspectorAgent

        Returns:
            CleaningResult with cleaned_df and transformation log
        """
        result = CleaningResult()

        # If no LLM available, skip straight to default cleaning
        if not self._llm_available:
            if self._llm_error:
                result.errors.append(f"Gemini setup failed: {self._llm_error}")
            result.cleaned_df, result.transformations = self._default_clean(df)
            result.llm_plan = ""
            result.success = True
            return result

        # Step 1: Build dataset profile
        profile = self._build_profile(df)

        # Step 2: Ask LLM for cleaning plan
        try:
            plan_json = self._ask_llm_for_plan(profile, inspection_issues)
            result.llm_plan = plan_json
        except Exception as exc:
            result.errors.append(f"LLM call failed: {exc}")
            # Fallback: apply sensible default cleaning
            result.cleaned_df, result.transformations = self._default_clean(df)
            result.success = True
            return result

        # Step 3: Parse JSON plan
        try:
            actions = self._parse_plan(plan_json)
        except Exception as exc:
            result.errors.append(f"Could not parse LLM plan: {exc}")
            result.cleaned_df, result.transformations = self._default_clean(df)
            result.success = True
            return result

        # Step 4: Execute actions
        cleaned_df, transformations, errors = self._execute_plan(df.copy(), actions)
        result.cleaned_df = cleaned_df
        result.transformations = transformations
        result.errors.extend(errors)
        result.success = True
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Profile builder
    # ──────────────────────────────────────────────────────────────────────

    def _build_profile(self, df: pd.DataFrame) -> dict:
        """Compact dataset profile sent to the LLM."""
        col_profiles = []
        for col in df.columns:
            series = df[col]
            missing = int(series.isna().sum())
            missing_pct = round(missing / len(df) * 100, 1) if len(df) else 0
            col_profiles.append({
                "name": col,
                "dtype": str(series.dtype),
                "missing": missing,
                "missing_pct": missing_pct,
                "unique": int(series.nunique()),
                "sample": [str(v) for v in series.dropna().head(3).tolist()],
            })

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "duplicates": int(df.duplicated().sum()),
            "column_profiles": col_profiles,
        }

    # ──────────────────────────────────────────────────────────────────────
    # LLM interaction
    # ──────────────────────────────────────────────────────────────────────

    def _ask_llm_for_plan(self, profile: dict, issues: list[str]) -> str:
        """Send profile to Gemini and retrieve a JSON action plan."""
        issues_text = "\n".join(f"- {i}" for i in issues) if issues else "None"
        prompt = f"""
You are a professional data cleaning expert. Analyse the dataset profile below and
produce a cleaning plan as a **JSON array** of action objects.

## Dataset Profile
{json.dumps(profile, indent=2)}

## Detected Issues
{issues_text}

## Output Format (strict JSON array, no markdown fences)
Return ONLY a valid JSON array where each element has these fields:
{{
  "action": "<action_type>",
  "column": "<column_name or null>",
  "params": {{ }}
}}

Supported action types:
- "drop_duplicates"      → params: {{}}
- "drop_column"          → column: str, params: {{}}
- "fill_missing"         → column: str, params: {{"strategy": "mean|median|mode|constant", "value": <only for constant>}}
- "drop_missing_rows"    → column: str, params: {{}}
- "convert_dtype"        → column: str, params: {{"to": "numeric|datetime|string|boolean"}}
- "strip_whitespace"     → column: str, params: {{}}
- "rename_to_snake_case" → column: str, params: {{}}

Rules:
- Only include actions that are truly necessary.
- For columns with >70% missing values, prefer "drop_column".
- For numeric columns with missing values, prefer mean/median fill.
- For categorical columns with missing values, prefer mode fill.
- Do NOT add markdown, comments, or explanation — only the JSON array.
"""
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

    # ──────────────────────────────────────────────────────────────────────
    # Plan parser
    # ──────────────────────────────────────────────────────────────────────

    def _parse_plan(self, raw: str) -> list[dict]:
        """
        Extract JSON array from LLM response even if wrapped in markdown.
        """
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        # Find first '[' and last ']'
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in LLM response.")
        return json.loads(cleaned[start : end + 1])

    # ──────────────────────────────────────────────────────────────────────
    # Plan executor
    # ──────────────────────────────────────────────────────────────────────

    def _execute_plan(
        self, df: pd.DataFrame, actions: list[dict]
    ) -> tuple[pd.DataFrame, list[dict], list[str]]:
        """
        Execute each action from the LLM plan on the DataFrame.

        Returns:
            (cleaned_df, transformation_log, error_list)
        """
        log: list[dict] = []
        errors: list[str] = []

        for action in actions:
            action_type = action.get("action", "")
            column = action.get("column")
            params = action.get("params", {})

            try:
                before_rows = len(df)

                if action_type == "drop_duplicates":
                    df = df.drop_duplicates()
                    dropped = before_rows - len(df)
                    log.append({
                        "action": "drop_duplicates",
                        "detail": f"Removed {dropped:,} duplicate row(s).",
                        "column": None,
                    })

                elif action_type == "drop_column":
                    if column and column in df.columns:
                        df = df.drop(columns=[column])
                        log.append({
                            "action": "drop_column",
                            "detail": f"Dropped column '{column}'.",
                            "column": column,
                        })

                elif action_type == "fill_missing":
                    if column and column in df.columns:
                        strategy = params.get("strategy", "mode")
                        df, detail = self._fill_missing(df, column, strategy, params)
                        log.append({
                            "action": "fill_missing",
                            "detail": detail,
                            "column": column,
                        })

                elif action_type == "drop_missing_rows":
                    if column and column in df.columns:
                        df = df.dropna(subset=[column])
                        dropped = before_rows - len(df)
                        log.append({
                            "action": "drop_missing_rows",
                            "detail": f"Dropped {dropped:,} rows with missing '{column}'.",
                            "column": column,
                        })

                elif action_type == "convert_dtype":
                    if column and column in df.columns:
                        to = params.get("to", "string")
                        df, detail = self._convert_dtype(df, column, to)
                        log.append({
                            "action": "convert_dtype",
                            "detail": detail,
                            "column": column,
                        })

                elif action_type == "strip_whitespace":
                    if column and column in df.columns:
                        if df[column].dtype == object:
                            df[column] = df[column].str.strip()
                            log.append({
                                "action": "strip_whitespace",
                                "detail": f"Stripped whitespace from '{column}'.",
                                "column": column,
                            })

                elif action_type == "rename_to_snake_case":
                    if column and column in df.columns:
                        new_name = re.sub(r"\W+", "_", column).strip("_").lower()
                        df = df.rename(columns={column: new_name})
                        log.append({
                            "action": "rename_to_snake_case",
                            "detail": f"Renamed '{column}' → '{new_name}'.",
                            "column": new_name,
                        })

            except Exception as exc:
                errors.append(f"Action '{action_type}' on '{column}' failed: {exc}")

        return df, log, errors

    # ──────────────────────────────────────────────────────────────────────
    # Helper methods
    # ──────────────────────────────────────────────────────────────────────

    def _fill_missing(
        self, df: pd.DataFrame, column: str, strategy: str, params: dict
    ) -> tuple[pd.DataFrame, str]:
        """Apply a fill strategy to a single column."""
        missing_before = df[column].isna().sum()

        if strategy == "mean":
            fill_val = df[column].mean()
        elif strategy == "median":
            fill_val = df[column].median()
        elif strategy == "mode":
            mode = df[column].mode()
            fill_val = mode.iloc[0] if not mode.empty else None
        elif strategy == "constant":
            fill_val = params.get("value", 0)
        else:
            fill_val = df[column].mode().iloc[0] if not df[column].mode().empty else None

        if fill_val is not None:
            df[column] = df[column].fillna(fill_val)

        return df, (
            f"Filled {missing_before:,} missing value(s) in '{column}' "
            f"using {strategy} ({fill_val!r})."
        )

    def _convert_dtype(
        self, df: pd.DataFrame, column: str, to: str
    ) -> tuple[pd.DataFrame, str]:
        """Convert a column's data type."""
        original = str(df[column].dtype)

        if to == "numeric":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif to == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce")
        elif to == "boolean":
            true_vals = {"true", "yes", "1"}
            df[column] = df[column].astype(str).str.lower().isin(true_vals)
        else:  # string
            df[column] = df[column].astype(str)

        return df, f"Converted '{column}' from {original} → {to}."

    # ──────────────────────────────────────────────────────────────────────
    # Default fallback cleaning (no LLM)
    # ──────────────────────────────────────────────────────────────────────

    def _default_clean(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[dict]]:
        """
        Apply sensible default cleaning without LLM assistance.
        Used when the API call fails.
        """
        log = []

        # Drop duplicates
        before = len(df)
        df = df.drop_duplicates()
        if len(df) < before:
            log.append({
                "action": "drop_duplicates",
                "detail": f"Removed {before - len(df):,} duplicate row(s).",
                "column": None,
            })

        # Per-column cleaning
        for col in df.columns:
            series = df[col]
            missing = series.isna().sum()

            # Drop columns with >70% missing
            if missing / len(df) > 0.7:
                df = df.drop(columns=[col])
                log.append({
                    "action": "drop_column",
                    "detail": f"Dropped '{col}' (>70% missing).",
                    "column": col,
                })
                continue

            if missing == 0:
                continue

            # Numeric: fill with median
            if pd.api.types.is_numeric_dtype(series):
                fill_val = series.median()
                df[col] = series.fillna(fill_val)
                log.append({
                    "action": "fill_missing",
                    "detail": f"Filled {missing:,} missing in '{col}' with median ({fill_val:.4g}).",
                    "column": col,
                })
            else:
                # Categorical: fill with mode
                mode = series.mode()
                if not mode.empty:
                    df[col] = series.fillna(mode.iloc[0])
                    log.append({
                        "action": "fill_missing",
                        "detail": f"Filled {missing:,} missing in '{col}' with mode ('{mode.iloc[0]}').",
                        "column": col,
                    })

        return df, log
