"""
agents/inspector.py - Inspector Agent for DataPilot AI

Performs deep data-quality analysis on a pandas DataFrame:
  - Missing value detection
  - Duplicate row detection
  - Data type audit (actual vs suggested)
  - Basic numeric statistics
  - Outlier detection via IQR
  - Cardinality & sample values per column
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from utils import (
    infer_correct_dtypes,
    get_column_summary,
    numeric_stats,
    safe_percentage,
)


@dataclass
class InspectionReport:
    """Structured result returned by InspectorAgent.run()."""

    # Dataset-level
    total_rows: int = 0
    total_columns: int = 0
    duplicate_rows: int = 0
    total_missing_cells: int = 0
    missing_pct: float = 0.0

    # Column-level
    column_summary: list[dict] = field(default_factory=list)
    dtype_issues: list[dict] = field(default_factory=list)   # cols with type mismatch
    outlier_info: list[dict] = field(default_factory=list)   # numeric outlier counts
    high_missing_cols: list[str] = field(default_factory=list)  # cols with >50 % missing

    # Statistics
    numeric_stats: pd.DataFrame | None = None

    # Human-readable issue list
    issues: list[str] = field(default_factory=list)


class InspectorAgent:
    """
    Agent responsible for comprehensive dataset quality inspection.

    Usage::

        agent = InspectorAgent()
        report = agent.run(df)
        print(report.issues)
    """

    MISSING_THRESHOLD = 50.0   # columns above this % missing are flagged
    OUTLIER_IQR_FACTOR = 1.5   # standard IQR multiplier

    def run(self, df: pd.DataFrame) -> InspectionReport:
        """
        Analyse the DataFrame and return an InspectionReport.

        Args:
            df: Raw uploaded DataFrame

        Returns:
            InspectionReport with all findings populated
        """
        report = InspectionReport()

        report.total_rows = len(df)
        report.total_columns = len(df.columns)

        # ── Duplicate rows ─────────────────────────────────────────────────
        report.duplicate_rows = int(df.duplicated().sum())
        if report.duplicate_rows:
            report.issues.append(
                f"🔁 {report.duplicate_rows:,} duplicate row(s) detected."
            )

        # ── Missing values ─────────────────────────────────────────────────
        total_cells = report.total_rows * report.total_columns
        report.total_missing_cells = int(df.isna().sum().sum())
        report.missing_pct = safe_percentage(report.total_missing_cells, total_cells)
        if report.total_missing_cells:
            report.issues.append(
                f"❓ {report.total_missing_cells:,} missing cell(s) "
                f"({report.missing_pct}% of all data)."
            )

        # ── Column-level summary ───────────────────────────────────────────
        report.column_summary = get_column_summary(df)

        # Columns with high missingness
        for col_info in report.column_summary:
            if col_info["Missing %"] > self.MISSING_THRESHOLD:
                report.high_missing_cols.append(col_info["Column"])
                report.issues.append(
                    f"⚠️  Column '{col_info['Column']}' has "
                    f"{col_info['Missing %']}% missing values."
                )

        # ── Data-type audit ────────────────────────────────────────────────
        suggested = infer_correct_dtypes(df)
        for col, suggested_type in suggested.items():
            current_type = str(df[col].dtype)
            # Flag when a column stored as object could be numeric or datetime
            if suggested_type in ("integer", "float", "datetime") and current_type == "object":
                report.dtype_issues.append(
                    {
                        "Column": col,
                        "Current Dtype": current_type,
                        "Suggested Dtype": suggested_type,
                    }
                )
                report.issues.append(
                    f"🔧 Column '{col}' is stored as object but looks like "
                    f"{suggested_type}."
                )

        # ── Numeric statistics ─────────────────────────────────────────────
        report.numeric_stats = numeric_stats(df)

        # ── Outlier detection (IQR method) ─────────────────────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.OUTLIER_IQR_FACTOR * iqr
            upper = q3 + self.OUTLIER_IQR_FACTOR * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())
            if outlier_count:
                report.outlier_info.append(
                    {
                        "Column": col,
                        "Outliers": outlier_count,
                        "Lower Fence": round(lower, 4),
                        "Upper Fence": round(upper, 4),
                    }
                )
                report.issues.append(
                    f"📊 Column '{col}' has {outlier_count:,} outlier(s) "
                    f"(IQR method)."
                )

        if not report.issues:
            report.issues.append("✅ Dataset looks clean — no major issues found!")

        return report
