"""
pipeline.py - Shared DataPilot workflow helpers.

This module contains the reusable orchestration layer used by both the
Streamlit UI and the FastAPI backend so the business logic stays in one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from agents import CleanerAgent, InspectorAgent, ReporterAgent
from agents.inspector import InspectionReport
from agents.cleaner import CleaningResult
from agents.reporter import CleaningReport

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Combined output from the shared DataPilot workflow."""

    inspection_report: InspectionReport
    cleaning_result: CleaningResult
    cleaning_report: Optional[CleaningReport] = None


def run_inspection(df: pd.DataFrame) -> InspectionReport:
    """Run the inspection stage on a DataFrame."""
    logger.info("Starting inspection for dataset shape %s", df.shape)
    inspector = InspectorAgent()
    return inspector.run(df)


def run_cleaning(df: pd.DataFrame, inspection_issues: list[str], api_key: str | None = None) -> CleaningResult:
    """Run the cleaning stage using the existing CleanerAgent logic."""
    cleaner = CleanerAgent(api_key=api_key or None)
    result = cleaner.run(df, inspection_issues)
    logger.info(
        "Completed cleaning for dataset shape %s -> %s",
        df.shape,
        result.cleaned_df.shape if result.cleaned_df is not None else None,
    )
    return result


def inspect_and_clean_dataframe(df: pd.DataFrame, api_key: str | None = None) -> PipelineResult:
    """Run the inspection and cleaning stages on a DataFrame."""
    inspection_report = run_inspection(df)
    cleaning_result = run_cleaning(df, inspection_report.issues, api_key=api_key)

    return PipelineResult(inspection_report=inspection_report, cleaning_result=cleaning_result)


def generate_cleaning_report(
    original_df: pd.DataFrame,
    cleaning_result: CleaningResult,
    api_key: str | None = None,
) -> CleaningReport:
    """Build the markdown and narrative report for a completed cleaning run."""
    if cleaning_result.cleaned_df is None:
        raise ValueError("Cannot build a cleaning report without a cleaned DataFrame.")

    reporter = ReporterAgent(api_key=api_key or None)
    return reporter.run(
        original_df=original_df,
        cleaned_df=cleaning_result.cleaned_df,
        transformations=cleaning_result.transformations,
        llm_plan=cleaning_result.llm_plan,
    )


def run_full_pipeline(df: pd.DataFrame, api_key: str | None = None) -> PipelineResult:
    """Run inspection, cleaning, and reporting in one shared workflow."""
    result = inspect_and_clean_dataframe(df, api_key=api_key)
    if result.cleaning_result.cleaned_df is None:
        raise RuntimeError("Cleaning completed without producing a DataFrame.")

    result.cleaning_report = generate_cleaning_report(
        original_df=df,
        cleaning_result=result.cleaning_result,
        api_key=api_key,
    )
    return result
