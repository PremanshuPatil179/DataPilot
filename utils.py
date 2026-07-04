"""
utils.py - Utility functions for DataPilot AI

Provides helper functions for data type detection, formatting,
and other shared logic across agents.
"""

import pandas as pd
import numpy as np
import io
import logging
import zipfile
from pathlib import Path
from typing import Any

from charset_normalizer import from_bytes


logger = logging.getLogger(__name__)

CSV_ENCODING_FALLBACKS = ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"]


def _read_upload_bytes(uploaded_file: Any) -> bytes:
    """Read all bytes from a Streamlit UploadedFile or file-like object."""
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()

    if hasattr(uploaded_file, "read"):
        current_pos = None
        if hasattr(uploaded_file, "tell") and hasattr(uploaded_file, "seek"):
            try:
                current_pos = uploaded_file.tell()
                uploaded_file.seek(0)
            except Exception:
                current_pos = None

        data = uploaded_file.read()

        if current_pos is not None:
            try:
                uploaded_file.seek(current_pos)
            except Exception:
                pass

        return data

    raise TypeError("Unsupported uploaded file object.")


def _detect_excel_format(data: bytes) -> str | None:
    """Detect Excel workbooks by signature, even if the extension is wrong."""
    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
                if "[Content_Types].xml" in names and any(
                    name.startswith("xl/") for name in names
                ):
                    return "xlsx"
        except zipfile.BadZipFile:
            pass

    if data.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        return "xls"

    return None


def _detect_text_encoding(data: bytes) -> tuple[str | None, str | None]:
    """Detect a likely text encoding from BOM or charset heuristics."""
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", "utf-8 BOM"
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return "utf-16", "utf-16 BOM"

    try:
        best = from_bytes(data).best()
        if best and best.encoding:
            return best.encoding, "charset_normalizer"
    except Exception:
        pass

    return None, None


def load_uploaded_tabular_file(uploaded_file: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load an uploaded CSV or Excel file robustly.

    Returns:
        (dataframe, metadata) where metadata includes the detected format,
        encoding, and any fallback attempts.
    """
    file_bytes = _read_upload_bytes(uploaded_file)
    file_name = getattr(uploaded_file, "name", "") or getattr(uploaded_file, "filename", "") or ""
    return load_tabular_file_from_bytes(file_bytes, file_name=file_name)


def load_tabular_file_from_bytes(file_bytes: bytes, file_name: str = "") -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a CSV or Excel file from raw bytes.

    This shared parser keeps Streamlit and FastAPI in sync.
    """
    metadata: dict[str, Any] = {
        "file_name": file_name,
        "source_format": "csv",
        "encoding": None,
        "detected_by": None,
        "attempts": [],
        "warnings": [],
    }

    excel_format = _detect_excel_format(file_bytes)
    if excel_format:
        metadata["source_format"] = excel_format
        metadata["detected_by"] = "file signature"
        logger.info("Loading %s as Excel based on file signature: %s", file_name or "upload", excel_format)

        excel_buffer = io.BytesIO(file_bytes)
        try:
            if excel_format == "xlsx":
                dataframe = pd.read_excel(excel_buffer, engine="openpyxl")
            else:
                dataframe = pd.read_excel(excel_buffer)
            logger.info("Loaded Excel file %s using read_excel", file_name or "upload")
            return dataframe, metadata
        except Exception as exc:
            raise RuntimeError(
                f"Detected an Excel workbook, but pandas could not read it: {exc}"
            ) from exc

    detected_encoding, detected_by = _detect_text_encoding(file_bytes)
    candidate_encodings: list[str] = []
    if detected_encoding:
        candidate_encodings.append(detected_encoding)
        metadata["detected_by"] = detected_by

    for encoding in CSV_ENCODING_FALLBACKS:
        if encoding not in candidate_encodings:
            candidate_encodings.append(encoding)

    last_error: Exception | None = None
    for encoding in candidate_encodings:
        try:
            text = file_bytes.decode(encoding)
            dataframe = pd.read_csv(io.StringIO(text))
            metadata["encoding"] = encoding
            metadata["attempts"].append(encoding)
            logger.info(
                "Loaded CSV file %s using encoding %s%s",
                file_name or "upload",
                encoding,
                f" (detected by {detected_by})" if encoding == detected_encoding and detected_by else "",
            )
            return dataframe, metadata
        except UnicodeDecodeError as exc:
            metadata["attempts"].append(f"{encoding}: decode error")
            last_error = exc
            continue
        except pd.errors.ParserError as exc:
            metadata["attempts"].append(f"{encoding}: parser error")
            last_error = exc
            continue
        except Exception as exc:
            metadata["attempts"].append(f"{encoding}: {type(exc).__name__}")
            last_error = exc
            continue

    raise RuntimeError(
        f"Could not read '{file_name or 'uploaded file'}' as CSV or Excel. Tried encodings: {candidate_encodings}. Last error: {last_error}"
    )


def infer_correct_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """
    Infer the correct/expected data types for each column.

    Args:
        df: Input DataFrame

    Returns:
        dict mapping column name -> suggested dtype label
    """
    suggestions = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            suggestions[col] = "unknown (all null)"
            continue

        # Try numeric
        try:
            numeric = pd.to_numeric(series, errors="raise")
            # Drop NaN before checking int-compatibility
            numeric_clean = numeric.dropna()
            try:
                if numeric_clean.empty or (numeric_clean == numeric_clean.astype("int64")).all():
                    suggestions[col] = "integer"
                else:
                    suggestions[col] = "float"
            except (ValueError, TypeError, OverflowError):
                suggestions[col] = "float"
            continue
        except (ValueError, TypeError):
            pass

        # Try datetime (infer_datetime_format removed in pandas 2.2)
        try:
            pd.to_datetime(series, errors="raise")
            suggestions[col] = "datetime"
            continue
        except (ValueError, TypeError, Exception):
            pass

        # Try boolean
        bool_vals = {"true", "false", "yes", "no", "1", "0"}
        if set(series.astype(str).str.lower().unique()).issubset(bool_vals):
            suggestions[col] = "boolean"
            continue

        suggestions[col] = "string"

    return suggestions


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV bytes for download.

    Args:
        df: DataFrame to convert

    Returns:
        UTF-8 encoded CSV bytes
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def format_number(n: Any) -> str:
    """Format a number with commas for display."""
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return str(n)


def truncate_string(s: str, max_len: int = 80) -> str:
    """Truncate a string and append ellipsis if too long."""
    return s if len(s) <= max_len else s[:max_len] + "..."


def safe_percentage(part: int, total: int) -> float:
    """Safely compute percentage, returning 0.0 if total is 0."""
    return round((part / total) * 100, 2) if total > 0 else 0.0


def get_column_summary(df: pd.DataFrame) -> list[dict]:
    """
    Build a per-column summary list for display.

    Args:
        df: Input DataFrame

    Returns:
        List of dicts with column metadata
    """
    summary = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        unique = int(series.nunique())
        dtype = str(series.dtype)
        sample = str(series.dropna().iloc[0]) if not series.dropna().empty else "N/A"

        summary.append({
            "Column": col,
            "Dtype": dtype,
            "Missing": missing,
            "Missing %": safe_percentage(missing, len(df)),
            "Unique": unique,
            "Sample Value": truncate_string(sample, 40),
        })
    return summary


def numeric_stats(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Return descriptive statistics for numeric columns.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame of statistics or None if no numeric columns
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None
    return numeric_df.describe().round(4)
