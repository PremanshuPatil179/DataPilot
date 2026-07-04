"""
database.py - SQLite persistence layer for DataPilot AI

Handles saving cleaned DataFrames into a local SQLite database
and querying stored tables.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime


# Default path for the SQLite database file
DB_PATH = Path(__file__).parent / "datapilot.db"


def get_connection(db_path: str = str(DB_PATH)) -> sqlite3.Connection:
    """
    Open and return a SQLite connection.

    Args:
        db_path: Path to the SQLite file (created if absent)

    Returns:
        sqlite3.Connection instance
    """
    return sqlite3.connect(db_path)


def save_dataframe(
    df: pd.DataFrame,
    table_name: str,
    db_path: str = str(DB_PATH),
    if_exists: str = "replace",
) -> str:
    """
    Save a DataFrame to an SQLite table.

    Args:
        df:          DataFrame to persist
        table_name:  Destination table name (sanitized internally)
        db_path:     Path to the SQLite database
        if_exists:   'replace' | 'append' | 'fail' (pandas convention)

    Returns:
        Confirmation message with row/column counts
    """
    # Sanitize table name: replace spaces and special chars with underscores
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)
    safe_name = safe_name.strip("_") or "cleaned_data"

    conn = get_connection(db_path)
    try:
        df.to_sql(safe_name, conn, if_exists=if_exists, index=False)
        conn.commit()
        return (
            f"✅ Saved {len(df):,} rows × {len(df.columns)} columns "
            f"into table **{safe_name}** in `{db_path}`."
        )
    except Exception as exc:
        return f"❌ Database save failed: {exc}"
    finally:
        conn.close()


def list_tables(db_path: str = str(DB_PATH)) -> list[str]:
    """
    List all user-created tables in the SQLite database.

    Args:
        db_path: Path to the SQLite database

    Returns:
        List of table name strings
    """
    if not Path(db_path).exists():
        return []
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def load_table(table_name: str, db_path: str = str(DB_PATH)) -> pd.DataFrame | None:
    """
    Load an SQLite table into a DataFrame.

    Args:
        table_name: Name of the table to load
        db_path:    Path to the SQLite database

    Returns:
        DataFrame or None if table doesn't exist
    """
    if not Path(db_path).exists():
        return None
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
    except Exception:
        return None
    finally:
        conn.close()


def get_table_info(table_name: str, db_path: str = str(DB_PATH)) -> dict:
    """
    Return metadata about a stored table.

    Args:
        table_name: Table to inspect
        db_path:    Path to the SQLite database

    Returns:
        Dict with row_count, column_count, columns list
    """
    df = load_table(table_name, db_path)
    if df is None:
        return {}
    return {
        "table_name": table_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }
