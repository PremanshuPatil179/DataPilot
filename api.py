"""
api.py - FastAPI backend for DataPilot.

Provides a production-friendly HTTP interface for the same cleaning workflow
used by the Streamlit application.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from dotenv import load_dotenv

from pipeline import inspect_and_clean_dataframe
from utils import dataframe_to_csv_bytes, load_tabular_file_from_bytes

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("datapilot.api")

app = FastAPI(
    title="DataPilot API",
    description="FastAPI backend for DataPilot data cleaning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


class APIError(Exception):
    """Domain-level error returned as JSON to API clients."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@app.exception_handler(APIError)
async def api_error_handler(_, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.message},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_, exc: Exception):
    logger.exception("Unhandled API error")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": "An unexpected error occurred while processing the dataset.",
        },
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "running", "service": "DataPilot API"}


@app.post("/clean")
async def clean(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if not filename:
        raise APIError("A filename is required.", status_code=400)
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        raise APIError(
            f"Unsupported file type '{suffix}'. Allowed types: CSV, XLS, XLSX.",
            status_code=400,
        )

    logger.info("Received clean request for %s", filename)

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise APIError("Uploaded file is empty.", status_code=400)

        dataframe, metadata = load_tabular_file_from_bytes(file_bytes, file_name=filename)
        logger.info(
            "Loaded %s as %s using %s",
            filename,
            metadata.get("source_format"),
            metadata.get("encoding") or metadata.get("detected_by") or "content detection",
        )

        pipeline_result = inspect_and_clean_dataframe(dataframe)
        cleaned_df = pipeline_result.cleaning_result.cleaned_df
        if cleaned_df is None:
            raise APIError("Cleaning did not produce an output dataset.", status_code=500)

        cleaned_csv = dataframe_to_csv_bytes(cleaned_df)
        base_name = Path(filename).stem or "cleaned_data"

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        try:
            tmp_file.write(cleaned_csv)
            tmp_file.flush()
            tmp_file.close()

            logger.info("Returning cleaned dataset for %s", filename)
            return FileResponse(
                path=tmp_file.name,
                media_type="text/csv",
                filename=f"{base_name}_cleaned.csv",
                background=BackgroundTask(os.unlink, tmp_file.name),
            )
        except Exception:
            tmp_file.close()
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass
            raise

    except APIError:
        raise
    except ValueError as exc:
        logger.warning("Validation failed for %s: %s", filename, exc)
        raise APIError(str(exc), status_code=400) from exc
    except RuntimeError as exc:
        logger.exception("Cleaning failed for %s", filename)
        raise APIError(f"Cleaning failed: {exc}", status_code=422) from exc
    except Exception as exc:
        logger.exception("Unexpected failure for %s", filename)
        raise APIError(f"Failed to clean file: {exc}", status_code=500) from exc


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
