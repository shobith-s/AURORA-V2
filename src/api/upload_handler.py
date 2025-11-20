"""
File upload handler with robust CSV parsing.

Handles CSV file uploads with automatic format detection and error recovery.
"""

from fastapi import UploadFile, HTTPException
from typing import Dict, Any
import pandas as pd
import tempfile
from pathlib import Path
import logging

from ..core.robust_parser import parse_csv_robust, RobustCSVParser

logger = logging.getLogger(__name__)


async def process_uploaded_csv(file: UploadFile) -> Dict[str, Any]:
    """
    Process an uploaded CSV file with robust parsing.

    Args:
        file: Uploaded file from FastAPI

    Returns:
        Dictionary containing:
        - dataframe: Parsed DataFrame as dict
        - metadata: Parsing metadata (encoding, delimiter, warnings, etc.)

    Raises:
        HTTPException: If file cannot be parsed
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Expected .csv, got {file.filename}"
        )

    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            tmp_path = tmp_file.name

        # Parse with robust parser
        parser = RobustCSVParser()
        result = parser.parse_file(tmp_path, use_streaming=False)

        # Clean up temp file
        Path(tmp_path).unlink()

        # Convert DataFrame to dict for JSON response
        df_dict = {}
        for col in result.dataframe.columns:
            # Convert to list, handling NaN values
            df_dict[col] = result.dataframe[col].where(
                pd.notna(result.dataframe[col]), None
            ).tolist()

        return {
            'success': True,
            'data': df_dict,
            'metadata': {
                'filename': file.filename,
                'encoding': result.encoding,
                'delimiter': result.delimiter,
                'rows_parsed': result.rows_parsed,
                'rows_skipped': result.rows_skipped,
                'columns_detected': result.columns_detected,
                'parsing_strategy': result.parsing_strategy,
                'warnings': result.warnings
            }
        }

    except Exception as e:
        logger.error(f"Error parsing CSV file {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse CSV: {str(e)}"
        )


async def validate_csv_columns(df_dict: Dict[str, list], required_columns: list = None) -> bool:
    """
    Validate that CSV has required columns.

    Args:
        df_dict: Dictionary representation of DataFrame
        required_columns: List of required column names

    Returns:
        True if valid

    Raises:
        HTTPException: If validation fails
    """
    if required_columns:
        missing = set(required_columns) - set(df_dict.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing)}"
            )

    return True
