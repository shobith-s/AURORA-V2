"""
Robust CSV Parser - Handles ANY CSV file format.

This module implements comprehensive CSV parsing that can handle:
- Multiple encodings (UTF-8, Latin-1, Windows-1252, etc.)
- Different delimiters (comma, tab, semicolon, pipe)
- Quoted fields with commas/newlines
- Malformed rows
- Large files (streaming support)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import chardet
import warnings
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of CSV parsing with metadata."""
    dataframe: pd.DataFrame
    encoding: str
    delimiter: str
    rows_parsed: int
    rows_skipped: int
    columns_detected: int
    parsing_strategy: str
    warnings: List[str]


class RobustCSVParser:
    """
    Production-grade CSV parser that handles edge cases.

    Features:
    - Auto-detect encoding
    - Try multiple delimiter strategies
    - Handle malformed rows gracefully
    - Stream large files
    - Comprehensive error reporting
    """

    # Security limits to prevent DOS attacks
    MAX_FILE_SIZE_BYTES = 100_000_000  # 100MB
    MAX_ROWS = 1_000_000              # 1 million rows
    MAX_COLUMNS = 1_000               # 1000 columns

    def __init__(
        self,
        max_sample_size: int = 100000,  # Bytes to sample for encoding
        chunk_size: int = 10000,         # Rows per chunk for large files
    ):
        self.max_sample_size = max_sample_size
        self.chunk_size = chunk_size

    def parse_file(
        self,
        file_path: str,
        use_streaming: bool = False
    ) -> ParseResult:
        """
        Parse CSV file with automatic format detection.

        Args:
            file_path: Path to CSV file
            use_streaming: If True, process in chunks (for large files)

        Returns:
            ParseResult with dataframe and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # SECURITY: Check file size limit to prevent DOS attacks
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File too large: {file_size:,} bytes. "
                f"Maximum allowed: {self.MAX_FILE_SIZE_BYTES:,} bytes ({self.MAX_FILE_SIZE_BYTES / 1_000_000:.0f}MB)"
            )

        # Step 1: Detect encoding
        encoding = self._detect_encoding(file_path)
        logger.info(f"Detected encoding: {encoding}")

        # Step 2: Detect delimiter
        delimiter = self._detect_delimiter(file_path, encoding)
        logger.info(f"Detected delimiter: {repr(delimiter)}")

        # Step 3: Parse with detected format
        try:
            if use_streaming:
                df = self._parse_streaming(file_path, encoding, delimiter)
            else:
                df = self._parse_standard(file_path, encoding, delimiter)

            # Step 4: Clean and validate
            df, warnings_list = self._clean_dataframe(df)

            # SECURITY: Validate row and column counts
            if len(df) > self.MAX_ROWS:
                raise ValueError(
                    f"Too many rows: {len(df):,} rows. "
                    f"Maximum allowed: {self.MAX_ROWS:,} rows"
                )

            if len(df.columns) > self.MAX_COLUMNS:
                raise ValueError(
                    f"Too many columns: {len(df.columns):,} columns. "
                    f"Maximum allowed: {self.MAX_COLUMNS:,} columns"
                )

            return ParseResult(
                dataframe=df,
                encoding=encoding,
                delimiter=delimiter,
                rows_parsed=len(df),
                rows_skipped=0,  # Would track bad rows
                columns_detected=len(df.columns),
                parsing_strategy="streaming" if use_streaming else "standard",
                warnings=warnings_list
            )

        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")
            raise ValueError(f"Could not parse CSV file: {e}")

    def _detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding using chardet.

        Samples the first portion of the file to detect encoding.
        """
        with open(file_path, 'rb') as f:
            raw_data = f.read(self.max_sample_size)

        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']

        logger.info(f"Encoding confidence: {confidence:.2f}")

        # Fallback to UTF-8 if confidence is too low
        if confidence < 0.7:
            logger.warning(f"Low encoding confidence ({confidence:.2f}), defaulting to UTF-8")
            encoding = 'utf-8'

        return encoding

    def _detect_delimiter(self, file_path: Path, encoding: str) -> str:
        """
        Detect CSV delimiter by analyzing first few rows.

        Tries common delimiters and picks the one that produces
        the most consistent column count.
        """
        delimiters = [',', '\t', ';', '|', ' ']

        best_delimiter = ','
        best_score = 0

        # Read first 10 lines
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            sample_lines = [f.readline() for _ in range(10)]

        for delimiter in delimiters:
            # Count columns in each line
            column_counts = [
                len(line.split(delimiter))
                for line in sample_lines if line.strip()
            ]

            if not column_counts:
                continue

            # Good delimiter has:
            # 1. Consistent column count across rows
            # 2. More than 1 column
            # 3. Not too many columns (likely wrong delimiter)

            avg_columns = np.mean(column_counts)
            std_columns = np.std(column_counts)

            # Score: high avg, low std, reasonable range
            if avg_columns > 1 and avg_columns < 100:
                score = avg_columns * (1 - std_columns / (avg_columns + 1))

                if score > best_score:
                    best_score = score
                    best_delimiter = delimiter

        return best_delimiter

    def _parse_standard(
        self,
        file_path: Path,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        """Parse entire file at once (for smaller files)."""

        return pd.read_csv(
            file_path,
            sep=delimiter,
            encoding=encoding,
            encoding_errors='replace',  # Replace bad chars with �
            on_bad_lines='warn',         # Log bad lines but continue
            engine='python',             # More forgiving parser
            low_memory=False,            # Don't guess dtypes
            skipinitialspace=True,       # Strip leading whitespace
        )

    def _parse_streaming(
        self,
        file_path: Path,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        """Parse file in chunks (for large files)."""

        chunks = []

        with pd.read_csv(
            file_path,
            sep=delimiter,
            encoding=encoding,
            encoding_errors='replace',
            on_bad_lines='warn',
            engine='python',
            chunksize=self.chunk_size,
            low_memory=False,
        ) as reader:
            for chunk in reader:
                chunks.append(chunk)

        # Concatenate all chunks
        return pd.concat(chunks, ignore_index=True)

    def _clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Clean parsed dataframe and return warnings.

        Handles common issues:
        - Unnamed columns (pandas artifacts)
        - Duplicate column names
        - Empty columns
        - Whitespace in column names
        """
        warnings_list = []

        # Issue 1: Remove unnamed columns (pandas adds these for malformed CSVs)
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            warnings_list.append(f"Removed {len(unnamed_cols)} unnamed columns")
            df = df.drop(columns=unnamed_cols)

        # Issue 2: Remove completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            warnings_list.append(f"Removed {len(empty_cols)} empty columns")
            df = df.drop(columns=empty_cols)

        # Issue 3: Handle duplicate column names
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            warnings_list.append(f"Found {len(duplicate_cols)} duplicate column names, renaming")
            # Add suffix to duplicates
            df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(
                df.columns
            )

        # Issue 4: Clean column names
        original_cols = df.columns.tolist()
        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
        df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscore

        renamed = sum(1 for a, b in zip(original_cols, df.columns) if a != b)
        if renamed > 0:
            warnings_list.append(f"Cleaned {renamed} column names (removed whitespace)")

        # Issue 5: Handle mixed-type columns
        for col in df.columns:
            type_counts = df[col].apply(type).value_counts()
            if len(type_counts) > 1:
                warnings_list.append(f"Column '{col}' has mixed types, converting to string")
                df[col] = df[col].astype(str)

        return df, warnings_list


def parse_csv_robust(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Convenience function for robust CSV parsing.

    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments passed to RobustCSVParser

    Returns:
        Parsed and cleaned DataFrame

    Example:
        >>> df = parse_csv_robust('messy_data.csv')
        >>> print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    """
    parser = RobustCSVParser(**kwargs)
    result = parser.parse_file(file_path)

    # Log warnings
    for warning in result.warnings:
        logger.warning(warning)

    logger.info(
        f"Successfully parsed: {result.rows_parsed} rows, "
        f"{result.columns_detected} columns "
        f"(encoding: {result.encoding}, delimiter: {repr(result.delimiter)})"
    )

    return result.dataframe


# Export main function
__all__ = ['parse_csv_robust', 'RobustCSVParser', 'ParseResult']
