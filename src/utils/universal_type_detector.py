"""
Universal Type Detector - Comprehensive semantic type detection for any CSV column.

Detects 10 semantic types using priority-based detection:
1. empty - All or mostly null values
2. url - HTTP/HTTPS URLs
3. phone - Phone numbers in various formats
4. email - Email addresses
5. identifier - Unique IDs, primary keys, hashes
6. datetime - Date and time values
7. boolean - True/False, Yes/No, 0/1
8. numeric - Numbers (integers, floats)
9. categorical - Low-cardinality text categories
10. text - Free-form text (names, descriptions)

Priority order ensures correct detection even when types overlap.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from enum import Enum


class SemanticType(Enum):
    """Semantic type classifications."""
    EMPTY = "empty"
    URL = "url"
    PHONE = "phone"
    EMAIL = "email"
    IDENTIFIER = "identifier"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"


@dataclass
class TypeDetectionResult:
    """Result of semantic type detection."""
    semantic_type: SemanticType
    confidence: float
    explanation: str
    detection_details: Dict[str, Any]


class UniversalTypeDetector:
    """
    Comprehensive semantic type detector for any CSV column.
    
    Uses priority-based detection to correctly classify columns:
    - Higher priority types are checked first
    - Each detection method returns (is_match, confidence, details)
    - First high-confidence match wins
    """
    
    # Detection thresholds
    EMPTY_THRESHOLD = 0.9  # 90% null = empty column
    HIGH_MATCH_THRESHOLD = 0.8  # 80% pattern match = confident detection
    MEDIUM_MATCH_THRESHOLD = 0.5  # 50% pattern match = possible detection
    ID_UNIQUE_THRESHOLD = 0.95  # 95% unique = likely identifier
    CATEGORICAL_CARDINALITY_MAX = 50  # Max unique values for categorical
    CATEGORICAL_RATIO_MAX = 0.5  # Max unique ratio for categorical
    
    # Pattern definitions
    URL_PATTERN = re.compile(r'^https?://[^\s<>"{}|\\^`\[\]]+$', re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    PHONE_PATTERNS = [
        re.compile(r'^\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}$'),  # International
        re.compile(r'^\(\d{3}\)\s*\d{3}[-.\s]?\d{4}$'),  # US (xxx) xxx-xxxx
        re.compile(r'^\d{10,15}$'),  # Plain digits
        re.compile(r'^\d{3}[-.\s]\d{3}[-.\s]\d{4}$'),  # xxx-xxx-xxxx
    ]
    ISO_DATETIME_PATTERN = re.compile(
        r'^\d{4}[-/]\d{2}[-/]\d{2}([T\s]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$'
    )
    DATE_PATTERNS = [
        re.compile(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$'),  # MM/DD/YY or DD/MM/YYYY
        re.compile(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$'),  # YYYY/MM/DD
        re.compile(r'^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$'),  # Month DD, YYYY
        re.compile(r'^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$'),  # DD Month YYYY
    ]
    
    # Identifier detection patterns
    ID_KEYWORDS = ['id', 'uuid', 'guid', 'hash', 'key', 'code', 'ref', 'num', 'number']
    UUID_PATTERN = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', re.IGNORECASE)
    
    def __init__(self):
        """Initialize the type detector."""
        pass
    
    def detect_semantic_type(
        self,
        column: pd.Series,
        column_name: str = ""
    ) -> TypeDetectionResult:
        """
        Detect the semantic type of a column using priority-based detection.
        
        Args:
            column: The pandas Series to analyze
            column_name: Name of the column (helps with identifier detection)
            
        Returns:
            TypeDetectionResult with semantic type, confidence, and details
        """
        # Priority-ordered detection methods
        detection_methods = [
            (self._detect_empty, SemanticType.EMPTY),
            (self._detect_url, SemanticType.URL),
            (self._detect_phone, SemanticType.PHONE),
            (self._detect_email, SemanticType.EMAIL),
            (self._detect_identifier, SemanticType.IDENTIFIER),
            (self._detect_datetime, SemanticType.DATETIME),
            (self._detect_boolean, SemanticType.BOOLEAN),
            (self._detect_numeric, SemanticType.NUMERIC),
            (self._detect_categorical, SemanticType.CATEGORICAL),
            (self._detect_text, SemanticType.TEXT),
        ]
        
        # Try each detection method in priority order
        for detect_fn, semantic_type in detection_methods:
            is_match, confidence, details = detect_fn(column, column_name)
            
            if is_match and confidence >= self.MEDIUM_MATCH_THRESHOLD:
                return TypeDetectionResult(
                    semantic_type=semantic_type,
                    confidence=confidence,
                    explanation=details.get('explanation', f'Detected as {semantic_type.value}'),
                    detection_details=details
                )
        
        # Fallback to text if nothing matches
        return TypeDetectionResult(
            semantic_type=SemanticType.TEXT,
            confidence=0.5,
            explanation="No specific pattern detected, treating as free-form text",
            detection_details={'fallback': True}
        )
    
    def _detect_empty(
        self,
        column: pd.Series,
        column_name: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect if column is empty (all/mostly null)."""
        if len(column) == 0:
            return True, 1.0, {'explanation': 'Empty column (no rows)'}
        
        null_ratio = column.isnull().sum() / len(column)
        
        if null_ratio >= self.EMPTY_THRESHOLD:
            return True, min(1.0, null_ratio + 0.05), {
                'explanation': f'Column is {null_ratio:.1%} null values',
                'null_ratio': null_ratio
            }
        
        return False, null_ratio, {'null_ratio': null_ratio}
    
    def _detect_url(
        self,
        column: pd.Series,
        column_name: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect URL columns."""
        non_null = column.dropna().astype(str)
        if len(non_null) == 0:
            return False, 0.0, {}
        
        # Check for URL pattern
        matches = non_null.apply(lambda x: bool(self.URL_PATTERN.match(str(x))))
        match_ratio = matches.sum() / len(non_null)
        
        # Also check column name hints
        name_hint = any(kw in column_name.lower() for kw in ['url', 'link', 'website', 'href', 'photo', 'image', 'img', 'src'])
        
        if match_ratio >= self.HIGH_MATCH_THRESHOLD or (match_ratio >= 0.5 and name_hint):
            confidence = match_ratio * 0.9 + (0.1 if name_hint else 0)
            return True, min(1.0, confidence), {
                'explanation': f'{match_ratio:.1%} values match URL pattern',
                'match_ratio': match_ratio,
                'name_hint': name_hint
            }
        
        return False, match_ratio, {'match_ratio': match_ratio}
    
    def _detect_phone(
        self,
        column: pd.Series,
        column_name: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect phone number columns."""
        non_null = column.dropna().astype(str)
        if len(non_null) == 0:
            return False, 0.0, {}
        
        # Check multiple phone patterns
        def is_phone(val):
            val_str = str(val).strip()
            return any(pattern.match(val_str) for pattern in self.PHONE_PATTERNS)
        
        matches = non_null.apply(is_phone)
        match_ratio = matches.sum() / len(non_null)
        
        # Check column name hints
        name_hint = any(kw in column_name.lower() for kw in ['phone', 'mobile', 'cell', 'tel', 'contact', 'fax'])
        
        if match_ratio >= self.HIGH_MATCH_THRESHOLD or (match_ratio >= 0.5 and name_hint):
            confidence = match_ratio * 0.85 + (0.15 if name_hint else 0)
            return True, min(1.0, confidence), {
                'explanation': f'{match_ratio:.1%} values match phone patterns',
                'match_ratio': match_ratio,
                'name_hint': name_hint
            }
        
        return False, match_ratio, {'match_ratio': match_ratio}
    
    def _detect_email(
        self,
        column: pd.Series,
        column_name: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect email columns."""
        non_null = column.dropna().astype(str)
        if len(non_null) == 0:
            return False, 0.0, {}
        
        matches = non_null.apply(lambda x: bool(self.EMAIL_PATTERN.match(str(x))))
        match_ratio = matches.sum() / len(non_null)
        
        # Check column name hints
        name_hint = any(kw in column_name.lower() for kw in ['email', 'mail', 'e-mail', 'e_mail'])
        
        if match_ratio >= self.HIGH_MATCH_THRESHOLD or (match_ratio >= 0.5 and name_hint):
            confidence = match_ratio * 0.9 + (0.1 if name_hint else 0)
            return True, min(1.0, confidence), {
                'explanation': f'{match_ratio:.1%} values match email pattern',
                'match_ratio': match_ratio,
                'name_hint': name_hint
            }
        
        return False, match_ratio, {'match_ratio': match_ratio}
    
    def _detect_identifier(
        self,
        column: pd.Series,
        column_name: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect identifier/ID columns (should be dropped or handled specially)."""
        non_null = column.dropna()
        if len(non_null) == 0:
            return False, 0.0, {}
        
        # Check uniqueness - IDs are typically unique
        unique_ratio = non_null.nunique() / len(non_null)
        
        # Check column name hints
        name_lower = column_name.lower()
        name_hint = any(kw in name_lower for kw in self.ID_KEYWORDS)
        
        # Check for UUID pattern
        non_null_str = non_null.astype(str)
        uuid_matches = non_null_str.apply(lambda x: bool(self.UUID_PATTERN.match(str(x))))
        uuid_ratio = uuid_matches.sum() / len(non_null_str) if len(non_null_str) > 0 else 0
        
        # Check for hash-like strings (alphanumeric, consistent length)
        if len(non_null_str) > 0:
            lengths = non_null_str.str.len()
            consistent_length = lengths.std() < 1 if len(lengths) > 1 else True
            avg_length = lengths.mean()
            is_hash_like = consistent_length and avg_length > 10 and non_null_str.str.match(r'^[a-zA-Z0-9]+$').mean() > 0.9
        else:
            is_hash_like = False
        
        # Determine if this is an identifier
        is_id = False
        confidence = 0.0
        reasons = []
        
        if uuid_ratio >= 0.8:
            is_id = True
            confidence = 0.95
            reasons.append('UUID pattern detected')
        elif unique_ratio >= self.ID_UNIQUE_THRESHOLD and name_hint:
            is_id = True
            confidence = 0.9
            reasons.append(f'{unique_ratio:.1%} unique values with ID-like name')
        elif unique_ratio >= self.ID_UNIQUE_THRESHOLD and is_hash_like:
            is_id = True
            confidence = 0.85
            reasons.append('Hash-like format with high uniqueness')
        elif name_hint and unique_ratio >= 0.8:
            is_id = True
            confidence = 0.75
            reasons.append(f'ID-like name with {unique_ratio:.1%} unique values')
        
        if is_id:
            return True, confidence, {
                'explanation': '; '.join(reasons),
                'unique_ratio': unique_ratio,
                'name_hint': name_hint,
                'uuid_ratio': uuid_ratio,
                'is_hash_like': is_hash_like
            }
        
        return False, unique_ratio * 0.5, {'unique_ratio': unique_ratio}
    
    def _detect_datetime(
        self,
        column: pd.Series,
        column_name: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect datetime columns."""
        # Check if already datetime type
        if pd.api.types.is_datetime64_any_dtype(column):
            return True, 0.98, {
                'explanation': 'Already datetime dtype',
                'is_native_datetime': True
            }
        
        non_null = column.dropna().astype(str)
        if len(non_null) == 0:
            return False, 0.0, {}
        
        # Check ISO datetime pattern
        iso_matches = non_null.apply(lambda x: bool(self.ISO_DATETIME_PATTERN.match(str(x))))
        iso_ratio = iso_matches.sum() / len(non_null)
        
        # Check other date patterns
        def matches_any_date_pattern(val):
            val_str = str(val)
            return any(pattern.match(val_str) for pattern in self.DATE_PATTERNS)
        
        date_matches = non_null.apply(matches_any_date_pattern)
        date_ratio = date_matches.sum() / len(non_null)
        
        # Column name hints
        name_hint = any(kw in column_name.lower() for kw in ['date', 'time', 'datetime', 'timestamp', 'created', 'updated', 'at', 'on'])
        
        # Determine match
        match_ratio = max(iso_ratio, date_ratio)
        
        if match_ratio >= self.HIGH_MATCH_THRESHOLD or (match_ratio >= 0.5 and name_hint):
            confidence = match_ratio * 0.9 + (0.1 if name_hint else 0)
            return True, min(1.0, confidence), {
                'explanation': f'{match_ratio:.1%} values match datetime patterns',
                'iso_ratio': iso_ratio,
                'date_ratio': date_ratio,
                'name_hint': name_hint
            }
        
        return False, match_ratio, {'iso_ratio': iso_ratio, 'date_ratio': date_ratio}
    
    def _detect_boolean(
        self,
        column: pd.Series,
        column_name: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect boolean columns."""
        non_null = column.dropna()
        if len(non_null) == 0:
            return False, 0.0, {}
        
        # Check cardinality first - boolean should have at most 2-3 unique values
        unique_count = non_null.nunique()
        if unique_count > 3:
            return False, 0.0, {'unique_count': unique_count}
        
        # Convert to lowercase strings for comparison
        non_null_str = non_null.astype(str).str.lower().str.strip()
        unique_values = set(non_null_str.unique())
        
        # Boolean patterns
        boolean_sets = [
            {'true', 'false'},
            {'t', 'f'},
            {'yes', 'no'},
            {'y', 'n'},
            {'1', '0'},
            {'1.0', '0.0'},
        ]
        
        for bool_set in boolean_sets:
            if unique_values.issubset(bool_set) or unique_values == bool_set:
                return True, 0.95, {
                    'explanation': f'Values match boolean pattern: {unique_values}',
                    'unique_values': list(unique_values),
                    'boolean_set': list(bool_set)
                }
        
        # Check for boolean with null
        if len(unique_values) <= 2:
            for bool_set in boolean_sets:
                if unique_values.issubset(bool_set):
                    return True, 0.9, {
                        'explanation': f'Values match boolean subset: {unique_values}',
                        'unique_values': list(unique_values)
                    }
        
        return False, 0.0, {'unique_count': unique_count, 'unique_values': list(unique_values)}
    
    def _detect_numeric(
        self,
        column: pd.Series,
        column_name: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect numeric columns."""
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(column):
            return True, 0.95, {
                'explanation': 'Native numeric dtype',
                'dtype': str(column.dtype)
            }
        
        # Try to convert to numeric
        non_null = column.dropna()
        if len(non_null) == 0:
            return False, 0.0, {}
        
        # Count how many can be converted to numeric
        numeric_converted = pd.to_numeric(non_null, errors='coerce')
        success_ratio = numeric_converted.notna().sum() / len(non_null)
        
        if success_ratio >= self.HIGH_MATCH_THRESHOLD:
            return True, success_ratio, {
                'explanation': f'{success_ratio:.1%} values are numeric',
                'success_ratio': success_ratio,
                'dtype': str(column.dtype)
            }
        
        return False, success_ratio, {'success_ratio': success_ratio}
    
    def _detect_categorical(
        self,
        column: pd.Series,
        column_name: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect categorical columns (low-cardinality text)."""
        non_null = column.dropna()
        if len(non_null) == 0:
            return False, 0.0, {}
        
        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null)
        
        # Categorical has low cardinality and moderate-to-low unique ratio
        if unique_count <= self.CATEGORICAL_CARDINALITY_MAX and unique_ratio <= self.CATEGORICAL_RATIO_MAX:
            confidence = 0.85 - (unique_ratio * 0.3)  # Lower unique ratio = higher confidence
            return True, max(0.6, confidence), {
                'explanation': f'Low cardinality ({unique_count} unique values)',
                'unique_count': unique_count,
                'unique_ratio': unique_ratio
            }
        
        return False, 0.0, {'unique_count': unique_count, 'unique_ratio': unique_ratio}
    
    def _detect_text(
        self,
        column: pd.Series,
        column_name: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect free-form text columns (names, descriptions, etc.)."""
        non_null = column.dropna().astype(str)
        if len(non_null) == 0:
            return False, 0.0, {}
        
        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null)
        avg_length = non_null.str.len().mean()
        
        # Text typically has:
        # - Higher cardinality than categorical
        # - Contains spaces (multiple words)
        # - Variable lengths
        has_spaces = non_null.str.contains(r'\s').mean()
        
        # High cardinality with spaces = likely text
        if unique_ratio > self.CATEGORICAL_RATIO_MAX or unique_count > self.CATEGORICAL_CARDINALITY_MAX:
            if has_spaces > 0.5 or avg_length > 10:
                return True, 0.8, {
                    'explanation': f'High cardinality text ({unique_count} unique values)',
                    'unique_count': unique_count,
                    'unique_ratio': unique_ratio,
                    'avg_length': avg_length,
                    'has_spaces_ratio': has_spaces
                }
        
        # Name-like columns
        name_hint = any(kw in column_name.lower() for kw in ['name', 'title', 'description', 'comment', 'note', 'text', 'bio', 'about'])
        if name_hint and avg_length > 3:
            return True, 0.75, {
                'explanation': f'Name-like column with text content',
                'unique_count': unique_count,
                'avg_length': avg_length,
                'name_hint': name_hint
            }
        
        return True, 0.5, {
            'explanation': 'Generic text column',
            'unique_count': unique_count,
            'avg_length': avg_length
        }


def get_type_detector() -> UniversalTypeDetector:
    """Get a singleton instance of the type detector."""
    return UniversalTypeDetector()
