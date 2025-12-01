"""
Tests for UniversalTypeDetector - Comprehensive semantic type detection.

Tests all 10 semantic types:
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
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.universal_type_detector import (
    UniversalTypeDetector,
    SemanticType,
    TypeDetectionResult,
    get_type_detector,
)


@pytest.fixture
def detector():
    """Create a fresh type detector for each test."""
    return UniversalTypeDetector()


class TestEmptyDetection:
    """Test detection of empty columns."""
    
    def test_all_null_column(self, detector):
        """Column with all null values should be detected as empty."""
        col = pd.Series([None, None, None, None])
        result = detector.detect_semantic_type(col, 'empty_col')
        assert result.semantic_type == SemanticType.EMPTY
        assert result.confidence >= 0.9
    
    def test_mostly_null_column(self, detector):
        """Column with >90% null should be detected as empty."""
        col = pd.Series([None, None, None, None, None, None, None, None, None, 'value'])
        result = detector.detect_semantic_type(col, 'mostly_empty')
        assert result.semantic_type == SemanticType.EMPTY
        assert result.confidence >= 0.9
    
    def test_partial_null_column(self, detector):
        """Column with 50% null should NOT be detected as empty."""
        col = pd.Series([1, 2, None, None, 5])
        result = detector.detect_semantic_type(col, 'partial_null')
        assert result.semantic_type != SemanticType.EMPTY
    
    def test_empty_series(self, detector):
        """Empty pandas Series should be detected as empty."""
        col = pd.Series([], dtype=object)
        result = detector.detect_semantic_type(col, 'empty_series')
        assert result.semantic_type == SemanticType.EMPTY
        assert result.confidence == 1.0


class TestURLDetection:
    """Test detection of URL columns."""
    
    def test_https_urls(self, detector):
        """HTTPS URLs should be detected."""
        col = pd.Series([
            'https://example.com',
            'https://google.com/search',
            'https://github.com/repo',
            'https://api.service.io/v1/data'
        ])
        result = detector.detect_semantic_type(col, 'website')
        assert result.semantic_type == SemanticType.URL
        assert result.confidence >= 0.8
    
    def test_http_urls(self, detector):
        """HTTP URLs should be detected."""
        col = pd.Series([
            'http://example.com',
            'http://localhost:8080',
            'http://192.168.1.1'
        ])
        result = detector.detect_semantic_type(col, 'links')
        assert result.semantic_type == SemanticType.URL
    
    def test_image_urls(self, detector):
        """Image URLs should be detected."""
        col = pd.Series([
            'https://image.com/photo1.jpg',
            'https://cdn.site.com/img2.png',
            'https://s3.aws.com/bucket/image3.gif'
        ])
        result = detector.detect_semantic_type(col, 'photoUrl')
        assert result.semantic_type == SemanticType.URL
        assert result.confidence >= 0.8
    
    def test_url_name_hint(self, detector):
        """Column named 'url' with partial matches should be detected."""
        col = pd.Series([
            'https://example.com',
            'not a url',
            'https://google.com'
        ])
        result = detector.detect_semantic_type(col, 'profile_url')
        # With name hint, should still detect as URL
        assert result.semantic_type == SemanticType.URL


class TestPhoneDetection:
    """Test detection of phone number columns."""
    
    def test_plain_digits(self, detector):
        """10-digit phone numbers should be detected."""
        col = pd.Series(['9876543210', '9988776655', '9123456789', '9090909090'])
        result = detector.detect_semantic_type(col, 'mobile')
        assert result.semantic_type == SemanticType.PHONE
        assert result.confidence >= 0.8
    
    def test_formatted_us_phones(self, detector):
        """US formatted phone numbers should be detected."""
        col = pd.Series([
            '(555) 123-4567',
            '(800) 555-1212',
            '(212) 555-1234'
        ])
        result = detector.detect_semantic_type(col, 'phone')
        assert result.semantic_type == SemanticType.PHONE
    
    def test_international_phones(self, detector):
        """International phone numbers should be detected."""
        col = pd.Series([
            '+1-555-123-4567',
            '+44-20-7123-4567',
            '+91-9876543210'
        ])
        result = detector.detect_semantic_type(col, 'contact')
        assert result.semantic_type == SemanticType.PHONE
    
    def test_phone_with_name_hint(self, detector):
        """Column named 'phone' with partial matches should be detected."""
        col = pd.Series(['9876543210', 'unknown', '9988776655', '9123456789'])
        result = detector.detect_semantic_type(col, 'phone_number')
        assert result.semantic_type == SemanticType.PHONE


class TestEmailDetection:
    """Test detection of email columns."""
    
    def test_standard_emails(self, detector):
        """Standard email addresses should be detected."""
        col = pd.Series([
            'user@example.com',
            'john.doe@company.org',
            'test123@email.co.uk'
        ])
        result = detector.detect_semantic_type(col, 'email')
        assert result.semantic_type == SemanticType.EMAIL
        assert result.confidence >= 0.8
    
    def test_mixed_case_emails(self, detector):
        """Mixed case emails should be detected."""
        col = pd.Series([
            'User@Example.COM',
            'John.Doe@Company.org',
            'TEST@email.co.uk'
        ])
        result = detector.detect_semantic_type(col, 'contact_email')
        assert result.semantic_type == SemanticType.EMAIL
    
    def test_special_char_emails(self, detector):
        """Emails with special characters should be detected."""
        col = pd.Series([
            'user+tag@example.com',
            'first.last@sub.domain.org',
            'user_name@email.co'
        ])
        result = detector.detect_semantic_type(col, 'user_email')
        assert result.semantic_type == SemanticType.EMAIL


class TestIdentifierDetection:
    """Test detection of identifier columns."""
    
    def test_uuid_column(self, detector):
        """UUID column should be detected as identifier."""
        col = pd.Series([
            '550e8400-e29b-41d4-a716-446655440000',
            '6ba7b810-9dad-11d1-80b4-00c04fd430c8',
            'f47ac10b-58cc-4372-a567-0e02b2c3d479'
        ])
        result = detector.detect_semantic_type(col, 'uuid')
        assert result.semantic_type == SemanticType.IDENTIFIER
        assert result.confidence >= 0.8
    
    def test_id_prefixed_column(self, detector):
        """Column with ID prefix and high uniqueness should be identifier."""
        col = pd.Series(['id_001', 'id_002', 'id_003', 'id_004', 'id_005', 'id_006', 'id_007', 'id_008'])
        result = detector.detect_semantic_type(col, 'customer_id')
        assert result.semantic_type == SemanticType.IDENTIFIER
    
    def test_hash_like_column(self, detector):
        """Hash-like alphanumeric strings should be detected as identifier."""
        col = pd.Series([
            'abc123def456',
            'xyz789uvw012',
            'qwe345rty678',
            'asd234fgh567'
        ])
        result = detector.detect_semantic_type(col, 'hash_code')
        assert result.semantic_type == SemanticType.IDENTIFIER
    
    def test_name_not_identifier(self, detector):
        """Column named 'name' should NOT be identifier even with high uniqueness."""
        col = pd.Series(['John Smith', 'Jane Doe', 'Bob Wilson', 'Alice Brown'])
        result = detector.detect_semantic_type(col, 'name')
        assert result.semantic_type != SemanticType.IDENTIFIER


class TestDatetimeDetection:
    """Test detection of datetime columns."""
    
    def test_iso_datetime(self, detector):
        """ISO format datetime should be detected."""
        col = pd.Series([
            '2023-01-15',
            '2023-06-20',
            '2023-12-31'
        ])
        result = detector.detect_semantic_type(col, 'date')
        assert result.semantic_type == SemanticType.DATETIME
        assert result.confidence >= 0.8
    
    def test_datetime_with_time(self, detector):
        """Full datetime with time should be detected."""
        col = pd.Series([
            '2023-01-15T10:30:00',
            '2023-06-20T14:45:30',
            '2023-12-31T23:59:59'
        ])
        result = detector.detect_semantic_type(col, 'timestamp')
        assert result.semantic_type == SemanticType.DATETIME
    
    def test_us_date_format(self, detector):
        """US date format (MM/DD/YYYY) should be detected."""
        col = pd.Series([
            '01/15/2023',
            '06/20/2023',
            '12/31/2023'
        ])
        result = detector.detect_semantic_type(col, 'order_date')
        assert result.semantic_type == SemanticType.DATETIME
    
    def test_native_datetime(self, detector):
        """Native pandas datetime should be detected."""
        col = pd.Series(pd.to_datetime(['2023-01-15', '2023-06-20', '2023-12-31']))
        result = detector.detect_semantic_type(col, 'created_at')
        assert result.semantic_type == SemanticType.DATETIME
        assert result.confidence >= 0.95


class TestBooleanDetection:
    """Test detection of boolean columns."""
    
    def test_true_false(self, detector):
        """True/False values should be detected as boolean."""
        col = pd.Series(['True', 'False', 'True', 'False'])
        result = detector.detect_semantic_type(col, 'is_active')
        assert result.semantic_type == SemanticType.BOOLEAN
        assert result.confidence >= 0.9
    
    def test_yes_no(self, detector):
        """Yes/No values should be detected as boolean."""
        col = pd.Series(['Yes', 'No', 'Yes', 'No'])
        result = detector.detect_semantic_type(col, 'subscribed')
        assert result.semantic_type == SemanticType.BOOLEAN
    
    def test_zero_one(self, detector):
        """0/1 values should be detected as boolean."""
        col = pd.Series([0, 1, 1, 0, 1])
        result = detector.detect_semantic_type(col, 'flag')
        assert result.semantic_type == SemanticType.BOOLEAN
    
    def test_tf_abbreviation(self, detector):
        """T/F abbreviations should be detected as boolean."""
        col = pd.Series(['T', 'F', 'T', 'T', 'F'])
        result = detector.detect_semantic_type(col, 'confirmed')
        assert result.semantic_type == SemanticType.BOOLEAN


class TestNumericDetection:
    """Test detection of numeric columns."""
    
    def test_integers(self, detector):
        """Integer column should be detected as numeric."""
        col = pd.Series([1, 2, 3, 100, 200, 300])
        result = detector.detect_semantic_type(col, 'count')
        assert result.semantic_type == SemanticType.NUMERIC
        assert result.confidence >= 0.9
    
    def test_floats(self, detector):
        """Float column should be detected as numeric."""
        col = pd.Series([1.5, 2.7, 3.14159, 100.0, 200.5])
        result = detector.detect_semantic_type(col, 'amount')
        assert result.semantic_type == SemanticType.NUMERIC
    
    def test_negative_numbers(self, detector):
        """Negative numbers should be detected as numeric."""
        col = pd.Series([-10, -5, 0, 5, 10])
        result = detector.detect_semantic_type(col, 'delta')
        assert result.semantic_type == SemanticType.NUMERIC
    
    def test_numeric_strings(self, detector):
        """Numeric strings should be detected as numeric."""
        col = pd.Series(['100', '200', '300', '400', '500'])
        result = detector.detect_semantic_type(col, 'value')
        assert result.semantic_type == SemanticType.NUMERIC


class TestCategoricalDetection:
    """Test detection of categorical columns."""
    
    def test_low_cardinality(self, detector):
        """Low cardinality column should be detected as categorical."""
        col = pd.Series(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'])
        result = detector.detect_semantic_type(col, 'category')
        assert result.semantic_type == SemanticType.CATEGORICAL
        assert result.confidence >= 0.6
    
    def test_status_column(self, detector):
        """Status column with few unique values should be categorical."""
        col = pd.Series(['active', 'inactive', 'pending', 'active', 'inactive'] * 10)
        result = detector.detect_semantic_type(col, 'status')
        assert result.semantic_type == SemanticType.CATEGORICAL
    
    def test_gender_column(self, detector):
        """Gender column should be detected as categorical."""
        col = pd.Series(['Male', 'Female', 'Male', 'Female', 'Male'] * 5)
        result = detector.detect_semantic_type(col, 'gender')
        assert result.semantic_type == SemanticType.CATEGORICAL


class TestTextDetection:
    """Test detection of text columns."""
    
    def test_person_names(self, detector):
        """Person names should be detected as text."""
        col = pd.Series(['John Smith', 'Jane Doe', 'Bob Wilson', 'Alice Brown'])
        result = detector.detect_semantic_type(col, 'name')
        assert result.semantic_type == SemanticType.TEXT
        assert result.confidence >= 0.7
    
    def test_product_names(self, detector):
        """Product names should be detected as text."""
        col = pd.Series(['iPhone 14 Pro', 'Samsung Galaxy S23', 'Google Pixel 7', 'OnePlus 11'])
        result = detector.detect_semantic_type(col, 'product_name')
        assert result.semantic_type == SemanticType.TEXT
    
    def test_descriptions(self, detector):
        """Description text should be detected as text."""
        col = pd.Series([
            'This is a long description with multiple words',
            'Another description that provides information',
            'Yet another detailed text description here'
        ])
        result = detector.detect_semantic_type(col, 'description')
        assert result.semantic_type == SemanticType.TEXT
    
    def test_car_names(self, detector):
        """Car names (from car dataset) should be detected as text."""
        col = pd.Series(['Maruti Swift', 'Honda City', 'Toyota Corolla', 'Hyundai Verna'])
        result = detector.detect_semantic_type(col, 'name')
        assert result.semantic_type == SemanticType.TEXT


class TestEdgeCases:
    """Test edge cases and tricky scenarios."""
    
    def test_mixed_types(self, detector):
        """Mixed type column should default to text."""
        col = pd.Series([1, 'two', 3.0, None, 'five'])
        result = detector.detect_semantic_type(col, 'mixed')
        # Should handle gracefully
        assert result.semantic_type in [SemanticType.TEXT, SemanticType.CATEGORICAL]
    
    def test_single_value(self, detector):
        """Single value column should be detected."""
        col = pd.Series(['only_value'])
        result = detector.detect_semantic_type(col, 'single')
        assert result.semantic_type is not None
    
    def test_numeric_looking_id(self, detector):
        """Numeric column that looks like ID should be handled."""
        col = pd.Series([10001, 10002, 10003, 10004, 10005])
        result = detector.detect_semantic_type(col, 'user_id')
        # Either numeric or identifier is acceptable
        assert result.semantic_type in [SemanticType.NUMERIC, SemanticType.IDENTIFIER]
    
    def test_special_characters_in_name(self, detector):
        """Column names with special characters should work."""
        col = pd.Series([1, 2, 3, 4, 5])
        result = detector.detect_semantic_type(col, 'column-with-dashes_and_underscores')
        assert result.semantic_type == SemanticType.NUMERIC


class TestGetTypeDetector:
    """Test the singleton getter function."""
    
    def test_get_type_detector(self):
        """get_type_detector should return a detector instance."""
        detector = get_type_detector()
        assert isinstance(detector, UniversalTypeDetector)
    
    def test_detector_works(self):
        """Detector from getter should work correctly."""
        detector = get_type_detector()
        col = pd.Series([1, 2, 3, 4, 5])
        result = detector.detect_semantic_type(col, 'numbers')
        assert result.semantic_type == SemanticType.NUMERIC
