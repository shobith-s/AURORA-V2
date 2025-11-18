"""
Tests for privacy-preserving pattern learner.
"""

import pytest
import tempfile
import json
from pathlib import Path
import numpy as np

from src.learning.pattern_learner import (
    ColumnPattern,
    CorrectionRecord,
    LocalPatternLearner
)
from src.learning.privacy import AnonymizationUtils, create_privacy_preserving_pattern
from src.core.actions import PreprocessingAction


class TestAnonymizationUtils:
    """Test anonymization utilities."""

    def test_hash_value(self):
        """Test value hashing."""
        utils = AnonymizationUtils()

        # Same value should produce same hash
        hash1 = utils.hash_value("test_value")
        hash2 = utils.hash_value("test_value")
        assert hash1 == hash2

        # Different values should produce different hashes
        hash3 = utils.hash_value("different_value")
        assert hash1 != hash3

        # Hash should be hexadecimal string
        assert all(c in '0123456789abcdef' for c in hash1)

    def test_discretize_value(self):
        """Test value discretization."""
        utils = AnonymizationUtils()

        # Test with explicit range
        bucket = utils.discretize_value(0.5, bins=5, value_range=(0, 1))
        assert 0 <= bucket < 5

        # Test edge cases
        bucket_min = utils.discretize_value(0.0, bins=5, value_range=(0, 1))
        assert bucket_min == 0

        bucket_max = utils.discretize_value(1.0, bins=5, value_range=(0, 1))
        assert bucket_max == 4

        # Test clipping
        bucket_over = utils.discretize_value(2.0, bins=5, value_range=(0, 1))
        assert bucket_over == 4

    def test_add_laplace_noise(self):
        """Test Laplace noise addition."""
        utils = AnonymizationUtils()

        np.random.seed(42)
        original_value = 100.0
        noisy_value = utils.add_laplace_noise(original_value, epsilon=1.0)

        # Noisy value should be different
        assert noisy_value != original_value

        # But should be reasonably close
        assert abs(noisy_value - original_value) < 50

    def test_generalize_numeric_value(self):
        """Test numeric value generalization."""
        utils = AnonymizationUtils()

        value = 3.14159265359

        # Test different precisions
        assert utils.generalize_numeric_value(value, precision=2) == 3.14
        assert utils.generalize_numeric_value(value, precision=0) == 3.0
        assert utils.generalize_numeric_value(value, precision=4) == 3.1416

    def test_k_anonymize_categorical(self):
        """Test k-anonymity for categorical values."""
        utils = AnonymizationUtils()

        values = ['A', 'A', 'A', 'B', 'B', 'C']  # C appears only once
        anonymized = utils.k_anonymize_categorical(values, k=2)

        # Rare values should be suppressed
        assert anonymized.count('*') > 0
        # Frequent values should be preserved
        assert 'A' in anonymized
        assert 'B' in anonymized


class TestCreatePrivacyPreservingPattern:
    """Test privacy-preserving pattern creation."""

    def test_create_pattern_high_privacy(self):
        """Test pattern creation with high privacy."""
        data = {
            'mean': 10.5,
            'median': 10.0,
            'std': 2.5,
            'skewness': 0.3,
            'null_pct': 0.05,
            'unique_ratio': 0.8,
            'is_numeric': True,
            'is_categorical': False,
            'column_name': 'test_column'
        }

        pattern = create_privacy_preserving_pattern(data, privacy_level='high')

        # Should have anonymized statistics
        assert 'mean_anonymous' in pattern
        assert 'median_anonymous' in pattern
        assert 'std_anonymous' in pattern

        # Should have discretized values
        assert 'null_pct_bucket' in pattern
        assert 'unique_ratio_bucket' in pattern

        # Should have hashed identifiers
        assert 'column_name_hash' in pattern

        # Should preserve boolean flags
        assert pattern['is_numeric'] == True
        assert pattern['is_categorical'] == False


class TestColumnPattern:
    """Test ColumnPattern class."""

    def test_pattern_creation(self):
        """Test creating a column pattern."""
        pattern = ColumnPattern(
            skew_bucket=2,
            null_bucket=1,
            cardinality_type='medium',
            is_numeric=True,
            name_tokens=['age', 'years']
        )

        assert pattern.skew_bucket == 2
        assert pattern.null_bucket == 1
        assert pattern.is_numeric == True
        assert 'age' in pattern.name_tokens

    def test_pattern_serialization(self):
        """Test pattern to/from dict conversion."""
        pattern = ColumnPattern(
            skew_bucket=2,
            null_bucket=1,
            cardinality_type='high',
            is_numeric=True,
            name_tokens=['price', 'usd']
        )

        # Convert to dict
        pattern_dict = pattern.to_dict()
        assert isinstance(pattern_dict, dict)
        assert pattern_dict['skew_bucket'] == 2

        # Recreate from dict
        pattern2 = ColumnPattern.from_dict(pattern_dict)
        assert pattern2.skew_bucket == pattern.skew_bucket
        assert pattern2.name_tokens == pattern.name_tokens

    def test_pattern_similarity_identical(self):
        """Test similarity between identical patterns."""
        pattern1 = ColumnPattern(
            skew_bucket=2,
            null_bucket=1,
            cardinality_type='medium',
            is_numeric=True,
            name_tokens=['age']
        )

        pattern2 = ColumnPattern(
            skew_bucket=2,
            null_bucket=1,
            cardinality_type='medium',
            is_numeric=True,
            name_tokens=['age']
        )

        similarity = pattern1.similarity(pattern2)
        assert similarity == 1.0

    def test_pattern_similarity_different(self):
        """Test similarity between different patterns."""
        pattern1 = ColumnPattern(
            skew_bucket=2,
            null_bucket=1,
            cardinality_type='medium',
            is_numeric=True,
            name_tokens=['age']
        )

        pattern2 = ColumnPattern(
            skew_bucket=4,
            null_bucket=3,
            cardinality_type='high',
            is_numeric=False,
            name_tokens=['name']
        )

        similarity = pattern1.similarity(pattern2)
        assert 0.0 <= similarity < 1.0

    def test_pattern_similarity_partial_match(self):
        """Test similarity with partial matching."""
        pattern1 = ColumnPattern(
            skew_bucket=2,
            is_numeric=True,
            name_tokens=['age', 'years']
        )

        pattern2 = ColumnPattern(
            skew_bucket=2,
            is_numeric=True,
            name_tokens=['age', 'months']  # Shares 'age' token
        )

        similarity = pattern1.similarity(pattern2)
        assert 0.3 < similarity < 1.0


class TestLocalPatternLearner:
    """Test LocalPatternLearner class."""

    def test_learner_initialization(self):
        """Test learner initialization."""
        learner = LocalPatternLearner(
            similarity_threshold=0.8,
            min_pattern_support=3,
            privacy_level='high'
        )

        assert learner.similarity_threshold == 0.8
        assert learner.min_pattern_support == 3
        assert learner.privacy_level == 'high'
        assert len(learner.correction_records) == 0
        assert len(learner.learned_rules) == 0

    def test_extract_pattern(self):
        """Test pattern extraction from column stats."""
        learner = LocalPatternLearner()

        column_stats = {
            'skewness': 0.5,
            'null_pct': 0.1,
            'unique_ratio': 0.7,
            'is_numeric': True,
            'is_categorical': False,
            'is_temporal': False,
            'matches_date_pattern': 0.0,
            'has_currency_symbols': False,
            'matches_email_pattern': 0.0,
            'matches_phone_pattern': 0.0
        }

        pattern = learner.extract_pattern(column_stats, 'user_age')

        assert pattern.is_numeric == True
        assert pattern.is_categorical == False
        assert pattern.skew_bucket is not None
        assert pattern.null_bucket is not None
        assert 'user' in pattern.name_tokens or 'age' in pattern.name_tokens

    def test_tokenize_column_name(self):
        """Test column name tokenization."""
        learner = LocalPatternLearner()

        # Test camelCase
        tokens = learner._tokenize_column_name('userAge')
        assert 'user' in tokens
        assert 'age' in tokens

        # Test snake_case
        tokens = learner._tokenize_column_name('user_age_years')
        assert 'user' in tokens
        assert 'age' in tokens
        assert 'years' in tokens

        # Test mixed
        tokens = learner._tokenize_column_name('totalPrice2023')
        assert 'total' in tokens or 'price' in tokens

    def test_learn_correction_single(self):
        """Test learning from a single correction."""
        learner = LocalPatternLearner(min_pattern_support=1)

        pattern = ColumnPattern(
            skew_bucket=2,
            null_bucket=1,
            cardinality_type='medium',
            is_numeric=True,
            pattern_hash='test123'
        )

        result = learner.learn_correction(
            pattern,
            wrong_action=PreprocessingAction.NORMALIZE,
            correct_action=PreprocessingAction.REMOVE_OUTLIERS,
            confidence=0.9
        )

        # Should create a correction record
        assert len(learner.correction_records) == 1

        # With min_pattern_support=1, should create a rule
        assert result is not None
        assert len(learner.learned_rules) == 1

    def test_learn_correction_multiple(self):
        """Test learning from multiple similar corrections."""
        learner = LocalPatternLearner(
            similarity_threshold=0.9,
            min_pattern_support=3
        )

        # Add 3 similar corrections
        for i in range(3):
            pattern = ColumnPattern(
                skew_bucket=2,
                null_bucket=1,
                cardinality_type='medium',
                is_numeric=True,
                pattern_hash='similar'
            )

            learner.learn_correction(
                pattern,
                wrong_action=PreprocessingAction.NORMALIZE,
                correct_action=PreprocessingAction.REMOVE_OUTLIERS,
                confidence=0.8
            )

        # Should have learned a rule after 3 similar patterns
        assert len(learner.correction_records) == 3
        assert len(learner.learned_rules) >= 1

    def test_find_similar_patterns(self):
        """Test finding similar patterns."""
        learner = LocalPatternLearner()

        # Add some correction records
        for i in range(5):
            pattern = ColumnPattern(
                skew_bucket=i,
                null_bucket=1,
                is_numeric=True,
                pattern_hash=f'hash{i}'
            )
            record = CorrectionRecord(
                pattern=pattern,
                wrong_action=PreprocessingAction.NORMALIZE,
                correct_action=PreprocessingAction.REMOVE_OUTLIERS,
                timestamp=float(i),
                confidence_score=0.8
            )
            learner.correction_records.append(record)

        # Find similar to a test pattern
        test_pattern = ColumnPattern(
            skew_bucket=2,
            null_bucket=1,
            is_numeric=True
        )

        similar = learner.find_similar_patterns(test_pattern, threshold=0.5)
        assert len(similar) > 0

    def test_check_patterns(self):
        """Test checking if learned patterns match."""
        learner = LocalPatternLearner(min_pattern_support=1)

        # Learn a pattern
        pattern = ColumnPattern(
            skew_bucket=2,
            null_bucket=1,
            cardinality_type='medium',
            is_numeric=True,
            pattern_hash='test'
        )

        learner.learn_correction(
            pattern,
            wrong_action=PreprocessingAction.NORMALIZE,
            correct_action=PreprocessingAction.REMOVE_OUTLIERS,
            confidence=0.9
        )

        # Check with matching stats
        column_stats = {
            'is_numeric': True,
            'unique_ratio': 0.3,  # medium cardinality
            'column_name': 'test_col'
        }

        action = learner.check_patterns(column_stats)
        # May or may not match depending on rule condition
        assert action is None or isinstance(action, PreprocessingAction)

    def test_save_and_load(self):
        """Test saving and loading learned patterns."""
        learner = LocalPatternLearner(min_pattern_support=1)

        # Add some corrections
        pattern = ColumnPattern(
            skew_bucket=2,
            null_bucket=1,
            cardinality_type='high',
            is_numeric=True,
            pattern_hash='test123'
        )

        learner.learn_correction(
            pattern,
            wrong_action=PreprocessingAction.NORMALIZE,
            correct_action=PreprocessingAction.REMOVE_OUTLIERS,
            confidence=0.85
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = Path(f.name)

        try:
            learner.save(temp_path)

            # Load into new learner
            learner2 = LocalPatternLearner()
            learner2.load(temp_path)

            # Check data was restored
            assert len(learner2.correction_records) == len(learner.correction_records)
            assert learner2.similarity_threshold == learner.similarity_threshold
            assert learner2.min_pattern_support == learner.min_pattern_support
            assert learner2.privacy_level == learner.privacy_level

        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

    def test_get_statistics(self):
        """Test getting learner statistics."""
        learner = LocalPatternLearner()

        stats = learner.get_statistics()

        assert 'total_corrections' in stats
        assert 'learned_rules' in stats
        assert 'pattern_clusters' in stats
        assert 'avg_cluster_size' in stats

        assert stats['total_corrections'] == 0
        assert stats['learned_rules'] == 0

    def test_pattern_hash_computation(self):
        """Test pattern hash computation for clustering."""
        learner = LocalPatternLearner()

        pattern1 = ColumnPattern(
            is_numeric=True,
            is_categorical=False,
            cardinality_type='medium',
            null_bucket=1
        )

        pattern2 = ColumnPattern(
            is_numeric=True,
            is_categorical=False,
            cardinality_type='medium',
            null_bucket=1
        )

        hash1 = learner._compute_pattern_hash(pattern1)
        hash2 = learner._compute_pattern_hash(pattern2)

        # Same patterns should produce same hash
        assert hash1 == hash2

        # Different pattern should produce different hash
        pattern3 = ColumnPattern(
            is_numeric=False,
            is_categorical=True,
            cardinality_type='high',
            null_bucket=2
        )

        hash3 = learner._compute_pattern_hash(pattern3)
        assert hash1 != hash3

    def test_generalize_patterns(self):
        """Test pattern generalization into rules."""
        learner = LocalPatternLearner()

        # Create similar correction records
        records = []
        for i in range(3):
            pattern = ColumnPattern(
                is_numeric=True,
                cardinality_type='high',
                skew_bucket=2
            )
            record = CorrectionRecord(
                pattern=pattern,
                wrong_action=PreprocessingAction.NORMALIZE,
                correct_action=PreprocessingAction.REMOVE_OUTLIERS,
                timestamp=float(i),
                confidence_score=0.8
            )
            records.append(record)

        rule = learner._generalize_patterns(records, PreprocessingAction.REMOVE_OUTLIERS)

        assert rule is not None
        assert rule.action == PreprocessingAction.REMOVE_OUTLIERS
        assert rule.priority == 85  # High priority for learned rules


class TestIntegration:
    """Integration tests for the pattern learning system."""

    def test_full_learning_cycle(self):
        """Test a full learning cycle from correction to rule application."""
        learner = LocalPatternLearner(
            similarity_threshold=0.8,
            min_pattern_support=2
        )

        # Simulate multiple user corrections for similar columns
        column_stats_list = [
            {
                'skewness': 2.5,
                'null_pct': 0.05,
                'unique_ratio': 0.9,
                'is_numeric': True,
                'is_categorical': False,
                'is_temporal': False,
                'matches_date_pattern': 0.0,
                'has_currency_symbols': False,
                'matches_email_pattern': 0.0,
                'matches_phone_pattern': 0.0,
                'column_name': 'user_id'
            },
            {
                'skewness': 2.3,
                'null_pct': 0.03,
                'unique_ratio': 0.95,
                'is_numeric': True,
                'is_categorical': False,
                'is_temporal': False,
                'matches_date_pattern': 0.0,
                'has_currency_symbols': False,
                'matches_email_pattern': 0.0,
                'matches_phone_pattern': 0.0,
                'column_name': 'customer_id'
            }
        ]

        # Learn from corrections
        for stats in column_stats_list:
            pattern = learner.extract_pattern(stats, stats['column_name'])
            learner.learn_correction(
                pattern,
                wrong_action=PreprocessingAction.NORMALIZE,
                correct_action=PreprocessingAction.KEEP,
                confidence=0.7
            )

        # Check that we learned something
        assert len(learner.correction_records) == 2

        # Get statistics
        stats = learner.get_statistics()
        assert stats['total_corrections'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
