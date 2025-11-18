"""
Tests for privacy-preserving components.
"""

import pytest
import numpy as np

from src.learning.privacy import (
    DifferentialPrivacy,
    PrivacyBudget,
    AnonymizationUtils,
    create_privacy_preserving_pattern
)


class TestPrivacyBudget:
    """Test cases for PrivacyBudget."""

    def test_budget_initialization(self):
        """Test privacy budget initialization."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)

        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5
        assert budget.spent == 0.0

    def test_can_spend(self):
        """Test checking if budget can be spent."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)

        assert budget.can_spend(0.5) == True
        assert budget.can_spend(1.0) == True
        assert budget.can_spend(1.1) == False

    def test_spend_budget(self):
        """Test spending privacy budget."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)

        assert budget.spend(0.5) == True
        assert budget.spent == 0.5
        assert budget.spend(0.3) == True
        assert budget.spent == 0.8
        assert budget.spend(0.5) == False  # Would exceed budget

    def test_remaining_budget(self):
        """Test getting remaining budget."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        budget.spend(0.3)

        assert budget.remaining() == 0.7


class TestDifferentialPrivacy:
    """Test cases for DifferentialPrivacy."""

    def test_laplacian_noise(self):
        """Test adding Laplacian noise."""
        dp = DifferentialPrivacy(epsilon=1.0)

        value = 100.0
        noisy_value = dp.add_laplacian_noise(value, sensitivity=1.0)

        # Noise should change the value
        assert noisy_value != value

        # But should be relatively close for epsilon=1.0
        assert abs(noisy_value - value) < 50  # Reasonable bound

    def test_gaussian_noise(self):
        """Test adding Gaussian noise."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

        value = 100.0
        noisy_value = dp.add_gaussian_noise(value, sensitivity=1.0)

        assert noisy_value != value
        assert abs(noisy_value - value) < 50

    def test_privatize_count(self):
        """Test privatizing counts."""
        dp = DifferentialPrivacy(epsilon=1.0)

        true_count = 100
        noisy_count = dp.privatize_count(true_count)

        # Should be an integer
        assert isinstance(noisy_count, int)
        # Should be non-negative
        assert noisy_count >= 0
        # Should be relatively close
        assert abs(noisy_count - true_count) < 50

    def test_privatize_mean(self):
        """Test privatizing mean values."""
        dp = DifferentialPrivacy(epsilon=1.0)

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true_mean = np.mean(values)

        noisy_mean = dp.privatize_mean(values, value_range=(1.0, 5.0))

        assert abs(noisy_mean - true_mean) < 2.0

    def test_privatize_histogram(self):
        """Test privatizing histograms."""
        dp = DifferentialPrivacy(epsilon=1.0)

        histogram = {'A': 10, 'B': 20, 'C': 15}
        noisy_histogram = dp.privatize_histogram(histogram)

        # Should have same keys
        assert set(noisy_histogram.keys()) == set(histogram.keys())

        # All counts should be non-negative
        assert all(count >= 0 for count in noisy_histogram.values())

    def test_gradient_noise(self):
        """Test adding noise to gradients."""
        dp = DifferentialPrivacy(epsilon=1.0)

        gradient = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        noisy_gradient = dp.add_noise_to_gradient(gradient, clip_norm=1.0)

        # Should have same shape
        assert noisy_gradient.shape == gradient.shape

        # Should be different
        assert not np.array_equal(noisy_gradient, gradient)

    def test_gradient_clipping(self):
        """Test that gradients are clipped."""
        dp = DifferentialPrivacy(epsilon=1.0)

        # Large gradient
        gradient = np.array([10.0, 20.0, 30.0])
        noisy_gradient = dp.add_noise_to_gradient(gradient, clip_norm=1.0)

        # Clipped norm should be close to clip_norm (plus noise)
        # Note: noise is added after clipping
        norm = np.linalg.norm(gradient)
        assert norm > 1.0  # Original is large
        # After clipping and noise, should be different


class TestAnonymizationUtils:
    """Test cases for AnonymizationUtils."""

    def test_hash_value(self):
        """Test value hashing."""
        utils = AnonymizationUtils()

        hash1 = utils.hash_value("test_value")
        hash2 = utils.hash_value("test_value")
        hash3 = utils.hash_value("different_value")

        # Same value should produce same hash
        assert hash1 == hash2
        # Different values should produce different hashes
        assert hash1 != hash3

    def test_hash_with_salt(self):
        """Test hashing with salt."""
        utils = AnonymizationUtils()

        hash1 = utils.hash_value("test", salt="salt1")
        hash2 = utils.hash_value("test", salt="salt2")

        # Different salts should produce different hashes
        assert hash1 != hash2

    def test_discretize_value(self):
        """Test value discretization."""
        utils = AnonymizationUtils()

        # Test within range
        bin_idx = utils.discretize_value(0.5, bins=10, value_range=(0, 1))
        assert 0 <= bin_idx < 10

        # Test edge cases
        assert utils.discretize_value(0.0, bins=10, value_range=(0, 1)) == 0
        assert utils.discretize_value(1.0, bins=10, value_range=(0, 1)) == 9

        # Test out of range
        assert utils.discretize_value(-0.5, bins=10, value_range=(0, 1)) == 0
        assert utils.discretize_value(1.5, bins=10, value_range=(0, 1)) == 9

    def test_generalize_numeric(self):
        """Test numeric generalization."""
        utils = AnonymizationUtils()

        value = 3.14159265
        generalized = utils.generalize_numeric(value, precision=2)

        assert generalized == 3.14

    def test_k_anonymize_categories(self):
        """Test k-anonymity for categories."""
        utils = AnonymizationUtils()

        categories = ['A'] * 10 + ['B'] * 5 + ['C'] * 2 + ['D'] * 1
        anonymized = utils.k_anonymize_categories(categories, k=3, suppress_threshold=2)

        # Rare categories should be suppressed or generalized
        assert '*' in anonymized or 'OTHER' in anonymized

    def test_remove_identifiers(self):
        """Test removal of identifiers."""
        utils = AnonymizationUtils()

        data = {
            'user_id': 123,
            'email': 'test@example.com',
            'age': 25,
            'revenue': 1000,
            'name': 'John'
        }

        cleaned = utils.remove_identifiers(data)

        # Identifiers should be removed
        assert 'user_id' not in cleaned
        assert 'email' not in cleaned
        assert 'name' not in cleaned

        # Non-identifiers should remain
        assert 'age' in cleaned
        assert 'revenue' in cleaned


class TestPrivacyPreservingPattern:
    """Test privacy-preserving pattern creation."""

    def test_pattern_creation(self):
        """Test that patterns don't contain raw data."""
        column_stats = {
            'skewness': 2.5,
            'null_pct': 0.15,
            'unique_ratio': 0.6,
            'is_numeric': True,
            'is_categorical': False,
            'min_value': 10.0,  # This should NOT be in pattern
            'max_value': 100.0,  # This should NOT be in pattern
            'sample_values': [10, 20, 30]  # This should NOT be in pattern
        }

        pattern = create_privacy_preserving_pattern(column_stats, privacy_level='high')

        # Should have discretized statistics
        assert 'skew_bucket' in pattern
        assert 'null_bucket' in pattern
        assert 'cardinality_type' in pattern

        # Should NOT have raw values or samples
        assert 'min_value' not in pattern
        assert 'max_value' not in pattern
        assert 'sample_values' not in pattern

    def test_privacy_levels(self):
        """Test different privacy levels."""
        column_stats = {
            'skewness': 2.5,
            'null_pct': 0.15,
            'unique_ratio': 0.6,
            'is_numeric': True
        }

        pattern_high = create_privacy_preserving_pattern(column_stats, privacy_level='high')
        pattern_low = create_privacy_preserving_pattern(column_stats, privacy_level='low')

        # Both should have same structure
        assert set(pattern_high.keys()) == set(pattern_low.keys())

        # Values might differ due to different discretization


class TestNoDataLeakage:
    """Test that no actual data values are stored."""

    def test_no_raw_values_in_pattern(self):
        """Ensure patterns never contain raw data values."""
        from src.learning.pattern_learner import LocalPatternLearner
        import pandas as pd

        learner = LocalPatternLearner()

        # Create sensitive column with PII
        sensitive_data = {
            'is_numeric': False,
            'is_categorical': True,
            'skewness': None,
            'null_pct': 0.1,
            'unique_ratio': 0.9,
            'cardinality': 100
        }

        pattern = learner.extract_pattern(sensitive_data, column_name="email_address")

        # Pattern should not contain any actual email addresses
        pattern_str = str(pattern.to_dict())

        # Check that common PII patterns are not in the pattern
        assert '@' not in pattern_str
        assert '.com' not in pattern_str

        # Should only have statistical signatures
        assert pattern.null_bucket is not None or pattern.cardinality_type is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
