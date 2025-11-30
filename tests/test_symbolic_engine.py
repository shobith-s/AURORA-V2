"""
Tests for the Symbolic Engine.
"""

import pytest
import pandas as pd
import numpy as np

from src.symbolic.engine import SymbolicEngine, ColumnStatistics
from src.core.actions import PreprocessingAction


class TestSymbolicEngine:
    """Test cases for SymbolicEngine."""

    @pytest.fixture
    def engine(self):
        """Create a symbolic engine instance."""
        return SymbolicEngine(confidence_threshold=0.9)

    def test_engine_initialization(self, engine):
        """Test that engine initializes correctly."""
        assert engine is not None
        assert engine.confidence_threshold == 0.9
        assert len(engine.rules) > 0

    def test_constant_column_detection(self, engine):
        """Test detection of constant columns."""
        # Create constant column
        column = pd.Series([42] * 100, name="constant")

        result = engine.evaluate(column)

        assert result.action == PreprocessingAction.DROP_COLUMN
        assert result.confidence > 0.9
        assert result.source == "symbolic"
        assert "constant" in result.explanation.lower()

    def test_mostly_null_detection(self, engine):
        """Test detection of columns with high null percentage."""
        # Create column with 70% nulls
        column = pd.Series([None] * 70 + list(range(30)), name="mostly_null")

        result = engine.evaluate(column)

        assert result.action == PreprocessingAction.DROP_COLUMN
        assert result.confidence > 0.9
        assert "null" in result.explanation.lower()

    def test_skewed_data_detection(self, engine):
        """Test detection of highly skewed data."""
        # Create highly skewed positive data with range > 100 and skewness > 2.0
        # (range_size > 100 is required for log transform to trigger)
        np.random.seed(42)
        # Use power of exponential to get high skewness (>2.0)
        # This generates data with skewness ~3.6 and range ~23000
        data = np.random.exponential(scale=100, size=1000) ** 1.5

        column = pd.Series(data, name="revenue")

        result = engine.evaluate(column)

        # Should recommend log transform, box-cox, or robust_scale for highly skewed data
        # Note: robust_scale may be chosen if outliers are present with higher confidence
        assert result.action in [
            PreprocessingAction.LOG_TRANSFORM,
            PreprocessingAction.LOG1P_TRANSFORM,
            PreprocessingAction.BOX_COX,
            PreprocessingAction.ROBUST_SCALE
        ]
        assert result.confidence > 0.7

    def test_normal_distribution_scaling(self, engine):
        """Test that normal distributions get standard scaling."""
        # Create normally distributed data
        np.random.seed(42)
        data = np.random.normal(50, 15, 1000)

        column = pd.Series(data, name="age")

        result = engine.evaluate(column)

        # Should recommend some form of scaling
        assert result.action in [
            PreprocessingAction.STANDARD_SCALE,
            PreprocessingAction.ROBUST_SCALE,
            PreprocessingAction.KEEP_AS_IS
        ]

    def test_outlier_detection(self, engine):
        """Test detection of data with outliers."""
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 900)
        outliers = np.array([200, 250, 300, -100, -150] * 20)  # 100 outliers
        data = np.concatenate([normal_data, outliers])
        np.random.shuffle(data)

        column = pd.Series(data, name="values")

        result = engine.evaluate(column)

        # Should detect outliers and recommend clipping or robust scaling
        assert result.action in [
            PreprocessingAction.CLIP_OUTLIERS,
            PreprocessingAction.WINSORIZE,
            PreprocessingAction.ROBUST_SCALE
        ]

    def test_low_cardinality_categorical(self, engine):
        """Test detection of low-cardinality categorical data."""
        # Create low-cardinality categorical
        categories = ['A', 'B', 'C', 'D', 'E']
        column = pd.Series(np.random.choice(categories, 1000), name="category")

        result = engine.evaluate(column)

        # Should recommend one-hot encoding
        assert result.action in [
            PreprocessingAction.ONEHOT_ENCODE,
            PreprocessingAction.LABEL_ENCODE
        ]

    def test_column_statistics_computation(self, engine):
        """Test that column statistics are computed correctly."""
        # Create test data
        data = [1, 2, 3, None, 5, 6, 7, 8, 9, 10]
        column = pd.Series(data, name="test")

        stats = engine.compute_column_statistics(column)

        assert stats.row_count == 10
        assert stats.null_count == 1
        assert stats.null_pct == 0.1
        assert stats.unique_count == 9
        assert stats.is_numeric == True

    def test_coverage_metric(self, engine):
        """Test that coverage metric is computed."""
        # Make several decisions
        for i in range(10):
            data = np.random.normal(0, 1, 100)
            column = pd.Series(data, name=f"col_{i}")
            engine.evaluate(column)

        coverage = engine.coverage()

        assert 0.0 <= coverage <= 1.0

    def test_add_custom_rule(self, engine):
        """Test adding a custom rule to the engine."""
        from src.symbolic.rules import Rule, RuleCategory

        # Create a custom rule
        custom_rule = Rule(
            name="CUSTOM_RULE",
            category=RuleCategory.DATA_QUALITY,
            action=PreprocessingAction.KEEP_AS_IS,
            condition=lambda stats: stats.get('is_numeric', False),
            confidence_fn=lambda stats: 0.95,
            explanation_fn=lambda stats: "Custom rule explanation",
            priority=100
        )

        initial_count = len(engine.rules)
        engine.add_rule(custom_rule)

        assert len(engine.rules) == initial_count + 1

    def test_alternatives_provided(self, engine):
        """Test that alternatives are provided when available."""
        # Create ambiguous case
        data = np.random.uniform(0, 100, 1000)
        column = pd.Series(data, name="score")

        result = engine.evaluate(column)

        # Should have alternatives for most cases
        assert isinstance(result.alternatives, list)

    def test_decision_explanation(self, engine):
        """Test that decisions include explanations."""
        data = [42] * 100
        column = pd.Series(data, name="constant")

        result = engine.evaluate(column)

        assert result.explanation is not None
        assert len(result.explanation) > 0
        assert isinstance(result.explanation, str)

    def test_confidence_scores(self, engine):
        """Test that confidence scores are in valid range."""
        data = np.random.normal(0, 1, 1000)
        column = pd.Series(data, name="test")

        result = engine.evaluate(column)

        assert 0.0 <= result.confidence <= 1.0

    def test_date_string_detection(self, engine):
        """Test detection of date strings."""
        # Create date strings
        dates = pd.date_range('2020-01-01', periods=100).strftime('%Y-%m-%d')
        column = pd.Series(dates, name="date_column")

        stats = engine.compute_column_statistics(column)

        assert stats.matches_iso_datetime > 0.8

    def test_boolean_detection(self, engine):
        """Test detection of boolean-like strings."""
        # Create boolean strings
        bools = np.random.choice(['True', 'False'], 100)
        column = pd.Series(bools, name="flag")

        stats = engine.compute_column_statistics(column)

        assert stats.matches_boolean_tf > 0.9


class TestColumnStatistics:
    """Test cases for ColumnStatistics."""

    def test_statistics_to_dict(self):
        """Test conversion of statistics to dictionary."""
        stats = ColumnStatistics(
            row_count=100,
            null_count=10,
            null_pct=0.1,
            unique_count=50,
            unique_ratio=0.5,
            dtype="float64",
            is_numeric=True
        )

        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert stats_dict['row_count'] == 100
        assert stats_dict['null_pct'] == 0.1
        assert stats_dict['is_numeric'] == True


class TestBestsellersScenario:
    """Test cases for the bestsellers-like preprocessing scenario."""

    @pytest.fixture
    def engine(self):
        """Create a symbolic engine instance."""
        return SymbolicEngine(confidence_threshold=0.9)

    def test_target_variable_preserved(self, engine):
        """Test that target variables like Price are preserved (keep_as_is)."""
        np.random.seed(42)
        price_data = np.random.uniform(8, 75, 500)
        column = pd.Series(price_data, name="Price")

        result = engine.evaluate(column, column_name="Price")

        assert result.action == PreprocessingAction.KEEP_AS_IS
        assert result.confidence >= 0.99
        assert "Target variable" in result.explanation

    def test_year_column_scaling(self, engine):
        """Test that Year columns get standard scaling, not log transform."""
        np.random.seed(42)
        year_data = np.random.randint(2009, 2020, 500)
        column = pd.Series(year_data, name="Year")

        result = engine.evaluate(column, column_name="Year")

        assert result.action == PreprocessingAction.STANDARD_SCALE
        assert result.confidence >= 0.95
        assert "Year" in result.explanation

    def test_reviews_log_transform(self, engine):
        """Test that highly skewed count data (Reviews) gets log1p transform."""
        np.random.seed(42)
        # Use 550 samples to get sufficient skewness (>1.5)
        reviews_data = np.random.exponential(scale=20000, size=550).astype(int)
        column = pd.Series(reviews_data, name="Reviews")

        result = engine.evaluate(column, column_name="Reviews")

        assert result.action == PreprocessingAction.LOG1P_TRANSFORM
        assert result.confidence >= 0.85
        assert "skewness" in result.explanation.lower() or "log1p" in result.explanation.lower()

    def test_genre_onehot_encode(self, engine):
        """Test that low cardinality categorical (Genre) gets one-hot encoding."""
        np.random.seed(42)
        genre_data = np.random.choice(['Fiction', 'Non Fiction'], 500)
        column = pd.Series(genre_data, name="Genre")

        result = engine.evaluate(column, column_name="Genre")

        assert result.action == PreprocessingAction.ONEHOT_ENCODE
        assert result.confidence >= 0.95

    def test_user_rating_scaling(self, engine):
        """Test that User Rating columns get standard scaling."""
        np.random.seed(42)
        rating_data = np.random.uniform(3.8, 4.9, 500)
        column = pd.Series(rating_data, name="User Rating")

        result = engine.evaluate(column, column_name="User Rating")

        assert result.action == PreprocessingAction.STANDARD_SCALE
        assert result.confidence >= 0.90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
