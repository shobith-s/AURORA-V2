"""
Minimal Synthetic Data Generator for Neural Oracle Training

Generates edge case data for training the neural oracle on ambiguous preprocessing scenarios.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from ..core.actions import PreprocessingAction


class SyntheticDataGenerator:
    """Generate synthetic training data for neural oracle."""

    def __init__(self, seed: int = 42):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)

    def generate_training_data(
        self,
        n_samples: int = 5000,
        ambiguous_only: bool = True
    ) -> Tuple[List[pd.Series], List[PreprocessingAction], List[str]]:
        """
        Generate training data for neural oracle.

        Args:
            n_samples: Number of samples to generate
            ambiguous_only: If True, generate only ambiguous edge cases

        Returns:
            Tuple of (columns, labels, difficulties)
        """
        columns = []
        labels = []
        difficulties = []

        # Edge case generators
        generators = [
            self._generate_skewed_data,
            self._generate_outlier_data,
            self._generate_mixed_type_data,
            self._generate_sparse_data,
            self._generate_high_cardinality_data,
            self._generate_bimodal_data,
            self._generate_zero_inflated_data,
        ]

        for _ in range(n_samples):
            # Pick random generator
            generator = self.rng.choice(generators)
            col, label, difficulty = generator()

            columns.append(col)
            labels.append(label)
            difficulties.append(difficulty)

        return columns, labels, difficulties

    def _generate_skewed_data(self) -> Tuple[pd.Series, PreprocessingAction, str]:
        """Generate highly skewed data (needs log transform)."""
        size = self.rng.randint(100, 1000)
        # Exponential distribution = high positive skew
        data = self.rng.exponential(scale=10, size=size)
        return pd.Series(data), PreprocessingAction.LOG_TRANSFORM, "hard"

    def _generate_outlier_data(self) -> Tuple[pd.Series, PreprocessingAction, str]:
        """Generate data with outliers (needs clipping or robust scaling)."""
        size = self.rng.randint(100, 1000)
        # Normal data with outliers
        data = self.rng.normal(loc=50, scale=10, size=size)
        # Add outliers
        n_outliers = int(size * 0.05)
        outlier_idx = self.rng.choice(size, n_outliers, replace=False)
        data[outlier_idx] = self.rng.uniform(200, 500, n_outliers)

        action = self.rng.choice([
            PreprocessingAction.CLIP_OUTLIERS,
            PreprocessingAction.ROBUST_SCALE
        ])
        return pd.Series(data), action, "medium"

    def _generate_mixed_type_data(self) -> Tuple[pd.Series, PreprocessingAction, str]:
        """Generate mixed numeric/string data (needs cleaning)."""
        size = self.rng.randint(100, 1000)
        data = []
        for _ in range(size):
            if self.rng.random() < 0.8:
                data.append(str(self.rng.randint(1, 100)))
            else:
                data.append(f"val_{self.rng.randint(1, 10)}")
        return pd.Series(data), PreprocessingAction.PARSE_NUMERIC, "hard"

    def _generate_sparse_data(self) -> Tuple[pd.Series, PreprocessingAction, str]:
        """Generate sparse data with many nulls."""
        size = self.rng.randint(100, 1000)
        data = self.rng.normal(loc=50, scale=10, size=size)
        # Make 30-60% null
        null_pct = self.rng.uniform(0.3, 0.6)
        null_idx = self.rng.choice(size, int(size * null_pct), replace=False)
        data = data.astype(object)
        data[null_idx] = None

        action = self.rng.choice([
            PreprocessingAction.DROP_COLUMN,
            PreprocessingAction.FILL_NULL_MEAN
        ])
        return pd.Series(data), action, "medium"

    def _generate_high_cardinality_data(self) -> Tuple[pd.Series, PreprocessingAction, str]:
        """Generate categorical with high cardinality (needs special encoding)."""
        size = self.rng.randint(100, 1000)
        n_categories = self.rng.randint(50, 200)
        data = [f"cat_{self.rng.randint(0, n_categories)}" for _ in range(size)]

        action = self.rng.choice([
            PreprocessingAction.HASH_ENCODE,
            PreprocessingAction.FREQUENCY_ENCODE,
            PreprocessingAction.TARGET_ENCODE
        ])
        return pd.Series(data), action, "hard"

    def _generate_bimodal_data(self) -> Tuple[pd.Series, PreprocessingAction, str]:
        """Generate bimodal distribution (might need binning)."""
        size = self.rng.randint(100, 1000)
        # Mix two normal distributions
        data1 = self.rng.normal(loc=20, scale=5, size=size//2)
        data2 = self.rng.normal(loc=80, scale=5, size=size - size//2)
        data = np.concatenate([data1, data2])
        self.rng.shuffle(data)

        action = self.rng.choice([
            PreprocessingAction.BINNING_EQUAL_FREQ,
            PreprocessingAction.STANDARD_SCALE
        ])
        return pd.Series(data), action, "medium"

    def _generate_zero_inflated_data(self) -> Tuple[pd.Series, PreprocessingAction, str]:
        """Generate zero-inflated data (many zeros plus normal values)."""
        size = self.rng.randint(100, 1000)
        data = np.zeros(size)
        # 20-40% non-zero
        non_zero_pct = self.rng.uniform(0.2, 0.4)
        non_zero_idx = self.rng.choice(size, int(size * non_zero_pct), replace=False)
        data[non_zero_idx] = self.rng.exponential(scale=10, size=len(non_zero_idx))

        return pd.Series(data), PreprocessingAction.LOG_TRANSFORM, "hard"
