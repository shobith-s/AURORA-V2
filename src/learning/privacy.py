"""
Privacy-preserving utilities for pattern learning.
Implements differential privacy, anonymization, and secure hashing.
"""

import hashlib
import numpy as np
from typing import Any, Dict, Optional, Tuple, List


class AnonymizationUtils:
    """Utilities for anonymizing data and creating privacy-preserving patterns."""

    @staticmethod
    def hash_value(value: str, algorithm: str = 'sha256') -> str:
        """
        Create a cryptographic hash of a value.

        Args:
            value: Value to hash
            algorithm: Hash algorithm (sha256, sha512, md5)

        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.new(algorithm)
        hasher.update(value.encode('utf-8'))
        return hasher.hexdigest()

    def discretize_value(
        self,
        value: float,
        bins: int = 5,
        value_range: Optional[Tuple[float, float]] = None
    ) -> int:
        """
        Discretize a continuous value into bins.

        Args:
            value: Value to discretize
            bins: Number of bins
            value_range: (min, max) range for binning

        Returns:
            Bin index (0 to bins-1)
        """
        # Handle NaN/None values (common for categorical columns)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0  # Default to first bin for undefined values

        if value_range is None:
            # Auto-range
            value_range = (value - abs(value), value + abs(value))

        min_val, max_val = value_range

        # Clip value to range
        value = np.clip(value, min_val, max_val)

        # Calculate bin
        if max_val == min_val:
            return 0

        bin_width = (max_val - min_val) / bins
        bin_index = int((value - min_val) / bin_width)

        # Handle edge case where value == max_val
        if bin_index >= bins:
            bin_index = bins - 1

        return bin_index

    def add_laplace_noise(
        self,
        value: float,
        epsilon: float = 1.0,
        sensitivity: float = 1.0
    ) -> float:
        """
        Add Laplace noise for differential privacy.

        Args:
            value: Original value
            epsilon: Privacy parameter (smaller = more private)
            sensitivity: Sensitivity of the query

        Returns:
            Value with noise added
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return value + noise

    def generalize_numeric_value(
        self,
        value: float,
        precision: int = 2
    ) -> float:
        """
        Generalize numeric value by reducing precision.

        Args:
            value: Original value
            precision: Number of decimal places

        Returns:
            Generalized value
        """
        return round(value, precision)

    def k_anonymize_categorical(
        self,
        values: List[str],
        k: int = 5
    ) -> List[str]:
        """
        Apply k-anonymity to categorical values.

        Args:
            values: List of categorical values
            k: Minimum group size

        Returns:
            Anonymized values (rare values replaced with '*')
        """
        from collections import Counter

        counts = Counter(values)
        anonymized = []

        for value in values:
            if counts[value] < k:
                anonymized.append('*')  # Suppress rare values
            else:
                anonymized.append(value)

        return anonymized


def create_privacy_preserving_pattern(
    data: Dict[str, Any],
    privacy_level: str = 'high'
) -> Dict[str, Any]:
    """
    Create a privacy-preserving pattern from raw data.

    Args:
        data: Original data dictionary
        privacy_level: 'low', 'medium', or 'high'

    Returns:
        Anonymized pattern dictionary
    """
    anonymizer = AnonymizationUtils()
    pattern = {}

    # Privacy parameters based on level
    privacy_params = {
        'low': {'epsilon': 2.0, 'precision': 3, 'bins': 10},
        'medium': {'epsilon': 1.0, 'precision': 2, 'bins': 5},
        'high': {'epsilon': 0.5, 'precision': 1, 'bins': 3}
    }

    params = privacy_params.get(privacy_level, privacy_params['medium'])

    # Process numeric statistics with noise
    for key in ['mean', 'median', 'std', 'skewness', 'kurtosis']:
        if key in data and data[key] is not None:
            # Add differential privacy noise
            noisy_value = anonymizer.add_laplace_noise(
                data[key],
                epsilon=params['epsilon']
            )
            # Generalize precision
            pattern[f'{key}_anonymous'] = anonymizer.generalize_numeric_value(
                noisy_value,
                precision=params['precision']
            )

    # Discretize values
    for key in ['null_pct', 'unique_ratio']:
        if key in data and data[key] is not None:
            pattern[f'{key}_bucket'] = anonymizer.discretize_value(
                data[key],
                bins=params['bins'],
                value_range=(0, 1)
            )

    # Hash identifiers
    for key in ['column_name', 'table_name', 'dataset_id']:
        if key in data and data[key]:
            pattern[f'{key}_hash'] = anonymizer.hash_value(str(data[key]))[:16]

    # Preserve boolean flags (low privacy risk)
    for key in ['is_numeric', 'is_categorical', 'is_temporal']:
        if key in data:
            pattern[key] = data[key]

    return pattern


def compute_differential_privacy_budget(
    num_queries: int,
    epsilon_per_query: float = 0.1
) -> float:
    """
    Compute total privacy budget consumed.

    Args:
        num_queries: Number of queries made
        epsilon_per_query: Privacy budget per query

    Returns:
        Total epsilon consumed
    """
    return num_queries * epsilon_per_query


def check_privacy_budget(
    consumed_epsilon: float,
    max_epsilon: float = 10.0
) -> bool:
    """
    Check if privacy budget is exhausted.

    Args:
        consumed_epsilon: Total epsilon consumed
        max_epsilon: Maximum allowed epsilon

    Returns:
        True if budget is available, False if exhausted
    """
    return consumed_epsilon < max_epsilon
