"""
Privacy-preserving utilities for pattern learning.
Implements differential privacy, anonymization, and secure hashing.
"""

import hashlib
import numpy as np
from typing import Any, Dict, Optional, Tuple, List


class PrivacyBudget:
    """Track and manage differential privacy budget."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize privacy budget.
        
        Args:
            epsilon: Privacy parameter (smaller = more private)
            delta: Failure probability
        """
        self.epsilon = epsilon
        self.delta = delta
        self.spent = 0.0
    
    def can_spend(self, amount: float) -> bool:
        """
        Check if budget can be spent.
        
        Args:
            amount: Amount to spend
            
        Returns:
            True if budget is available
        """
        return (self.spent + amount) <= self.epsilon
    
    def spend(self, amount: float) -> bool:
        """
        Spend privacy budget.
        
        Args:
            amount: Amount to spend
            
        Returns:
            True if spent successfully, False if insufficient budget
        """
        if self.can_spend(amount):
            self.spent += amount
            return True
        return False
    
    def remaining(self) -> float:
        """Get remaining budget."""
        return max(0.0, self.epsilon - self.spent)


class DifferentialPrivacy:
    """Implement differential privacy mechanisms."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy.
        
        Args:
            epsilon: Privacy parameter
            delta: Failure probability
        """
        self.budget = PrivacyBudget(epsilon, delta)
        self.anonymizer = AnonymizationUtils()
    
    def add_noise(self, value: float, sensitivity: float = 1.0) -> Optional[float]:
        """
        Add Laplace noise to a value.
        
        Args:
            value: Original value
            sensitivity: Sensitivity of the query
            
        Returns:
            Noisy value or None if budget exhausted
        """
        epsilon_cost = 0.1  # Cost per query
        
        if not self.budget.can_spend(epsilon_cost):
            return None
        
        self.budget.spend(epsilon_cost)
        return self.anonymizer.add_laplace_noise(value, self.budget.epsilon, sensitivity)
    
    def add_laplacian_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """
        Add Laplace noise to a value.
        
        Args:
            value: Original value
            sensitivity: Sensitivity of the query
            
        Returns:
            Noisy value
        """
        return self.anonymizer.add_laplace_noise(value, self.budget.epsilon, sensitivity)
    
    def add_gaussian_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """
        Add Gaussian noise to a value (for (epsilon, delta)-differential privacy).
        
        Args:
            value: Original value
            sensitivity: Sensitivity of the query
            
        Returns:
            Noisy value
        """
        # Calculate noise scale for Gaussian mechanism
        # sigma >= sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.budget.delta)) / self.budget.epsilon
        noise = np.random.normal(0, sigma)
        return value + noise
    
    def privatize_count(self, count: int, sensitivity: float = 1.0) -> int:
        """
        Privatize a count value.
        
        Args:
            count: True count
            sensitivity: Sensitivity (default 1 for counts)
            
        Returns:
            Noisy count (non-negative integer)
        """
        noisy = self.add_laplacian_noise(float(count), sensitivity)
        return max(0, int(round(noisy)))
    
    def privatize_mean(self, values: np.ndarray, value_range: Tuple[float, float]) -> float:
        """
        Privatize a mean value.
        
        Args:
            values: Array of values
            value_range: (min, max) range of values
            
        Returns:
            Noisy mean
        """
        true_mean = np.mean(values)
        min_val, max_val = value_range
        sensitivity = (max_val - min_val) / len(values)
        return self.add_laplacian_noise(true_mean, sensitivity)
    
    def privatize_histogram(self, histogram: Dict[str, int]) -> Dict[str, int]:
        """
        Privatize a histogram.
        
        Args:
            histogram: Dictionary mapping categories to counts
            
        Returns:
            Dictionary with noisy counts
        """
        noisy_histogram = {}
        for key, count in histogram.items():
            noisy_histogram[key] = self.privatize_count(count)
        return noisy_histogram
    
    def add_noise_to_gradient(self, gradient: np.ndarray, clip_norm: float = 1.0) -> np.ndarray:
        """
        Add noise to gradient with clipping for DP-SGD.
        
        Args:
            gradient: Original gradient
            clip_norm: Clipping norm
            
        Returns:
            Noisy clipped gradient
        """
        # Clip gradient
        norm = np.linalg.norm(gradient)
        if norm > clip_norm:
            gradient = gradient * (clip_norm / norm)
        
        # Add Gaussian noise
        noise_scale = clip_norm * np.sqrt(2 * np.log(1.25 / self.budget.delta)) / self.budget.epsilon
        noise = np.random.normal(0, noise_scale, gradient.shape)
        
        return gradient + noise
    
    def privatize_statistics(
        self,
        statistics: Dict[str, float],
        sensitivity: float = 1.0
    ) -> Dict[str, float]:
        """
        Add noise to multiple statistics.
        
        Args:
            statistics: Dictionary of statistics
            sensitivity: Sensitivity per statistic
            
        Returns:
            Dictionary of privatized statistics
        """
        privatized = {}
        
        for key, value in statistics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                noisy = self.add_noise(value, sensitivity)
                if noisy is not None:
                    privatized[key] = noisy
                else:
                    privatized[key] = value  # Return original if budget exhausted
            else:
                privatized[key] = value
        
        return privatized


class AnonymizationUtils:
    """Utilities for anonymizing data and creating privacy-preserving patterns."""

    @staticmethod
    def hash_value(value: str, algorithm: str = 'sha256', salt: Optional[str] = None) -> str:
        """
        Create a cryptographic hash of a value.

        Args:
            value: Value to hash
            algorithm: Hash algorithm (sha256, sha512, md5)
            salt: Optional salt for hashing

        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.new(algorithm)
        if salt:
            hasher.update(salt.encode('utf-8'))
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
    
    def generalize_numeric(
        self,
        value: float,
        precision: int = 2
    ) -> float:
        """
        Alias for generalize_numeric_value for backward compatibility.

        Args:
            value: Original value
            precision: Number of decimal places

        Returns:
            Generalized value
        """
        return self.generalize_numeric_value(value, precision)

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
    
    def k_anonymize_categories(
        self,
        categories: List[str],
        k: int = 5,
        suppress_threshold: int = 2
    ) -> List[str]:
        """
        Apply k-anonymity to categories with suppression.

        Args:
            categories: List of categorical values
            k: Minimum group size
            suppress_threshold: Threshold below which to suppress

        Returns:
            Anonymized categories
        """
        from collections import Counter

        counts = Counter(categories)
        anonymized = []

        for category in categories:
            if counts[category] < suppress_threshold:
                anonymized.append('*')  # Suppress rare values
            elif counts[category] < k:
                anonymized.append('OTHER')  # Generalize
            else:
                anonymized.append(category)

        return anonymized
    
    def remove_identifiers(
        self,
        data: Dict[str, Any],
        identifier_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Remove identifier fields from data.

        Args:
            data: Original data dictionary
            identifier_keys: List of keys to remove (auto-detect if None)

        Returns:
            Data with identifiers removed
        """
        if identifier_keys is None:
            # Common identifier patterns
            identifier_keys = ['id', 'user_id', 'email', 'name', 'username', 
                             'phone', 'ssn', 'address', 'ip_address']
        
        cleaned = {}
        for key, value in data.items():
            # Check if key contains identifier pattern
            is_identifier = any(pattern in key.lower() for pattern in identifier_keys)
            if not is_identifier:
                cleaned[key] = value
        
        return cleaned


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
