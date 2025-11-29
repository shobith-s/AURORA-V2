"""
Validation and Metrics System for AURORA

This module provides:
1. Performance metrics tracking (existing)
2. Statistical validation of preprocessing decisions (new)

Statistical validation uses industry-standard tests with proper citations:
- Anderson-Darling normality test (Anderson & Darling, 1952)
- IQR outlier detection (Tukey, 1977)
- Shannon Entropy (Shannon, 1948)
- Coefficient of Variation (Pearson, 1896)
"""

from .metrics_tracker import MetricsTracker, PerformanceMetrics
from .statistical_validator import StatisticalValidator
from .consistency_validator import ConsistencyValidator
from .validation_result import ValidationResult, ValidationMetric

__all__ = [
    'MetricsTracker',
    'PerformanceMetrics',
    'StatisticalValidator',
    'ConsistencyValidator',
    'ValidationResult',
    'ValidationMetric',
]
