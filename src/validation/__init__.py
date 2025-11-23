"""
Validation and Metrics System for AURORA

This module tracks real usage metrics to prove the system's value:
- Performance benchmarking (time saved, quality improved)
- User feedback collection
- Comparison with alternatives
- Success metrics and analytics
"""

from .metrics_tracker import MetricsTracker, PerformanceMetrics
from .benchmarking import BenchmarkRunner, BenchmarkResult
from .feedback_collector import FeedbackCollector, UserFeedback
from .validation_dashboard import ValidationDashboard

__all__ = [
    'MetricsTracker',
    'PerformanceMetrics',
    'BenchmarkRunner',
    'BenchmarkResult',
    'FeedbackCollector',
    'UserFeedback',
    'ValidationDashboard'
]
