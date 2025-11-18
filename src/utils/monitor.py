"""
Performance monitoring and metrics tracking for AURORA.
Tracks latency, throughput, accuracy, and resource usage.
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
import numpy as np


@dataclass
class PerformanceMetrics:
    """Performance metrics for a component."""
    component: str
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_per_sec: float
    error_rate: float
    memory_mb: float
    cpu_percent: float
    total_calls: int
    successful_calls: int
    failed_calls: int


@dataclass
class DecisionMetrics:
    """Metrics for preprocessing decisions."""
    timestamp: str
    decision_id: str
    column_name: str
    action: str
    confidence: float
    source: str
    latency_ms: float
    num_rows: int


class PerformanceMonitor:
    """
    Real-time performance monitoring for AURORA components.
    Tracks latency, throughput, and resource usage.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of recent measurements to keep
        """
        self.window_size = window_size
        self.component_metrics: Dict[str, Dict[str, Any]] = {}
        self.decision_history: deque = deque(maxlen=window_size)
        self.lock = threading.Lock()

        # Initialize component trackers
        self._init_component('symbolic_engine')
        self._init_component('neural_oracle')
        self._init_component('pattern_learner')
        self._init_component('feature_extractor')
        self._init_component('overall_pipeline')

    def _init_component(self, component: str):
        """Initialize tracking for a component."""
        self.component_metrics[component] = {
            'latencies': deque(maxlen=self.window_size),
            'timestamps': deque(maxlen=self.window_size),
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'start_time': time.time()
        }

    def record_call(
        self,
        component: str,
        latency_ms: float,
        success: bool = True
    ):
        """
        Record a component call.

        Args:
            component: Component name
            latency_ms: Latency in milliseconds
            success: Whether the call succeeded
        """
        with self.lock:
            if component not in self.component_metrics:
                self._init_component(component)

            metrics = self.component_metrics[component]
            metrics['latencies'].append(latency_ms)
            metrics['timestamps'].append(time.time())
            metrics['total_calls'] += 1

            if success:
                metrics['successful_calls'] += 1
            else:
                metrics['failed_calls'] += 1

    def record_decision(
        self,
        decision_id: str,
        column_name: str,
        action: str,
        confidence: float,
        source: str,
        latency_ms: float,
        num_rows: int
    ):
        """Record a preprocessing decision."""
        with self.lock:
            decision = DecisionMetrics(
                timestamp=datetime.now().isoformat(),
                decision_id=decision_id,
                column_name=column_name,
                action=action,
                confidence=confidence,
                source=source,
                latency_ms=latency_ms,
                num_rows=num_rows
            )
            self.decision_history.append(decision)

    def get_component_metrics(self, component: str) -> Optional[PerformanceMetrics]:
        """Get metrics for a specific component."""
        with self.lock:
            if component not in self.component_metrics:
                return None

            metrics = self.component_metrics[component]
            latencies = list(metrics['latencies'])

            if not latencies:
                return PerformanceMetrics(
                    component=component,
                    avg_latency_ms=0.0,
                    p50_latency_ms=0.0,
                    p95_latency_ms=0.0,
                    p99_latency_ms=0.0,
                    throughput_per_sec=0.0,
                    error_rate=0.0,
                    memory_mb=0.0,
                    cpu_percent=0.0,
                    total_calls=0,
                    successful_calls=0,
                    failed_calls=0
                )

            # Calculate latency percentiles
            avg_latency = np.mean(latencies)
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)

            # Calculate throughput (calls per second)
            elapsed_time = time.time() - metrics['start_time']
            throughput = metrics['total_calls'] / elapsed_time if elapsed_time > 0 else 0

            # Calculate error rate
            error_rate = (
                metrics['failed_calls'] / metrics['total_calls']
                if metrics['total_calls'] > 0 else 0
            )

            # Get system resources
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=0.1)

            return PerformanceMetrics(
                component=component,
                avg_latency_ms=float(avg_latency),
                p50_latency_ms=float(p50),
                p95_latency_ms=float(p95),
                p99_latency_ms=float(p99),
                throughput_per_sec=float(throughput),
                error_rate=float(error_rate),
                memory_mb=float(memory_mb),
                cpu_percent=float(cpu_percent),
                total_calls=metrics['total_calls'],
                successful_calls=metrics['successful_calls'],
                failed_calls=metrics['failed_calls']
            )

    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get metrics for all components."""
        return {
            component: self.get_component_metrics(component)
            for component in self.component_metrics.keys()
        }

    def get_recent_decisions(self, limit: int = 100) -> List[DecisionMetrics]:
        """Get recent decisions."""
        with self.lock:
            decisions = list(self.decision_history)
            return decisions[-limit:] if limit else decisions

    def get_summary(self) -> Dict[str, Any]:
        """Get overall system summary."""
        all_metrics = self.get_all_metrics()

        # Overall statistics
        total_calls = sum(m.total_calls for m in all_metrics.values() if m)
        successful_calls = sum(m.successful_calls for m in all_metrics.values() if m)
        failed_calls = sum(m.failed_calls for m in all_metrics.values() if m)

        # Decision source breakdown
        decisions = list(self.decision_history)
        source_breakdown = {}
        if decisions:
            for decision in decisions:
                source_breakdown[decision.source] = source_breakdown.get(decision.source, 0) + 1

        # Confidence distribution
        confidence_avg = np.mean([d.confidence for d in decisions]) if decisions else 0.0
        confidence_std = np.std([d.confidence for d in decisions]) if decisions else 0.0

        # System resources
        memory_info = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()

        return {
            'overview': {
                'total_calls': total_calls,
                'successful_calls': successful_calls,
                'failed_calls': failed_calls,
                'success_rate': successful_calls / total_calls if total_calls > 0 else 0.0,
                'total_decisions': len(decisions)
            },
            'decision_sources': source_breakdown,
            'confidence_stats': {
                'avg': float(confidence_avg),
                'std': float(confidence_std)
            },
            'system_resources': {
                'memory_total_gb': memory_info.total / (1024 ** 3),
                'memory_available_gb': memory_info.available / (1024 ** 3),
                'memory_percent': memory_info.percent,
                'cpu_count': cpu_count,
                'cpu_percent': psutil.cpu_percent(interval=0.1)
            },
            'component_metrics': {
                name: asdict(metrics) if metrics else None
                for name, metrics in all_metrics.items()
            }
        }

    def reset(self):
        """Reset all metrics."""
        with self.lock:
            for component in self.component_metrics.keys():
                self._init_component(component)
            self.decision_history.clear()


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, monitor: PerformanceMonitor, component: str):
        """
        Initialize timer.

        Args:
            monitor: Performance monitor instance
            component: Component name
        """
        self.monitor = monitor
        self.component = component
        self.start_time = None
        self.success = True

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        elapsed_ms = (time.time() - self.start_time) * 1000
        self.success = exc_type is None
        self.monitor.record_call(self.component, elapsed_ms, self.success)
        return False  # Don't suppress exceptions


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def reset_monitor():
    """Reset the global monitor."""
    global _global_monitor
    _global_monitor = PerformanceMonitor()
