"""
Track accuracy and usage by decision layer.

This proves which layers work best and identifies improvement areas.
Metrics are persisted to file for long-term tracking.
"""

from typing import Dict, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class LayerStats:
    """Statistics for a single decision layer."""

    total_decisions: int = 0
    correct_decisions: int = 0
    total_confidence: float = 0.0

    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage."""
        return (
            self.correct_decisions / self.total_decisions * 100
            if self.total_decisions > 0 else 0.0
        )

    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence."""
        return (
            self.total_confidence / self.total_decisions
            if self.total_decisions > 0 else 0.0
        )

    def usage_percentage(self, total: int) -> float:
        """Calculate usage percentage."""
        return (
            self.total_decisions / total * 100
            if total > 0 else 0.0
        )


class LayerMetrics:
    """
    Track metrics for each decision layer.

    Layers:
    - learned: Learned patterns from corrections
    - symbolic: Symbolic rule engine
    - neural: Neural oracle
    - meta_learning: Meta-learning heuristics
    - conservative_fallback: Ultra-conservative fallback (when all else fails)
    """

    def __init__(self, persistence_file: Optional[Path] = None):
        self.stats = {
            'learned': LayerStats(),
            'symbolic': LayerStats(),
            'neural': LayerStats(),
            'meta_learning': LayerStats(),
            'conservative_fallback': LayerStats()
        }
        self.persistence_file = persistence_file

        # Load existing stats if available
        if persistence_file and persistence_file.exists():
            self.load()

    def record_decision(
        self,
        layer: str,
        confidence: float,
        was_correct: Optional[bool] = None
    ):
        """
        Record a decision from a layer.

        Args:
            layer: Which layer made the decision
            confidence: Decision confidence (0-1)
            was_correct: Whether it was correct (None if unknown)
        """
        # Auto-create stats for unknown layers (defensive programming)
        if layer not in self.stats:
            self.stats[layer] = LayerStats()

        stats = self.stats[layer]
        stats.total_decisions += 1
        stats.total_confidence += confidence

        if was_correct is not None and was_correct:
            stats.correct_decisions += 1

    def record_correction(self, layer: str):
        """
        Record that a decision was corrected (was wrong).

        Args:
            layer: Which layer made the wrong decision
        """
        # Correction means it was wrong, so don't increment correct_decisions
        # This is tracked when record_decision is called with was_correct=False
        pass

    def get_summary(self) -> Dict:
        """Get summary of all layer metrics."""

        total_decisions = sum(s.total_decisions for s in self.stats.values())

        return {
            'total_decisions': total_decisions,
            'by_layer': {
                layer: {
                    'decisions': stats.total_decisions,
                    'usage_pct': stats.usage_percentage(total_decisions),
                    'accuracy_pct': stats.accuracy,
                    'avg_confidence': stats.avg_confidence
                }
                for layer, stats in self.stats.items()
            },
            'overall_accuracy': (
                sum(s.correct_decisions for s in self.stats.values()) /
                total_decisions * 100
                if total_decisions > 0 else 0
            )
        }

    def save(self):
        """Save metrics to file."""
        if not self.persistence_file:
            return

        self.persistence_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            layer: asdict(stats)
            for layer, stats in self.stats.items()
        }

        with open(self.persistence_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load metrics from file."""
        if not self.persistence_file or not self.persistence_file.exists():
            return

        with open(self.persistence_file, 'r') as f:
            data = json.load(f)

        for layer, stats_dict in data.items():
            if layer in self.stats:
                self.stats[layer] = LayerStats(**stats_dict)

    def reset(self):
        """Reset all metrics."""
        for layer in self.stats:
            self.stats[layer] = LayerStats()

        if self.persistence_file and self.persistence_file.exists():
            self.persistence_file.unlink()
