"""
Real-time Metrics Tracking System

Tracks actual usage to prove AURORA's value:
- Time saved vs manual preprocessing
- Decision quality metrics
- User satisfaction scores
- Learning effectiveness
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import time


@dataclass
class DecisionMetrics:
    """Metrics for a single preprocessing decision."""
    decision_id: str
    timestamp: datetime
    column_name: str
    action_taken: str
    confidence: float
    source: str  # symbolic, neural, learned
    processing_time_ms: float
    user_accepted: Optional[bool] = None  # Did user accept or override?
    user_rating: Optional[int] = None  # 1-5 stars
    alternative_chosen: Optional[str] = None  # If user overrode
    explanation_helpful: Optional[bool] = None
    time_saved_estimate_seconds: float = 0.0  # Estimated time saved vs manual


@dataclass
class SessionMetrics:
    """Metrics for a user session."""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    decisions_made: int = 0
    decisions_accepted: int = 0
    decisions_overridden: int = 0
    total_processing_time_ms: float = 0.0
    total_time_saved_seconds: float = 0.0
    user_satisfaction: Optional[int] = None  # 1-5 overall rating
    learned_something: bool = False
    would_recommend: Optional[bool] = None


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    # Usage stats
    total_decisions: int = 0
    total_users: int = 0
    total_sessions: int = 0

    # Quality metrics
    average_confidence: float = 0.0
    acceptance_rate: float = 0.0  # % of decisions accepted without override
    override_rate: float = 0.0

    # Time metrics
    average_processing_time_ms: float = 0.0
    total_time_saved_hours: float = 0.0
    average_time_saved_per_decision_seconds: float = 0.0

    # Decision source breakdown
    symbolic_decisions: int = 0
    neural_decisions: int = 0
    learned_decisions: int = 0

    # User satisfaction
    average_user_rating: float = 0.0
    explanation_helpfulness_rate: float = 0.0
    learning_rate: float = 0.0  # % of users who learned something
    recommendation_rate: float = 0.0  # % who would recommend

    # Comparison metrics
    time_vs_manual_percentage: float = 0.0  # How much faster than manual
    quality_vs_manual_percentage: float = 0.0


class MetricsTracker:
    """
    Tracks all metrics to prove AURORA's value.

    This provides the evidence that:
    - AURORA saves time
    - Decisions are high quality
    - Users are satisfied
    - Learning is effective
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/validation/metrics.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory tracking
        self.decisions: List[DecisionMetrics] = []
        self.sessions: List[SessionMetrics] = []
        self.current_session: Optional[SessionMetrics] = None

        # Load existing data
        self._load()

    def start_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Start tracking a new user session."""
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"

        self.current_session = SessionMetrics(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now()
        )
        return session_id

    def end_session(
        self,
        satisfaction: Optional[int] = None,
        learned_something: bool = False,
        would_recommend: Optional[bool] = None
    ):
        """End the current session and save metrics."""
        if self.current_session:
            self.current_session.end_time = datetime.now()
            self.current_session.user_satisfaction = satisfaction
            self.current_session.learned_something = learned_something
            self.current_session.would_recommend = would_recommend

            self.sessions.append(self.current_session)
            self.current_session = None
            self._save()

    def track_decision(
        self,
        decision_id: str,
        column_name: str,
        action_taken: str,
        confidence: float,
        source: str,
        processing_time_ms: float,
        estimated_manual_time_seconds: float = 60.0  # Default: 1 minute manual
    ) -> DecisionMetrics:
        """Track a preprocessing decision."""
        decision = DecisionMetrics(
            decision_id=decision_id,
            timestamp=datetime.now(),
            column_name=column_name,
            action_taken=action_taken,
            confidence=confidence,
            source=source,
            processing_time_ms=processing_time_ms,
            time_saved_estimate_seconds=estimated_manual_time_seconds - (processing_time_ms / 1000)
        )

        self.decisions.append(decision)

        # Update session metrics
        if self.current_session:
            self.current_session.decisions_made += 1
            self.current_session.total_processing_time_ms += processing_time_ms
            self.current_session.total_time_saved_seconds += decision.time_saved_estimate_seconds

        self._save()
        return decision

    def record_user_feedback(
        self,
        decision_id: str,
        accepted: bool,
        rating: Optional[int] = None,
        alternative_chosen: Optional[str] = None,
        explanation_helpful: Optional[bool] = None
    ):
        """Record user feedback on a decision."""
        # Find the decision
        decision = next((d for d in self.decisions if d.decision_id == decision_id), None)
        if decision:
            decision.user_accepted = accepted
            decision.user_rating = rating
            decision.alternative_chosen = alternative_chosen
            decision.explanation_helpful = explanation_helpful

            # Update session metrics
            if self.current_session:
                if accepted:
                    self.current_session.decisions_accepted += 1
                else:
                    self.current_session.decisions_overridden += 1

            self._save()

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculate aggregated performance metrics."""
        if not self.decisions:
            return PerformanceMetrics()

        # Calculate metrics
        total_decisions = len(self.decisions)
        unique_users = len(set(s.user_id for s in self.sessions))

        # Quality metrics
        decisions_with_feedback = [d for d in self.decisions if d.user_accepted is not None]
        accepted = sum(1 for d in decisions_with_feedback if d.user_accepted)
        acceptance_rate = (accepted / len(decisions_with_feedback) * 100) if decisions_with_feedback else 0.0

        avg_confidence = sum(d.confidence for d in self.decisions) / total_decisions

        # Time metrics
        avg_processing_time = sum(d.processing_time_ms for d in self.decisions) / total_decisions
        total_time_saved = sum(d.time_saved_estimate_seconds for d in self.decisions)
        avg_time_saved_per_decision = total_time_saved / total_decisions if total_decisions > 0 else 0.0

        # Source breakdown
        source_counts = {}
        for d in self.decisions:
            source_counts[d.source] = source_counts.get(d.source, 0) + 1

        # User satisfaction
        sessions_with_rating = [s for s in self.sessions if s.user_satisfaction is not None]
        avg_rating = (
            sum(s.user_satisfaction for s in sessions_with_rating) / len(sessions_with_rating)
            if sessions_with_rating else 0.0
        )

        decisions_with_expl_feedback = [d for d in self.decisions if d.explanation_helpful is not None]
        expl_helpful_rate = (
            sum(1 for d in decisions_with_expl_feedback if d.explanation_helpful) / len(decisions_with_expl_feedback) * 100
            if decisions_with_expl_feedback else 0.0
        )

        sessions_with_learning = [s for s in self.sessions if s.learned_something]
        learning_rate = len(sessions_with_learning) / len(self.sessions) * 100 if self.sessions else 0.0

        sessions_with_recommendation = [s for s in self.sessions if s.would_recommend is not None]
        recommendation_rate = (
            sum(1 for s in sessions_with_recommendation if s.would_recommend) / len(sessions_with_recommendation) * 100
            if sessions_with_recommendation else 0.0
        )

        # Estimate time saved vs manual
        # Assuming manual preprocessing takes ~60 seconds per column on average
        avg_manual_time_seconds = 60.0
        time_vs_manual_percentage = (
            ((avg_manual_time_seconds - avg_time_saved_per_decision) / avg_manual_time_seconds * 100)
            if avg_manual_time_seconds > 0 else 0.0
        )

        return PerformanceMetrics(
            total_decisions=total_decisions,
            total_users=unique_users,
            total_sessions=len(self.sessions),
            average_confidence=avg_confidence,
            acceptance_rate=acceptance_rate,
            override_rate=100 - acceptance_rate,
            average_processing_time_ms=avg_processing_time,
            total_time_saved_hours=total_time_saved / 3600,
            average_time_saved_per_decision_seconds=avg_time_saved_per_decision,
            symbolic_decisions=source_counts.get('symbolic', 0),
            neural_decisions=source_counts.get('neural', 0),
            learned_decisions=source_counts.get('learned', 0),
            average_user_rating=avg_rating,
            explanation_helpfulness_rate=expl_helpful_rate,
            learning_rate=learning_rate,
            recommendation_rate=recommendation_rate,
            time_vs_manual_percentage=time_vs_manual_percentage
        )

    def get_recent_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for recent activity."""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)

        recent_decisions = [
            d for d in self.decisions
            if d.timestamp.timestamp() > cutoff_time
        ]

        recent_sessions = [
            s for s in self.sessions
            if s.start_time.timestamp() > cutoff_time
        ]

        return {
            "time_period_hours": hours,
            "decisions_made": len(recent_decisions),
            "sessions": len(recent_sessions),
            "unique_users": len(set(s.user_id for s in recent_sessions)),
            "average_confidence": (
                sum(d.confidence for d in recent_decisions) / len(recent_decisions)
                if recent_decisions else 0.0
            ),
            "time_saved_hours": sum(d.time_saved_estimate_seconds for d in recent_decisions) / 3600
        }

    def _save(self):
        """Save metrics to disk."""
        data = {
            "decisions": [
                {
                    "decision_id": d.decision_id,
                    "timestamp": d.timestamp.isoformat(),
                    "column_name": d.column_name,
                    "action_taken": d.action_taken,
                    "confidence": d.confidence,
                    "source": d.source,
                    "processing_time_ms": d.processing_time_ms,
                    "user_accepted": d.user_accepted,
                    "user_rating": d.user_rating,
                    "alternative_chosen": d.alternative_chosen,
                    "explanation_helpful": d.explanation_helpful,
                    "time_saved_estimate_seconds": d.time_saved_estimate_seconds
                }
                for d in self.decisions
            ],
            "sessions": [
                {
                    "session_id": s.session_id,
                    "user_id": s.user_id,
                    "start_time": s.start_time.isoformat(),
                    "end_time": s.end_time.isoformat() if s.end_time else None,
                    "decisions_made": s.decisions_made,
                    "decisions_accepted": s.decisions_accepted,
                    "decisions_overridden": s.decisions_overridden,
                    "total_processing_time_ms": s.total_processing_time_ms,
                    "total_time_saved_seconds": s.total_time_saved_seconds,
                    "user_satisfaction": s.user_satisfaction,
                    "learned_something": s.learned_something,
                    "would_recommend": s.would_recommend
                }
                for s in self.sessions
            ]
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load existing metrics from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Load decisions
            self.decisions = [
                DecisionMetrics(
                    decision_id=d["decision_id"],
                    timestamp=datetime.fromisoformat(d["timestamp"]),
                    column_name=d["column_name"],
                    action_taken=d["action_taken"],
                    confidence=d["confidence"],
                    source=d["source"],
                    processing_time_ms=d["processing_time_ms"],
                    user_accepted=d.get("user_accepted"),
                    user_rating=d.get("user_rating"),
                    alternative_chosen=d.get("alternative_chosen"),
                    explanation_helpful=d.get("explanation_helpful"),
                    time_saved_estimate_seconds=d.get("time_saved_estimate_seconds", 0.0)
                )
                for d in data.get("decisions", [])
            ]

            # Load sessions
            self.sessions = [
                SessionMetrics(
                    session_id=s["session_id"],
                    user_id=s["user_id"],
                    start_time=datetime.fromisoformat(s["start_time"]),
                    end_time=datetime.fromisoformat(s["end_time"]) if s.get("end_time") else None,
                    decisions_made=s.get("decisions_made", 0),
                    decisions_accepted=s.get("decisions_accepted", 0),
                    decisions_overridden=s.get("decisions_overridden", 0),
                    total_processing_time_ms=s.get("total_processing_time_ms", 0.0),
                    total_time_saved_seconds=s.get("total_time_saved_seconds", 0.0),
                    user_satisfaction=s.get("user_satisfaction"),
                    learned_something=s.get("learned_something", False),
                    would_recommend=s.get("would_recommend")
                )
                for s in data.get("sessions", [])
            ]
        except Exception as e:
            print(f"Error loading metrics: {e}")


# Singleton instance
_metrics_tracker: Optional[MetricsTracker] = None


def get_metrics_tracker() -> MetricsTracker:
    """Get the global metrics tracker instance."""
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker()
    return _metrics_tracker
