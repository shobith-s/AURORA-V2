"""
Validation Dashboard - Aggregates all metrics for display

Combines:
- Performance metrics
- Benchmarking results
- User feedback
- Testimonials

Into a comprehensive dashboard view.
"""

from typing import Dict, Any, List
from .metrics_tracker import get_metrics_tracker, PerformanceMetrics
from .feedback_collector import get_feedback_collector
from .benchmarking import BenchmarkRunner, create_benchmark_summary


class ValidationDashboard:
    """
    Aggregates all validation data for display.
    """

    def __init__(self):
        self.metrics_tracker = get_metrics_tracker()
        self.feedback_collector = get_feedback_collector()

    def get_complete_dashboard(self) -> Dict[str, Any]:
        """
        Get all validation data in one comprehensive view.

        Returns dashboard data ready for frontend display.
        """
        # Get performance metrics
        perf_metrics = self.metrics_tracker.get_performance_metrics()
        recent_stats = self.metrics_tracker.get_recent_stats(hours=24)

        # Get feedback summary
        feedback_stats = self.feedback_collector.get_summary_stats()
        testimonials = self.feedback_collector.get_testimonials(limit=5)

        # Build dashboard
        dashboard = {
            "overview": {
                "total_decisions": perf_metrics.total_decisions,
                "total_users": perf_metrics.total_users,
                "total_sessions": perf_metrics.total_sessions,
                "time_saved_hours": perf_metrics.total_time_saved_hours,
                "average_confidence": perf_metrics.average_confidence,
            },

            "performance": {
                "acceptance_rate": perf_metrics.acceptance_rate,
                "override_rate": perf_metrics.override_rate,
                "average_processing_time_ms": perf_metrics.average_processing_time_ms,
                "average_time_saved_per_decision_seconds": perf_metrics.average_time_saved_per_decision_seconds,
                "time_vs_manual_improvement_percentage": perf_metrics.time_vs_manual_percentage,
            },

            "decision_sources": {
                "symbolic": perf_metrics.symbolic_decisions,
                "neural": perf_metrics.neural_decisions,
                "learned": perf_metrics.learned_decisions,
                "meta_learning": perf_metrics.meta_learning_decisions,
            },

            "user_satisfaction": {
                "average_rating": perf_metrics.average_user_rating,
                "explanation_helpfulness_rate": perf_metrics.explanation_helpfulness_rate,
                "learning_rate": perf_metrics.learning_rate,
                "recommendation_rate": perf_metrics.recommendation_rate,
            },

            "recent_activity": recent_stats,

            "feedback_summary": feedback_stats,

            "testimonials": testimonials,

            "key_stats": self._generate_key_stats(perf_metrics, feedback_stats),

            "proof_points": self._generate_proof_points(perf_metrics, feedback_stats)
        }

        return dashboard

    def _generate_key_stats(
        self,
        perf_metrics: PerformanceMetrics,
        feedback_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate key statistics for hero section."""
        stats = []

        # Time saved
        if perf_metrics.total_time_saved_hours > 0:
            stats.append({
                "label": "Total Time Saved",
                "value": f"{perf_metrics.total_time_saved_hours:.1f} hours",
                "icon": "clock",
                "color": "blue",
                "description": f"Across {perf_metrics.total_decisions} decisions"
            })

        # User satisfaction
        if feedback_stats.get("total_responses", 0) > 0:
            stats.append({
                "label": "User Satisfaction",
                "value": f"{feedback_stats['average_rating']:.1f}/5",
                "icon": "star",
                "color": "yellow",
                "description": f"From {feedback_stats['total_responses']} users"
            })

        # Recommendation rate
        if feedback_stats.get("would_recommend_percentage", 0) > 0:
            stats.append({
                "label": "Would Recommend",
                "value": f"{feedback_stats['would_recommend_percentage']:.0f}%",
                "icon": "thumbs-up",
                "color": "green",
                "description": "Users would recommend AURORA"
            })

        # Learning rate
        if perf_metrics.learning_rate > 0:
            stats.append({
                "label": "Users Learned",
                "value": f"{perf_metrics.learning_rate:.0f}%",
                "icon": "graduation-cap",
                "color": "purple",
                "description": "Reported learning something new"
            })

        # Acceptance rate
        if perf_metrics.acceptance_rate > 0:
            stats.append({
                "label": "Decision Acceptance",
                "value": f"{perf_metrics.acceptance_rate:.0f}%",
                "icon": "check",
                "color": "green",
                "description": "Decisions accepted without override"
            })

        # Average time saved
        if perf_metrics.average_time_saved_per_decision_seconds > 0:
            stats.append({
                "label": "Avg Time Saved",
                "value": f"{perf_metrics.average_time_saved_per_decision_seconds:.0f}s",
                "icon": "zap",
                "color": "yellow",
                "description": "Per preprocessing decision"
            })

        return stats

    def _generate_proof_points(
        self,
        perf_metrics: PerformanceMetrics,
        feedback_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate proof points for marketing/showcase."""
        points = []

        if perf_metrics.total_decisions > 0:
            points.append(
                f"✅ Processed {perf_metrics.total_decisions:,} preprocessing decisions "
                f"for {perf_metrics.total_users} users"
            )

        if perf_metrics.total_time_saved_hours > 0:
            points.append(
                f"⏱️ Saved {perf_metrics.total_time_saved_hours:.1f} hours of manual preprocessing work"
            )

        if perf_metrics.acceptance_rate > 70:
            points.append(
                f"🎯 {perf_metrics.acceptance_rate:.0f}% of decisions accepted without modification"
            )

        if perf_metrics.average_confidence >= 0.80:
            points.append(
                f"💪 Average confidence score of {perf_metrics.average_confidence:.0%} across all decisions"
            )

        if feedback_stats.get("total_responses", 0) > 0:
            if feedback_stats.get("would_recommend_percentage", 0) > 80:
                points.append(
                    f"⭐ {feedback_stats['would_recommend_percentage']:.0f}% of users would recommend AURORA"
                )

            if feedback_stats.get("learned_something_percentage", 0) > 70:
                points.append(
                    f"🎓 {feedback_stats['learned_something_percentage']:.0f}% of users learned something new"
                )

            if feedback_stats.get("average_explanation_quality", 0) >= 4.0:
                points.append(
                    f"📖 Explanation quality rated {feedback_stats['average_explanation_quality']:.1f}/5"
                )

        if perf_metrics.time_vs_manual_percentage > 0:
            points.append(
                f"🚀 {perf_metrics.time_vs_manual_percentage:.0f}% faster than manual preprocessing"
            )

        return points

    def get_metrics_for_export(self) -> str:
        """Export all metrics in markdown format for reports."""
        dashboard = self.get_complete_dashboard()

        md = "# AURORA Validation Report\n\n"
        md += f"**Generated:** {dashboard['recent_activity']['time_period_hours']}h data\n\n"

        md += "## Key Metrics\n\n"
        for stat in dashboard['key_stats']:
            md += f"- **{stat['label']}:** {stat['value']} - {stat['description']}\n"

        md += "\n## Proof Points\n\n"
        for point in dashboard['proof_points']:
            md += f"{point}\n"

        md += "\n## User Feedback Summary\n\n"
        feedback = dashboard['feedback_summary']
        if feedback.get('total_responses', 0) > 0:
            md += f"- Total Responses: {feedback['total_responses']}\n"
            md += f"- Average Rating: {feedback['average_rating']:.1f}/5\n"
            md += f"- Would Recommend: {feedback['would_recommend_percentage']:.0f}%\n"
            md += f"- Learned Something: {feedback['learned_something_percentage']:.0f}%\n"

        md += "\n## Testimonials\n\n"
        for t in dashboard['testimonials']:
            md += f"### {t['user_id']} ({t['rating']}/5 stars)\n"
            md += f"> {t['testimonial']}\n\n"
            md += f"*Use case: {t['use_case']}*\n\n"

        return md
