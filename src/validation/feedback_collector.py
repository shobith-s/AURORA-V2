"""
User Feedback Collection System

Collects structured feedback to prove value:
- Decision acceptance/rejection
- User ratings
- Testimonials
- Learning outcomes
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
from pathlib import Path


@dataclass
class UserFeedback:
    """Structured user feedback."""
    feedback_id: str
    user_id: str
    timestamp: datetime

    # Overall experience
    overall_rating: int  # 1-5 stars
    would_recommend: bool
    learned_something: bool

    # Specific feedback
    time_saved_perception: str  # "saved_time", "same_time", "took_longer"
    ease_of_use: int  # 1-5
    explanation_quality: int  # 1-5
    confidence_in_decisions: int  # 1-5

    # Open-ended
    what_worked_well: str
    what_needs_improvement: str
    use_case: str  # What were they trying to do?

    # Testimonial
    willing_to_be_quoted: bool = False
    testimonial: Optional[str] = None


class FeedbackCollector:
    """Collects and analyzes user feedback."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/validation/feedback.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.feedback_entries: List[UserFeedback] = []
        self._load()

    def collect_feedback(
        self,
        user_id: str,
        overall_rating: int,
        would_recommend: bool,
        learned_something: bool,
        time_saved_perception: str,
        ease_of_use: int,
        explanation_quality: int,
        confidence_in_decisions: int,
        what_worked_well: str,
        what_needs_improvement: str,
        use_case: str,
        willing_to_be_quoted: bool = False,
        testimonial: Optional[str] = None
    ) -> UserFeedback:
        """Collect structured feedback from a user."""
        import time

        feedback = UserFeedback(
            feedback_id=f"feedback_{int(time.time() * 1000)}",
            user_id=user_id,
            timestamp=datetime.now(),
            overall_rating=overall_rating,
            would_recommend=would_recommend,
            learned_something=learned_something,
            time_saved_perception=time_saved_perception,
            ease_of_use=ease_of_use,
            explanation_quality=explanation_quality,
            confidence_in_decisions=confidence_in_decisions,
            what_worked_well=what_worked_well,
            what_needs_improvement=what_needs_improvement,
            use_case=use_case,
            willing_to_be_quoted=willing_to_be_quoted,
            testimonial=testimonial
        )

        self.feedback_entries.append(feedback)
        self._save()
        return feedback

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from feedback."""
        if not self.feedback_entries:
            return {
                "total_responses": 0,
                "message": "No feedback collected yet"
            }

        total = len(self.feedback_entries)

        return {
            "total_responses": total,
            "average_rating": sum(f.overall_rating for f in self.feedback_entries) / total,
            "would_recommend_percentage": sum(1 for f in self.feedback_entries if f.would_recommend) / total * 100,
            "learned_something_percentage": sum(1 for f in self.feedback_entries if f.learned_something) / total * 100,
            "time_saved_perception": {
                "saved_time": sum(1 for f in self.feedback_entries if f.time_saved_perception == "saved_time") / total * 100,
                "same_time": sum(1 for f in self.feedback_entries if f.time_saved_perception == "same_time") / total * 100,
                "took_longer": sum(1 for f in self.feedback_entries if f.time_saved_perception == "took_longer") / total * 100,
            },
            "average_ease_of_use": sum(f.ease_of_use for f in self.feedback_entries) / total,
            "average_explanation_quality": sum(f.explanation_quality for f in self.feedback_entries) / total,
            "average_confidence": sum(f.confidence_in_decisions for f in self.feedback_entries) / total,
            "testimonials_available": sum(1 for f in self.feedback_entries if f.willing_to_be_quoted and f.testimonial)
        }

    def get_testimonials(self, limit: int = 5) -> List[Dict[str, str]]:
        """Get quotable testimonials."""
        testimonials = [
            {
                "user_id": f.user_id,
                "rating": f.overall_rating,
                "testimonial": f.testimonial,
                "use_case": f.use_case,
                "timestamp": f.timestamp.strftime("%Y-%m-%d")
            }
            for f in self.feedback_entries
            if f.willing_to_be_quoted and f.testimonial
        ]

        # Sort by rating (highest first)
        testimonials.sort(key=lambda x: x["rating"], reverse=True)

        return testimonials[:limit]

    def _save(self):
        """Save feedback to disk."""
        data = {
            "feedback": [
                {
                    "feedback_id": f.feedback_id,
                    "user_id": f.user_id,
                    "timestamp": f.timestamp.isoformat(),
                    "overall_rating": f.overall_rating,
                    "would_recommend": f.would_recommend,
                    "learned_something": f.learned_something,
                    "time_saved_perception": f.time_saved_perception,
                    "ease_of_use": f.ease_of_use,
                    "explanation_quality": f.explanation_quality,
                    "confidence_in_decisions": f.confidence_in_decisions,
                    "what_worked_well": f.what_worked_well,
                    "what_needs_improvement": f.what_needs_improvement,
                    "use_case": f.use_case,
                    "willing_to_be_quoted": f.willing_to_be_quoted,
                    "testimonial": f.testimonial
                }
                for f in self.feedback_entries
            ]
        }

        with open(self.storage_path, 'w') as file:
            json.dump(data, file, indent=2)

    def _load(self):
        """Load existing feedback from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, 'r') as file:
                data = json.load(file)

            self.feedback_entries = [
                UserFeedback(
                    feedback_id=f["feedback_id"],
                    user_id=f["user_id"],
                    timestamp=datetime.fromisoformat(f["timestamp"]),
                    overall_rating=f["overall_rating"],
                    would_recommend=f["would_recommend"],
                    learned_something=f["learned_something"],
                    time_saved_perception=f["time_saved_perception"],
                    ease_of_use=f["ease_of_use"],
                    explanation_quality=f["explanation_quality"],
                    confidence_in_decisions=f["confidence_in_decisions"],
                    what_worked_well=f["what_worked_well"],
                    what_needs_improvement=f["what_needs_improvement"],
                    use_case=f["use_case"],
                    willing_to_be_quoted=f.get("willing_to_be_quoted", False),
                    testimonial=f.get("testimonial")
                )
                for f in data.get("feedback", [])
            ]
        except Exception as e:
            print(f"Error loading feedback: {e}")


# Singleton instance
_feedback_collector: Optional[FeedbackCollector] = None


def get_feedback_collector() -> FeedbackCollector:
    """Get the global feedback collector instance."""
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector()
    return _feedback_collector
