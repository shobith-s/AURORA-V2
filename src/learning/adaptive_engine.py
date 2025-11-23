"""Adaptive learning engine for persistent correction storage and rule creation."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from numbers import Number
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from ..database.connection import SessionLocal, init_db
from ..database.models import CorrectionRecord, LearnedRule
from .privacy import create_privacy_preserving_pattern

logger = logging.getLogger(__name__)


class AdaptiveLearningEngine:
    """Persist corrections and derive high-level learned rules."""

    def __init__(
        self,
        db_url: str,
        min_support: int = 5,
        similarity_threshold: float = 0.85,
        epsilon: float = 1.0,
    ) -> None:
        self.db_url = db_url
        self.min_support = min_support
        self.similarity_threshold = similarity_threshold
        self.epsilon = epsilon

        # Ensure database schema exists
        try:
            init_db()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not initialize learning database: %s", exc)

    def _get_session(self) -> Session:
        return SessionLocal()

    def record_correction(
        self,
        user_id: str,
        column_stats: Dict[str, Any],
        wrong_action: str,
        correct_action: str,
        confidence: Optional[float],
    ) -> Dict[str, Any]:
        """Store a correction and optionally create a learned rule."""

        session = self._get_session()
        try:
            fingerprint = create_privacy_preserving_pattern(column_stats, privacy_level="medium")
            fingerprint = self._sanitize_for_json(fingerprint)
            pattern_hash = self._hash_fingerprint(fingerprint)

            record = CorrectionRecord(
                user_id=user_id,
                timestamp=time.time(),
                pattern_hash=pattern_hash,
                statistical_fingerprint=fingerprint,
                wrong_action=wrong_action,
                correct_action=correct_action,
                system_confidence=float(confidence) if confidence is not None else None,
            )
            session.add(record)
            session.commit()

            rule_info = self._maybe_create_rule(
                session=session,
                user_id=user_id,
                pattern_hash=pattern_hash,
                fingerprint=fingerprint,
                recommended_action=correct_action,
            )

            return {"recorded": True, "pattern_hash": pattern_hash, **rule_info}
        except Exception as exc:  # pragma: no cover - defensive
            session.rollback()
            logger.error("Failed to record correction: %s", exc)
            return {"recorded": False, "error": str(exc)}
        finally:
            session.close()

    def _hash_fingerprint(self, fingerprint: Dict[str, Any]) -> str:
        serialized = json.dumps(fingerprint, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def _maybe_create_rule(
        self,
        session: Session,
        user_id: str,
        pattern_hash: str,
        fingerprint: Dict[str, Any],
        recommended_action: str,
    ) -> Dict[str, Any]:
        """Create a learned rule when support threshold is reached."""

        try:
            existing_rule = session.query(LearnedRule).filter_by(rule_name=pattern_hash).first()
            if existing_rule:
                return {"new_rule_created": False, "rule_name": existing_rule.rule_name}

            support_count = (
                session.query(CorrectionRecord)
                .filter(CorrectionRecord.pattern_hash == pattern_hash)
                .count()
            )

            if support_count < self.min_support:
                return {
                    "new_rule_created": False,
                    "correction_support": support_count,
                    "corrections_needed_for_production": max(self.min_support - support_count, 0),
                }

            rule = LearnedRule(
                user_id=user_id,
                rule_name=pattern_hash,
                pattern_template=fingerprint,
                recommended_action=recommended_action,
                base_confidence=self.similarity_threshold,
                support_count=support_count,
                created_at=time.time(),
                validation_successes=0,
                validation_failures=0,
                is_active=True,
                performance_score=0.5,
            )
            session.add(rule)
            session.commit()

            return {
                "new_rule_created": True,
                "rule_name": rule.rule_name,
                "rule_confidence": rule.base_confidence,
                "rule_support": rule.support_count,
            }
        except Exception as exc:  # pragma: no cover - defensive
            session.rollback()
            logger.error("Failed to create learned rule: %s", exc)
            return {"new_rule_created": False, "error": str(exc)}

    def _sanitize_for_json(self, value: Any) -> Any:
        """Recursively convert values to JSON-serializable types."""

        if isinstance(value, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_for_json(v) for v in value]
        if isinstance(value, tuple):
            return [self._sanitize_for_json(v) for v in value]
        if isinstance(value, bool) or value is None:
            return bool(value) if value is not None else None
        if isinstance(value, Number):
            return float(value) if isinstance(value, complex) else value
        # Fallback: coerce to string to guarantee serialization without raising
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)
