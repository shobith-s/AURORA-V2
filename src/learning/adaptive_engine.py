"""Adaptive learning engine for persistent correction storage and rule creation."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import random
from numbers import Number
from typing import Any, Dict, Optional, List

from sqlalchemy.orm import Session
from sqlalchemy import func

from ..database.connection import SessionLocal, init_db
from ..database.models import CorrectionRecord, LearnedRule
from .privacy import create_privacy_preserving_pattern

logger = logging.getLogger(__name__)


class AdaptiveLearningEngine:
    """Persist corrections and derive high-level learned rules with validation and A/B testing."""

    def __init__(
        self,
        db_url: str,
        min_support: int = 10,  # CHANGED: 5 → 10 corrections per pattern
        similarity_threshold: float = 0.85,
        epsilon: float = 1.0,
        validation_sample_size: int = 20,  # NEW: Minimum validations needed
        ab_test_min_decisions: int = 100,  # NEW: Minimum decisions for A/B test
        ab_test_success_threshold: float = 0.80,  # NEW: Accuracy threshold for promotion
    ) -> None:
        self.db_url = db_url
        self.min_support = min_support
        self.similarity_threshold = similarity_threshold
        self.epsilon = epsilon
        self.validation_sample_size = validation_sample_size
        self.ab_test_min_decisions = ab_test_min_decisions
        self.ab_test_success_threshold = ab_test_success_threshold

        # Ensure database schema exists
        try:
            init_db()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not initialize learning database: %s", exc)

    def _get_session(self) -> Session:
        return SessionLocal()

    def _identify_pattern_type(self, column_stats: Dict[str, Any]) -> str:
        """
        Identify pattern type from column statistics.
        
        Returns pattern type like: "numeric_high_skewness", "categorical_high_cardinality"
        """
        dtype = column_stats.get('dtype', column_stats.get('detected_dtype', 'unknown'))
        is_numeric = column_stats.get('is_numeric', False)

        # Numeric patterns
        if is_numeric or dtype in ['numeric', 'integer', 'float', 'int64', 'float64']:
            skewness = abs(column_stats.get('skewness', 0) or 0)
            null_pct = column_stats.get('null_pct', column_stats.get('null_percentage', 0))
            outlier_pct = column_stats.get('outlier_pct', column_stats.get('outlier_percentage', 0))

            if null_pct > 0.5:
                return 'numeric_high_nulls'
            elif null_pct > 0.1:
                return 'numeric_medium_nulls'
            elif skewness > 2.0:
                return 'numeric_high_skewness'
            elif skewness > 1.0:
                return 'numeric_medium_skewness'
            elif outlier_pct > 0.1:
                return 'numeric_many_outliers'
            else:
                return 'numeric_normal'

        # Categorical patterns
        elif column_stats.get('is_categorical') or dtype in ['categorical', 'object']:
            unique_ratio = column_stats.get('unique_ratio', 0)
            cardinality = column_stats.get('unique_count', 0)

            if unique_ratio > 0.9:
                return 'categorical_high_uniqueness'
            elif cardinality > 50:
                return 'categorical_high_cardinality'
            elif cardinality < 10:
                return 'categorical_low_cardinality'
            else:
                return 'categorical_medium_cardinality'

        return 'unknown'

    def record_correction(
        self,
        user_id: str,
        column_stats: Dict[str, Any],
        wrong_action: str,
        correct_action: str,
        confidence: Optional[float],
    ) -> Dict[str, Any]:
        """Store a correction and optionally create a learned rule with validation."""

        session = self._get_session()
        try:
            fingerprint = create_privacy_preserving_pattern(column_stats, privacy_level="medium")
            fingerprint = self._sanitize_for_json(fingerprint)
            pattern_hash = self._hash_fingerprint(fingerprint)
            pattern_type = self._identify_pattern_type(column_stats)

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
                pattern_type=pattern_type,
                fingerprint=fingerprint,
                recommended_action=correct_action,
            )

            return {"recorded": True, "pattern_hash": pattern_hash, "pattern_type": pattern_type, **rule_info}
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
        pattern_type: str,
        fingerprint: Dict[str, Any],
        recommended_action: str,
    ) -> Dict[str, Any]:
        """Create a learned rule when support threshold is reached, with validation."""

        try:
            existing_rule = session.query(LearnedRule).filter_by(rule_name=pattern_hash).first()
            if existing_rule:
                return {"new_rule_created": False, "rule_name": existing_rule.rule_name}

            # Count corrections for this PATTERN TYPE (not just pattern hash)
            pattern_type_corrections = (
                session.query(CorrectionRecord)
                .join(
                    session.query(CorrectionRecord.pattern_hash)
                    .filter(CorrectionRecord.pattern_hash == pattern_hash)
                    .subquery(),
                    CorrectionRecord.pattern_hash == pattern_hash
                )
                .count()
            )

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
                    "pattern_type": pattern_type,
                    "pattern_type_corrections": pattern_type_corrections,
                }

            # Create rule but start in A/B test mode
            rule = LearnedRule(
                user_id=user_id,
                rule_name=pattern_hash,
                pattern_template=fingerprint,
                recommended_action=recommended_action,
                base_confidence=self.similarity_threshold,
                support_count=support_count,
                created_at=time.time(),
                pattern_type=pattern_type,
                corrections_per_pattern=pattern_type_corrections,
                validation_successes=0,
                validation_failures=0,
                validation_passed=False,
                is_active=False,  # Start inactive until validated
                ab_test_group='treatment',  # Start in A/B test
                ab_test_start=time.time(),
                ab_test_decisions=0,
                ab_test_corrections=0,
                ab_test_accuracy=0.0,
                performance_score=0.5,
            )
            session.add(rule)
            session.commit()

            logger.info(f"Created new rule '{pattern_hash}' for pattern type '{pattern_type}' with {support_count} corrections. Starting A/B test.")

            return {
                "new_rule_created": True,
                "rule_name": rule.rule_name,
                "rule_confidence": rule.base_confidence,
                "rule_support": rule.support_count,
                "pattern_type": pattern_type,
                "ab_test_started": True,
                "validation_required": True,
            }
        except Exception as exc:  # pragma: no cover - defensive
            session.rollback()
            logger.error("Failed to create learned rule: %s", exc)
            return {"new_rule_created": False, "error": str(exc)}

    def validate_rule(self, rule_id: int, validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a learned rule on held-out data.
        
        Args:
            rule_id: ID of the rule to validate
            validation_data: List of {column_stats, expected_action} dicts
            
        Returns:
            Validation results with accuracy and pass/fail status
        """
        session = self._get_session()
        try:
            rule = session.query(LearnedRule).filter_by(id=rule_id).first()
            if not rule:
                return {"error": "Rule not found"}

            correct = 0
            total = len(validation_data)

            # FIXED: Use similarity matching instead of exact hash matching
            # Rules match on similarity during inference, so validation must too
            for item in validation_data:
                # Import here to avoid circular dependency
                from ..learning.rule_converter import compute_pattern_similarity

                # Check if pattern matches using similarity threshold (consistent with inference)
                similarity = compute_pattern_similarity(
                    item['column_stats'],
                    rule.pattern_template,
                    self.similarity_threshold
                )

                # Rule matches if similarity >= threshold AND action matches
                if similarity >= self.similarity_threshold and item['expected_action'] == rule.recommended_action:
                    correct += 1

            accuracy = correct / total if total > 0 else 0.0
            passed = accuracy >= self.ab_test_success_threshold and total >= self.validation_sample_size

            # Update rule
            rule.validation_accuracy = accuracy
            rule.validation_sample_size = total
            rule.validation_passed = passed
            if passed:
                rule.validation_successes += 1
            else:
                rule.validation_failures += 1
            rule.last_validation = time.time()

            session.commit()

            logger.info(f"Validated rule '{rule.rule_name}': {accuracy:.2%} accuracy on {total} samples. Passed: {passed}")

            return {
                "rule_id": rule_id,
                "accuracy": accuracy,
                "sample_size": total,
                "passed": passed,
                "threshold": self.ab_test_success_threshold,
            }
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to validate rule: {exc}")
            return {"error": str(exc)}
        finally:
            session.close()

    def record_ab_test_decision(self, rule_id: int, was_correct: bool) -> None:
        """Record an A/B test decision result."""
        session = self._get_session()
        try:
            rule = session.query(LearnedRule).filter_by(id=rule_id).first()
            if rule and rule.ab_test_group == 'treatment':
                rule.ab_test_decisions += 1
                if was_correct:
                    rule.ab_test_corrections += 1
                
                # Update accuracy
                if rule.ab_test_decisions > 0:
                    rule.ab_test_accuracy = rule.ab_test_corrections / rule.ab_test_decisions
                
                session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to record A/B test decision: {exc}")
        finally:
            session.close()

    def evaluate_ab_test(self, rule_id: int) -> Dict[str, Any]:
        """
        Evaluate A/B test results and determine if rule should be promoted.
        
        Returns:
            Evaluation results with recommendation to promote/reject
        """
        session = self._get_session()
        try:
            rule = session.query(LearnedRule).filter_by(id=rule_id).first()
            if not rule:
                return {"error": "Rule not found"}

            if rule.ab_test_group != 'treatment':
                return {"error": "Rule is not in A/B test"}

            # Check if enough decisions have been made
            if rule.ab_test_decisions < self.ab_test_min_decisions:
                return {
                    "rule_id": rule_id,
                    "status": "in_progress",
                    "decisions": rule.ab_test_decisions,
                    "required_decisions": self.ab_test_min_decisions,
                    "accuracy": rule.ab_test_accuracy,
                    "recommendation": "continue_testing",
                }

            # Evaluate performance
            should_promote = (
                rule.ab_test_accuracy >= self.ab_test_success_threshold and
                rule.validation_passed
            )

            return {
                "rule_id": rule_id,
                "status": "complete",
                "decisions": rule.ab_test_decisions,
                "accuracy": rule.ab_test_accuracy,
                "threshold": self.ab_test_success_threshold,
                "validation_passed": rule.validation_passed,
                "recommendation": "promote" if should_promote else "reject",
            }
        except Exception as exc:
            logger.error(f"Failed to evaluate A/B test: {exc}")
            return {"error": str(exc)}
        finally:
            session.close()

    def promote_to_production(self, rule_id: int) -> Dict[str, Any]:
        """Promote a successful A/B test rule to production."""
        session = self._get_session()
        try:
            rule = session.query(LearnedRule).filter_by(id=rule_id).first()
            if not rule:
                return {"error": "Rule not found"}

            evaluation = self.evaluate_ab_test(rule_id)
            if evaluation.get("recommendation") != "promote":
                return {"error": "Rule does not meet promotion criteria", "evaluation": evaluation}

            rule.ab_test_group = 'production'
            rule.is_active = True
            rule.performance_score = rule.ab_test_accuracy
            session.commit()

            logger.info(f"Promoted rule '{rule.rule_name}' to production with {rule.ab_test_accuracy:.2%} accuracy")

            return {
                "promoted": True,
                "rule_id": rule_id,
                "rule_name": rule.rule_name,
                "accuracy": rule.ab_test_accuracy,
            }
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to promote rule: {exc}")
            return {"error": str(exc)}
        finally:
            session.close()

    def reject_rule(self, rule_id: int) -> Dict[str, Any]:
        """Reject a failed A/B test rule."""
        session = self._get_session()
        try:
            rule = session.query(LearnedRule).filter_by(id=rule_id).first()
            if not rule:
                return {"error": "Rule not found"}

            rule.is_active = False
            rule.ab_test_group = 'rejected'
            session.commit()

            logger.info(f"Rejected rule '{rule.rule_name}' with {rule.ab_test_accuracy:.2%} accuracy")

            return {
                "rejected": True,
                "rule_id": rule_id,
                "rule_name": rule.rule_name,
                "accuracy": rule.ab_test_accuracy,
            }
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to reject rule: {exc}")
            return {"error": str(exc)}
        finally:
            session.close()

    def get_active_rules(self) -> List[LearnedRule]:
        """Get all active production rules."""
        session = self._get_session()
        try:
            return session.query(LearnedRule).filter_by(
                is_active=True,
                ab_test_group='production'
            ).all()
        finally:
            session.close()

    def get_ab_test_rules(self) -> List[LearnedRule]:
        """Get all rules currently in A/B testing."""
        session = self._get_session()
        try:
            return session.query(LearnedRule).filter_by(
                ab_test_group='treatment'
            ).all()
        finally:
            session.close()

    def should_use_ab_test_rule(self, rule: LearnedRule) -> bool:
        """Determine if this request should use the A/B test rule (50% split)."""
        if rule.ab_test_group != 'treatment':
            return False
        return random.random() < 0.5  # 50% traffic split

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

    def record_validation_result(
        self,
        rule_id: int,
        validation_passed: bool,
        validation_score: float
    ) -> None:
        """
        Record validation result for a learned rule (SIMPLE VERSION).
        
        Just tracks pass/fail rate - no complex logic.
        
        Args:
            rule_id: ID of the learned rule
            validation_passed: Whether validation passed
            validation_score: Validation score (0.0-1.0)
        """
        session = self._get_session()
        try:
            rule = session.query(LearnedRule).filter_by(id=rule_id).first()
            if rule:
                # Simple tracking: increment counters
                if validation_passed:
                    rule.validation_successes = (rule.validation_successes or 0) + 1
                else:
                    rule.validation_failures = (rule.validation_failures or 0) + 1
                
                # Update average validation score (simple moving average)
                total_validations = (rule.validation_successes or 0) + (rule.validation_failures or 0)
                if total_validations > 0:
                    current_avg = rule.validation_accuracy or 0.0
                    # Simple update: new_avg = (old_avg * (n-1) + new_score) / n
                    rule.validation_accuracy = (current_avg * (total_validations - 1) + validation_score) / total_validations
                
                session.commit()
                logger.debug(f"Recorded validation for rule {rule_id}: passed={validation_passed}, score={validation_score:.2f}")
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to record validation result: {exc}")
        finally:
            session.close()
