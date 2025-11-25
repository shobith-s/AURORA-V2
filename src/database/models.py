"""
SQLAlchemy models for AURORA adaptive learning system.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class CorrectionRecord(Base):
    """Stores user corrections (privacy-preserved)."""
    __tablename__ = 'corrections'

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    timestamp = Column(Float, index=True)

    # Pattern identification (hashed, not raw data)
    pattern_hash = Column(String(32), index=True)
    statistical_fingerprint = Column(JSON)  # Anonymized stats only

    # Actions
    wrong_action = Column(String)
    correct_action = Column(String)
    system_confidence = Column(Float)

    # Validation tracking
    was_validated = Column(Boolean, default=False)
    validation_result = Column(Boolean, nullable=True)


class LearnedRule(Base):
    """Rules learned from correction patterns with validation and A/B testing."""
    __tablename__ = 'learned_rules'

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    rule_name = Column(String, unique=True, index=True)

    # Rule definition
    pattern_template = Column(JSON)
    recommended_action = Column(String)
    base_confidence = Column(Float)

    # Learning metadata
    support_count = Column(Integer)
    created_at = Column(Float)

    # Pattern type tracking (NEW)
    pattern_type = Column(String, index=True)  # e.g., 'numeric_high_skewness'
    corrections_per_pattern = Column(Integer, default=0)

    # Validation tracking (ENHANCED)
    validation_successes = Column(Integer, default=0)
    validation_failures = Column(Integer, default=0)
    last_validation = Column(Float, nullable=True)
    validation_accuracy = Column(Float, default=0.0)  # NEW
    validation_sample_size = Column(Integer, default=0)  # NEW
    validation_passed = Column(Boolean, default=False)  # NEW

    # A/B testing (ENHANCED)
    is_active = Column(Boolean, default=True)
    performance_score = Column(Float, default=0.5)
    ab_test_group = Column(String, default='control')  # NEW: 'control', 'treatment', 'production'
    ab_test_start = Column(Float, nullable=True)  # NEW
    ab_test_decisions = Column(Integer, default=0)  # NEW
    ab_test_corrections = Column(Integer, default=0)  # NEW
    ab_test_accuracy = Column(Float, default=0.0)  # NEW


class ModelVersion(Base):
    """Tracks ML model versions and performance."""
    __tablename__ = 'model_versions'

    id = Column(Integer, primary_key=True)
    version_name = Column(String, unique=True)
    model_type = Column(String)

    # Model location
    model_uri = Column(String)

    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)

    # Training info
    trained_at = Column(Float)
    training_samples = Column(Integer)
    training_duration_seconds = Column(Float)

    # Deployment status
    status = Column(String)
    deployed_at = Column(Float, nullable=True)
