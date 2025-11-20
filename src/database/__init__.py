"""
Database module for AURORA adaptive learning system.

Provides SQLAlchemy models and session management for:
- Correction records (privacy-preserved)
- Learned rules
- Model versions
"""

from .models import Base, CorrectionRecord, LearnedRule, ModelVersion
from .connection import get_db, SessionLocal, engine

__all__ = [
    'Base',
    'CorrectionRecord',
    'LearnedRule',
    'ModelVersion',
    'get_db',
    'SessionLocal',
    'engine',
]
