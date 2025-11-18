"""
API request/response schemas for AURORA.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class PreprocessRequest(BaseModel):
    """Request schema for preprocessing a column."""

    column_data: List[Any] = Field(
        ...,
        description="Column data as a list"
    )
    column_name: str = Field(
        default="",
        description="Name of the column"
    )
    target_available: bool = Field(
        default=False,
        description="Whether a target variable is available"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the column"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "column_data": [1, 2, 3, 100, 200, 300],
                "column_name": "revenue",
                "target_available": False,
                "metadata": {"dtype": "numeric"}
            }
        }


class AlternativeAction(BaseModel):
    """Alternative preprocessing action with confidence."""

    action: str
    confidence: float


class PreprocessResponse(BaseModel):
    """Response schema for preprocessing."""

    action: str = Field(..., description="Recommended preprocessing action")
    confidence: float = Field(..., description="Confidence score (0-1)")
    source: str = Field(..., description="Decision source: 'symbolic', 'neural', or 'learned'")
    explanation: str = Field(..., description="Human-readable explanation")
    alternatives: List[AlternativeAction] = Field(
        default=[],
        description="Alternative actions with confidence scores"
    )
    parameters: Dict[str, Any] = Field(
        default={},
        description="Action-specific parameters"
    )
    decision_id: Optional[str] = Field(
        None,
        description="Unique ID for this decision"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "action": "log_transform",
                "confidence": 0.92,
                "source": "symbolic",
                "explanation": "High positive skewness (3.45) in positive data",
                "alternatives": [
                    {"action": "box_cox", "confidence": 0.85},
                    {"action": "yeo_johnson", "confidence": 0.78}
                ],
                "parameters": {},
                "decision_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


class CorrectionRequest(BaseModel):
    """Request schema for submitting a correction."""

    column_data: List[Any] = Field(
        ...,
        description="Column data (only used for pattern extraction)"
    )
    column_name: str = Field(
        default="",
        description="Name of the column"
    )
    wrong_action: str = Field(
        ...,
        description="Action that was incorrect"
    )
    correct_action: str = Field(
        ...,
        description="Correct action"
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence of the wrong prediction"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "column_data": [1, 2, 3, 100, 200, 300],
                "column_name": "revenue",
                "wrong_action": "standard_scale",
                "correct_action": "log_transform",
                "confidence": 0.75
            }
        }


class CorrectionResponse(BaseModel):
    """Response schema for correction submission."""

    learned: bool = Field(..., description="Whether the correction was learned")
    pattern_recorded: bool = Field(..., description="Whether the pattern was recorded")
    new_rule_created: bool = Field(..., description="Whether a new rule was created")
    rule_name: Optional[str] = Field(None, description="Name of the new rule if created")
    rule_confidence: Optional[float] = Field(None, description="Confidence of the new rule")
    similar_patterns_count: Optional[int] = Field(
        None,
        description="Number of similar patterns found"
    )
    applicable_to: Optional[str] = Field(
        None,
        description="Description of what the rule applies to"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "learned": True,
                "pattern_recorded": True,
                "new_rule_created": True,
                "rule_name": "LEARNED_LOG_TRANSFORM_5",
                "rule_confidence": 0.88,
                "similar_patterns_count": 7,
                "applicable_to": "~7 similar cases"
            }
        }


class ExplanationResponse(BaseModel):
    """Response schema for decision explanation."""

    decision_id: str
    action: str
    confidence: float
    source: str
    explanation: str
    rule_used: Optional[str] = None
    alternatives: List[AlternativeAction] = []
    feature_importance: Optional[Dict[str, float]] = None
    context: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "decision_id": "550e8400-e29b-41d4-a716-446655440000",
                "action": "log_transform",
                "confidence": 0.92,
                "source": "symbolic",
                "explanation": "High positive skewness (3.45) in positive data",
                "rule_used": "LOG_TRANSFORM_HIGH_SKEW",
                "alternatives": [
                    {"action": "box_cox", "confidence": 0.85}
                ],
                "context": {
                    "skewness": 3.45,
                    "null_pct": 0.02,
                    "is_numeric": True
                }
            }
        }


class StatisticsResponse(BaseModel):
    """Response schema for system statistics."""

    total_decisions: int
    learned_decisions: int
    symbolic_decisions: int
    neural_decisions: int
    high_confidence_decisions: int
    learned_pct: float
    symbolic_pct: float
    neural_pct: float
    high_confidence_pct: float
    avg_time_ms: float
    symbolic_coverage: float
    learning: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "total_decisions": 1000,
                "learned_decisions": 150,
                "symbolic_decisions": 750,
                "neural_decisions": 100,
                "high_confidence_decisions": 900,
                "learned_pct": 15.0,
                "symbolic_pct": 75.0,
                "neural_pct": 10.0,
                "high_confidence_pct": 90.0,
                "avg_time_ms": 0.85,
                "symbolic_coverage": 0.82,
                "learning": {
                    "total_corrections": 45,
                    "learned_rules": 12,
                    "pattern_clusters": 8
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    components: Dict[str, str]

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "symbolic_engine": "ok",
                    "neural_oracle": "ok",
                    "pattern_learner": "ok"
                }
            }
        }


class BatchPreprocessRequest(BaseModel):
    """Request schema for batch preprocessing."""

    columns: Dict[str, List[Any]] = Field(
        ...,
        description="Dictionary mapping column names to data"
    )
    target_column: Optional[str] = Field(
        None,
        description="Name of the target column"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "columns": {
                    "age": [25, 30, 35, 40, 45],
                    "revenue": [1000, 2000, 15000, 3000, 4000],
                    "category": ["A", "B", "A", "C", "B"]
                },
                "target_column": None
            }
        }


class BatchPreprocessResponse(BaseModel):
    """Response schema for batch preprocessing."""

    results: Dict[str, PreprocessResponse]
    summary: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "results": {
                    "age": {
                        "action": "standard_scale",
                        "confidence": 0.9,
                        "source": "symbolic",
                        "explanation": "Normal distribution without outliers"
                    }
                },
                "summary": {
                    "total_columns": 3,
                    "avg_confidence": 0.87,
                    "source_breakdown": {
                        "symbolic": 2,
                        "neural": 1
                    }
                }
            }
        }
