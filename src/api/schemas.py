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
    context: Optional[str] = Field(
        default="general",
        description="Analysis context/goal (general, regression, classification)"
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
    # Phase 1: Enhanced features
    enhanced_features: Optional[Dict[str, Any]] = Field(
        None,
        description="Enhanced feature analysis (statistical tests, patterns, distribution)"
    )
    cache_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Cache hit information (level, similarity)"
    )
    # Phase 3: Confidence warnings
    warning: Optional[str] = Field(
        None,
        description="Warning message for low confidence decisions"
    )
    require_manual_review: bool = Field(
        default=False,
        description="Whether this decision requires manual review"
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
    production_ready: bool = Field(False, description="Whether this pattern is production-ready")
    pattern_corrections: int = Field(0, description="Number of corrections for this pattern")
    corrections_needed_for_training: int = Field(0, description="Corrections needed to compute adjustment")
    corrections_needed_for_production: int = Field(0, description="Corrections needed for production use")
    preferred_action: Optional[str] = Field(None, description="Preferred action learned from corrections")
    message: Optional[str] = Field(None, description="User-friendly status message")

    class Config:
        json_schema_extra = {
            "example": {
                "learned": True,
                "pattern_recorded": True,
                "new_rule_created": True,
                "rule_name": "numeric_high_skewness",
                "rule_confidence": 0.20,
                "similar_patterns_count": 10,
                "applicable_to": "Similar columns matching 'numeric_high_skewness' pattern",
                "production_ready": True,
                "pattern_corrections": 10,
                "corrections_needed_for_training": 0,
                "corrections_needed_for_production": 0,
                "preferred_action": "log_transform",
                "message": "✓ PRODUCTION: Adjustments active!"
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


class ExecutePipelineRequest(BaseModel):
    """Request schema for executing preprocessing pipeline."""

    columns: Dict[str, List[Any]] = Field(
        ...,
        description="Dictionary mapping column names to data"
    )
    actions: Dict[str, str] = Field(
        ...,
        description="Dictionary mapping column names to preprocessing actions"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "columns": {
                    "age": [25, 30, 35, 40, 45],
                    "revenue": [1000, 2000, 15000, 3000, 4000],
                    "category": ["A", "B", "A", "C", "B"]
                },
                "actions": {
                    "age": "standard_scale",
                    "revenue": "log_transform",
                    "category": "onehot_encode"
                }
            }
        }


class ExecutePipelineResponse(BaseModel):
    """Response schema for pipeline execution."""

    success: bool = Field(..., description="Whether pipeline executed successfully")
    processed_data: Dict[str, List[Any]] = Field(..., description="Processed column data")
    applied_actions: Dict[str, str] = Field(..., description="Actions that were applied")
    skipped_columns: List[str] = Field(default=[], description="Columns that were skipped")
    errors: Dict[str, str] = Field(default={}, description="Errors per column if any")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "processed_data": {
                    "age": [-1.2, -0.4, 0.4, 1.2, 2.0],
                    "revenue": [6.9, 7.6, 9.6, 8.0, 8.3],
                    "category_A": [1, 0, 1, 0, 0],
                    "category_B": [0, 1, 0, 0, 1],
                    "category_C": [0, 0, 0, 1, 0]
                },
                "applied_actions": {
                    "age": "standard_scale",
                    "revenue": "log_transform",
                    "category": "onehot_encode"
                },
                "skipped_columns": [],
                "errors": {}
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
    context: Optional[str] = Field(
        default="general",
        description="Analysis context/goal (general, regression, classification)"
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


class ColumnHealthMetrics(BaseModel):
    """Health metrics for a single column."""

    column_name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Detected data type")
    health_score: float = Field(..., description="Overall health score (0-100)")
    anomalies: List[str] = Field(default=[], description="List of detected anomalies")

    # Quality metrics
    null_count: int = Field(default=0, description="Number of null values")
    null_pct: float = Field(default=0.0, description="Percentage of null values")
    duplicate_count: int = Field(default=0, description="Number of duplicate values")
    duplicate_pct: float = Field(default=0.0, description="Percentage of duplicates")
    unique_count: int = Field(default=0, description="Number of unique values")
    unique_ratio: float = Field(default=0.0, description="Ratio of unique values")

    # Numeric-specific metrics
    outlier_count: Optional[int] = Field(None, description="Number of outliers (numeric only)")
    outlier_pct: Optional[float] = Field(None, description="Percentage of outliers")
    skewness: Optional[float] = Field(None, description="Distribution skewness")
    kurtosis: Optional[float] = Field(None, description="Distribution kurtosis")
    mean: Optional[float] = Field(None, description="Mean value")
    std: Optional[float] = Field(None, description="Standard deviation")
    cv: Optional[float] = Field(None, description="Coefficient of variation")

    # Categorical-specific metrics
    cardinality: Optional[int] = Field(None, description="Number of categories")
    is_imbalanced: Optional[bool] = Field(None, description="Whether categorical is imbalanced")

    # Severity classification
    severity: str = Field(..., description="Overall severity: 'healthy', 'warning', 'critical'")


class BatchHealthResponse(BaseModel):
    """Response schema for batch data health analysis."""

    overall_health_score: float = Field(..., description="Overall dataset health score (0-100)")
    healthy_columns: int = Field(..., description="Number of healthy columns")
    warning_columns: int = Field(..., description="Number of columns with warnings")
    critical_columns: int = Field(..., description="Number of critical columns")
    column_health: Dict[str, ColumnHealthMetrics] = Field(..., description="Per-column health metrics")


class BatchPreprocessResponse(BaseModel):
    """Response schema for batch preprocessing."""

    results: Dict[str, PreprocessResponse]
    summary: Dict[str, Any]
    health: Optional[BatchHealthResponse] = Field(None, description="Data health analysis")

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


# Phase 1: New schemas for cache and drift detection

class CacheStatsResponse(BaseModel):
    """Response schema for cache statistics."""

    total_queries: int = Field(..., description="Total cache queries")
    l1_hits: int = Field(..., description="L1 (exact match) hits")
    l2_hits: int = Field(..., description="L2 (similarity) hits")
    l3_hits: int = Field(..., description="L3 (pattern) hits")
    misses: int = Field(..., description="Cache misses")
    hit_rate: float = Field(..., description="Overall hit rate (0-1)")
    cache_size: int = Field(..., description="Current cache size")
    pattern_rules: int = Field(..., description="Number of learned pattern rules")

    class Config:
        json_schema_extra = {
            "example": {
                "total_queries": 1000,
                "l1_hits": 450,
                "l2_hits": 120,
                "l3_hits": 80,
                "misses": 350,
                "hit_rate": 0.65,
                "cache_size": 650,
                "pattern_rules": 5
            }
        }


class DriftStatus(BaseModel):
    """Drift status for a single column."""

    column_name: str
    drift_detected: bool
    drift_score: float
    severity: str  # 'none', 'low', 'medium', 'high', 'critical'
    recommendation: str
    p_value: float
    test_used: str
    changes: Dict[str, Any]
    timestamp: float


class DriftMonitoringResponse(BaseModel):
    """Response schema for drift monitoring status."""

    monitored_columns: int = Field(..., description="Number of columns being monitored")
    columns_with_drift: int = Field(..., description="Number of columns with detected drift")
    critical_columns: List[str] = Field(default=[], description="Columns with critical drift")
    high_priority_columns: List[str] = Field(default=[], description="Columns with high drift")
    requires_retraining: bool = Field(..., description="Whether model retraining is recommended")
    drift_reports: List[DriftStatus] = Field(default=[], description="Detailed drift reports")

    class Config:
        json_schema_extra = {
            "example": {
                "monitored_columns": 10,
                "columns_with_drift": 3,
                "critical_columns": ["age"],
                "high_priority_columns": ["income", "score"],
                "requires_retraining": True,
                "drift_reports": []
            }
        }


class ChatQueryRequest(BaseModel):
    """Request schema for chatbot query."""
    
    question: str = Field(..., description="User's question")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context (current dataframe info, recent results)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the statistics for revenue?",
                "context": {
                    "current_columns": ["revenue", "price", "quantity"],
                    "row_count": 1000
                }
            }
        }


class ChatQueryResponse(BaseModel):
    """Response schema for chatbot query."""
    
    answer: str = Field(..., description="AI assistant's answer")
    confidence: float = Field(default=1.0, description="Confidence in answer (0-1)")
    suggestions: List[str] = Field(default=[], description="Suggested follow-up questions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The revenue column has a mean of $1,234.56...",
                "confidence": 0.95,
                "suggestions": [
                    "What preprocessing do you recommend for revenue?",
                    "Show me the distribution of revenue"
                ]
            }
        }
