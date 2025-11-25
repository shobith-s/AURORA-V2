"""
FastAPI server for AURORA preprocessing system.
Provides REST API for intelligent data preprocessing with real-time recommendations.
"""

__version__ = "1.0.0"

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import os

from .schemas import (
    PreprocessRequest,
    PreprocessResponse,
    CorrectionRequest,
    CorrectionResponse,
    ExplanationResponse,
    StatisticsResponse,
    HealthResponse,
    BatchPreprocessRequest,
    BatchPreprocessResponse,
    ExecutePipelineRequest,
    ExecutePipelineResponse,
    AlternativeAction,
    CacheStatsResponse
)
from ..core.preprocessor import IntelligentPreprocessor, get_preprocessor
from ..utils.monitor import get_monitor
from ..database.connection import init_db
from ..learning.adaptive_engine import AdaptiveLearningEngine
import time
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance monitor
monitor = get_monitor()

# Create FastAPI app
app = FastAPI(
    title="AURORA - Intelligent Data Preprocessing System",
    description="Production-ready intelligent data preprocessing with symbolic rules, neural intelligence, and privacy-preserving learning",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS - environment-based for security
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000"  # Default for development
).split(",")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # ✅ SECURE: Whitelist only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Explicit methods only
    allow_headers=["Authorization", "Content-Type"],  # Explicit headers only
)

# Global preprocessor instance
preprocessor: IntelligentPreprocessor = None

# Global learning engine instance (for Option B - MVP with Learning)
learning_engine: AdaptiveLearningEngine = None

# Decision cache (for explanations)
decision_cache: Dict[str, Dict[str, Any]] = {}
MAX_CACHE_SIZE = 1000


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types and other non-JSON-serializable types to native Python types.
    NumPy 2.0 compatible.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):  # Abstract base class for all numpy ints (NumPy 2.0 compatible)
        return int(obj)
    elif isinstance(obj, (np.floating,)):  # Abstract base class for all numpy floats (NumPy 2.0 compatible)
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_json_serializable(obj.tolist())
    else:
        return obj


@app.on_event("startup")
async def startup_event():
    """Initialize the preprocessor and learning system on startup."""
    global preprocessor, learning_engine
    logger.info("Starting AURORA preprocessing system...")

    # Initialize database
    try:
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}. Using in-memory storage.")

    # Initialize learning engine (Option B: with validation and A/B testing)
    try:
        db_url = os.getenv("DATABASE_URL", "sqlite:///./aurora.db")
        learning_engine = AdaptiveLearningEngine(
            db_url=db_url,
            min_support=10,  # CHANGED: 5 → 10 corrections per pattern
            similarity_threshold=0.85,
            epsilon=1.0,
            validation_sample_size=20,
            ab_test_min_decisions=100,
            ab_test_success_threshold=0.80
        )
        logger.info("Adaptive learning engine initialized with validation and A/B testing")
    except Exception as e:
        logger.warning(f"Learning engine initialization failed: {e}")

    # Initialize preprocessor
    try:
        preprocessor = get_preprocessor(
            confidence_threshold=0.75,  # CHANGED: 0.9 → 0.75 for more neural participation
            use_neural_oracle=True,
            enable_learning=True
        )
        logger.info("AURORA initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AURORA: {e}")
        # Create basic preprocessor without neural oracle
        preprocessor = IntelligentPreprocessor(
            confidence_threshold=0.75,
            use_neural_oracle=False,
            enable_learning=True
        )
        logger.warning("AURORA initialized in fallback mode (no neural oracle)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AURORA preprocessing system...")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "AURORA - Intelligent Data Preprocessing System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint for monitoring and load balancers.

    Returns:
        HealthResponse with overall status and component-level health
    """
    components = {}
    all_healthy = True

    # Check symbolic engine (critical)
    try:
        if hasattr(preprocessor, 'symbolic_engine') and preprocessor.symbolic_engine is not None:
            components["symbolic_engine"] = "ok"
        else:
            components["symbolic_engine"] = "unavailable"
            all_healthy = False
    except Exception as e:
        logger.error(f"Health check - symbolic engine error: {e}")
        components["symbolic_engine"] = "error"
        all_healthy = False

    # Check pattern learner (optional)
    try:
        if preprocessor.enable_learning:
            components["pattern_learner"] = "ok" if preprocessor.pattern_learner else "initializing"
        else:
            components["pattern_learner"] = "disabled"
    except Exception as e:
        logger.warning(f"Health check - pattern learner error: {e}")
        components["pattern_learner"] = "error"

    # Check adaptive rules (optional)
    try:
        if preprocessor.enable_learning:
            components["adaptive_rules"] = "ok" if preprocessor.adaptive_rules else "unavailable"
        else:
            components["adaptive_rules"] = "disabled"
    except Exception as e:
        logger.warning(f"Health check - adaptive rules error: {e}")
        components["adaptive_rules"] = "error"

    # Check neural oracle (optional)
    try:
        if preprocessor.use_neural_oracle:
            components["neural_oracle"] = "ok" if preprocessor.neural_oracle else "loading"
        else:
            components["neural_oracle"] = "disabled"
    except Exception as e:
        logger.warning(f"Health check - neural oracle error: {e}")
        components["neural_oracle"] = "error"

    # Check database (optional)
    try:
        if learning_engine:
            components["database"] = "ok"
        else:
            components["database"] = "disabled"
    except Exception as e:
        logger.warning(f"Health check - database error: {e}")
        components["database"] = "error"

    # Check cache (optional)
    try:
        if preprocessor.enable_cache:
            components["cache"] = "ok" if preprocessor.cache else "unavailable"
        else:
            components["cache"] = "disabled"
    except Exception as e:
        logger.warning(f"Health check - cache error: {e}")
        components["cache"] = "error"

    overall_status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        components=components
    )


# New API Endpoints for Redesigned Frontend

@app.get("/stats")
async def get_system_stats():
    """Get system-wide statistics for dashboard."""
    try:
        stats = preprocessor.stats.copy()
        total = stats.get('total_decisions', 0)
        avg_confidence = 0.0
        if total > 0:
            high_conf = stats.get('high_confidence_decisions', 0)
            avg_confidence = (high_conf / total) * 100
        
        avg_time = 0.0
        if total > 0:
            avg_time = stats.get('total_time_ms', 0) / total
        
        rule_count = len(preprocessor.symbolic_engine.rules) if hasattr(preprocessor, 'symbolic_engine') else 0
        
        return {
            "total_decisions": total,
            "symbolic_decisions": stats.get('symbolic_decisions', 0),
            "meta_decisions": stats.get('meta_learning_decisions', 0),
            "neural_decisions": stats.get('neural_decisions', 0),
            "high_confidence_decisions": stats.get('high_confidence_decisions', 0),
            "avg_confidence": round(avg_confidence, 2),
            "avg_time_ms": round(avg_time, 2),
            "active_rules": rule_count
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return {"total_decisions": 0, "error": str(e)}


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_column(request: PreprocessRequest):
    """
    Preprocess a single column and get recommendations.

    Args:
        request: Preprocessing request with column data

    Returns:
        Preprocessing recommendation with action, confidence, and explanation
    """
    try:
        # Input validation
        if not request.column_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="column_data cannot be empty"
            )

        if len(request.column_data) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="column_data must contain at least 2 values"
            )

        if len(request.column_data) > 1000000:  # 1M row limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="column_data exceeds maximum size of 1,000,000 rows"
            )

        if not request.column_name or not request.column_name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="column_name cannot be empty"
            )

        # Time the operation
        start_time = time.time()

        # Process column
        result = preprocessor.preprocess_column(
            column=request.column_data,
            column_name=request.column_name,
            target_available=request.target_available,
            metadata=request.metadata,
            context=request.context
        )

        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        monitor.record_call('overall_pipeline', latency_ms, success=True)
        monitor.record_decision(
            decision_id=result.decision_id,
            column_name=request.column_name,
            action=result.action.value,
            confidence=result.confidence,
            source=result.source,
            latency_ms=latency_ms,
            num_rows=len(request.column_data)
        )

        # Cache decision for explanation endpoint
        if result.decision_id and len(decision_cache) < MAX_CACHE_SIZE:
            decision_cache[result.decision_id] = {
                "result": result,
                "request": request.dict()
            }

        # Convert to response format
        alternatives = [
            AlternativeAction(
                action=action.value,
                confidence=float(conf) if isinstance(conf, (np.floating,)) else conf
            )
            for action, conf in result.alternatives
        ]

        # Convert parameters to JSON-serializable types
        json_safe_parameters = convert_to_json_serializable(result.parameters)

        response = PreprocessResponse(
            action=result.action.value,
            confidence=float(result.confidence),
            source=result.source,
            explanation=result.explanation,
            alternatives=alternatives,
            parameters=json_safe_parameters,
            decision_id=result.decision_id,
            enhanced_features=None,
            cache_info=None
        )

        # Use FastAPI's jsonable_encoder to ensure ALL types are JSON-safe
        return JSONResponse(content=jsonable_encoder(response))

    except Exception as e:
        logger.error(f"Error preprocessing column: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {str(e)}"
        )


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, (np.bool_, np.generic)) and hasattr(obj, 'item'):
        # Handle numpy scalars including bool_
        return obj.item()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


@app.post("/explain/enhanced")
async def explain_enhanced(request: PreprocessRequest):
    """
    Get enhanced explanation for a column decision with markdown report.
    """
    try:
        # Preprocess to get the decision and details
        result = preprocessor.preprocess_column(
            column=request.column_data,
            column_name=request.column_name,
            target_available=request.target_available,
            metadata=request.metadata,
            context=request.context
        )
        
        # Generate markdown report
        markdown_report = f"""
## Decision Analysis for "{request.column_name}"

### Recommended Action: **{result.action.value.replace('_', ' ').title()}**

**Confidence:** {result.confidence * 100:.1f}%  
**Source:** {result.source}

### Explanation

{result.explanation}

### Why This Action?

"""
        
        # Get metadata from result.context (that's where stats are stored)
        metadata = result.context if result.context else {}
        
        # Convert numpy types in metadata to native Python types
        metadata = convert_numpy_types(metadata)
        
        # Add metadata insights if available
        if metadata:
            if metadata.get("skewness"):
                markdown_report += f"- **Skewness:** {metadata['skewness']:.2f}\n"
            if metadata.get("null_pct"):
                markdown_report += f"- **Missing Values:** {metadata['null_pct'] * 100:.1f}%\n"
            if metadata.get("outlier_pct"):
                markdown_report += f"- **Outliers:** {metadata['outlier_pct'] * 100:.1f}%\n"
            if metadata.get("unique_ratio"):
                markdown_report += f"- **Unique Ratio:** {metadata['unique_ratio'] * 100:.1f}%\n"
        
        # Add alternatives section
        alternatives_serializable = []
        if result.alternatives:
            markdown_report += "\n### Alternative Approaches\n\n"
            for alt in result.alternatives[:3]:  # Top 3 alternatives
                # Handle both tuple and dict formats
                if isinstance(alt, tuple):
                    action, conf = alt
                    action_name = action.value if hasattr(action, 'value') else str(action)
                    alternatives_serializable.append({
                        "action": action_name,
                        "confidence": float(conf)
                    })
                else:
                    action_name = alt.get("action", "unknown")
                    conf = alt.get("confidence", 0)
                    alternatives_serializable.append({
                        "action": action_name,
                        "confidence": float(conf)
                    })
                action_name = action_name.replace('_', ' ').title()
                markdown_report += f"- **{action_name}** ({conf * 100:.1f}% confidence)\n"
        
        # Construct response matching frontend expectations
        response = {
            "decision": {
                "action": result.action.value,
                "confidence": float(result.confidence),
                "source": result.source,
                "decision_id": result.decision_id
            },
            "markdown_report": markdown_report,
            "alternatives": alternatives_serializable,
            "metadata": metadata,  # Frontend expects this field
            "context": {
                "column_name": request.column_name,
                "sample_size": len(request.column_data),
                "context": request.context
            }
        }
        
        # Final safety conversion
        return convert_numpy_types(response)
        
    except Exception as e:
        import traceback
        logger.error(f"Explanation error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate enhanced explanation: {str(e)}")


def compute_column_health(column: pd.Series, column_name: str) -> 'ColumnHealthMetrics':
    """Compute health metrics for a single column."""
    from ..api.schemas import ColumnHealthMetrics

    anomalies = []
    health_score = 100.0

    # Basic stats
    total_count = len(column)
    null_count = int(column.isnull().sum())
    null_pct = float(null_count / total_count) if total_count > 0 else 0.0

    # Duplicates
    duplicate_count = int(total_count - column.nunique())
    duplicate_pct = float(duplicate_count / total_count) if total_count > 0 else 0.0

    # Unique values
    unique_count = int(column.nunique())
    unique_ratio = float(unique_count / total_count) if total_count > 0 else 0.0

    # Detect data type - check actual pandas dtype first
    non_null = column.dropna()
    if len(non_null) == 0:
        data_type = "empty"
        anomalies.append("All values are null")
        health_score = 0.0
    else:
        # Check if already numeric dtype
        if pd.api.types.is_numeric_dtype(column):
            data_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(column):
            data_type = "datetime"
        # For object/string types, try to infer
        elif pd.api.types.is_object_dtype(column) or pd.api.types.is_string_dtype(column):
            try:
                # Try to convert to numeric
                numeric_column = pd.to_numeric(column, errors='coerce')
                # If most values successfully converted, it's numeric
                successful_conversions = numeric_column.notna().sum()
                if successful_conversions / len(column) > 0.8:  # 80% threshold
                    column = numeric_column  # Use the numeric version
                    data_type = "numeric"
                else:
                    # Check unique ratio to determine if categorical
                    if unique_ratio < 0.5:  # Less than 50% unique -> categorical
                        data_type = "categorical"
                    else:
                        data_type = "text"  # High unique ratio -> text
            except:
                # Default to categorical for object types with low unique ratio
                if unique_ratio < 0.5:
                    data_type = "categorical"
                else:
                    data_type = "text"
        else:
            # Fallback for unknown types
            data_type = "categorical"

    # Numeric-specific metrics
    outlier_count = None
    outlier_pct = None
    skewness = None
    kurtosis = None
    mean = None
    std = None
    cv = None

    if data_type == "numeric":
        # Recalculate non_null with the numeric column
        non_null = column.dropna()
        if len(non_null) > 0:
            try:
                from scipy import stats
                mean = float(non_null.mean())
                std = float(non_null.std())
                cv = float(abs(std / mean)) if mean != 0 else None

                # Skewness and kurtosis
                if len(non_null) > 2:
                    skewness = float(non_null.skew())
                    kurtosis = float(non_null.kurtosis())

                # Outlier detection using IQR
                Q1 = non_null.quantile(0.25)
                Q3 = non_null.quantile(0.75)
                IQR = Q3 - Q1
                outliers = (non_null < (Q1 - 1.5 * IQR)) | (non_null > (Q3 + 1.5 * IQR))
                outlier_count = int(outliers.sum())
                outlier_pct = float(outlier_count / len(non_null)) if len(non_null) > 0 else 0.0

                # Anomaly detection for numeric
                if outlier_pct and outlier_pct > 0.1:
                    anomalies.append(f"High outlier percentage ({outlier_pct:.1%})")
                    health_score -= 15
                if skewness and abs(skewness) > 2:
                    anomalies.append(f"High skewness ({skewness:.2f})")
                    health_score -= 10
                if cv and cv > 2:
                    anomalies.append(f"High coefficient of variation ({cv:.2f})")
                    health_score -= 5
            except:
                pass

    # Categorical-specific metrics
    cardinality = None
    is_imbalanced = None

    if data_type == "categorical":
        cardinality = unique_count
        # Check for imbalance
        if len(non_null) > 0:
            value_counts = non_null.value_counts()
            max_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
            is_imbalanced = (max_freq / len(non_null)) > 0.9

            if is_imbalanced:
                anomalies.append(f"Imbalanced categories ({max_freq / len(non_null):.1%} in top category)")
                health_score -= 10
            if cardinality > 100:
                anomalies.append(f"High cardinality ({cardinality} categories)")
                health_score -= 15
            elif cardinality == 1:
                anomalies.append("Only one unique value (constant column)")
                health_score -= 30

    # General anomalies
    if null_pct > 0.5:
        anomalies.append(f"High null percentage ({null_pct:.1%})")
        health_score -= 30
    elif null_pct > 0.2:
        anomalies.append(f"Moderate null percentage ({null_pct:.1%})")
        health_score -= 15
    elif null_pct > 0.05:
        anomalies.append(f"Low null percentage ({null_pct:.1%})")
        health_score -= 5

    if unique_ratio > 0.95 and total_count > 50:
        anomalies.append(f"Nearly all values unique ({unique_ratio:.1%}) - likely ID column")
        health_score -= 20

    if unique_count == 1:
        anomalies.append("All values are identical")
        health_score -= 30

    # Determine severity
    if health_score >= 80:
        severity = "healthy"
    elif health_score >= 50:
        severity = "warning"
    else:
        severity = "critical"

    return ColumnHealthMetrics(
        column_name=column_name,
        data_type=data_type,
        health_score=max(0.0, health_score),
        anomalies=anomalies,
        null_count=null_count,
        null_pct=null_pct,
        duplicate_count=duplicate_count,
        duplicate_pct=duplicate_pct,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        outlier_count=outlier_count,
        outlier_pct=outlier_pct,
        skewness=skewness,
        kurtosis=kurtosis,
        mean=mean,
        std=std,
        cv=cv,
        cardinality=cardinality,
        is_imbalanced=is_imbalanced,
        severity=severity
    )


@app.post("/batch", response_model=BatchPreprocessResponse)
async def batch_preprocess(request: BatchPreprocessRequest):
    """
    Preprocess multiple columns in batch.

    Args:
        request: Batch preprocessing request

    Returns:
        Preprocessing recommendations for all columns
    """
    try:
        # Time the operation
        start_time = time.time()

        # Convert to DataFrame
        df = pd.DataFrame(request.columns)

        # Process all columns
        results_dict = preprocessor.preprocess_dataframe(
            df,
            target_column=request.target_column,
            context=request.context
        )

        # Record overall pipeline latency
        batch_latency_ms = (time.time() - start_time) * 1000
        monitor.record_call('overall_pipeline', batch_latency_ms, success=True)

        # Compute health metrics for all columns
        from ..api.schemas import BatchHealthResponse, ColumnHealthMetrics
        column_health = {}
        healthy_count = 0
        warning_count = 0
        critical_count = 0

        for col_name in df.columns:
            health = compute_column_health(df[col_name], col_name)
            column_health[col_name] = health

            if health.severity == "healthy":
                healthy_count += 1
            elif health.severity == "warning":
                warning_count += 1
            else:
                critical_count += 1

        # Overall health score
        overall_health = sum(h.health_score for h in column_health.values()) / len(column_health) if column_health else 0.0

        health_response = BatchHealthResponse(
            overall_health_score=float(overall_health),
            healthy_columns=int(healthy_count),
            warning_columns=int(warning_count),
            critical_columns=int(critical_count),
            column_health=column_health
        )

        # Convert results to response format (INCLUDE ALL COLUMNS, even healthy ones)
        results = {}
        total_confidence = 0.0
        source_breakdown = {}

        for col_name, result in results_dict.items():
            # Record decision in monitor (for all columns)
            monitor.record_decision(
                decision_id=result.decision_id,
                column_name=col_name,
                action=result.action.value,
                confidence=result.confidence,
                source=result.source,
                latency_ms=batch_latency_ms / len(results_dict),  # Approximate per-column latency
                num_rows=len(df)
            )

            # Convert alternatives to JSON-serializable format
            alternatives = [
                AlternativeAction(
                    action=action.value,
                    confidence=float(conf) if isinstance(conf, (np.floating,)) else conf
                )
                for action, conf in result.alternatives
            ]

            # Convert parameters to JSON-serializable types
            json_safe_parameters = convert_to_json_serializable(result.parameters)

            # Create response with all fields explicitly set and converted
            results[col_name] = PreprocessResponse(
                action=result.action.value,
                confidence=float(result.confidence),
                source=result.source,
                explanation=result.explanation,
                alternatives=alternatives,
                parameters=json_safe_parameters,
                decision_id=result.decision_id,
                enhanced_features=None,  # Phase 1 field - not yet implemented
                cache_info=None  # Phase 1 field - not yet implemented
            )

            total_confidence += float(result.confidence)
            source_breakdown[result.source] = source_breakdown.get(result.source, 0) + 1

        # Create summary - ensure all values are JSON-serializable
        summary = {
            "total_columns": int(len(results_dict)),  # Total columns analyzed
            "processed_columns": int(len(results_dict)),  # All columns
            "avg_confidence": float(total_confidence / len(results_dict)) if len(results_dict) > 0 else 0.0,
            "source_breakdown": {k: int(v) for k, v in source_breakdown.items()}
        }

        # Convert summary to JSON-safe format just to be absolutely sure
        json_safe_summary = convert_to_json_serializable(summary)

        # Create response
        response = BatchPreprocessResponse(
            results=results,
            summary=json_safe_summary,
            health=health_response
        )

        # Use FastAPI's jsonable_encoder to ensure ALL types are JSON-safe
        # This catches any numpy types that might have slipped through
        return JSONResponse(content=jsonable_encoder(response))

    except Exception as e:
        import traceback
        logger.error(f"Error in batch preprocessing: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch preprocessing failed: {str(e)}"
        )


@app.post("/execute", response_model=ExecutePipelineResponse)
async def execute_pipeline(request: ExecutePipelineRequest):
    """
    Execute preprocessing pipeline on data.

    Args:
        request: Data and preprocessing actions to apply

    Returns:
        Processed data ready for download
    """
    try:
        from sklearn.preprocessing import (
            StandardScaler, MinMaxScaler, RobustScaler,
            LabelEncoder, OneHotEncoder
        )
        from scipy.stats import boxcox
        import warnings
        warnings.filterwarnings('ignore')

        # Convert to DataFrame
        df = pd.DataFrame(request.columns)
        processed_df = df.copy()
        applied_actions = {}
        skipped_columns = []
        errors = {}

        for col_name, action in request.actions.items():
            if col_name not in df.columns:
                skipped_columns.append(col_name)
                continue

            try:
                column = df[col_name].copy()

                # Skip if action is keep_as_is
                if action.lower() in ['keep_as_is', 'keep']:
                    applied_actions[col_name] = 'keep_as_is'
                    continue

                # Handle nulls first (if action is not about nulls)
                if action not in ['fill_null_mean', 'fill_null_median', 'fill_null_mode']:
                    # Drop rows with nulls for this column temporarily
                    non_null_mask = column.notna()
                    if non_null_mask.sum() == 0:
                        errors[col_name] = "All values are null"
                        continue
                    column_clean = column[non_null_mask]
                else:
                    column_clean = column

                # Apply preprocessing based on action
                if action == 'standard_scale':
                    scaler = StandardScaler()
                    processed_df[col_name] = scaler.fit_transform(column_clean.values.reshape(-1, 1)).flatten()
                    applied_actions[col_name] = action

                elif action == 'minmax_scale':
                    scaler = MinMaxScaler()
                    processed_df[col_name] = scaler.fit_transform(column_clean.values.reshape(-1, 1)).flatten()
                    applied_actions[col_name] = action

                elif action == 'robust_scale':
                    scaler = RobustScaler()
                    processed_df[col_name] = scaler.fit_transform(column_clean.values.reshape(-1, 1)).flatten()
                    applied_actions[col_name] = action

                elif action == 'log_transform':
                    # Ensure positive values
                    if (column_clean > 0).all():
                        processed_df[col_name] = np.log(column_clean)
                        applied_actions[col_name] = action
                    else:
                        errors[col_name] = "Log transform requires all positive values"

                elif action == 'log1p_transform':
                    processed_df[col_name] = np.log1p(column_clean)
                    applied_actions[col_name] = action

                elif action == 'sqrt_transform':
                    if (column_clean >= 0).all():
                        processed_df[col_name] = np.sqrt(column_clean)
                        applied_actions[col_name] = action
                    else:
                        errors[col_name] = "Sqrt transform requires non-negative values"

                elif action == 'box_cox':
                    if (column_clean > 0).all():
                        transformed, _ = boxcox(column_clean)
                        processed_df[col_name] = transformed
                        applied_actions[col_name] = action
                    else:
                        errors[col_name] = "Box-Cox requires all positive values"

                elif action == 'onehot_encode':
                    # One-hot encoding creates multiple columns
                    dummies = pd.get_dummies(column_clean, prefix=col_name)
                    # Drop original column
                    processed_df = processed_df.drop(columns=[col_name])
                    # Add dummy columns
                    for dummy_col in dummies.columns:
                        processed_df[dummy_col] = dummies[dummy_col]
                    applied_actions[col_name] = action

                elif action == 'label_encode':
                    encoder = LabelEncoder()
                    processed_df[col_name] = encoder.fit_transform(column_clean.astype(str))
                    applied_actions[col_name] = action

                elif action == 'fill_null_mean':
                    if pd.api.types.is_numeric_dtype(column):
                        processed_df[col_name] = column.fillna(column.mean())
                        applied_actions[col_name] = action
                    else:
                        errors[col_name] = "Mean imputation requires numeric column"

                elif action == 'fill_null_median':
                    if pd.api.types.is_numeric_dtype(column):
                        processed_df[col_name] = column.fillna(column.median())
                        applied_actions[col_name] = action
                    else:
                        errors[col_name] = "Median imputation requires numeric column"

                elif action == 'fill_null_mode':
                    mode_value = column.mode()[0] if not column.mode().empty else None
                    if mode_value is not None:
                        processed_df[col_name] = column.fillna(mode_value)
                        applied_actions[col_name] = action
                    else:
                        errors[col_name] = "No mode value found"

                elif action == 'drop_column':
                    processed_df = processed_df.drop(columns=[col_name])
                    applied_actions[col_name] = action

                else:
                    skipped_columns.append(col_name)
                    errors[col_name] = f"Unknown action: {action}"

            except Exception as e:
                logger.error(f"Error processing column {col_name} with action {action}: {e}")
                errors[col_name] = str(e)
                skipped_columns.append(col_name)

        # Convert DataFrame to dictionary of lists
        processed_data = {}
        for col in processed_df.columns:
            # Convert to native Python types
            values = processed_df[col].tolist()
            processed_data[col] = convert_to_json_serializable(values)

        response = ExecutePipelineResponse(
            success=len(errors) == 0,
            processed_data=processed_data,
            applied_actions=applied_actions,
            skipped_columns=skipped_columns,
            errors=errors
        )

        return JSONResponse(content=jsonable_encoder(response))

    except Exception as e:
        import traceback
        logger.error(f"Error executing pipeline: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {str(e)}"
        )


@app.post("/correct", response_model=CorrectionResponse)
async def submit_correction(request: CorrectionRequest):
    """
    Submit a correction to learn from (privacy-preserving).

    This endpoint records corrections persistently and creates learned rules
    after seeing similar patterns 5+ times.

    Args:
        request: Correction with wrong and correct actions

    Returns:
        Learning result with information about rule creation
    """
    try:
        # Process correction with in-memory pattern learner
        result = preprocessor.process_correction(
            column=request.column_data,
            column_name=request.column_name,
            wrong_action=request.wrong_action,
            correct_action=request.correct_action,
            confidence=request.confidence
        )

        # ALSO record in persistent learning engine (Option B - MVP with Learning)
        if learning_engine:
            # Compute statistics for privacy-preserved storage
            import pandas as pd
            import numpy as np
            column = pd.Series(request.column_data, name=request.column_name)
            stats = preprocessor.symbolic_engine.compute_column_statistics(
                column, request.column_name
            )
            stats_dict = stats.to_dict()

            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                """Recursively convert numpy types to native Python types."""
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return convert_numpy_types(obj.tolist())
                elif pd.isna(obj):
                    return None
                else:
                    return obj

            stats_dict = convert_numpy_types(stats_dict)

            # Record correction persistently (with privacy preservation)
            user_id = "default_user"  # TODO: Get from JWT when auth is enabled
            learning_result = learning_engine.record_correction(
                user_id=user_id,
                column_stats=stats_dict,  # Privacy-preserved stats only!
                wrong_action=request.wrong_action,
                correct_action=request.correct_action,
                confidence=request.confidence
            )

            # Add persistent learning info to response
            result['persistent_learning'] = {
                'recorded': learning_result.get('recorded', False),
                'new_rule_created': learning_result.get('new_rule_created', False),
                'pattern_hash': learning_result.get('pattern_hash'),
            }

            if learning_result.get('new_rule_created'):
                result['rule_info'] = {
                    'rule_name': learning_result.get('rule_name'),
                    'rule_confidence': learning_result.get('rule_confidence'),
                    'rule_support': learning_result.get('rule_support'),
                }
                logger.info(f"✨ New rule created: {learning_result.get('rule_name')}")

        # Convert all numpy types in result to native Python types
        def convert_numpy_types_in_result(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_numpy_types_in_result(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types_in_result(item) for item in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_types_in_result(obj.tolist())
            elif pd.isna(obj):
                return None
            else:
                return obj

        result = convert_numpy_types_in_result(result)

        # Map preprocessor result to CorrectionResponse schema
        response_data = {
            'learned': bool(result.get('learned', False)),
            'pattern_recorded': bool(result.get('learned', False)),  # If learned, pattern was recorded
            'new_rule_created': bool(result.get('adjustment_active', False)),  # Adjustment = rule created
            'rule_name': result.get('pattern_category'),  # Pattern category is the rule name
            'rule_confidence': float(result.get('confidence_boost', '0').replace('+', '')) if 'confidence_boost' in result else None,
            'similar_patterns_count': int(result.get('correction_support', 0)) if result.get('correction_support') else None,
            'applicable_to': result.get('applicable_to'),
            'production_ready': bool(result.get('production_ready', False)),
            'pattern_corrections': int(result.get('pattern_corrections', 0)),
            'corrections_needed_for_training': int(result.get('corrections_needed_for_training', 0)),
            'corrections_needed_for_production': int(result.get('corrections_needed_for_production', 0)),
            'preferred_action': result.get('preferred_action'),
            'message': result.get('message')
        }

        return CorrectionResponse(**response_data)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid correction: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing correction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Correction processing failed: {str(e)}"
        )


@app.get("/explain/{decision_id}", response_model=ExplanationResponse)
async def explain_decision(decision_id: str):
    """
    Get detailed explanation for a decision.

    Args:
        decision_id: Decision ID from preprocessing response

    Returns:
        Detailed explanation with context
    """
    if decision_id not in decision_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Decision {decision_id} not found in cache"
        )

    cached = decision_cache[decision_id]
    result = cached["result"]

    # Convert alternatives to JSON-serializable format
    alternatives = [
        AlternativeAction(
            action=action.value,
            confidence=float(conf) if isinstance(conf, (np.floating,)) else conf
        )
        for action, conf in result.alternatives
    ]

    # Convert context to JSON-serializable types
    json_safe_context = convert_to_json_serializable(result.context) if result.context else None

    response = ExplanationResponse(
        decision_id=decision_id,
        action=result.action.value,
        confidence=float(result.confidence),
        source=result.source,
        explanation=result.explanation,
        alternatives=alternatives,
        context=json_safe_context
    )

    # Use FastAPI's jsonable_encoder to ensure ALL types are JSON-safe
    return JSONResponse(content=jsonable_encoder(response))


@app.get("/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """
    Get system statistics.

    Returns:
        Statistics about preprocessing decisions and performance
    """
    try:
        stats = preprocessor.get_statistics()
        return StatisticsResponse(**stats)

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@app.post("/statistics/reset")
async def reset_statistics():
    """Reset all statistics."""
    try:
        preprocessor.reset_statistics()
        return {"message": "Statistics reset successfully"}

    except Exception as e:
        logger.error(f"Error resetting statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset statistics: {str(e)}"
        )


@app.post("/patterns/save")
async def save_patterns(filepath: str = "patterns.json"):
    """
    Save learned patterns to disk.

    Args:
        filepath: Path to save patterns

    Returns:
        Success message
    """
    try:
        path = Path(filepath)
        preprocessor.save_learned_patterns(path)
        return {"message": f"Patterns saved to {filepath}"}

    except Exception as e:
        logger.error(f"Error saving patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save patterns: {str(e)}"
        )


@app.post("/patterns/load")
async def load_patterns(filepath: str = "patterns.json"):
    """
    Load learned patterns from disk.

    Args:
        filepath: Path to load patterns from

    Returns:
        Success message
    """
    try:
        path = Path(filepath)
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pattern file not found: {filepath}"
            )

        preprocessor.load_learned_patterns(path)
        return {"message": f"Patterns loaded from {filepath}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load patterns: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


@app.get("/metrics/performance")
async def get_performance_metrics():
    """Get detailed performance metrics for all components."""
    try:
        # Get monitor stats (real-time performance tracking)
        summary = monitor.get_summary()

        # Get preprocessor stats (decision statistics and learning)
        preprocessor_stats = preprocessor.get_statistics()

        # Merge both sources - monitor stats take precedence for overlapping keys
        # but we add preprocessor-specific stats like learning
        merged_stats = {
            **summary,
            'overview': {
                **summary.get('overview', {}),
                # Add total_decisions from preprocessor if monitor is empty
                'total_decisions': summary.get('overview', {}).get('total_decisions', 0) or preprocessor_stats.get('total_decisions', 0)
            },
            'learning': preprocessor_stats.get('learning', {
                'learned_rules': 0,
                'total_corrections': 0,
                'pattern_clusters': 0
            })
        }

        return merged_stats
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@app.get("/metrics/decisions")
async def get_recent_decisions(limit: int = 100):
    """Get recent preprocessing decisions with metrics."""
    try:
        decisions = monitor.get_recent_decisions(limit=limit)
        return {
            "decisions": [
                {
                    "timestamp": d.timestamp,
                    "decision_id": d.decision_id,
                    "column_name": d.column_name,
                    "action": d.action,
                    "confidence": d.confidence,
                    "source": d.source,
                    "latency_ms": d.latency_ms,
                    "num_rows": d.num_rows
                }
                for d in decisions
            ],
            "total": len(decisions)
        }
    except Exception as e:
        logger.error(f"Error getting decisions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get decisions: {str(e)}"
        )


@app.get("/metrics/realtime")
async def get_realtime_metrics():
    """Get real-time system metrics."""
    try:
        import psutil

        return {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024 ** 3),
            "active_requests": len(decision_cache),
            "recent_decisions": len(monitor.get_recent_decisions(limit=10))
        }
    except Exception as e:
        logger.error(f"Error getting realtime metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get realtime metrics: {str(e)}"
        )


# Phase 1: Removed - Cache endpoints deleted (cache system removed in Option B)
# /cache/stats and /cache/clear endpoints removed





@app.get("/metrics/neural_oracle")
async def get_neural_oracle_metrics():
    """
    Get neural oracle specific metrics and training status.

    Returns:
        Neural oracle performance, model info, and learning statistics
    """
    try:
        from pathlib import Path
        import json

        metrics = {
            "model_loaded": preprocessor.neural_oracle.model is not None if preprocessor.neural_oracle else False,
            "model_info": {},
            "training_history": {},
            "usage_stats": {}
        }

        # Get model information
        if preprocessor.neural_oracle and preprocessor.neural_oracle.model:
            oracle = preprocessor.neural_oracle

            metrics["model_info"] = {
                "model_size_kb": oracle.get_model_size() / 1024,
                "num_actions": len(oracle.action_encoder),
                "feature_names": oracle.feature_names,
                "top_features": [
                    {"name": name, "importance": importance}
                    for name, importance in oracle.get_top_features(top_k=5)
                ]
            }

            # Load training metadata if available
            model_path = Path(__file__).parent.parent.parent / "models" / "neural_oracle_v1.json"
            if model_path.exists():
                with open(model_path, 'r') as f:
                    metrics["training_history"] = json.load(f)

        # Get usage statistics from monitor
        component_stats = monitor.get_component_metrics('neural_oracle')
        if component_stats:
            metrics["usage_stats"] = {
                "total_calls": component_stats.total_calls,
                "avg_latency_ms": component_stats.avg_latency_ms,
                "p95_latency_ms": component_stats.p95_latency_ms,
                "success_rate": (component_stats.successful_calls / component_stats.total_calls * 100)
                                if component_stats.total_calls > 0 else 0
            }

        return metrics

    except Exception as e:
        logger.error(f"Error getting neural oracle metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get neural oracle metrics: {str(e)}"
        )


@app.get("/metrics/learning")
async def get_learning_metrics():
    """
    Get adaptive learning progress and correction statistics.

    Returns:
        Learning engine statistics, correction counts, rule creation progress
    """
    try:
        from ..database.connection import SessionLocal
        from ..database.models import CorrectionRecord, LearnedRule
        from sqlalchemy import func

        db = SessionLocal()
        try:
            # Get correction statistics
            total_corrections = db.query(func.count(CorrectionRecord.id)).scalar() or 0

            # Get action distribution of corrections
            corrections_by_action = db.query(
                CorrectionRecord.correct_action,
                func.count(CorrectionRecord.id).label('count')
            ).group_by(CorrectionRecord.correct_action).all()

            # Get learned rules
            total_rules = db.query(func.count(LearnedRule.id)).scalar() or 0
            active_rules = db.query(func.count(LearnedRule.id)).filter(
                LearnedRule.is_active == True
            ).scalar() or 0

            # Get top learned rules by support
            top_rules = db.query(LearnedRule).order_by(
                LearnedRule.support_count.desc()
            ).limit(10).all()

            # Calculate learning velocity (corrections per day)
            from datetime import datetime, timedelta
            last_week = datetime.utcnow() - timedelta(days=7)
            recent_corrections = db.query(func.count(CorrectionRecord.id)).filter(
                CorrectionRecord.timestamp >= last_week
            ).scalar() or 0

            return {
                "corrections": {
                    "total": total_corrections,
                    "last_7_days": recent_corrections,
                    "velocity_per_day": recent_corrections / 7,
                    "by_action": [
                        {"action": action, "count": count}
                        for action, count in corrections_by_action
                    ]
                },
                "learned_rules": {
                    "total": total_rules,
                    "active": active_rules,
                    "inactive": total_rules - active_rules,
                    "top_rules": [
                        {
                            "rule_name": rule.rule_name,
                            "action": rule.recommended_action,
                            "support_count": rule.support_count,
                            "confidence": rule.base_confidence,
                            "accuracy": (rule.validation_successes /
                                       (rule.validation_successes + rule.validation_failures) * 100)
                                      if (rule.validation_successes + rule.validation_failures) > 0 else 0
                        }
                        for rule in top_rules
                    ]
                },
                "learning_engine": {
                    "enabled": learning_engine is not None,
                    "min_support": learning_engine.min_support if learning_engine else 5,
                    "similarity_threshold": learning_engine.similarity_threshold if learning_engine else 0.85
                }
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error getting learning metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get learning metrics: {str(e)}"
        )


@app.get("/metrics/dashboard")
async def get_dashboard_metrics():
    """
    Get comprehensive dashboard metrics for UI display.

    Returns:
        All key metrics in one response for dashboard rendering
    """
    try:
        # Get all metrics in parallel
        performance = await get_performance_metrics()
        neural_oracle = await get_neural_oracle_metrics()
        learning = await get_learning_metrics()
        realtime = await get_realtime_metrics()

        # Get decision source breakdown
        from ..database.connection import SessionLocal
        db = SessionLocal()
        try:
            # Use preprocessor stats for source breakdown
            stats = preprocessor.get_statistics()

            decision_sources = {
                "learned": stats.get('learned_patterns_used', 0),
                "symbolic": stats.get('symbolic_decisions', 0),
                "neural": stats.get('neural_decisions', 0)
            }

            total_decisions = sum(decision_sources.values())
            decision_percentages = {
                source: (count / total_decisions * 100) if total_decisions > 0 else 0
                for source, count in decision_sources.items()
            }

        finally:
            db.close()

        return {
            "timestamp": time.time(),
            "overview": {
                "total_decisions": performance.get('overview', {}).get('total_decisions', 0),
                "avg_latency_ms": performance.get('overview', {}).get('avg_latency_ms', 0),
                "system_cpu": realtime.get('cpu_percent', 0),
                "system_memory": realtime.get('memory_percent', 0),
                "uptime_hours": (time.time() - preprocessor.get_statistics().get('start_time', time.time())) / 3600
            },
            "decision_sources": {
                "counts": decision_sources,
                "percentages": decision_percentages
            },
            "neural_oracle": neural_oracle,
            "learning": learning,
            "performance_by_component": performance.get('components', {})
        }

    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard metrics: {str(e)}"
        )



# =============================================================================
# VALIDATION & METRICS ENDPOINTS
# =============================================================================

from ..validation.metrics_tracker import get_metrics_tracker

# Initialize validation components
metrics_tracker = get_metrics_tracker()



@app.get("/validation/metrics")
async def get_performance_metrics():
    """Get detailed performance metrics."""
    try:
        metrics = metrics_tracker.get_performance_metrics()
        return {
            "success": True,
            "metrics": {
                "usage": {
                    "total_decisions": metrics.total_decisions,
                    "total_users": metrics.total_users,
                    "total_sessions": metrics.total_sessions,
                },
                "quality": {
                    "average_confidence": metrics.average_confidence,
                    "acceptance_rate": metrics.acceptance_rate,
                    "override_rate": metrics.override_rate,
                },
                "performance": {
                    "average_processing_time_ms": metrics.average_processing_time_ms,
                    "total_time_saved_hours": metrics.total_time_saved_hours,
                    "average_time_saved_per_decision_seconds": metrics.average_time_saved_per_decision_seconds,
                    "time_vs_manual_percentage": metrics.time_vs_manual_percentage,
                },
                "decision_sources": {
                    "symbolic": metrics.symbolic_decisions,
                    "neural": metrics.neural_decisions,
                    "learned": metrics.learned_decisions,
                    "meta_learning": metrics.meta_learning_decisions,
                },
                "user_satisfaction": {
                    "average_rating": metrics.average_user_rating,
                    "explanation_helpfulness_rate": metrics.explanation_helpfulness_rate,
                    "learning_rate": metrics.learning_rate,
                    "recommendation_rate": metrics.recommendation_rate,
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )






@app.post("/validation/track-decision")
async def track_decision_for_validation(
    decision_id: str,
    column_name: str,
    action_taken: str,
    confidence: float,
    source: str,
    processing_time_ms: float,
    user_id: str = "anonymous"
):
    """
    Track a preprocessing decision for validation metrics.

    Called automatically after each preprocessing decision.
    """
    try:
        # Ensure session exists
        if not metrics_tracker.current_session:
            metrics_tracker.start_session(user_id)

        # Track the decision
        metrics_tracker.track_decision(
            decision_id=decision_id,
            column_name=column_name,
            action_taken=action_taken,
            confidence=confidence,
            source=source,
            processing_time_ms=processing_time_ms,
            estimated_manual_time_seconds=60.0  # Default estimate
        )

        return {"success": True, "message": "Decision tracked"}

    except Exception as e:
        logger.error(f"Error tracking decision: {e}", exc_info=True)
        # Don't fail the request if tracking fails
        return {"success": False, "message": "Tracking failed but operation completed"}


@app.post("/validation/record-user-action")
async def record_user_action(
    decision_id: str,
    accepted: bool,
    rating: Optional[int] = None,
    alternative_chosen: Optional[str] = None,
    explanation_helpful: Optional[bool] = None
):
    """
    Record user action on a decision (accept/override/rate).
    """
    try:
        metrics_tracker.record_user_feedback(
            decision_id=decision_id,
            accepted=accepted,
            rating=rating,
            alternative_chosen=alternative_chosen,
            explanation_helpful=explanation_helpful
        )

        return {"success": True, "message": "User action recorded"}

    except Exception as e:
        logger.error(f"Error recording user action: {e}", exc_info=True)
        return {"success": False, "message": "Recording failed"}
