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
from typing import Dict, Any
import logging

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
    AlternativeAction,
    CacheStatsResponse,
    DriftMonitoringResponse,
    DriftStatus
)
from ..core.preprocessor import IntelligentPreprocessor, get_preprocessor
from ..utils.monitor import get_monitor, PerformanceTimer
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global preprocessor instance
preprocessor: IntelligentPreprocessor = None

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
    """Initialize the preprocessor on startup."""
    global preprocessor
    logger.info("Starting AURORA preprocessing system...")

    try:
        preprocessor = get_preprocessor(
            confidence_threshold=0.9,
            use_neural_oracle=True,
            enable_learning=True
        )
        logger.info("AURORA initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AURORA: {e}")
        # Create basic preprocessor without neural oracle
        preprocessor = IntelligentPreprocessor(
            confidence_threshold=0.9,
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
    """Health check endpoint."""
    components = {
        "symbolic_engine": "ok",
        "pattern_learner": "ok" if preprocessor.enable_learning else "disabled",
        "neural_oracle": "ok" if preprocessor.use_neural_oracle and preprocessor.neural_oracle else "unavailable"
    }

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components=components
    )


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
        # Time the operation
        start_time = time.time()

        # Process column
        result = preprocessor.preprocess_column(
            column=request.column_data,
            column_name=request.column_name,
            target_available=request.target_available,
            metadata=request.metadata
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
            enhanced_features=None,  # Phase 1 field - not yet implemented
            cache_info=None  # Phase 1 field - not yet implemented
        )

        # Use FastAPI's jsonable_encoder to ensure ALL types are JSON-safe
        return JSONResponse(content=jsonable_encoder(response))

    except Exception as e:
        logger.error(f"Error preprocessing column: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {str(e)}"
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
            target_column=request.target_column
        )

        # Record overall pipeline latency
        batch_latency_ms = (time.time() - start_time) * 1000
        monitor.record_call('overall_pipeline', batch_latency_ms, success=True)

        # Convert results to response format
        results = {}
        total_confidence = 0.0
        source_breakdown = {}
        processed_count = 0  # Columns that actually need preprocessing

        for col_name, result in results_dict.items():
            # Record decision in monitor (for all columns, not just those that need preprocessing)
            monitor.record_decision(
                decision_id=result.decision_id,
                column_name=col_name,
                action=result.action.value,
                confidence=result.confidence,
                source=result.source,
                latency_ms=batch_latency_ms / len(results_dict),  # Approximate per-column latency
                num_rows=len(df)
            )

            # Skip columns that don't need preprocessing (action = "keep")
            # These columns are already clean and should not be shown to the user
            if result.action.value.lower() == 'keep':
                continue

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
            processed_count += 1

        # Create summary - ensure all values are JSON-serializable
        summary = {
            "total_columns": int(len(results_dict)),  # Total columns analyzed
            "processed_columns": int(processed_count),  # Columns that need preprocessing
            "avg_confidence": float(total_confidence / processed_count) if processed_count > 0 else 0.0,
            "source_breakdown": {k: int(v) for k, v in source_breakdown.items()}
        }

        # Convert summary to JSON-safe format just to be absolutely sure
        json_safe_summary = convert_to_json_serializable(summary)

        # Create response
        response = BatchPreprocessResponse(
            results=results,
            summary=json_safe_summary
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


@app.post("/correct", response_model=CorrectionResponse)
async def submit_correction(request: CorrectionRequest):
    """
    Submit a correction to learn from (privacy-preserving).

    Args:
        request: Correction with wrong and correct actions

    Returns:
        Learning result
    """
    try:
        result = preprocessor.process_correction(
            column=request.column_data,
            column_name=request.column_name,
            wrong_action=request.wrong_action,
            correct_action=request.correct_action,
            confidence=request.confidence
        )

        return CorrectionResponse(**result)

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


# Phase 1: Cache and Drift Detection Endpoints

@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_statistics():
    """
    Get intelligent cache statistics.

    Returns cache hit rates, levels, and pattern rules.
    """
    try:
        from ..features.intelligent_cache import get_cache

        cache = get_cache()
        stats = cache.get_stats()

        return CacheStatsResponse(
            total_queries=stats['total_queries'],
            l1_hits=stats['l1_hits'],
            l2_hits=stats['l2_hits'],
            l3_hits=stats['l3_hits'],
            misses=stats['misses'],
            hit_rate=stats['hit_rate'],
            cache_size=stats['cache_size'],
            pattern_rules=stats['pattern_rules']
        )
    except Exception as e:
        logger.error(f"Error getting cache statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache statistics: {str(e)}"
        )


@app.post("/cache/clear")
async def clear_cache():
    """Clear the intelligent cache."""
    try:
        from ..features.intelligent_cache import clear_cache

        clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@app.get("/drift/status", response_model=DriftMonitoringResponse)
async def get_drift_status():
    """
    Get drift monitoring status for all tracked columns.

    Returns drift detection results and retraining recommendations.
    """
    try:
        from ..monitoring.drift_detector import get_drift_detector

        detector = get_drift_detector()

        # Get recent drift reports
        drift_reports = []
        critical_columns = []
        high_priority_columns = []
        columns_with_drift = 0

        for col_name, profile in detector.reference_profiles.items():
            # Find recent drift report for this column
            recent_reports = [r for r in detector.drift_history
                            if r.column_name == col_name]

            if recent_reports:
                latest = recent_reports[-1]

                drift_status = DriftStatus(
                    column_name=latest.column_name,
                    drift_detected=latest.drift_detected,
                    drift_score=latest.drift_score,
                    severity=latest.severity,
                    recommendation=latest.recommendation,
                    p_value=latest.p_value,
                    test_used=latest.test_used,
                    changes=latest.changes,
                    timestamp=latest.timestamp
                )
                drift_reports.append(drift_status)

                if latest.drift_detected:
                    columns_with_drift += 1

                    if latest.severity == 'critical':
                        critical_columns.append(col_name)
                    elif latest.severity == 'high':
                        high_priority_columns.append(col_name)

        requires_retraining = len(critical_columns) > 0 or len(high_priority_columns) >= 3

        return DriftMonitoringResponse(
            monitored_columns=len(detector.reference_profiles),
            columns_with_drift=columns_with_drift,
            critical_columns=critical_columns,
            high_priority_columns=high_priority_columns,
            requires_retraining=requires_retraining,
            drift_reports=drift_reports
        )
    except Exception as e:
        logger.error(f"Error getting drift status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get drift status: {str(e)}"
        )


@app.post("/drift/set_reference")
async def set_drift_reference(request: PreprocessRequest):
    """
    Set reference distribution for drift detection.

    Args:
        request: Column data to use as reference
    """
    try:
        from ..monitoring.drift_detector import get_drift_detector

        detector = get_drift_detector()
        column_series = pd.Series(request.column_data)

        detector.set_reference(
            column_name=request.column_name,
            column_data=column_series
        )

        return {
            "message": f"Reference set for column '{request.column_name}'",
            "column_name": request.column_name,
            "sample_size": len(request.column_data)
        }
    except Exception as e:
        logger.error(f"Error setting drift reference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set drift reference: {str(e)}"
        )


@app.post("/drift/check")
async def check_drift(request: PreprocessRequest):
    """
    Check a column for drift against its reference.

    Args:
        request: Column data to check for drift

    Returns:
        Drift detection report
    """
    try:
        from ..monitoring.drift_detector import get_drift_detector

        detector = get_drift_detector()
        column_series = pd.Series(request.column_data)

        report = detector.detect_drift(
            column_name=request.column_name,
            new_data=column_series
        )

        return DriftStatus(
            column_name=report.column_name,
            drift_detected=report.drift_detected,
            drift_score=report.drift_score,
            severity=report.severity,
            recommendation=report.recommendation,
            p_value=report.p_value,
            test_used=report.test_used,
            changes=report.changes,
            timestamp=report.timestamp
        )
    except Exception as e:
        logger.error(f"Error checking drift: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check drift: {str(e)}"
        )
