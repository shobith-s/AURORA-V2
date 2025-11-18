"""
FastAPI server for AURORA preprocessing system.
Provides REST API for intelligent data preprocessing.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    AlternativeAction
)
from ..core.preprocessor import IntelligentPreprocessor, get_preprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AURORA - Intelligent Data Preprocessing System",
    description="Production-ready intelligent data preprocessing with symbolic rules, neural intelligence, and privacy-preserving learning",
    version="1.0.0",
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
        # Process column
        result = preprocessor.preprocess_column(
            column=request.column_data,
            column_name=request.column_name,
            target_available=request.target_available,
            metadata=request.metadata
        )

        # Cache decision for explanation endpoint
        if result.decision_id and len(decision_cache) < MAX_CACHE_SIZE:
            decision_cache[result.decision_id] = {
                "result": result,
                "request": request.dict()
            }

        # Convert to response format
        alternatives = [
            AlternativeAction(action=action.value, confidence=conf)
            for action, conf in result.alternatives
        ]

        return PreprocessResponse(
            action=result.action.value,
            confidence=result.confidence,
            source=result.source,
            explanation=result.explanation,
            alternatives=alternatives,
            parameters=result.parameters,
            decision_id=result.decision_id
        )

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
        # Convert to DataFrame
        df = pd.DataFrame(request.columns)

        # Process all columns
        results_dict = preprocessor.preprocess_dataframe(
            df,
            target_column=request.target_column
        )

        # Convert results to response format
        results = {}
        total_confidence = 0.0
        source_breakdown = {}

        for col_name, result in results_dict.items():
            alternatives = [
                AlternativeAction(action=action.value, confidence=conf)
                for action, conf in result.alternatives
            ]

            results[col_name] = PreprocessResponse(
                action=result.action.value,
                confidence=result.confidence,
                source=result.source,
                explanation=result.explanation,
                alternatives=alternatives,
                parameters=result.parameters,
                decision_id=result.decision_id
            )

            total_confidence += result.confidence
            source_breakdown[result.source] = source_breakdown.get(result.source, 0) + 1

        # Create summary
        summary = {
            "total_columns": len(results),
            "avg_confidence": total_confidence / len(results) if results else 0.0,
            "source_breakdown": source_breakdown
        }

        return BatchPreprocessResponse(
            results=results,
            summary=summary
        )

    except Exception as e:
        logger.error(f"Error in batch preprocessing: {e}")
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

    alternatives = [
        AlternativeAction(action=action.value, confidence=conf)
        for action, conf in result.alternatives
    ]

    return ExplanationResponse(
        decision_id=decision_id,
        action=result.action.value,
        confidence=result.confidence,
        source=result.source,
        explanation=result.explanation,
        alternatives=alternatives,
        context=result.context
    )


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
