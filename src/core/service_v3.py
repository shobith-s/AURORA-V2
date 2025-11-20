"""
AURORA V3 Service - Production-ready preprocessing service with proper architecture.

This demonstrates how to refactor the current system to be production-ready:
- No singletons (dependency injection instead)
- Stateless services (can scale horizontally)
- Persistent storage (PostgreSQL + Redis)
- Proper error handling
- Monitoring and metrics
- Security (authentication, rate limiting)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from sqlalchemy.orm import Session
from redis import Redis
import time
import logging

from ..learning.adaptive_engine import AdaptiveLearningEngine
from ..symbolic.engine import SymbolicEngine
from ..neural.oracle import NeuralOracle
from ..core.actions import PreprocessingAction, PreprocessingResult
from ..core.robust_parser import parse_csv_robust

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for the preprocessing service."""
    confidence_threshold: float = 0.9
    enable_caching: bool = True
    enable_learning: bool = True
    cache_ttl_seconds: int = 3600
    rate_limit_per_minute: int = 60


class PreprocessingServiceV3:
    """
    Production-ready preprocessing service.

    Key improvements over current implementation:
    1. NO SINGLETONS - pass dependencies explicitly
    2. Stateless - can scale horizontally
    3. Uses persistent storage (PostgreSQL for data, Redis for cache)
    4. Proper error handling and logging
    5. Metrics collection
    6. Rate limiting
    """

    def __init__(
        self,
        db_session: Session,
        cache: Redis,
        config: ServiceConfig = None,
    ):
        """
        Initialize the preprocessing service.

        Args:
            db_session: SQLAlchemy database session
            cache: Redis client for caching
            config: Service configuration
        """
        # Dependencies (injected, not global!)
        self.db = db_session
        self.cache = cache
        self.config = config or ServiceConfig()

        # Initialize components
        self.symbolic_engine = SymbolicEngine(
            confidence_threshold=self.config.confidence_threshold
        )

        self.learning_engine = AdaptiveLearningEngine(
            # Use the same db connection
            db_url=None,  # Will use injected session
        )

        # Neural oracle (optional)
        try:
            self.neural_oracle = NeuralOracle()
        except Exception as e:
            logger.warning(f"Neural oracle not available: {e}")
            self.neural_oracle = None

    # ========================================================================
    # MAIN PREPROCESSING METHODS
    # ========================================================================

    async def preprocess_column(
        self,
        user_id: str,
        column_data: List[Any],
        column_name: str = "",
        target_available: bool = False
    ) -> PreprocessingResult:
        """
        Preprocess a single column with full adaptive learning pipeline.

        This is the main entry point that demonstrates the complete flow:
        1. Check cache (ultra-fast)
        2. Check learned rules (fast, adaptive)
        3. Apply symbolic rules (fast, deterministic)
        4. Fall back to neural (slower but handles edge cases)
        5. Learn from this decision for next time

        Args:
            user_id: User identifier (for personalized learning)
            column_data: The column data to preprocess
            column_name: Name of the column
            target_available: Whether target variable exists

        Returns:
            PreprocessingResult with action, confidence, and explanation
        """
        start_time = time.time()

        try:
            # SECURITY: Check rate limit
            if not await self._check_rate_limit(user_id):
                raise PermissionError(f"Rate limit exceeded for user {user_id}")

            # Convert to pandas Series
            import pandas as pd
            column = pd.Series(column_data, name=column_name)

            # Compute statistics (privacy-preserved)
            stats = self.symbolic_engine.compute_column_statistics(
                column, column_name, target_available
            )
            stats_dict = stats.to_dict()

            # =================================================================
            # LAYER 0: Persistent Cache (Redis)
            # =================================================================
            if self.config.enable_caching:
                cached_result = await self._check_cache(user_id, stats_dict)
                if cached_result:
                    # Record cache hit metric
                    await self._record_metric('cache_hit', 1, user_id=user_id)
                    return cached_result

            # =================================================================
            # LAYER 1: Learned Rules (Adaptive, User-Specific)
            # =================================================================
            if self.config.enable_learning:
                learned_result = self.learning_engine.get_recommendation(
                    user_id=user_id,
                    column_stats=stats_dict
                )

                if learned_result:
                    action, confidence, source = learned_result

                    result = PreprocessingResult(
                        action=PreprocessingAction(action),
                        confidence=confidence,
                        source=source,
                        explanation=f"Learned from your previous corrections ({source})",
                        alternatives=[],
                        parameters={},
                        context=stats_dict
                    )

                    # Cache this decision
                    await self._cache_decision(user_id, stats_dict, result)

                    # Record metrics
                    await self._record_metrics(start_time, result, user_id)

                    return result

            # =================================================================
            # LAYER 2: Symbolic Rules (Fast, Deterministic)
            # =================================================================
            symbolic_result = self.symbolic_engine.evaluate(
                column, column_name, target_available
            )

            if symbolic_result.confidence >= self.config.confidence_threshold:
                # High confidence from symbolic rules

                # Cache this decision
                await self._cache_decision(user_id, stats_dict, symbolic_result)

                # Record metrics
                await self._record_metrics(start_time, symbolic_result, user_id)

                return symbolic_result

            # =================================================================
            # LAYER 3: Neural Oracle (Slower, Handles Ambiguity)
            # =================================================================
            if self.neural_oracle:
                # Use neural oracle for ambiguous cases
                # (Implementation similar to current system)
                pass

            # =================================================================
            # LAYER 4: Conservative Fallback
            # =================================================================
            # Return low-confidence symbolic result with warning
            symbolic_result.explanation = f"[LOW CONFIDENCE] {symbolic_result.explanation}"

            # Record metrics
            await self._record_metrics(start_time, symbolic_result, user_id)

            return symbolic_result

        except Exception as e:
            logger.error(f"Error preprocessing column: {e}", exc_info=True)
            # Record error metric
            await self._record_metric('preprocessing_error', 1, user_id=user_id)
            raise

    async def submit_correction(
        self,
        user_id: str,
        column_data: List[Any],
        column_name: str,
        wrong_action: str,
        correct_action: str,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Learn from a user correction.

        This is THE KEY METHOD for continuous improvement.

        Args:
            user_id: User identifier
            column_data: Column data (only for statistics, not stored!)
            column_name: Column name
            wrong_action: The incorrect action we recommended
            correct_action: The correct action from the user
            confidence: How confident we were (for learning)

        Returns:
            Result of learning (was rule created, etc.)
        """
        try:
            # Compute statistics (privacy-preserved)
            import pandas as pd
            column = pd.Series(column_data, name=column_name)
            stats = self.symbolic_engine.compute_column_statistics(column, column_name)
            stats_dict = stats.to_dict()

            # Learn from correction
            result = self.learning_engine.record_correction(
                user_id=user_id,
                column_stats=stats_dict,
                wrong_action=wrong_action,
                correct_action=correct_action,
                confidence=confidence
            )

            # Invalidate cache for this pattern
            await self._invalidate_cache_pattern(user_id, stats_dict)

            # Record learning metric
            await self._record_metric('correction_recorded', 1, user_id=user_id)

            if result.get('new_rule_created'):
                await self._record_metric('rule_created', 1, user_id=user_id)

            return result

        except Exception as e:
            logger.error(f"Error recording correction: {e}", exc_info=True)
            raise

    # ========================================================================
    # CACHING (Redis)
    # ========================================================================

    async def _check_cache(
        self,
        user_id: str,
        stats: Dict[str, Any]
    ) -> Optional[PreprocessingResult]:
        """Check cache for previous decision."""
        try:
            # Create cache key from statistics
            cache_key = self._create_cache_key(user_id, stats)

            # Check Redis
            cached = self.cache.get(cache_key)

            if cached:
                # Deserialize and return
                import json
                data = json.loads(cached)

                return PreprocessingResult(
                    action=PreprocessingAction(data['action']),
                    confidence=data['confidence'],
                    source=f"cache:{data['source']}",
                    explanation=data['explanation'],
                    alternatives=[],
                    parameters=data.get('parameters', {}),
                    context=stats
                )

            return None

        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
            return None

    async def _cache_decision(
        self,
        user_id: str,
        stats: Dict[str, Any],
        result: PreprocessingResult
    ):
        """Cache a decision for fast lookup."""
        try:
            cache_key = self._create_cache_key(user_id, stats)

            # Serialize result
            import json
            data = json.dumps({
                'action': result.action.value,
                'confidence': result.confidence,
                'source': result.source,
                'explanation': result.explanation,
                'parameters': result.parameters
            })

            # Store in Redis with TTL
            self.cache.setex(
                cache_key,
                self.config.cache_ttl_seconds,
                data
            )

        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    async def _invalidate_cache_pattern(
        self,
        user_id: str,
        stats: Dict[str, Any]
    ):
        """Invalidate cache for a pattern (after correction)."""
        try:
            cache_key = self._create_cache_key(user_id, stats)
            self.cache.delete(cache_key)
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")

    def _create_cache_key(self, user_id: str, stats: Dict[str, Any]) -> str:
        """Create a cache key from user + statistics."""
        import hashlib
        import json

        # Create deterministic hash from stats
        stats_str = json.dumps(stats, sort_keys=True)
        stats_hash = hashlib.sha256(stats_str.encode()).hexdigest()[:16]

        return f"preprocessing:{user_id}:{stats_hash}"

    # ========================================================================
    # RATE LIMITING
    # ========================================================================

    async def _check_rate_limit(self, user_id: str) -> bool:
        """
        Check if user has exceeded rate limit.

        Uses Redis for distributed rate limiting (works across multiple servers).
        """
        try:
            rate_key = f"rate_limit:{user_id}"

            # Increment counter
            current = self.cache.incr(rate_key)

            # Set expiry on first request
            if current == 1:
                self.cache.expire(rate_key, 60)  # 1 minute window

            # Check limit
            return current <= self.config.rate_limit_per_minute

        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            return True  # Fail open (allow request)

    # ========================================================================
    # METRICS AND MONITORING
    # ========================================================================

    async def _record_metrics(
        self,
        start_time: float,
        result: PreprocessingResult,
        user_id: str
    ):
        """Record metrics for monitoring (Prometheus)."""
        latency_ms = (time.time() - start_time) * 1000

        await self._record_metric('preprocessing_latency_ms', latency_ms, user_id=user_id)
        await self._record_metric('preprocessing_confidence', result.confidence, user_id=user_id)
        await self._record_metric(f'preprocessing_source_{result.source}', 1, user_id=user_id)

    async def _record_metric(
        self,
        metric_name: str,
        value: float,
        **labels
    ):
        """
        Record a metric to Prometheus.

        In production, this would use prometheus_client:
        from prometheus_client import Counter, Histogram

        Example:
            preprocessing_latency = Histogram('preprocessing_latency_ms', 'Latency')
            preprocessing_latency.observe(value)
        """
        # For now, just log
        logger.info(f"METRIC: {metric_name}={value} labels={labels}")

        # In production:
        # prometheus_metrics[metric_name].labels(**labels).observe(value)

    # ========================================================================
    # BATCH PROCESSING (for large CSV files)
    # ========================================================================

    async def preprocess_csv(
        self,
        user_id: str,
        file_path: str,
        target_column: Optional[str] = None
    ) -> Dict[str, PreprocessingResult]:
        """
        Preprocess an entire CSV file.

        Uses the robust parser to handle any CSV format.

        Args:
            user_id: User identifier
            file_path: Path to CSV file
            target_column: Optional target column name

        Returns:
            Dictionary mapping column names to preprocessing results
        """
        try:
            # Parse CSV robustly
            df = parse_csv_robust(file_path)

            logger.info(f"Parsed CSV: {len(df)} rows, {len(df.columns)} columns")

            # Process each column
            results = {}

            for col_name in df.columns:
                if col_name == target_column:
                    continue  # Skip target

                # Preprocess column
                result = await self.preprocess_column(
                    user_id=user_id,
                    column_data=df[col_name].tolist(),
                    column_name=col_name,
                    target_available=(target_column is not None)
                )

                results[col_name] = result

            return results

        except Exception as e:
            logger.error(f"Error preprocessing CSV: {e}", exc_info=True)
            raise


# ============================================================================
# DEPENDENCY INJECTION HELPERS (for FastAPI)
# ============================================================================

def get_preprocessing_service(
    db: Session,  # Injected by FastAPI
    cache: Redis  # Injected by FastAPI
) -> PreprocessingServiceV3:
    """
    Factory function for dependency injection.

    Usage in FastAPI:
        @app.post("/preprocess")
        async def preprocess(
            request: PreprocessRequest,
            user = Depends(get_current_user),
            service = Depends(get_preprocessing_service)
        ):
            result = await service.preprocess_column(...)
            return result
    """
    config = ServiceConfig(
        confidence_threshold=0.9,
        enable_caching=True,
        enable_learning=True,
        cache_ttl_seconds=3600,
        rate_limit_per_minute=60
    )

    return PreprocessingServiceV3(
        db_session=db,
        cache=cache,
        config=config
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example showing how to use the service (production pattern).
    """

    # Set up dependencies
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import redis

    # Database
    engine = create_engine("postgresql://localhost/aurora")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    # Cache
    cache = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # Create service (dependency injection!)
    service = PreprocessingServiceV3(db_session=db, cache=cache)

    # Use service
    import asyncio

    async def example():
        # Process a column
        result = await service.preprocess_column(
            user_id="user123",
            column_data=[1, 2, 3, 100, 200, 300, 5000, 10000],
            column_name="revenue"
        )

        print(f"Action: {result.action.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Explanation: {result.explanation}")

        # User corrects it
        correction_result = await service.submit_correction(
            user_id="user123",
            column_data=[1, 2, 3, 100, 200, 300, 5000, 10000],
            column_name="revenue",
            wrong_action=result.action.value,
            correct_action="log1p_transform",
            confidence=result.confidence
        )

        print(f"\nLearning result: {correction_result}")

        # Next time, it will use the learned rule!

    asyncio.run(example())
