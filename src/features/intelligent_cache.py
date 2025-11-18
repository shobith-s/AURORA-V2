"""
Multi-Level Intelligent Cache - Phase 1 Improvements.

Three-level caching strategy:
- L1: Exact feature match (hash-based)
- L2: Similar features (cosine similarity)
- L3: Pattern-based (rule matching)

Expected impact: 10-50x speedup on repeated similar columns.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import time


@dataclass
class CacheEntry:
    """Entry in the cache with metadata."""
    features: Any  # Feature dict or array
    decision: Any  # Preprocessing decision
    column_name: str
    timestamp: float
    hit_count: int = 0
    last_accessed: float = None

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp


class MultiLevelCache:
    """
    Multi-level caching system for preprocessing decisions.

    L1: Exact match (O(1) hash lookup)
    L2: Similar features (O(k) where k = cache size, with optimizations)
    L3: Pattern match (O(p) where p = number of patterns)
    """

    def __init__(self, max_size: int = 10000, similarity_threshold: float = 0.95):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries to cache
            similarity_threshold: Minimum cosine similarity for L2 cache hit
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

        # L1: Exact match cache (hash -> CacheEntry)
        self.exact_cache: Dict[str, CacheEntry] = {}

        # L2: Similar features cache (feature vectors for similarity search)
        self.feature_vectors: List[Tuple[str, np.ndarray]] = []  # (hash, features)

        # L3: Pattern cache (pattern -> decision)
        self.pattern_cache: Dict[str, Any] = {}

        # Statistics
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'misses': 0,
            'total_queries': 0
        }

    def get(self, features: Any, column_name: str = "") -> Tuple[Optional[Any], Optional[str]]:
        """
        Try to get decision from cache.

        Args:
            features: Features dict or array
            column_name: Name of column

        Returns:
            (decision, cache_level) where cache_level is 'l1', 'l2', 'l3', or None
        """
        self.stats['total_queries'] += 1

        # Convert features to dict and array
        if hasattr(features, 'to_dict'):
            features_dict = features.to_dict()
            features_array = features.to_array()
        elif isinstance(features, dict):
            features_dict = features
            features_array = self._dict_to_array(features_dict)
        else:
            features_array = np.array(features)
            features_dict = {}

        # L1: Exact match
        feature_hash = self._hash_features(features_dict)
        if feature_hash in self.exact_cache:
            entry = self.exact_cache[feature_hash]
            entry.hit_count += 1
            entry.last_accessed = time.time()
            self.stats['l1_hits'] += 1
            return entry.decision, 'l1'

        # L2: Similar features
        similar = self._find_similar(features_array, self.similarity_threshold)
        if similar:
            sim_hash, similarity = similar
            if sim_hash in self.exact_cache:
                entry = self.exact_cache[sim_hash]
                entry.hit_count += 1
                entry.last_accessed = time.time()
                self.stats['l2_hits'] += 1
                return entry.decision, 'l2'

        # L3: Pattern match
        pattern_decision = self._match_pattern(features_dict, column_name)
        if pattern_decision:
            self.stats['l3_hits'] += 1
            return pattern_decision, 'l3'

        # Cache miss
        self.stats['misses'] += 1
        return None, None

    def set(self, features: Any, decision: Any, column_name: str = ""):
        """
        Add to cache.

        Args:
            features: Features dict or array
            decision: Preprocessing decision
            column_name: Name of column
        """
        # Convert features
        if hasattr(features, 'to_dict'):
            features_dict = features.to_dict()
            features_array = features.to_array()
        elif isinstance(features, dict):
            features_dict = features
            features_array = self._dict_to_array(features_dict)
        else:
            features_array = np.array(features)
            features_dict = {}

        feature_hash = self._hash_features(features_dict)

        # Add to L1 cache
        entry = CacheEntry(
            features=features_dict,
            decision=decision,
            column_name=column_name,
            timestamp=time.time()
        )

        self.exact_cache[feature_hash] = entry

        # Add to L2 similarity index
        self.feature_vectors.append((feature_hash, features_array))

        # Learn pattern for L3 (if applicable)
        self._learn_pattern(features_dict, decision, column_name)

        # Evict if cache is full
        if len(self.exact_cache) > self.max_size:
            self._evict_lru()

    def _hash_features(self, features_dict: Dict) -> str:
        """Create hash from features for exact matching."""

        # Convert numpy types to native Python types for JSON serialization
        def convert_for_json(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_for_json(obj.tolist())
            else:
                return obj

        # Convert features to JSON-serializable format
        json_safe_features = convert_for_json(features_dict)

        # Sort keys for consistent hashing
        feature_str = json.dumps(json_safe_features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()

    def _find_similar(self, query_features: np.ndarray,
                     threshold: float) -> Optional[Tuple[str, float]]:
        """
        Find similar features using cosine similarity.

        Args:
            query_features: Query feature vector
            threshold: Minimum similarity threshold

        Returns:
            (hash, similarity) of most similar entry, or None
        """
        if len(self.feature_vectors) == 0:
            return None

        best_sim = 0.0
        best_hash = None

        # Normalize query
        query_norm = np.linalg.norm(query_features)
        if query_norm == 0:
            return None

        query_normalized = query_features / query_norm

        # Find most similar (linear search for now; could use FAISS for large scale)
        for feat_hash, feat_vector in self.feature_vectors:
            # Cosine similarity
            feat_norm = np.linalg.norm(feat_vector)
            if feat_norm == 0:
                continue

            similarity = np.dot(query_normalized, feat_vector / feat_norm)

            if similarity > best_sim:
                best_sim = similarity
                best_hash = feat_hash

        if best_sim >= threshold:
            return best_hash, best_sim

        return None

    def _match_pattern(self, features_dict: Dict, column_name: str) -> Optional[Any]:
        """
        Match against learned patterns.

        Pattern examples:
        - All columns named "*_id" with unique_ratio > 0.95 -> DROP
        - All columns with email_ratio > 0.8 -> KEEP (or special handling)
        """
        # Pattern 1: ID columns
        if column_name.endswith('_id') or column_name.endswith('_ID'):
            unique_ratio = features_dict.get('unique_ratio', 0)
            if unique_ratio > 0.95:
                pattern_key = 'id_column_drop'
                if pattern_key in self.pattern_cache:
                    return self.pattern_cache[pattern_key]

        # Pattern 2: Email columns
        email_ratio = features_dict.get('email_ratio', 0)
        if email_ratio > 0.8:
            pattern_key = 'email_column'
            if pattern_key in self.pattern_cache:
                return self.pattern_cache[pattern_key]

        # Pattern 3: Constant columns
        unique_ratio = features_dict.get('unique_ratio', 0)
        if unique_ratio < 0.01:  # Less than 1% unique
            pattern_key = 'constant_column_drop'
            if pattern_key in self.pattern_cache:
                return self.pattern_cache[pattern_key]

        # Pattern 4: High null columns
        null_pct = features_dict.get('null_percentage', 0)
        if null_pct > 70:
            pattern_key = 'high_null_drop'
            if pattern_key in self.pattern_cache:
                return self.pattern_cache[pattern_key]

        return None

    def _learn_pattern(self, features_dict: Dict, decision: Any, column_name: str):
        """
        Learn patterns from decisions.

        After seeing multiple similar decisions, create a pattern rule.
        """
        # Check if this is a pattern-worthy decision
        unique_ratio = features_dict.get('unique_ratio', 0)
        null_pct = features_dict.get('null_percentage', 0)

        # Learn ID column pattern
        if (column_name.endswith('_id') or column_name.endswith('_ID')) and unique_ratio > 0.95:
            pattern_key = 'id_column_drop'
            if pattern_key not in self.pattern_cache:
                self.pattern_cache[pattern_key] = decision

        # Learn constant column pattern
        if unique_ratio < 0.01:
            pattern_key = 'constant_column_drop'
            if pattern_key not in self.pattern_cache:
                self.pattern_cache[pattern_key] = decision

        # Learn high null pattern
        if null_pct > 70:
            pattern_key = 'high_null_drop'
            if pattern_key not in self.pattern_cache:
                self.pattern_cache[pattern_key] = decision

    def _evict_lru(self):
        """Evict least recently used entries."""
        # Find entry with lowest access time
        min_time = float('inf')
        evict_hash = None

        for hash_key, entry in self.exact_cache.items():
            if entry.last_accessed < min_time:
                min_time = entry.last_accessed
                evict_hash = hash_key

        if evict_hash:
            # Remove from exact cache
            del self.exact_cache[evict_hash]

            # Remove from feature vectors
            self.feature_vectors = [
                (h, f) for h, f in self.feature_vectors if h != evict_hash
            ]

    def _dict_to_array(self, features_dict: Dict) -> np.ndarray:
        """Convert feature dict to array."""
        # Extract numeric values in consistent order
        keys = sorted(features_dict.keys())
        values = []
        for k in keys:
            v = features_dict[k]
            if isinstance(v, bool):
                values.append(float(v))
            elif isinstance(v, (int, float)):
                values.append(float(v))
            else:
                values.append(0.0)
        return np.array(values, dtype=np.float32)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats['total_queries']
        if total == 0:
            hit_rate = 0.0
        else:
            hits = self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['l3_hits']
            hit_rate = hits / total

        return {
            'total_queries': total,
            'l1_hits': self.stats['l1_hits'],
            'l2_hits': self.stats['l2_hits'],
            'l3_hits': self.stats['l3_hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'cache_size': len(self.exact_cache),
            'pattern_rules': len(self.pattern_cache)
        }

    def clear(self):
        """Clear all caches."""
        self.exact_cache.clear()
        self.feature_vectors.clear()
        self.pattern_cache.clear()
        self.stats = {k: 0 for k in self.stats}

    def warm_up(self, columns_and_decisions: List[Tuple[Any, Any, str]]):
        """
        Warm up cache with known decisions.

        Args:
            columns_and_decisions: List of (features, decision, column_name) tuples
        """
        for features, decision, column_name in columns_and_decisions:
            self.set(features, decision, column_name)


# Global cache instance
_global_cache: Optional[MultiLevelCache] = None


def get_cache() -> MultiLevelCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MultiLevelCache()
    return _global_cache


def clear_cache():
    """Clear global cache."""
    global _global_cache
    if _global_cache:
        _global_cache.clear()
