"""
Feature extraction caching system.
Caches expensive feature computations to improve performance.
"""

import hashlib
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class FeatureCache:
    """
    Cache for expensive feature extraction operations.
    Uses content-based hashing for cache keys.
    """

    def __init__(
        self,
        cache_dir: str = "./cache/features",
        max_cache_size_mb: int = 500,
        ttl_hours: int = 24
    ):
        """
        Initialize feature cache.

        Args:
            cache_dir: Directory to store cache files
            max_cache_size_mb: Maximum cache size in MB
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.ttl = timedelta(hours=ttl_hours)

        # Metadata file
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

        # Statistics
        self.hits = 0
        self.misses = 0

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_hash(self, data: Any) -> str:
        """
        Compute hash for cache key.

        Args:
            data: Data to hash (DataFrame, array, dict, etc.)

        Returns:
            Hash string
        """
        if isinstance(data, pd.DataFrame):
            # Hash DataFrame content
            content = data.to_json().encode('utf-8')
        elif isinstance(data, np.ndarray):
            # Hash array content
            content = data.tobytes()
        elif isinstance(data, dict):
            # Hash dict as JSON
            content = json.dumps(data, sort_keys=True).encode('utf-8')
        else:
            # Hash string representation
            content = str(data).encode('utf-8')

        return hashlib.sha256(content).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_expired(self, cache_key: str) -> bool:
        """Check if cache entry is expired."""
        if cache_key not in self.metadata:
            return True

        created_at = datetime.fromisoformat(self.metadata[cache_key]['created_at'])
        return datetime.now() - created_at > self.ttl

    def get(self, data: Any, feature_name: str = "") -> Optional[Any]:
        """
        Get cached features if available.

        Args:
            data: Input data
            feature_name: Optional feature name for namespacing

        Returns:
            Cached features or None
        """
        # Compute cache key
        data_hash = self._compute_hash(data)
        cache_key = f"{feature_name}_{data_hash}" if feature_name else data_hash

        # Check if cached and not expired
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists() and not self._is_expired(cache_key):
            try:
                with open(cache_path, 'rb') as f:
                    features = pickle.load(f)

                self.hits += 1
                self.metadata[cache_key]['last_accessed'] = datetime.now().isoformat()
                self.metadata[cache_key]['access_count'] = \
                    self.metadata[cache_key].get('access_count', 0) + 1

                return features
            except Exception as e:
                # Cache corrupted, remove it
                cache_path.unlink()
                if cache_key in self.metadata:
                    del self.metadata[cache_key]

        self.misses += 1
        return None

    def set(self, data: Any, features: Any, feature_name: str = ""):
        """
        Cache extracted features.

        Args:
            data: Input data
            features: Extracted features to cache
            feature_name: Optional feature name for namespacing
        """
        # Compute cache key
        data_hash = self._compute_hash(data)
        cache_key = f"{feature_name}_{data_hash}" if feature_name else data_hash

        cache_path = self._get_cache_path(cache_key)

        # Save features
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)

        # Update metadata
        file_size = cache_path.stat().st_size
        self.metadata[cache_key] = {
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'size_bytes': file_size,
            'feature_name': feature_name,
            'access_count': 0
        }

        self._save_metadata()

        # Check cache size and evict if necessary
        self._enforce_cache_size()

    def compute_or_get(
        self,
        data: Any,
        compute_fn: Callable[[Any], Any],
        feature_name: str = ""
    ) -> Any:
        """
        Get cached features or compute and cache them.

        Args:
            data: Input data
            compute_fn: Function to compute features if not cached
            feature_name: Optional feature name for namespacing

        Returns:
            Features (from cache or computed)
        """
        # Try to get from cache
        cached = self.get(data, feature_name)
        if cached is not None:
            return cached

        # Compute features
        features = compute_fn(data)

        # Cache them
        self.set(data, features, feature_name)

        return features

    def _enforce_cache_size(self):
        """Enforce maximum cache size by evicting old entries."""
        total_size = sum(entry['size_bytes'] for entry in self.metadata.values())

        if total_size <= self.max_cache_size:
            return

        # Sort by last access time (LRU eviction)
        sorted_entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1]['last_accessed']
        )

        # Evict oldest entries until under size limit
        for cache_key, entry in sorted_entries:
            if total_size <= self.max_cache_size:
                break

            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()

            total_size -= entry['size_bytes']
            del self.metadata[cache_key]

        self._save_metadata()

    def clear(self, feature_name: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            feature_name: If provided, only clear entries for this feature
        """
        if feature_name:
            # Clear specific feature
            keys_to_remove = [
                key for key, entry in self.metadata.items()
                if entry.get('feature_name') == feature_name
            ]
        else:
            # Clear all
            keys_to_remove = list(self.metadata.keys())

        for cache_key in keys_to_remove:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
            if cache_key in self.metadata:
                del self.metadata[cache_key]

        self._save_metadata()

    def cleanup_expired(self):
        """Remove expired cache entries."""
        expired_keys = [
            key for key in self.metadata.keys()
            if self._is_expired(key)
        ]

        for cache_key in expired_keys:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
            del self.metadata[cache_key]

        self._save_metadata()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary of statistics
        """
        total_size = sum(entry['size_bytes'] for entry in self.metadata.values())
        total_entries = len(self.metadata)

        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0

        return {
            'total_entries': total_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_cache_size / (1024 * 1024),
            'utilization_pct': (total_size / self.max_cache_size) * 100,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'entries': list(self.metadata.keys())
        }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"FeatureCache(entries={stats['total_entries']}, "
            f"size={stats['total_size_mb']:.1f}MB, "
            f"hit_rate={stats['hit_rate']*100:.1f}%)"
        )


class InMemoryFeatureCache:
    """
    Lightweight in-memory feature cache.
    Useful for short-lived processes or testing.
    """

    def __init__(self, max_entries: int = 100):
        """
        Initialize in-memory cache.

        Args:
            max_entries: Maximum number of entries to keep
        """
        self.max_entries = max_entries
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}

        self.hits = 0
        self.misses = 0

    def _compute_hash(self, data: Any) -> str:
        """Compute hash for cache key."""
        if isinstance(data, pd.DataFrame):
            content = data.to_json().encode('utf-8')
        elif isinstance(data, np.ndarray):
            content = data.tobytes()
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True).encode('utf-8')
        else:
            content = str(data).encode('utf-8')

        return hashlib.sha256(content).hexdigest()[:16]

    def get(self, data: Any, feature_name: str = "") -> Optional[Any]:
        """Get cached features."""
        data_hash = self._compute_hash(data)
        cache_key = f"{feature_name}_{data_hash}" if feature_name else data_hash

        if cache_key in self.cache:
            self.hits += 1
            self.access_times[cache_key] = datetime.now()
            return self.cache[cache_key]

        self.misses += 1
        return None

    def set(self, data: Any, features: Any, feature_name: str = ""):
        """Cache features."""
        data_hash = self._compute_hash(data)
        cache_key = f"{feature_name}_{data_hash}" if feature_name else data_hash

        # Evict if at capacity
        if len(self.cache) >= self.max_entries and cache_key not in self.cache:
            # Remove least recently used
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_times[lru_key]

        self.cache[cache_key] = features
        self.access_times[cache_key] = datetime.now()

    def compute_or_get(
        self,
        data: Any,
        compute_fn: Callable[[Any], Any],
        feature_name: str = ""
    ) -> Any:
        """Get cached features or compute and cache them."""
        cached = self.get(data, feature_name)
        if cached is not None:
            return cached

        features = compute_fn(data)
        self.set(data, features, feature_name)
        return features

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0

        return {
            'total_entries': len(self.cache),
            'max_entries': self.max_entries,
            'utilization_pct': (len(self.cache) / self.max_entries) * 100,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


# Global cache instance
_global_cache: Optional[FeatureCache] = None


def get_global_cache(
    cache_dir: str = "./cache/features",
    max_cache_size_mb: int = 500
) -> FeatureCache:
    """
    Get or create global feature cache instance.

    Args:
        cache_dir: Directory for cache
        max_cache_size_mb: Maximum cache size in MB

    Returns:
        Global cache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = FeatureCache(
            cache_dir=cache_dir,
            max_cache_size_mb=max_cache_size_mb
        )

    return _global_cache
