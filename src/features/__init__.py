"""
Feature extraction modules for AURORA preprocessing.

Available extractors:
- MinimalFeatureExtractor: 20 basic features (for neural oracle v2)
- EnhancedFeatureExtractor: 30 features (for enhanced neural oracle)
- MetaLearningFeatureExtractor: 62 features (for meta-learning training)
"""

from .minimal_extractor import (
    MinimalFeatures,
    MinimalFeatureExtractor,
    get_feature_extractor,
)

from .enhanced_extractor import (
    EnhancedFeatures,
    EnhancedFeatureExtractor,
    MetaLearningFeatures,
    MetaLearningFeatureExtractor,
    get_enhanced_extractor,
    get_meta_learning_extractor,
)

__all__ = [
    # Minimal extractor
    'MinimalFeatures',
    'MinimalFeatureExtractor',
    'get_feature_extractor',
    # Enhanced extractor
    'EnhancedFeatures',
    'EnhancedFeatureExtractor',
    'get_enhanced_extractor',
    # Meta-learning extractor
    'MetaLearningFeatures',
    'MetaLearningFeatureExtractor',
    'get_meta_learning_extractor',
]
