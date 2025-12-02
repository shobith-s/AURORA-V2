"""
Feature extraction modules for AURORA preprocessing.

Available extractors:
- MinimalFeatureExtractor: 20 basic features (for neural oracle v2)
- EnhancedFeatureExtractor: 30 features (for enhanced neural oracle)
- MetaLearningFeatureExtractor: 62 features (for meta-learning training)
- MetaFeatureExtractor: 40 features (for hybrid preprocessing oracle)
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

from .meta_extractor import (
    MetaFeatures,
    MetaFeatureExtractor,
    get_meta_feature_extractor,
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
    # Meta extractor (hybrid oracle)
    'MetaFeatures',
    'MetaFeatureExtractor',
    'get_meta_feature_extractor',
]
