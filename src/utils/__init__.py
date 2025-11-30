"""
AURORA Utilities Package.

Provides universal preprocessing utilities:
- UniversalTypeDetector: Semantic type detection for any column
- TargetDetector: Target variable detection
- SafeTransforms: Safe transformation wrappers
- PreprocessingValidator: Pre-execution validation
"""

from .universal_type_detector import (
    UniversalTypeDetector,
    SemanticType,
    TypeDetectionResult,
    get_type_detector,
)

from .target_detector import (
    TargetDetector,
    TargetDetectionResult,
    get_target_detector,
)

from .safe_transforms import (
    SafeTransforms,
    SafeTransformResult,
    TransformResult,
    get_safe_transforms,
)

from .preprocessing_validator import (
    PreprocessingValidator,
    PreprocessingValidation,
    ValidationCheck,
    ValidationResult,
    get_preprocessing_validator,
)

__all__ = [
    # Type Detection
    'UniversalTypeDetector',
    'SemanticType',
    'TypeDetectionResult',
    'get_type_detector',
    # Target Detection
    'TargetDetector',
    'TargetDetectionResult',
    'get_target_detector',
    # Safe Transforms
    'SafeTransforms',
    'SafeTransformResult',
    'TransformResult',
    'get_safe_transforms',
    # Validation
    'PreprocessingValidator',
    'PreprocessingValidation',
    'ValidationCheck',
    'ValidationResult',
    'get_preprocessing_validator',
]
