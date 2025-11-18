"""
Action definitions for AURORA preprocessing system.
Defines all possible preprocessing actions with their metadata.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


class ActionCategory(Enum):
    """Categories of preprocessing actions."""
    DATA_QUALITY = "data_quality"
    TYPE_CONVERSION = "type_conversion"
    SCALING = "scaling"
    TRANSFORMATION = "transformation"
    ENCODING = "encoding"
    FEATURE_ENGINEERING = "feature_engineering"
    DOMAIN_SPECIFIC = "domain_specific"


class PreprocessingAction(Enum):
    """All available preprocessing actions."""

    # Data Quality Actions
    DROP_COLUMN = "drop_column"
    DROP_IF_MOSTLY_NULL = "drop_if_mostly_null"
    DROP_IF_CONSTANT = "drop_if_constant"
    DROP_IF_ALL_UNIQUE = "drop_if_all_unique"
    REMOVE_DUPLICATES = "remove_duplicates"
    FILL_NULL_MEAN = "fill_null_mean"
    FILL_NULL_MEDIAN = "fill_null_median"
    FILL_NULL_MODE = "fill_null_mode"
    FILL_NULL_FORWARD = "fill_null_forward"
    FILL_NULL_BACKWARD = "fill_null_backward"
    FILL_NULL_INTERPOLATE = "fill_null_interpolate"

    # Type Conversion Actions
    PARSE_DATETIME = "parse_datetime"
    PARSE_BOOLEAN = "parse_boolean"
    PARSE_NUMERIC = "parse_numeric"
    PARSE_JSON = "parse_json"
    PARSE_CATEGORICAL = "parse_categorical"

    # Scaling Actions
    STANDARD_SCALE = "standard_scale"
    MINMAX_SCALE = "minmax_scale"
    ROBUST_SCALE = "robust_scale"
    MAXABS_SCALE = "maxabs_scale"
    NORMALIZE_L1 = "normalize_l1"
    NORMALIZE_L2 = "normalize_l2"

    # Transformation Actions
    LOG_TRANSFORM = "log_transform"
    LOG1P_TRANSFORM = "log1p_transform"
    SQRT_TRANSFORM = "sqrt_transform"
    BOX_COX = "box_cox"
    YEO_JOHNSON = "yeo_johnson"
    QUANTILE_TRANSFORM = "quantile_transform"
    POWER_TRANSFORM = "power_transform"

    # Outlier Handling
    CLIP_OUTLIERS = "clip_outliers"
    WINSORIZE = "winsorize"
    REMOVE_OUTLIERS = "remove_outliers"
    CAP_FLOOR_OUTLIERS = "cap_floor_outliers"

    # Encoding Actions
    ONEHOT_ENCODE = "onehot_encode"
    LABEL_ENCODE = "label_encode"
    ORDINAL_ENCODE = "ordinal_encode"
    TARGET_ENCODE = "target_encode"
    FREQUENCY_ENCODE = "frequency_encode"
    BINARY_ENCODE = "binary_encode"
    HASH_ENCODE = "hash_encode"

    # Feature Engineering
    BINNING_EQUAL_WIDTH = "binning_equal_width"
    BINNING_EQUAL_FREQ = "binning_equal_freq"
    BINNING_CUSTOM = "binning_custom"
    POLYNOMIAL_FEATURES = "polynomial_features"
    INTERACTION_FEATURES = "interaction_features"

    # Domain-Specific Actions
    CURRENCY_NORMALIZE = "currency_normalize"
    PERCENTAGE_TO_DECIMAL = "percentage_to_decimal"
    PHONE_STANDARDIZE = "phone_standardize"
    EMAIL_VALIDATE = "email_validate"
    URL_PARSE = "url_parse"
    TEXT_LOWERCASE = "text_lowercase"
    TEXT_UPPERCASE = "text_uppercase"
    TEXT_CLEAN = "text_clean"

    # No Action
    KEEP_AS_IS = "keep_as_is"


@dataclass
class ActionMetadata:
    """Metadata for a preprocessing action."""
    action: PreprocessingAction
    category: ActionCategory
    description: str
    requirements: List[str]
    parameters: Dict[str, Any]
    reversible: bool
    preserves_nulls: bool


# Action metadata registry
ACTION_REGISTRY: Dict[PreprocessingAction, ActionMetadata] = {
    # Data Quality Actions
    PreprocessingAction.DROP_COLUMN: ActionMetadata(
        action=PreprocessingAction.DROP_COLUMN,
        category=ActionCategory.DATA_QUALITY,
        description="Remove column entirely",
        requirements=["low_information_content"],
        parameters={},
        reversible=False,
        preserves_nulls=False
    ),

    PreprocessingAction.DROP_IF_MOSTLY_NULL: ActionMetadata(
        action=PreprocessingAction.DROP_IF_MOSTLY_NULL,
        category=ActionCategory.DATA_QUALITY,
        description="Drop column if null percentage > threshold",
        requirements=["null_pct"],
        parameters={"threshold": 0.6},
        reversible=False,
        preserves_nulls=False
    ),

    PreprocessingAction.DROP_IF_CONSTANT: ActionMetadata(
        action=PreprocessingAction.DROP_IF_CONSTANT,
        category=ActionCategory.DATA_QUALITY,
        description="Drop if all values are the same",
        requirements=["unique_count"],
        parameters={},
        reversible=False,
        preserves_nulls=False
    ),

    PreprocessingAction.DROP_IF_ALL_UNIQUE: ActionMetadata(
        action=PreprocessingAction.DROP_IF_ALL_UNIQUE,
        category=ActionCategory.DATA_QUALITY,
        description="Drop if all values are unique (likely ID)",
        requirements=["cardinality"],
        parameters={},
        reversible=False,
        preserves_nulls=False
    ),

    # Scaling Actions
    PreprocessingAction.STANDARD_SCALE: ActionMetadata(
        action=PreprocessingAction.STANDARD_SCALE,
        category=ActionCategory.SCALING,
        description="Standardize to mean=0, std=1",
        requirements=["numeric"],
        parameters={},
        reversible=True,
        preserves_nulls=True
    ),

    PreprocessingAction.ROBUST_SCALE: ActionMetadata(
        action=PreprocessingAction.ROBUST_SCALE,
        category=ActionCategory.SCALING,
        description="Scale using median and IQR (robust to outliers)",
        requirements=["numeric"],
        parameters={},
        reversible=True,
        preserves_nulls=True
    ),

    # Transformation Actions
    PreprocessingAction.LOG_TRANSFORM: ActionMetadata(
        action=PreprocessingAction.LOG_TRANSFORM,
        category=ActionCategory.TRANSFORMATION,
        description="Apply log transformation (requires positive values)",
        requirements=["numeric", "positive"],
        parameters={},
        reversible=True,
        preserves_nulls=True
    ),

    PreprocessingAction.BOX_COX: ActionMetadata(
        action=PreprocessingAction.BOX_COX,
        category=ActionCategory.TRANSFORMATION,
        description="Box-Cox power transformation",
        requirements=["numeric", "positive"],
        parameters={},
        reversible=True,
        preserves_nulls=True
    ),

    # Encoding Actions
    PreprocessingAction.ONEHOT_ENCODE: ActionMetadata(
        action=PreprocessingAction.ONEHOT_ENCODE,
        category=ActionCategory.ENCODING,
        description="One-hot encode categorical variable",
        requirements=["categorical", "low_cardinality"],
        parameters={"max_categories": 10},
        reversible=True,
        preserves_nulls=True
    ),

    PreprocessingAction.TARGET_ENCODE: ActionMetadata(
        action=PreprocessingAction.TARGET_ENCODE,
        category=ActionCategory.ENCODING,
        description="Encode categories by target mean",
        requirements=["categorical", "high_cardinality", "target_available"],
        parameters={},
        reversible=False,
        preserves_nulls=True
    ),

    # Keep as is
    PreprocessingAction.KEEP_AS_IS: ActionMetadata(
        action=PreprocessingAction.KEEP_AS_IS,
        category=ActionCategory.DATA_QUALITY,
        description="No preprocessing needed",
        requirements=[],
        parameters={},
        reversible=True,
        preserves_nulls=True
    ),
}


@dataclass
class PreprocessingResult:
    """Result of a preprocessing decision."""
    action: PreprocessingAction
    confidence: float
    source: str  # 'symbolic', 'neural', or 'learned'
    explanation: str
    alternatives: List[tuple[PreprocessingAction, float]]  # [(action, confidence), ...]
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    decision_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "source": self.source,
            "explanation": self.explanation,
            "alternatives": [(a.value, c) for a, c in self.alternatives],
            "parameters": self.parameters,
            "decision_id": self.decision_id
        }


def get_action_metadata(action: PreprocessingAction) -> ActionMetadata:
    """Get metadata for an action."""
    return ACTION_REGISTRY.get(action, None)


def get_actions_by_category(category: ActionCategory) -> List[PreprocessingAction]:
    """Get all actions in a category."""
    return [
        action for action, metadata in ACTION_REGISTRY.items()
        if metadata.category == category
    ]


def is_action_applicable(
    action: PreprocessingAction,
    column_properties: Dict[str, Any]
) -> bool:
    """Check if an action is applicable given column properties."""
    metadata = get_action_metadata(action)
    if not metadata:
        return False

    # Check requirements
    for req in metadata.requirements:
        if req == "numeric" and not column_properties.get("is_numeric", False):
            return False
        elif req == "categorical" and not column_properties.get("is_categorical", False):
            return False
        elif req == "positive" and not column_properties.get("all_positive", False):
            return False
        elif req == "low_cardinality" and column_properties.get("cardinality", float('inf')) > 10:
            return False
        elif req == "high_cardinality" and column_properties.get("cardinality", 0) <= 50:
            return False

    return True
