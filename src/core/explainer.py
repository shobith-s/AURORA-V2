"""
Enhanced Explanation System for AURORA

Provides clear, detailed, human-readable explanations that build user trust.
"""

from typing import Dict, Any, List
from .actions import PreprocessingAction
import pandas as pd


class ExplanationGenerator:
    """Generate clear, trustworthy explanations for preprocessing decisions."""

    # Detailed explanation templates
    ACTION_EXPLANATIONS = {
        PreprocessingAction.DROP_COLUMN: {
            "why": "This column should be removed from the dataset",
            "reasons": {
                "high_null": "Over {null_pct:.0f}% of values are missing - too sparse to be useful",
                "constant": "All values are identical - provides no information for analysis",
                "low_variance": "Values barely vary (std dev: {std:.4f}) - unlikely to be predictive"
            },
            "impact": "Removing this column will improve model performance and reduce noise",
            "alternative": "If you need this data, consider collecting more complete records first"
        },

        PreprocessingAction.LOG_TRANSFORM: {
            "why": "Apply logarithmic transformation to normalize the data",
            "reasons": {
                "high_skew": "Data is heavily skewed (skewness: {skew:.2f}) - most values cluster at one end",
                "exponential": "Values follow an exponential pattern - log makes them linear",
                "positive_only": "All values are positive, making log transformation safe"
            },
            "impact": "After log transform, the data will have a more normal distribution, improving ML model performance",
            "alternative": "Could also use square root or Box-Cox transform, but log is simpler and often works best"
        },

        PreprocessingAction.STANDARD_SCALE: {
            "why": "Standardize values to have mean=0 and standard deviation=1",
            "reasons": {
                "large_range": "Values range from {min:.1f} to {max:.1f} - needs normalization",
                "normal_dist": "Data follows a normal distribution - standardization preserves this",
                "model_requirement": "Many ML algorithms (SVM, Neural Networks) require standardized inputs"
            },
            "impact": "All features will be on the same scale, preventing large values from dominating the model",
            "alternative": "Use MinMax scaling if you need values between 0 and 1 instead"
        },

        PreprocessingAction.ROBUST_SCALE: {
            "why": "Scale data using median and IQR (robust to outliers)",
            "reasons": {
                "outliers_present": "Detected {outlier_pct:.1f}% outliers - standard scaling would be distorted",
                "heavy_tails": "Distribution has heavy tails - robust scaling handles this better",
                "preserve_outliers": "Outliers may be important - robust scaling reduces their influence without removing them"
            },
            "impact": "Data will be scaled while preserving the relative importance of outliers",
            "alternative": "Could remove outliers first with CLIP_OUTLIERS, but you'd lose potentially important data"
        },

        PreprocessingAction.MINMAX_SCALE: {
            "why": "Scale values to range [0, 1]",
            "reasons": {
                "bounded_needed": "Algorithm requires values in [0, 1] range",
                "preserve_zero": "Original zero values will remain zero after scaling",
                "no_outliers": "No extreme outliers detected - safe to use min-max"
            },
            "impact": "All values will be squeezed into [0, 1] range, making them directly comparable",
            "alternative": "Use standard scaling if you don't need strict bounds"
        },

        PreprocessingAction.ONE_HOT_ENCODE: {
            "why": "Convert categories into binary columns (one column per category)",
            "reasons": {
                "low_cardinality": "Only {n_categories} unique categories - won't create too many columns",
                "categorical": "This is categorical data - ML models need numeric inputs",
                "no_order": "Categories have no natural order (e.g., colors, names)"
            },
            "impact": "Each category becomes its own binary feature (1 if present, 0 if not)",
            "alternative": "Use label encoding if categories have a natural order, or target encoding if you have many categories"
        },

        PreprocessingAction.LABEL_ENCODE: {
            "why": "Convert categories to numeric labels (0, 1, 2, ...)",
            "reasons": {
                "ordered": "Categories have natural order (e.g., low < medium < high)",
                "tree_model": "Tree-based models handle label encoding well",
                "save_space": "More memory-efficient than one-hot encoding"
            },
            "impact": "Each unique category gets a unique number, preserving ordinality",
            "alternative": "Use one-hot encoding if categories don't have a natural order"
        },

        PreprocessingAction.TARGET_ENCODE: {
            "why": "Replace categories with their average target value",
            "reasons": {
                "high_cardinality": "{n_categories} unique categories - too many for one-hot encoding",
                "predictive": "Category values correlate with target variable",
                "efficient": "Reduces dimensions while preserving predictive power"
            },
            "impact": "High-cardinality categorical becomes a single numeric feature encoding predictive information",
            "alternative": "Use frequency encoding or hash encoding as simpler alternatives"
        },

        PreprocessingAction.FREQUENCY_ENCODE: {
            "why": "Replace categories with their frequency count",
            "reasons": {
                "high_cardinality": "{n_categories} categories - need dimension reduction",
                "frequency_matters": "How often a category appears is informative",
                "no_target_leak": "Safer than target encoding - no risk of data leakage"
            },
            "impact": "Common categories get high values, rare ones get low values",
            "alternative": "Use target encoding if you have the target variable available"
        },

        PreprocessingAction.CLIP_OUTLIERS: {
            "why": "Cap extreme values at reasonable limits",
            "reasons": {
                "extreme_outliers": "Found {outlier_pct:.1f}% extreme outliers (beyond {n_std}σ)",
                "distort_model": "Outliers would dominate the model and skew predictions",
                "likely_errors": "Extreme values may be data entry errors or anomalies"
            },
            "impact": "Outliers will be capped at the 1st and 99th percentiles, reducing their influence",
            "alternative": "Use REMOVE_OUTLIERS to delete them entirely, or ROBUST_SCALE to reduce their influence"
        },

        PreprocessingAction.FILL_NULL_MEAN: {
            "why": "Fill missing values with the column's average",
            "reasons": {
                "moderate_missing": "{null_pct:.1f}% missing values - imputation is viable",
                "normal_dist": "Data is roughly normal - mean is a good central value",
                "preserve_rows": "Keep all rows instead of dropping incomplete records"
            },
            "impact": "Missing values will be replaced with {mean:.2f} (the column mean)",
            "alternative": "Use IMPUTE_MEDIAN if data has outliers, or IMPUTE_MODE for categorical data"
        },

        PreprocessingAction.FILL_NULL_MEDIAN: {
            "why": "Fill missing values with the middle value (median)",
            "reasons": {
                "moderate_missing": "{null_pct:.1f}% missing values - imputation is viable",
                "skewed": "Data is skewed - median is more robust than mean",
                "outliers": "Outliers present - median won't be affected by them"
            },
            "impact": "Missing values will be replaced with {median:.2f} (the column median)",
            "alternative": "Use IMPUTE_MEAN if data is normally distributed"
        },

        PreprocessingAction.BINNING_EQUAL_FREQ: {
            "why": "Group continuous values into equal-frequency bins",
            "reasons": {
                "non_linear": "Relationship with target is non-linear - binning captures patterns",
                "many_values": "Too many unique values - binning reduces complexity",
                "outlier_robust": "Quantile bins are robust to outliers"
            },
            "impact": "Continuous values become categorical bins, each containing equal numbers of samples",
            "alternative": "Use BIN_UNIFORM for equal-width bins instead of equal-frequency"
        }
    }

    @staticmethod
    def generate_explanation(
        action: PreprocessingAction,
        confidence: float,
        source: str,
        context: Dict[str, Any],
        column_name: str
    ) -> str:
        """
        Generate a comprehensive, clear explanation.

        Args:
            action: The recommended action
            confidence: Confidence score (0-1)
            source: Decision source (symbolic/neural/learned)
            context: Statistical context about the column
            column_name: Name of the column

        Returns:
            Clear, detailed explanation string
        """
        # Get base template
        template = ExplanationGenerator.ACTION_EXPLANATIONS.get(action)

        if not template:
            # Fallback for actions without templates
            return ExplanationGenerator._generate_simple_explanation(
                action, confidence, source, context, column_name
            )

        # Build explanation parts
        parts = []

        # 1. Header with action and confidence
        confidence_text = ExplanationGenerator._confidence_to_text(confidence)
        parts.append(f"📊 **Recommendation for '{column_name}'**: {action.value}")
        parts.append(f"🎯 **Confidence**: {confidence:.1%} ({confidence_text})")
        parts.append(f"🔍 **Source**: {source.title()}")
        parts.append("")

        # 2. Why this action
        parts.append(f"**Why this action?**")
        parts.append(template["why"])
        parts.append("")

        # 3. Specific reasons with data
        parts.append(f"**Reasons based on your data:**")
        reasons = ExplanationGenerator._select_reasons(template["reasons"], context)
        for i, reason in enumerate(reasons, 1):
            parts.append(f"{i}. {reason}")
        parts.append("")

        # 4. Impact
        parts.append(f"**Impact on your data:**")
        parts.append(template["impact"])
        parts.append("")

        # 5. Alternative
        parts.append(f"**Alternative approaches:**")
        parts.append(template["alternative"])
        parts.append("")

        # 6. Key statistics
        parts.append(f"**Key Statistics:**")
        stats = ExplanationGenerator._format_key_stats(context)
        for stat in stats:
            parts.append(f"  • {stat}")

        return "\n".join(parts)

    @staticmethod
    def _confidence_to_text(confidence: float) -> str:
        """Convert confidence score to human text."""
        if confidence >= 0.95:
            return "Very High - Highly reliable recommendation"
        elif confidence >= 0.85:
            return "High - Reliable recommendation"
        elif confidence >= 0.75:
            return "Good - Solid recommendation"
        elif confidence >= 0.65:
            return "Moderate - Review recommended"
        elif confidence >= 0.50:
            return "Low - Manual review strongly advised"
        else:
            return "Very Low - Use with caution"

    @staticmethod
    def _select_reasons(reason_templates: Dict[str, str], context: Dict[str, Any]) -> List[str]:
        """Select and format relevant reasons based on context."""
        reasons = []

        # Check which reasons apply based on context
        if "null_percentage" in context and context["null_percentage"] > 50:
            reasons.append(reason_templates.get("high_null", "").format(**context))

        if "is_constant" in context and context["is_constant"]:
            reasons.append(reason_templates.get("constant", ""))

        if "skewness" in context and abs(context.get("skewness", 0)) > 2:
            reasons.append(reason_templates.get("high_skew", "").format(**context))

        if "outlier_percentage" in context and context["outlier_percentage"] > 5:
            reasons.append(reason_templates.get("outliers_present", "").format(**context))

        if "unique_ratio" in context and context["unique_ratio"] < 0.05:
            reasons.append(reason_templates.get("low_cardinality", "").format(**context))

        if "unique_count" in context and context["unique_count"] > 50:
            reasons.append(reason_templates.get("high_cardinality", "").format(
                n_categories=context["unique_count"]
            ))

        if "min_value" in context and "max_value" in context:
            reasons.append(reason_templates.get("large_range", "").format(**context))

        if "all_positive" in context and context.get("all_positive", False):
            reasons.append(reason_templates.get("positive_only", ""))

        # If no specific reasons matched, use first available
        if not reasons and reason_templates:
            first_reason = list(reason_templates.values())[0]
            try:
                reasons.append(first_reason.format(**context))
            except (KeyError, ValueError):
                reasons.append(first_reason)

        return reasons[:3]  # Max 3 reasons

    @staticmethod
    def _format_key_stats(context: Dict[str, Any]) -> List[str]:
        """Format key statistics in human-readable form."""
        stats = []

        if "row_count" in context:
            stats.append(f"Total rows: {context['row_count']:,}")

        if "null_percentage" in context:
            stats.append(f"Missing values: {context['null_percentage']:.1f}%")

        if "unique_count" in context:
            stats.append(f"Unique values: {context['unique_count']:,}")

        if "mean_value" in context:
            stats.append(f"Mean: {context['mean_value']:.2f}")

        if "std_dev" in context:
            stats.append(f"Std deviation: {context['std_dev']:.2f}")

        if "skewness" in context:
            skew = context['skewness']
            direction = "right" if skew > 0 else "left"
            stats.append(f"Skewness: {skew:.2f} ({direction}-skewed)")

        if "outlier_percentage" in context and context['outlier_percentage'] > 0:
            stats.append(f"Outliers: {context['outlier_percentage']:.1f}%")

        return stats

    @staticmethod
    def _generate_simple_explanation(
        action: PreprocessingAction,
        confidence: float,
        source: str,
        context: Dict[str, Any],
        column_name: str
    ) -> str:
        """Generate a simple explanation when no template is available."""
        confidence_text = ExplanationGenerator._confidence_to_text(confidence)

        parts = [
            f"📊 **Recommendation for '{column_name}'**: {action.value}",
            f"🎯 **Confidence**: {confidence:.1%} ({confidence_text})",
            f"🔍 **Source**: {source.title()}",
            "",
            f"**Based on the data characteristics**, this action is recommended.",
            ""
        ]

        # Add statistics
        if context:
            parts.append("**Key Statistics:**")
            stats = ExplanationGenerator._format_key_stats(context)
            for stat in stats:
                parts.append(f"  • {stat}")

        return "\n".join(parts)


# Singleton instance
_explainer = None


def get_explainer() -> ExplanationGenerator:
    """Get the global explanation generator instance."""
    global _explainer
    if _explainer is None:
        _explainer = ExplanationGenerator()
    return _explainer
