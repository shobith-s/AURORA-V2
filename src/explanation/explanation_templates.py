"""
Template registry for generating rich explanations.
"""

from typing import Dict, Any
from .enhanced_explanation import (
    EnhancedExplanation,
    ExplanationSection,
    AlternativeExplanation,
    StatisticalEvidence,
    ExplanationSeverity
)


class ExplanationTemplateRegistry:
    """Registry of explanation templates for different preprocessing actions."""
    
    def get_log_transform_explanation(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for log transform."""
        skewness = stats.get('skewness', 0)
        min_val = stats.get('min_value', 0)
        max_val = stats.get('max_value', 0)
        
        # Why section
        why_section = ExplanationSection(
            title="Why Log Transform?",
            content=(
                "The data is highly skewed, meaning most values cluster at one end "
                "of the distribution. Log transformation will normalize the distribution, "
                "making it more suitable for machine learning algorithms."
            ),
            evidence=[
                StatisticalEvidence(
                    metric="Skewness",
                    value=skewness,
                    threshold=1.5,
                    comparison="greater than"
                )
            ],
            severity=ExplanationSeverity.INFO
        )
        
        # Evidence section
        evidence_section = ExplanationSection(
            title="Statistical Evidence",
            content=(
                f"The data ranges from {min_val:.2f} to {max_val:.2f} with high skewness ({skewness:.2f}). "
                "This indicates an exponential or power-law distribution."
            ),
            evidence=[],
            severity=ExplanationSeverity.INFO
        )
        
        # Impact section
        impact_section = ExplanationSection(
            title="Expected Impact",
            content=(
                "After log transformation, the data will have a more normal distribution. "
                "This will improve model performance and make relationships more linear."
            ),
            evidence=[],
            severity=ExplanationSeverity.SUCCESS
        )
        
        # Alternatives
        alternatives = [
            AlternativeExplanation(
                action="Square Root Transform",
                reason="Less aggressive than log transform",
                pros=["Handles moderate skewness", "Preserves zero values"],
                cons=["Less effective for high skewness"]
            ),
            AlternativeExplanation(
                action="Standard Scaling",
                reason="Simple normalization without transformation",
                pros=["Preserves distribution shape", "Fast"],
                cons=["Doesn't address skewness"]
            )
        ]
        
        return {
            'why_section': why_section,
            'evidence_section': evidence_section,
            'impact_section': impact_section,
            'alternatives': alternatives
        }
    
    def get_standard_scale_explanation(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for standard scaling."""
        mean = stats.get('mean', 0)
        std = stats.get('std', 1)
        min_val = stats.get('min_value', 0)
        max_val = stats.get('max_value', 0)
        
        why_section = ExplanationSection(
            title="Why Standard Scaling?",
            content=(
                "The data has a reasonable distribution but needs to be normalized "
                "to have mean=0 and standard deviation=1. This is required for many "
                "machine learning algorithms."
            ),
            evidence=[
                StatisticalEvidence(
                    metric="Mean",
                    value=mean,
                    threshold=None,
                    comparison=""
                ),
                StatisticalEvidence(
                    metric="Std Dev",
                    value=std,
                    threshold=None,
                    comparison=""
                )
            ],
            severity=ExplanationSeverity.INFO
        )
        
        evidence_section = ExplanationSection(
            title="Statistical Evidence",
            content=(
                f"The data ranges from {min_val:.2f} to {max_val:.2f} with mean {mean:.2f} "
                f"and standard deviation {std:.2f}."
            ),
            evidence=[],
            severity=ExplanationSeverity.INFO
        )
        
        impact_section = ExplanationSection(
            title="Expected Impact",
            content=(
                "All features will be on the same scale, preventing features with "
                "large values from dominating the model."
            ),
            evidence=[],
            severity=ExplanationSeverity.SUCCESS
        )
        
        alternatives = [
            AlternativeExplanation(
                action="MinMax Scaling",
                reason="Scale to range [0, 1]",
                pros=["Bounded output", "Preserves zero values"],
                cons=["Sensitive to outliers"]
            ),
            AlternativeExplanation(
                action="Robust Scaling",
                reason="Scale using median and IQR",
                pros=["Resistant to outliers"],
                cons=["May not center at zero"]
            )
        ]
        
        return {
            'why_section': why_section,
            'evidence_section': evidence_section,
            'impact_section': impact_section,
            'alternatives': alternatives
        }
    
    def get_drop_column_explanation(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for dropping a column."""
        null_pct = stats.get('null_pct', 0) * 100
        unique_count = stats.get('unique_count', 0)
        row_count = stats.get('row_count', 1)
        
        why_section = ExplanationSection(
            title="Why Drop This Column?",
            content=(
                "This column has significant data quality issues that make it unsuitable "
                "for analysis or modeling."
            ),
            evidence=[
                StatisticalEvidence(
                    metric="Null Percentage",
                    value=null_pct,
                    threshold=80.0,
                    comparison="greater than"
                )
            ],
            severity=ExplanationSeverity.WARNING
        )
        
        evidence_section = ExplanationSection(
            title="Statistical Evidence",
            content=(
                f"{null_pct:.1f}% of values are missing. "
                f"Only {unique_count} unique values out of {row_count} rows."
            ),
            evidence=[],
            severity=ExplanationSeverity.WARNING
        )
        
        impact_section = ExplanationSection(
            title="Expected Impact",
            content=(
                "Removing this column will reduce noise and improve model performance. "
                "The missing data would require imputation which could introduce bias."
            ),
            evidence=[],
            severity=ExplanationSeverity.INFO
        )
        
        alternatives = [
            AlternativeExplanation(
                action="Fill Nulls",
                reason="Impute missing values",
                pros=["Preserves column"],
                cons=["May introduce bias", "Reduces data quality"]
            )
        ]
        
        return {
            'why_section': why_section,
            'evidence_section': evidence_section,
            'impact_section': impact_section,
            'alternatives': alternatives
        }
    
    def get_onehot_encode_explanation(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for one-hot encoding."""
        cardinality = stats.get('cardinality', 0)
        
        why_section = ExplanationSection(
            title="Why One-Hot Encoding?",
            content=(
                "This is a categorical column with moderate cardinality. "
                "One-hot encoding will convert it to binary features suitable for ML models."
            ),
            evidence=[
                StatisticalEvidence(
                    metric="Cardinality",
                    value=cardinality,
                    threshold=10,
                    comparison="less than"
                )
            ],
            severity=ExplanationSeverity.INFO
        )
        
        evidence_section = ExplanationSection(
            title="Statistical Evidence",
            content=(
                f"The column has {cardinality} unique categories. "
                "This is suitable for one-hot encoding."
            ),
            evidence=[],
            severity=ExplanationSeverity.INFO
        )
        
        impact_section = ExplanationSection(
            title="Expected Impact",
            content=(
                f"This will create {cardinality} new binary columns. "
                "The categorical relationships will be preserved without imposing order."
            ),
            evidence=[],
            severity=ExplanationSeverity.SUCCESS
        )
        
        alternatives = [
            AlternativeExplanation(
                action="Label Encoding",
                reason="Convert to integers",
                pros=["Fewer columns", "Faster"],
                cons=["Implies ordering that may not exist"]
            )
        ]
        
        return {
            'why_section': why_section,
            'evidence_section': evidence_section,
            'impact_section': impact_section,
            'alternatives': alternatives
        }
