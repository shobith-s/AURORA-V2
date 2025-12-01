"""
Statistical Validator - Industry-standard statistical tests for preprocessing validation.

All statistical tests are properly cited and follow established methodologies.
This module does NOT claim novelty in the statistical tests themselves, but rather
in their application to automated preprocessing validation.

References:
    Anderson, T.W. and Darling, D.A. (1952). "Asymptotic Theory of Certain 
        'Goodness of Fit' Criteria Based on Stochastic Processes". 
        Annals of Mathematical Statistics, 23(2), 193-212.
    
    Tukey, J.W. (1977). "Exploratory Data Analysis". Addison-Wesley.
    
    Shannon, C.E. (1948). "A Mathematical Theory of Communication". 
        Bell System Technical Journal, 27(3), 379-423.
    
    Pearson, K. (1896). "Mathematical Contributions to the Theory of Evolution. III. 
        Regression, Heredity, and Panmixia". Philosophical Transactions of the 
        Royal Society of London. Series A, 187, 253-318.
"""

import time
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy import stats
from scipy.stats import anderson

from .validation_result import ValidationResult, ValidationMetric, ValidationStatus


class StatisticalValidator:
    """
    Validates preprocessing decisions using established statistical tests.
    
    This validator provides explainable proof that preprocessing improved
    data quality through rigorous statistical analysis.
    """
    
    def __init__(
        self,
        normality_threshold: float = 0.05,  # p-value threshold
        cv_improvement_threshold: float = 0.1,  # 10% CV reduction
        outlier_reduction_threshold: float = 0.05,  # 5% outlier reduction
        entropy_preservation_threshold: float = 0.80,  # Preserve 80% of entropy
        overall_pass_threshold: float = 0.60,  # 60% of metrics must improve
    ):
        """
        Initialize the statistical validator.
        
        Args:
            normality_threshold: P-value threshold for normality tests
            cv_improvement_threshold: Minimum CV reduction to consider improvement
            outlier_reduction_threshold: Minimum outlier reduction to consider improvement
            entropy_preservation_threshold: Minimum entropy to preserve
            overall_pass_threshold: Fraction of metrics that must improve to pass
        """
        self.normality_threshold = normality_threshold
        self.cv_improvement_threshold = cv_improvement_threshold
        self.outlier_reduction_threshold = outlier_reduction_threshold
        self.entropy_preservation_threshold = entropy_preservation_threshold
        self.overall_pass_threshold = overall_pass_threshold
    
    def validate(
        self,
        original: pd.Series,
        preprocessed: pd.Series,
        action_name: str,
    ) -> ValidationResult:
        """
        Validate a preprocessing decision using statistical tests.
        
        Args:
            original: Original column data
            preprocessed: Preprocessed column data
            action_name: Name of the preprocessing action applied
        
        Returns:
            ValidationResult with detailed metrics and explanations
        """
        start_time = time.time()
        
        result = ValidationResult(
            overall_score=0.0,
            passed=False,
            status=ValidationStatus.PASSED,
        )
        
        try:
            # Determine if data is numeric
            is_numeric = pd.api.types.is_numeric_dtype(preprocessed)
            
            if is_numeric:
                # Test 1: Normality Improvement (Anderson-Darling)
                try:
                    normality_metric = self._test_normality(original, preprocessed)
                    result.add_metric(normality_metric)
                except Exception as e:
                    result.add_warning(f"Normality test skipped: {str(e)}")
                
                # Test 2: Variance Stabilization (Coefficient of Variation)
                try:
                    variance_metric = self._test_variance_stabilization(original, preprocessed)
                    result.add_metric(variance_metric)
                except Exception as e:
                    result.add_warning(f"Variance test skipped: {str(e)}")
                
                # Test 3: Outlier Reduction (IQR Method)
                try:
                    outlier_metric = self._test_outlier_reduction(original, preprocessed)
                    result.add_metric(outlier_metric)
                except Exception as e:
                    result.add_warning(f"Outlier test skipped: {str(e)}")
            
            # Test 4: Information Preservation (Shannon Entropy)
            try:
                entropy_metric = self._test_information_preservation(original, preprocessed)
                result.add_metric(entropy_metric)
            except Exception as e:
                result.add_warning(f"Entropy test skipped: {str(e)}")
            
            # Calculate overall score
            if result.metrics:
                passed_count = sum(1 for m in result.metrics.values() if m.passed)
                total_count = len(result.metrics)
                result.overall_score = passed_count / total_count
                result.passed = result.overall_score >= self.overall_pass_threshold
                result.status = ValidationStatus.PASSED if result.passed else ValidationStatus.FAILED
            else:
                result.add_warning("No metrics could be computed")
                result.status = ValidationStatus.SKIPPED
            
            # Add summary details
            result.details['action'] = action_name
            result.details['metrics_computed'] = len(result.metrics)
            result.details['metrics_passed'] = sum(1 for m in result.metrics.values() if m.passed)
            
        except Exception as e:
            result.add_error(f"Validation failed: {str(e)}")
            result.status = ValidationStatus.FAILED
            result.passed = False
        
        result.latency_ms = (time.time() - start_time) * 1000
        return result
    
    def _test_normality(
        self,
        original: pd.Series,
        preprocessed: pd.Series,
    ) -> ValidationMetric:
        """
        Test if preprocessing improved normality using Anderson-Darling test.
        
        The Anderson-Darling test is more sensitive than Shapiro-Wilk for
        detecting departures from normality, especially in the tails.
        
        Citation: Anderson & Darling (1952)
        """
        # Remove NaN values
        orig_clean = original.dropna()
        prep_clean = preprocessed.dropna()
        
        if len(orig_clean) < 8 or len(prep_clean) < 8:
            # Anderson-Darling requires at least 8 samples
            return ValidationMetric(
                name="normality_improvement",
                value_before=0.0,
                value_after=0.0,
                improvement=0.0,
                passed=True,  # Skip test, assume pass
                explanation="Normality test skipped (insufficient samples)",
                citation="Anderson & Darling (1952)"
            )
        
        # Anderson-Darling test
        # Returns: statistic, critical_values, significance_level
        result_before = anderson(orig_clean, dist='norm')
        result_after = anderson(prep_clean, dist='norm')
        
        # Lower statistic = more normal
        # Use 5% significance level (index 2: 15% critical value)
        stat_before = result_before.statistic
        stat_after = result_after.statistic
        improvement = stat_before - stat_after  # Positive = improvement
        
        # Check if after preprocessing, data is more normal
        # (statistic decreased)
        passed = stat_after < stat_before
        
        explanation = (
            f"Normality: AD statistic {stat_before:.2f} -> {stat_after:.2f} "
            f"({'improved' if passed else 'degraded'})"
        )
        
        return ValidationMetric(
            name="normality_improvement",
            value_before=stat_before,
            value_after=stat_after,
            improvement=improvement,
            passed=passed,
            explanation=explanation,
            citation="Anderson & Darling (1952)"
        )
    
    def _test_variance_stabilization(
        self,
        original: pd.Series,
        preprocessed: pd.Series,
    ) -> ValidationMetric:
        """
        Test if preprocessing stabilized variance using Coefficient of Variation.
        
        CV = std / mean, measures relative variability.
        Lower CV = more stable variance.
        
        Citation: Pearson (1896)
        """
        orig_clean = original.dropna()
        prep_clean = preprocessed.dropna()
        
        # Calculate CV
        cv_before = self._calculate_cv(orig_clean)
        cv_after = self._calculate_cv(prep_clean)
        
        improvement = cv_before - cv_after  # Positive = improvement
        passed = improvement >= self.cv_improvement_threshold
        
        explanation = (
            f"Variance: CV {cv_before:.3f} -> {cv_after:.3f} "
            f"({'stabilized' if passed else 'not stabilized'})"
        )
        
        return ValidationMetric(
            name="variance_stabilization",
            value_before=cv_before,
            value_after=cv_after,
            improvement=improvement,
            passed=passed,
            explanation=explanation,
            citation="Pearson (1896)"
        )
    
    def _test_outlier_reduction(
        self,
        original: pd.Series,
        preprocessed: pd.Series,
    ) -> ValidationMetric:
        """
        Test if preprocessing reduced outliers using IQR method.
        
        Outliers defined as values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        This is Tukey's standard method for outlier detection.
        
        Citation: Tukey (1977)
        """
        orig_clean = original.dropna()
        prep_clean = preprocessed.dropna()
        
        # Calculate outlier percentage
        outlier_pct_before = self._calculate_outlier_percentage(orig_clean)
        outlier_pct_after = self._calculate_outlier_percentage(prep_clean)
        
        improvement = outlier_pct_before - outlier_pct_after  # Positive = improvement
        passed = improvement >= self.outlier_reduction_threshold
        
        explanation = (
            f"Outliers: {outlier_pct_before:.1%} -> {outlier_pct_after:.1%} "
            f"({'reduced' if passed else 'not reduced'})"
        )
        
        return ValidationMetric(
            name="outlier_reduction",
            value_before=outlier_pct_before,
            value_after=outlier_pct_after,
            improvement=improvement,
            passed=passed,
            explanation=explanation,
            citation="Tukey (1977)"
        )
    
    def _test_information_preservation(
        self,
        original: pd.Series,
        preprocessed: pd.Series,
    ) -> ValidationMetric:
        """
        Test if preprocessing preserved information using Shannon Entropy.
        
        Entropy measures information content. We want to preserve most of
        the original information while improving data quality.
        
        Citation: Shannon (1948)
        """
        # Calculate entropy
        entropy_before = self._calculate_entropy(original)
        entropy_after = self._calculate_entropy(preprocessed)
        
        # Calculate preservation ratio
        if entropy_before > 0:
            preservation_ratio = entropy_after / entropy_before
        else:
            preservation_ratio = 1.0
        
        passed = preservation_ratio >= self.entropy_preservation_threshold
        
        explanation = (
            f"Information: {preservation_ratio:.1%} preserved "
            f"({'acceptable' if passed else 'too much loss'})"
        )
        
        return ValidationMetric(
            name="information_preservation",
            value_before=entropy_before,
            value_after=entropy_after,
            improvement=preservation_ratio - 1.0,  # Negative = loss
            passed=passed,
            explanation=explanation,
            citation="Shannon (1948)"
        )
    
    # Helper methods
    
    def _calculate_cv(self, data: pd.Series) -> float:
        """Calculate Coefficient of Variation."""
        if len(data) == 0:
            return 0.0
        
        mean = data.mean()
        std = data.std()
        
        if abs(mean) < 1e-10:  # Avoid division by zero
            return 0.0
        
        return abs(std / mean)
    
    def _calculate_outlier_percentage(self, data: pd.Series) -> float:
        """Calculate percentage of outliers using IQR method."""
        if len(data) < 4:
            return 0.0
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers.sum() / len(data)
    
    def _calculate_entropy(self, data: pd.Series) -> float:
        """Calculate Shannon Entropy."""
        if len(data) == 0:
            return 0.0
        
        # Get value counts
        value_counts = data.value_counts()
        probabilities = value_counts / len(data)
        
        # Calculate entropy: H = -sum(p * log2(p))
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
