"""
Enhanced Feature Extractor - Meta-Learning Edition
====================================================
Extracts 60+ features for meta-learning based neural oracle training.
Maintains backward compatibility with existing neural oracle.

Feature Categories (60+ features total):
- Basic statistics (10): from MinimalFeatureExtractor
- Distribution features (8): quantiles, gaps, dispersion, distribution type
- Text features (5): length, diversity, patterns
- Column name semantics (20): price, date, id, rating, count, etc.
- Domain-specific patterns (15): year, rating, currency, percentage, code
- Temporal features (4): autocorrelation, monotonicity, seasonality
- Advanced patterns (5): embedded nulls, diversity metrics

For Meta-Learning Training:
- Use MetaLearningFeatureExtractor for full 60+ features
- Use EnhancedFeatureExtractor for 30 features (backward compatible)
- Use MinimalFeatureExtractor for 20 features (original)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import re

from .minimal_extractor import MinimalFeatures, MinimalFeatureExtractor


@dataclass
class EnhancedFeatures:
    """Enhanced feature set with 30 features for superior data understanding."""
    
    # ============= ORIGINAL 10 FEATURES (maintained for compatibility) =============
    # Basic statistics (4)
    null_percentage: float
    unique_ratio: float
    skewness: float
    outlier_percentage: float
    
    # Pattern detection (3)
    entropy: float
    pattern_complexity: float
    multimodality_score: float
    
    # Metadata (3)
    cardinality_bucket: int
    detected_dtype: int
    column_name_signal: float
    
    # ============= NEW 20 FEATURES (Phase 1 enhancement) =============
    
    # Distribution features (5)
    quantile_25: float = 0.0
    quantile_75: float = 0.0
    mode_frequency: float = 0.0  # Frequency of most common value
    has_value_gaps: float = 0.0  # Gaps in numeric range (0-1)
    coefficient_dispersion: float = 0.0  # IQR / median
    
    # Text features (3)
    avg_string_length: float = 0.0
    char_diversity: float = 0.0  # Unique chars / total chars
    numeric_string_ratio: float = 0.0  # % strings that are numeric
    
    # Semantic features (5)
    semantic_type: int = 4  # 0=id, 1=metric, 2=attribute, 3=temporal, 4=target, 5=other
    domain_category: int = 5  # 0=finance, 1=customer, 2=product, 3=temporal, 4=geo, 5=other
    naming_quality_score: float = 0.0  # Follows naming conventions (0-1)
    is_foreign_key_candidate: float = 0.0  # Probability 0-1
    is_primary_key_candidate: float = 0.0  # Probability 0-1
    
    # Temporal features (2)
    autocorrelation_lag1: float = 0.0  # First-order autocorrelation
    is_monotonic_score: float = 0.0  # 0=not, 0.5=mostly, 1.0=fully monotonic
    
    # Advanced pattern features (5)
    has_embedded_nulls: float = 0.0  # "N/A", "null", "None" etc.
    url_host_diversity: float = 0.0  # Unique hosts / total URLs
    email_domain_diversity: float = 0.0  # Unique domains / total emails
    json_complexity: float = 0.0  # Average nesting depth
    delimited_pattern_score: float = 0.0  # Comma/pipe separated values
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input (30 features)."""
        return np.array([
            # Original 10
            self.null_percentage,
            self.unique_ratio,
            self.skewness,
            self.outlier_percentage,
            self.entropy,
            self.pattern_complexity,
            self.multimodality_score,
            float(self.cardinality_bucket),
            float(self.detected_dtype),
            self.column_name_signal,
            # New 20
            self.quantile_25,
            self.quantile_75,
            self.mode_frequency,
            self.has_value_gaps,
            self.coefficient_dispersion,
            self.avg_string_length,
            self.char_diversity,
            self.numeric_string_ratio,
            float(self.semantic_type),
            float(self.domain_category),
            self.naming_quality_score,
            self.is_foreign_key_candidate,
            self.is_primary_key_candidate,
            self.autocorrelation_lag1,
            self.is_monotonic_score,
            self.has_embedded_nulls,
            self.url_host_diversity,
            self.email_domain_diversity,
            self.json_complexity,
            self.delimited_pattern_score
        ], dtype=np.float32)
    
    def to_minimal(self) -> MinimalFeatures:
        """Convert to minimal features for backward compatibility."""
        return MinimalFeatures(
            null_percentage=self.null_percentage,
            unique_ratio=self.unique_ratio,
            skewness=self.skewness,
            outlier_percentage=self.outlier_percentage,
            entropy=self.entropy,
            pattern_complexity=self.pattern_complexity,
            multimodality_score=self.multimodality_score,
            cardinality_bucket=self.cardinality_bucket,
            detected_dtype=self.detected_dtype,
            column_name_signal=self.column_name_signal
        )


@dataclass
class MetaLearningFeatures:
    """
    Meta-learning feature set with 60+ features for neural oracle training.
    
    This comprehensive feature set is designed for curriculum-based meta-learning
    where the model learns optimal preprocessing actions from actual ML performance.
    
    Feature Groups:
    - Basic (10): null_pct, unique_ratio, skewness, outlier_pct, entropy, etc.
    - Column Name Semantics (20): name_contains_price, name_contains_date, etc.
    - Domain Patterns (15): looks_like_year, looks_like_rating, looks_like_currency
    - Distribution (8): distribution_type, mode_spike_ratio, multimodal_score, etc.
    - Text/Categorical (5): avg_length, char_diversity, etc.
    - Temporal (4): autocorrelation, monotonicity, seasonality_score
    """
    
    # ============= BASIC FEATURES (10) =============
    null_percentage: float = 0.0
    unique_ratio: float = 0.0
    skewness: float = 0.0
    outlier_percentage: float = 0.0
    entropy: float = 0.0
    pattern_complexity: float = 0.0
    multimodality_score: float = 0.0
    cardinality_bucket: int = 0
    detected_dtype: int = 0
    column_name_signal: float = 0.0
    
    # ============= COLUMN NAME SEMANTIC FEATURES (20) =============
    name_contains_id: float = 0.0
    name_contains_date: float = 0.0
    name_contains_time: float = 0.0
    name_contains_price: float = 0.0
    name_contains_cost: float = 0.0
    name_contains_amount: float = 0.0
    name_contains_count: float = 0.0
    name_contains_num: float = 0.0
    name_contains_rating: float = 0.0
    name_contains_score: float = 0.0
    name_contains_pct: float = 0.0
    name_contains_ratio: float = 0.0
    name_contains_code: float = 0.0
    name_contains_type: float = 0.0
    name_contains_category: float = 0.0
    name_contains_status: float = 0.0
    name_contains_name: float = 0.0
    name_contains_desc: float = 0.0
    name_is_short: float = 0.0  # Name length <= 3
    name_is_long: float = 0.0   # Name length >= 20
    
    # ============= DOMAIN-SPECIFIC PATTERN FEATURES (15) =============
    looks_like_year: float = 0.0  # Values in 1900-2100 range
    looks_like_age: float = 0.0   # Values in 0-120 range
    looks_like_rating: float = 0.0  # Values in 0-5 or 0-10 bounded
    looks_like_percentage: float = 0.0  # Values in 0-100 range
    looks_like_currency: float = 0.0  # Has $ or currency patterns
    looks_like_zipcode: float = 0.0  # 5-digit patterns
    looks_like_phone: float = 0.0  # Phone number patterns
    looks_like_email: float = 0.0  # Email patterns detected
    looks_like_url: float = 0.0  # URL patterns detected
    looks_like_boolean: float = 0.0  # True/False, Yes/No, 0/1 patterns
    looks_like_categorical: float = 0.0  # Low cardinality, repeated values
    looks_like_ordinal: float = 0.0  # Ordered categories
    looks_like_code: float = 0.0  # Short alphanumeric codes
    looks_like_freetext: float = 0.0  # Long variable strings
    has_special_chars: float = 0.0  # Contains special characters
    
    # ============= DISTRIBUTION FEATURES (8) =============
    distribution_type: int = 0  # 0=unknown, 1=normal, 2=uniform, 3=exponential, 4=bimodal
    mode_spike_ratio: float = 0.0  # Ratio of mode count to total
    has_outliers: float = 0.0  # Binary: outliers detected
    quantile_25: float = 0.0
    quantile_75: float = 0.0
    coefficient_dispersion: float = 0.0
    range_normalized: float = 0.0  # (max-min)/mean
    has_zeros: float = 0.0  # Contains zero values
    
    # ============= TEXT/CATEGORICAL FEATURES (5) =============
    avg_string_length: float = 0.0
    char_diversity: float = 0.0
    numeric_string_ratio: float = 0.0
    has_mixed_case: float = 0.0
    word_count_avg: float = 0.0
    
    # ============= TEMPORAL FEATURES (4) =============
    autocorrelation_lag1: float = 0.0
    is_monotonic_score: float = 0.0
    seasonality_score: float = 0.0
    trend_strength: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input (62 features)."""
        return np.array([
            # Basic (10)
            self.null_percentage,
            self.unique_ratio,
            self.skewness,
            self.outlier_percentage,
            self.entropy,
            self.pattern_complexity,
            self.multimodality_score,
            float(self.cardinality_bucket),
            float(self.detected_dtype),
            self.column_name_signal,
            # Column Name Semantics (20)
            self.name_contains_id,
            self.name_contains_date,
            self.name_contains_time,
            self.name_contains_price,
            self.name_contains_cost,
            self.name_contains_amount,
            self.name_contains_count,
            self.name_contains_num,
            self.name_contains_rating,
            self.name_contains_score,
            self.name_contains_pct,
            self.name_contains_ratio,
            self.name_contains_code,
            self.name_contains_type,
            self.name_contains_category,
            self.name_contains_status,
            self.name_contains_name,
            self.name_contains_desc,
            self.name_is_short,
            self.name_is_long,
            # Domain Patterns (15)
            self.looks_like_year,
            self.looks_like_age,
            self.looks_like_rating,
            self.looks_like_percentage,
            self.looks_like_currency,
            self.looks_like_zipcode,
            self.looks_like_phone,
            self.looks_like_email,
            self.looks_like_url,
            self.looks_like_boolean,
            self.looks_like_categorical,
            self.looks_like_ordinal,
            self.looks_like_code,
            self.looks_like_freetext,
            self.has_special_chars,
            # Distribution (8)
            float(self.distribution_type),
            self.mode_spike_ratio,
            self.has_outliers,
            self.quantile_25,
            self.quantile_75,
            self.coefficient_dispersion,
            self.range_normalized,
            self.has_zeros,
            # Text/Categorical (5)
            self.avg_string_length,
            self.char_diversity,
            self.numeric_string_ratio,
            self.has_mixed_case,
            self.word_count_avg,
            # Temporal (4)
            self.autocorrelation_lag1,
            self.is_monotonic_score,
            self.seasonality_score,
            self.trend_strength,
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for inspection."""
        return {
            # Basic
            'null_percentage': self.null_percentage,
            'unique_ratio': self.unique_ratio,
            'skewness': self.skewness,
            'outlier_percentage': self.outlier_percentage,
            'entropy': self.entropy,
            'pattern_complexity': self.pattern_complexity,
            'multimodality_score': self.multimodality_score,
            'cardinality_bucket': float(self.cardinality_bucket),
            'detected_dtype': float(self.detected_dtype),
            'column_name_signal': self.column_name_signal,
            # Column Name Semantics
            'name_contains_id': self.name_contains_id,
            'name_contains_date': self.name_contains_date,
            'name_contains_time': self.name_contains_time,
            'name_contains_price': self.name_contains_price,
            'name_contains_cost': self.name_contains_cost,
            'name_contains_amount': self.name_contains_amount,
            'name_contains_count': self.name_contains_count,
            'name_contains_num': self.name_contains_num,
            'name_contains_rating': self.name_contains_rating,
            'name_contains_score': self.name_contains_score,
            'name_contains_pct': self.name_contains_pct,
            'name_contains_ratio': self.name_contains_ratio,
            'name_contains_code': self.name_contains_code,
            'name_contains_type': self.name_contains_type,
            'name_contains_category': self.name_contains_category,
            'name_contains_status': self.name_contains_status,
            'name_contains_name': self.name_contains_name,
            'name_contains_desc': self.name_contains_desc,
            'name_is_short': self.name_is_short,
            'name_is_long': self.name_is_long,
            # Domain Patterns
            'looks_like_year': self.looks_like_year,
            'looks_like_age': self.looks_like_age,
            'looks_like_rating': self.looks_like_rating,
            'looks_like_percentage': self.looks_like_percentage,
            'looks_like_currency': self.looks_like_currency,
            'looks_like_zipcode': self.looks_like_zipcode,
            'looks_like_phone': self.looks_like_phone,
            'looks_like_email': self.looks_like_email,
            'looks_like_url': self.looks_like_url,
            'looks_like_boolean': self.looks_like_boolean,
            'looks_like_categorical': self.looks_like_categorical,
            'looks_like_ordinal': self.looks_like_ordinal,
            'looks_like_code': self.looks_like_code,
            'looks_like_freetext': self.looks_like_freetext,
            'has_special_chars': self.has_special_chars,
            # Distribution
            'distribution_type': float(self.distribution_type),
            'mode_spike_ratio': self.mode_spike_ratio,
            'has_outliers': self.has_outliers,
            'quantile_25': self.quantile_25,
            'quantile_75': self.quantile_75,
            'coefficient_dispersion': self.coefficient_dispersion,
            'range_normalized': self.range_normalized,
            'has_zeros': self.has_zeros,
            # Text/Categorical
            'avg_string_length': self.avg_string_length,
            'char_diversity': self.char_diversity,
            'numeric_string_ratio': self.numeric_string_ratio,
            'has_mixed_case': self.has_mixed_case,
            'word_count_avg': self.word_count_avg,
            # Temporal
            'autocorrelation_lag1': self.autocorrelation_lag1,
            'is_monotonic_score': self.is_monotonic_score,
            'seasonality_score': self.seasonality_score,
            'trend_strength': self.trend_strength,
        }
    
    def to_minimal(self) -> MinimalFeatures:
        """Convert to minimal features for backward compatibility."""
        return MinimalFeatures(
            null_percentage=self.null_percentage,
            unique_ratio=self.unique_ratio,
            skewness=self.skewness,
            outlier_percentage=self.outlier_percentage,
            entropy=self.entropy,
            pattern_complexity=self.pattern_complexity,
            multimodality_score=self.multimodality_score,
            cardinality_bucket=self.cardinality_bucket,
            detected_dtype=self.detected_dtype,
            column_name_signal=self.column_name_signal
        )
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of all feature names in order."""
        return [
            # Basic
            'null_percentage', 'unique_ratio', 'skewness', 'outlier_percentage',
            'entropy', 'pattern_complexity', 'multimodality_score',
            'cardinality_bucket', 'detected_dtype', 'column_name_signal',
            # Column Name Semantics
            'name_contains_id', 'name_contains_date', 'name_contains_time',
            'name_contains_price', 'name_contains_cost', 'name_contains_amount',
            'name_contains_count', 'name_contains_num', 'name_contains_rating',
            'name_contains_score', 'name_contains_pct', 'name_contains_ratio',
            'name_contains_code', 'name_contains_type', 'name_contains_category',
            'name_contains_status', 'name_contains_name', 'name_contains_desc',
            'name_is_short', 'name_is_long',
            # Domain Patterns
            'looks_like_year', 'looks_like_age', 'looks_like_rating',
            'looks_like_percentage', 'looks_like_currency', 'looks_like_zipcode',
            'looks_like_phone', 'looks_like_email', 'looks_like_url',
            'looks_like_boolean', 'looks_like_categorical', 'looks_like_ordinal',
            'looks_like_code', 'looks_like_freetext', 'has_special_chars',
            # Distribution
            'distribution_type', 'mode_spike_ratio', 'has_outliers',
            'quantile_25', 'quantile_75', 'coefficient_dispersion',
            'range_normalized', 'has_zeros',
            # Text/Categorical
            'avg_string_length', 'char_diversity', 'numeric_string_ratio',
            'has_mixed_case', 'word_count_avg',
            # Temporal
            'autocorrelation_lag1', 'is_monotonic_score',
            'seasonality_score', 'trend_strength',
        ]


class EnhancedFeatureExtractor(MinimalFeatureExtractor):
    """
    Enhanced feature extractor with 30 features.
    Extends MinimalFeatureExtractor for backward compatibility.
    """
    
    def __init__(self, use_cache: bool = True):
        super().__init__(use_cache)
        
        # Semantic keyword mappings
        self.semantic_keywords = {
            'id': ['id', 'key', 'uuid', 'guid', 'identifier', 'pk_', 'fk_'],
            'metric': ['amount', 'price', 'cost', 'revenue', 'sales', 'value', 
                      'count', 'quantity', 'total', 'sum', 'avg', 'rate'],
            'attribute': ['name', 'type', 'category', 'status', 'description', 
                         'label', 'tag', 'group', 'class'],
            'temporal': ['date', 'time', 'timestamp', 'created', 'updated', 
                        'modified', 'datetime', 'year', 'month', 'day'],
            'target': ['target', 'label', 'class', 'outcome', 'result', 'y', 
                      'prediction', 'response']
        }
        
        self.domain_keywords = {
            'finance': ['price', 'cost', 'revenue', 'amount', 'payment', 'balance',
                       'interest', 'fee', 'tax', 'discount', 'profit'],
            'customer': ['customer', 'user', 'client', 'account', 'subscriber',
                        'member', 'visitor', 'guest'],
            'product': ['product', 'item', 'sku', 'inventory', 'stock', 'catalog'],
            'temporal': ['date', 'time', 'timestamp', 'period', 'duration'],
            'geo': ['country', 'city', 'state', 'zip', 'postal', 'address',
                   'location', 'region', 'latitude', 'longitude', 'geo']
        }
        
        # Embedded null patterns
        self.null_patterns = [
            r'\bna\b', r'\bn/a\b', r'\bnull\b', r'\bnone\b', r'\bnan\b',
            r'\bmissing\b', r'\bunknown\b', r'^-+$', r'^\?+$', r'^\*+$'
        ]
    
    def extract_enhanced(
        self,
        column: pd.Series,
        column_name: str = "",
        force_recompute: bool = False
    ) -> EnhancedFeatures:
        """
        Extract all 30 enhanced features.
        
        Args:
            column: The column data
            column_name: Name of the column
            force_recompute: Force recomputation even if cached
        
        Returns:
            EnhancedFeatures object with 30 features
        """
        # Get base minimal features first
        minimal = self.extract(column, column_name, force_recompute)
        
        # Compute additional features
        enhanced_dict = self._compute_enhanced_features(column, column_name, minimal)
        
        # Combine minimal and enhanced
        return EnhancedFeatures(
            # Original 10
            **minimal.to_dict(),
            # New 20
            **enhanced_dict
        )
    
    def _compute_enhanced_features(
        self,
        column: pd.Series,
        column_name: str,
        minimal: MinimalFeatures
    ) -> Dict[str, float]:
        """Compute the 20 new enhanced features."""
        
        features = {}
        
        # Distribution features
        features.update(self._compute_distribution_features(column, minimal))
        
        # Text features
        features.update(self._compute_text_features(column))
        
        # Semantic features
        features.update(self._compute_semantic_features(column, column_name, minimal))
        
        # Temporal features
        features.update(self._compute_temporal_features(column))
        
        # Advanced pattern features
        features.update(self._compute_advanced_pattern_features(column))
        
        return features
    
    def _compute_distribution_features(self, column: pd.Series, minimal: MinimalFeatures) -> Dict:
        """Compute distribution features."""
        features = {
            'quantile_25': 0.0,
            'quantile_75': 0.0,
            'mode_frequency': 0.0,
            'has_value_gaps': 0.0,
            'coefficient_dispersion': 0.0
        }
        
        if minimal.detected_dtype == 0:  # numeric
            non_null = column.dropna()
            if len(non_null) > 0:
                # Quantiles
                features['quantile_25'] = float(non_null.quantile(0.25))
                features['quantile_75'] = float(non_null.quantile(0.75))
                
                # Mode frequency
                mode_count = non_null.value_counts().iloc[0] if len(non_null) > 0 else 0
                features['mode_frequency'] = mode_count / len(non_null) if len(non_null) > 0 else 0
                
                # Coefficient of dispersion (IQR / median)
                median = non_null.median()
                if median != 0:
                    iqr = features['quantile_75'] - features['quantile_25']
                    features['coefficient_dispersion'] = abs(iqr / median)
                
                # Value gaps (for integers)
                if non_null.dtype in [np.int32, np.int64]:
                    unique_vals = sorted(non_null.unique())
                    if len(unique_vals) > 1:
                        expected_range = unique_vals[-1] - unique_vals[0] + 1
                        actual_count = len(unique_vals)
                        features['has_value_gaps'] = 1.0 - (actual_count / expected_range) if expected_range > 0 else 0.0
        else:
            # For categorical, mode frequency is still useful
            if len(column) > 0:
                mode_count = column.value_counts().iloc[0] if len(column.dropna()) > 0 else 0
                features['mode_frequency'] = mode_count / len(column)
        
        return features
    
    def _compute_text_features(self, column: pd.Series) -> Dict:
        """Compute text-specific features."""
        features = {
            'avg_string_length': 0.0,
            'char_diversity': 0.0,
            'numeric_string_ratio': 0.0
        }
        
        if pd.api.types.is_object_dtype(column) or pd.api.types.is_string_dtype(column):
            sample = column.dropna().head(1000).astype(str)
            if len(sample) > 0:
                # Average string length
                lengths = sample.str.len()
                features['avg_string_length'] = float(lengths.mean())
                
                # Character diversity
                all_text = ''.join(sample)
                if len(all_text) > 0:
                    unique_chars = len(set(all_text))
                    features['char_diversity'] = unique_chars / len(all_text)
                
                # Numeric string ratio
                numeric_pattern = r'^-?\d+\.?\d*$'
                numeric_count = sample.str.match(numeric_pattern).sum()
                features['numeric_string_ratio'] = numeric_count / len(sample)
        
        return features
    
    def _compute_semantic_features(
        self, 
        column: pd.Series, 
        column_name: str,
        minimal: MinimalFeatures
    ) -> Dict:
        """Compute semantic understanding features."""
        features = {
            'semantic_type': 5,  # 5 = other (default)
            'domain_category': 5,  # 5 = other (default)
            'naming_quality_score': 0.0,
            'is_foreign_key_candidate': 0.0,
            'is_primary_key_candidate': 0.0
        }
        
        col_lower = column_name.lower()
        
        # Semantic type detection
        for sem_type, keywords in enumerate(['id', 'metric', 'attribute', 'temporal', 'target']):
            if any(kw in col_lower for kw in self.semantic_keywords[keywords]):
                features['semantic_type'] = sem_type
                break
        
        # Domain category detection
        for domain_idx, (domain, keywords) in enumerate(self.domain_keywords.items()):
            if any(kw in col_lower for kw in keywords):
                features['domain_category'] = domain_idx
                break
        
        # Naming quality (follows conventions)
        quality_score = 0.0
        # Snake_case or camelCase
        if '_' in column_name or column_name[0].islower():
            quality_score += 0.3
        # Descriptive (length > 3)
        if len(column_name) > 3:
            quality_score += 0.2
        # No special chars except underscore
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column_name):
            quality_score += 0.3
        # Not too long
        if len(column_name) < 30:
            quality_score += 0.2
        features['naming_quality_score'] = min(1.0, quality_score)
        
        # Primary key candidate (high unique, low/no nulls, has "id" in name)
        if minimal.unique_ratio > 0.95 and minimal.null_percentage < 0.01:
            pk_score = 0.5
            if any(kw in col_lower for kw in ['id', 'key', 'pk']):
                pk_score += 0.5
            features['is_primary_key_candidate'] = min(1.0, pk_score)
        
        # Foreign key candidate (moderate unique, has "id" in name but not primary)
        if 0.1 < minimal.unique_ratio < 0.95:
            if any(kw in col_lower for kw in ['_id', 'fk_', 'ref_']):
                features['is_foreign_key_candidate'] = 0.8
        
        return features
    
    def _compute_temporal_features(self, column: pd.Series) -> Dict:
        """Compute temporal/sequential features."""
        features = {
            'autocorrelation_lag1': 0.0,
            'is_monotonic_score': 0.0
        }
        
        if pd.api.types.is_numeric_dtype(column):
            non_null = column.dropna()
            if len(non_null) > 10:  # Need enough data
                try:
                    # Autocorrelation lag 1
                    if len(non_null) > 1:
                        shifted = non_null.shift(1).dropna()
                        aligned = non_null.iloc[1:]
                        if len(shifted) > 0 and len(aligned) > 0:
                            corr = np.corrcoef(aligned, shifted)[0, 1]
                            features['autocorrelation_lag1'] = float(corr) if not np.isnan(corr) else 0.0
                    
                    # Monotonicity
                    increasing = (non_null.diff().dropna() >= 0).sum() / len(non_null) if len(non_null) > 1 else 0
                    decreasing = (non_null.diff().dropna() <= 0).sum() / len(non_null) if len(non_null) > 1 else 0
                    features['is_monotonic_score'] = max(increasing, decreasing)
                except:
                    pass
        
        return features
    
    def _compute_advanced_pattern_features(self, column: pd.Series) -> Dict:
        """Compute advanced pattern recognition features."""
        features = {
            'has_embedded_nulls': 0.0,
            'url_host_diversity': 0.0,
            'email_domain_diversity': 0.0,
            'json_complexity': 0.0,
            'delimited_pattern_score': 0.0
        }
        
        if pd.api.types.is_object_dtype(column) or pd.api.types.is_string_dtype(column):
            sample = column.dropna().head(1000).astype(str)
            if len(sample) == 0:
                return features
            
            # Embedded nulls
            null_count = 0
            for pattern in self.null_patterns:
                null_count += sample.str.contains(pattern, case=False, regex=True).sum()
            features['has_embedded_nulls'] = min(1.0, null_count / len(sample))
            
            # URL host diversity
            url_pattern = r'https?://([^/]+)'
            urls = sample.str.extract(url_pattern, expand=False).dropna()
            if len(urls) > 0:
                features['url_host_diversity'] = urls.nunique() / len(urls)
            
            # Email domain diversity
            email_pattern = r'@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            domains = sample.str.extract(email_pattern, expand=False).dropna()
            if len(domains) > 0:
                features['email_domain_diversity'] = domains.nunique() / len(domains)
            
            # JSON complexity (average nesting depth)
            json_count = 0
            total_depth = 0
            for val in sample.head(100):
                if val.strip().startswith(('{', '[')):
                    try:
                        import json
                        obj = json.loads(val)
                        depth = self._json_depth(obj)
                        total_depth += depth
                        json_count += 1
                    except:
                        pass
            if json_count > 0:
                features['json_complexity'] = total_depth / json_count
            
            # Delimited patterns (CSV-like within cell)
            delimited_count = sample.str.contains(r'[,|;]\s*\w', regex=True).sum()
            features['delimited_pattern_score'] = delimited_count / len(sample)
        
        return features
    
    def _json_depth(self, obj, depth=0):
        """Calculate JSON nesting depth."""
        if isinstance(obj, dict):
            return max([self._json_depth(v, depth + 1) for v in obj.values()] + [depth])
        elif isinstance(obj, list) and obj:
            return max([self._json_depth(item, depth + 1) for item in obj] + [depth])
        return depth


# Singleton instance
_enhanced_extractor_instance: Optional[EnhancedFeatureExtractor] = None


def get_enhanced_extractor() -> EnhancedFeatureExtractor:
    """Get the global enhanced feature extractor instance."""
    global _enhanced_extractor_instance
    if _enhanced_extractor_instance is None:
        _enhanced_extractor_instance = EnhancedFeatureExtractor()
    return _enhanced_extractor_instance


class MetaLearningFeatureExtractor(EnhancedFeatureExtractor):
    """
    Feature extractor for meta-learning training with 60+ features.
    
    This extractor is used by the Colab training notebook to extract
    comprehensive features for training the neural oracle ensemble.
    
    Features are organized into groups:
    - Basic statistics (10)
    - Column name semantics (20)
    - Domain-specific patterns (15)
    - Distribution fingerprints (8)
    - Text/Categorical features (5)
    - Temporal features (4)
    
    Total: 62 features
    """
    
    # Column name patterns for semantic features
    NAME_PATTERNS = {
        'id': ['id', 'key', 'uuid', 'guid', 'pk', 'fk', 'idx', 'index'],
        'date': ['date', 'dt', 'day', 'created', 'updated', 'modified'],
        'time': ['time', 'hour', 'minute', 'second', 'timestamp', 'ts'],
        'price': ['price', 'cost', 'fee', 'charge', 'msrp', 'retail'],
        'cost': ['cost', 'expense', 'spend', 'expenditure'],
        'amount': ['amount', 'amt', 'total', 'sum', 'balance'],
        'count': ['count', 'cnt', 'num', 'qty', 'quantity', 'number'],
        'num': ['num', 'no', 'number', 'n_', '_n'],
        'rating': ['rating', 'rate', 'star', 'review', 'score'],
        'score': ['score', 'point', 'grade', 'rank', 'ranking'],
        'pct': ['pct', 'percent', 'percentage', 'ratio', '%'],
        'ratio': ['ratio', 'rate', 'proportion', 'share'],
        'code': ['code', 'cd', 'sku', 'upc', 'barcode', 'isbn'],
        'type': ['type', 'typ', 'kind', 'class'],
        'category': ['category', 'cat', 'group', 'segment', 'genre'],
        'status': ['status', 'state', 'flag', 'active', 'enabled'],
        'name': ['name', 'title', 'label', 'description'],
        'desc': ['desc', 'description', 'comment', 'note', 'text'],
    }
    
    def __init__(self, use_cache: bool = True):
        super().__init__(use_cache)
    
    def extract_meta_features(
        self,
        column: pd.Series,
        column_name: str = "",
        force_recompute: bool = False
    ) -> MetaLearningFeatures:
        """
        Extract all 62 meta-learning features from a column.
        
        Args:
            column: The pandas Series to extract features from
            column_name: Name of the column (used for semantic features)
            force_recompute: Force recomputation even if cached
        
        Returns:
            MetaLearningFeatures object with 62 features
        """
        # Get base features using parent extractors
        minimal = self.extract(column, column_name, force_recompute)
        
        # Extract all feature groups
        name_features = self._extract_name_features(column_name)
        domain_features = self._extract_domain_features(column, minimal)
        distribution_features = self._extract_distribution_features(column, minimal)
        text_features = self._extract_text_cat_features(column)
        temporal_features = self._extract_temporal_features(column, minimal)
        
        # Combine into MetaLearningFeatures
        return MetaLearningFeatures(
            # Basic features (from minimal)
            null_percentage=minimal.null_percentage,
            unique_ratio=minimal.unique_ratio,
            skewness=minimal.skewness,
            outlier_percentage=minimal.outlier_percentage,
            entropy=minimal.entropy,
            pattern_complexity=minimal.pattern_complexity,
            multimodality_score=minimal.multimodality_score,
            cardinality_bucket=minimal.cardinality_bucket,
            detected_dtype=minimal.detected_dtype,
            column_name_signal=minimal.column_name_signal,
            
            # Column name semantic features
            **name_features,
            
            # Domain-specific pattern features
            **domain_features,
            
            # Distribution features
            **distribution_features,
            
            # Text/Categorical features
            **text_features,
            
            # Temporal features
            **temporal_features,
        )
    
    def _extract_name_features(self, column_name: str) -> Dict[str, float]:
        """Extract 20 column name semantic features."""
        name_lower = column_name.lower() if column_name else ""
        
        features = {}
        
        # Pattern matching for each category
        for pattern_name, keywords in self.NAME_PATTERNS.items():
            key = f'name_contains_{pattern_name}'
            features[key] = float(any(kw in name_lower for kw in keywords))
        
        # Name length features
        features['name_is_short'] = float(len(column_name) <= 3) if column_name else 0.0
        features['name_is_long'] = float(len(column_name) >= 20) if column_name else 0.0
        
        return features
    
    def _extract_domain_features(
        self,
        column: pd.Series,
        minimal: MinimalFeatures
    ) -> Dict[str, float]:
        """Extract 15 domain-specific pattern features."""
        features = {
            'looks_like_year': 0.0,
            'looks_like_age': 0.0,
            'looks_like_rating': 0.0,
            'looks_like_percentage': 0.0,
            'looks_like_currency': 0.0,
            'looks_like_zipcode': 0.0,
            'looks_like_phone': 0.0,
            'looks_like_email': 0.0,
            'looks_like_url': 0.0,
            'looks_like_boolean': 0.0,
            'looks_like_categorical': 0.0,
            'looks_like_ordinal': 0.0,
            'looks_like_code': 0.0,
            'looks_like_freetext': 0.0,
            'has_special_chars': 0.0,
        }
        
        non_null = column.dropna()
        if len(non_null) == 0:
            return features
        
        sample = non_null.head(1000)
        
        # Numeric domain detection
        if minimal.detected_dtype == 0:  # numeric
            numeric_vals = pd.to_numeric(sample, errors='coerce').dropna()
            if len(numeric_vals) > 0:
                min_val, max_val = numeric_vals.min(), numeric_vals.max()
                mean_val = numeric_vals.mean()
                
                # Year detection (1900-2100)
                if 1900 <= min_val and max_val <= 2100 and mean_val > 1950:
                    features['looks_like_year'] = 1.0
                
                # Age detection (0-120)
                if 0 <= min_val and max_val <= 120 and mean_val > 0:
                    features['looks_like_age'] = 0.8 if max_val < 100 else 0.5
                
                # Rating detection (0-5 or 0-10)
                if min_val >= 0 and max_val <= 10:
                    if max_val <= 5:
                        features['looks_like_rating'] = 1.0
                    else:
                        features['looks_like_rating'] = 0.8
                
                # Percentage detection (0-100)
                if 0 <= min_val and max_val <= 100:
                    features['looks_like_percentage'] = 0.7
        
        # String pattern detection
        if minimal.detected_dtype in [1, 2]:  # categorical or text
            str_sample = sample.astype(str)
            
            # Boolean detection
            bool_patterns = {'true', 'false', 'yes', 'no', '0', '1', 'y', 'n', 't', 'f'}
            unique_lower = set(str_sample.str.lower().unique())
            if len(unique_lower) <= 3 and unique_lower.issubset(bool_patterns):
                features['looks_like_boolean'] = 1.0
            
            # Email detection
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            email_match = str_sample.str.match(email_pattern).mean()
            features['looks_like_email'] = float(email_match > 0.5)
            
            # URL detection
            url_pattern = r'^https?://'
            url_match = str_sample.str.match(url_pattern).mean()
            features['looks_like_url'] = float(url_match > 0.5)
            
            # Phone detection
            phone_pattern = r'^\+?[\d\s\-\(\)]{7,}$'
            phone_match = str_sample.str.match(phone_pattern).mean()
            features['looks_like_phone'] = float(phone_match > 0.5)
            
            # Zipcode detection (5 digit)
            zip_pattern = r'^\d{5}(-\d{4})?$'
            zip_match = str_sample.str.match(zip_pattern).mean()
            features['looks_like_zipcode'] = float(zip_match > 0.5)
            
            # Currency detection
            currency_pattern = r'^[\$€£¥]\s*[\d,]+\.?\d*$'
            currency_match = str_sample.str.match(currency_pattern).mean()
            features['looks_like_currency'] = float(currency_match > 0.3)
            
            # Code detection (short alphanumeric)
            code_pattern = r'^[A-Z0-9]{2,10}$'
            code_match = str_sample.str.match(code_pattern).mean()
            features['looks_like_code'] = float(code_match > 0.5)
            
            # Categorical detection (low cardinality)
            if minimal.unique_ratio < 0.1 and minimal.cardinality_bucket <= 1:
                features['looks_like_categorical'] = 1.0
            elif minimal.unique_ratio < 0.3:
                features['looks_like_categorical'] = 0.5
            
            # Free text detection (long, variable strings)
            avg_len = str_sample.str.len().mean()
            len_std = str_sample.str.len().std()
            if avg_len > 50 and len_std > 20:
                features['looks_like_freetext'] = 1.0
            elif avg_len > 20:
                features['looks_like_freetext'] = 0.5
            
            # Special characters detection
            special_pattern = r'[!@#$%^&*()+=\[\]{};\':"\\|,.<>?/~`]'
            special_match = str_sample.str.contains(special_pattern, regex=True).mean()
            features['has_special_chars'] = float(special_match > 0.1)
            
            # Ordinal detection (ordered categories like Low/Medium/High)
            ordinal_indicators = {'low', 'medium', 'high', 'small', 'large', 'xs', 's', 'm', 'l', 'xl'}
            if len(unique_lower) <= 5 and unique_lower.intersection(ordinal_indicators):
                features['looks_like_ordinal'] = 1.0
        
        return features
    
    def _extract_distribution_features(
        self,
        column: pd.Series,
        minimal: MinimalFeatures
    ) -> Dict[str, float]:
        """Extract 8 distribution fingerprint features."""
        features = {
            'distribution_type': 0,  # 0=unknown, 1=normal, 2=uniform, 3=exponential, 4=bimodal
            'mode_spike_ratio': 0.0,
            'has_outliers': 0.0,
            'quantile_25': 0.0,
            'quantile_75': 0.0,
            'coefficient_dispersion': 0.0,
            'range_normalized': 0.0,
            'has_zeros': 0.0,
        }
        
        if minimal.detected_dtype != 0:  # Not numeric
            return features
        
        non_null = column.dropna()
        if len(non_null) < 10:
            return features
        
        numeric_vals = pd.to_numeric(non_null, errors='coerce').dropna()
        if len(numeric_vals) < 10:
            return features
        
        # Basic stats
        mean_val = numeric_vals.mean()
        std_val = numeric_vals.std()
        q25 = numeric_vals.quantile(0.25)
        q75 = numeric_vals.quantile(0.75)
        min_val = numeric_vals.min()
        max_val = numeric_vals.max()
        
        features['quantile_25'] = float(q25)
        features['quantile_75'] = float(q75)
        
        # Mode spike ratio
        mode_count = numeric_vals.value_counts().iloc[0]
        features['mode_spike_ratio'] = float(mode_count / len(numeric_vals))
        
        # Has outliers (using IQR method)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outlier_count = ((numeric_vals < lower_bound) | (numeric_vals > upper_bound)).sum()
        features['has_outliers'] = float(outlier_count > 0)
        
        # Coefficient of dispersion
        if mean_val != 0:
            features['coefficient_dispersion'] = float(abs(iqr / mean_val))
        
        # Range normalized
        if mean_val != 0:
            features['range_normalized'] = float((max_val - min_val) / abs(mean_val))
        
        # Has zeros
        features['has_zeros'] = float((numeric_vals == 0).any())
        
        # Distribution type detection
        if std_val > 0 and len(numeric_vals) >= 30:
            # Check normality using skewness and kurtosis
            skew = minimal.skewness
            if -0.5 <= skew <= 0.5:
                features['distribution_type'] = 1  # Normal-ish
            elif skew > 1.5:
                features['distribution_type'] = 3  # Exponential-ish
            
            # Check bimodality
            if minimal.multimodality_score > 0.555:  # Hartigan's dip test threshold
                features['distribution_type'] = 4  # Bimodal
            
            # Check uniformity (low variance relative to range)
            if features['range_normalized'] > 0 and features['coefficient_dispersion'] < 0.3:
                features['distribution_type'] = 2  # Uniform-ish
        
        return features
    
    def _extract_text_cat_features(self, column: pd.Series) -> Dict[str, float]:
        """Extract 5 text/categorical features."""
        features = {
            'avg_string_length': 0.0,
            'char_diversity': 0.0,
            'numeric_string_ratio': 0.0,
            'has_mixed_case': 0.0,
            'word_count_avg': 0.0,
        }
        
        non_null = column.dropna()
        if len(non_null) == 0:
            return features
        
        # Only process string-like columns
        if not (pd.api.types.is_object_dtype(column) or pd.api.types.is_string_dtype(column)):
            return features
        
        sample = non_null.head(1000).astype(str)
        
        # Average string length
        lengths = sample.str.len()
        features['avg_string_length'] = float(lengths.mean())
        
        # Character diversity
        all_text = ''.join(sample)
        if len(all_text) > 0:
            unique_chars = len(set(all_text))
            features['char_diversity'] = float(unique_chars / min(len(all_text), 1000))
        
        # Numeric string ratio
        numeric_pattern = r'^-?\d+\.?\d*$'
        numeric_match = sample.str.match(numeric_pattern)
        features['numeric_string_ratio'] = float(numeric_match.mean())
        
        # Mixed case detection
        has_upper = sample.str.contains(r'[A-Z]', regex=True)
        has_lower = sample.str.contains(r'[a-z]', regex=True)
        mixed = (has_upper & has_lower).mean()
        features['has_mixed_case'] = float(mixed)
        
        # Average word count
        word_counts = sample.str.split().str.len()
        features['word_count_avg'] = float(word_counts.mean())
        
        return features
    
    def _extract_temporal_features(
        self,
        column: pd.Series,
        minimal: MinimalFeatures
    ) -> Dict[str, float]:
        """Extract 4 temporal/sequential features."""
        features = {
            'autocorrelation_lag1': 0.0,
            'is_monotonic_score': 0.0,
            'seasonality_score': 0.0,
            'trend_strength': 0.0,
        }
        
        if minimal.detected_dtype != 0:  # Not numeric
            return features
        
        non_null = column.dropna()
        if len(non_null) < 20:
            return features
        
        numeric_vals = pd.to_numeric(non_null, errors='coerce').dropna()
        if len(numeric_vals) < 20:
            return features
        
        try:
            # Autocorrelation lag 1
            shifted = numeric_vals.shift(1).dropna()
            aligned = numeric_vals.iloc[1:]
            if len(shifted) > 1 and len(aligned) > 1:
                corr = np.corrcoef(aligned, shifted)[0, 1]
                if not np.isnan(corr):
                    features['autocorrelation_lag1'] = float(corr)
            
            # Monotonicity score
            diffs = numeric_vals.diff().dropna()
            if len(diffs) > 0:
                increasing = (diffs >= 0).mean()
                decreasing = (diffs <= 0).mean()
                features['is_monotonic_score'] = float(max(increasing, decreasing))
            
            # Trend strength (correlation with index)
            if len(numeric_vals) > 5:
                idx = np.arange(len(numeric_vals))
                corr_with_idx = np.corrcoef(idx, numeric_vals.values)[0, 1]
                if not np.isnan(corr_with_idx):
                    features['trend_strength'] = float(abs(corr_with_idx))
            
            # Seasonality score (simplified - check for periodic patterns)
            # Use autocorrelation at potential seasonal lags
            if len(numeric_vals) >= 50:
                for lag in [7, 12, 24]:  # Common seasonal periods
                    if lag < len(numeric_vals) // 2:
                        shifted_lag = numeric_vals.shift(lag).dropna()
                        aligned_lag = numeric_vals.iloc[lag:]
                        if len(shifted_lag) > 1 and len(aligned_lag) > 1:
                            corr_lag = np.corrcoef(aligned_lag, shifted_lag)[0, 1]
                            if not np.isnan(corr_lag) and abs(corr_lag) > 0.3:
                                features['seasonality_score'] = max(
                                    features['seasonality_score'],
                                    float(abs(corr_lag))
                                )
        except Exception:
            pass
        
        return features


# Singleton instance for meta-learning extractor
_meta_extractor_instance: Optional[MetaLearningFeatureExtractor] = None


def get_meta_learning_extractor() -> MetaLearningFeatureExtractor:
    """Get the global meta-learning feature extractor instance."""
    global _meta_extractor_instance
    if _meta_extractor_instance is None:
        _meta_extractor_instance = MetaLearningFeatureExtractor()
    return _meta_extractor_instance
