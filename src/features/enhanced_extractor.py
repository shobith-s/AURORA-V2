"""
Enhanced Feature Extractor - Phase 1 Upgrade
==============================================
Extends the minimal 10-feature extractor to 30 features for better data understanding.
Maintains backward compatibility with existing neural oracle.

New Features Added:
- Distribution features (5): quantiles, gaps, dispersion
- Text features (3): length, diversity, numeric ratio
- Semantic features (5): type tags, domain, naming quality
- Temporal features (2): autocorrelation, monotonicity
- Advanced patterns (5): embedded nulls, diversity metrics
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
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
