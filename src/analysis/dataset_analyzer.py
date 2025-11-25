"""
Dataset Analyzer - Inter-Column Relationship Detection
=======================================================

Analyzes relationships between columns to detect:
- Primary keys and composite keys
- Numeric and categorical correlations
- Schema patterns (transactional, time-series, etc.)
- Foreign key candidates

Author: AURORA Team
"""

from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


@dataclass
class AnalysisResult:
    """Results from dataset-level analysis."""
    
    # Primary keys
    primary_key_candidates: List[str] = field(default_factory=list)
    composite_key_candidates: List[List[str]] = field(default_factory=list)
    
    # Correlations
    numeric_correlations: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    categorical_associations: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    
    # Schema detection
    schema_type: str = "unknown"
    schema_confidence: float = 0.0
    schema_features: Dict[str, Any] = field(default_factory=dict)
    
    # Foreign key suggestions
    foreign_key_candidates: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    total_rows: int = 0
    total_columns: int = 0
    analysis_timestamp: Optional[str] = None


class DatasetAnalyzer:
    """
    Analyze inter-column relationships and dataset-level patterns.
    
    This analyzer performs cross-column analysis to understand:
    - Which columns are unique identifiers (primary keys)
    - How columns relate to each other (correlations)
    - What type of data schema this represents
    - Potential foreign key relationships
    """
    
    def __init__(
        self,
        uniqueness_threshold: float = 0.995,
        correlation_threshold: float = 0.7,
        association_threshold: float = 0.5,
        max_composite_key_size: int = 3
    ):
        """
        Initialize the dataset analyzer.
        
        Args:
            uniqueness_threshold: Minimum ratio for primary key detection (default 0.995)
            correlation_threshold: Minimum Pearson correlation to report (default 0.7)
            association_threshold: Minimum Cramér's V to report (default 0.5)
            max_composite_key_size: Maximum columns in composite key (default 3)
        """
        self.uniqueness_threshold = uniqueness_threshold
        self.correlation_threshold = correlation_threshold
        self.association_threshold = association_threshold
        self.max_composite_key_size = max_composite_key_size
    
    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        """
        Perform comprehensive dataset analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            AnalysisResult with all detected patterns
        """
        result = AnalysisResult(
            total_rows=len(df),
            total_columns=len(df.columns),
            analysis_timestamp=pd.Timestamp.now().isoformat()
        )
        
        # Primary key detection
        result.primary_key_candidates = self.detect_primary_keys(df)
        result.composite_key_candidates = self.detect_composite_keys(df)
        
        # Correlation analysis
        result.numeric_correlations = self.find_numeric_correlations(df)
        result.categorical_associations = self.find_categorical_associations(df)
        
        # Schema type identification
        schema_type, confidence, features = self.identify_schema_type(df)
        result.schema_type = schema_type
        result.schema_confidence = confidence
        result.schema_features = features
        
        # Foreign key candidates
        result.foreign_key_candidates = self.analyze_foreign_key_candidates(df)
        
        return result
    
    def detect_primary_keys(self, df: pd.DataFrame) -> List[str]:
        """
        Detect single-column primary key candidates.
        
        A column is considered a primary key candidate if:
        1. Uniqueness ratio > threshold (default 99.5%)
        2. No null values
        3. Has reasonable cardinality (not too low)
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names that are primary key candidates
        """
        candidates = []
        
        for col in df.columns:
            # Skip if has nulls
            if df[col].isnull().any():
                continue
            
            # Calculate uniqueness ratio
            unique_ratio = df[col].nunique() / len(df)
            
            # Must be highly unique
            if unique_ratio >= self.uniqueness_threshold:
                # Check cardinality is reasonable (> 10% of rows)
                if df[col].nunique() >= len(df) * 0.1:
                    candidates.append(col)
        
        return candidates
    
    def detect_composite_keys(self, df: pd.DataFrame) -> List[List[str]]:
        """
        Detect composite key candidates (combinations of columns).
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column combinations that form composite keys
        """
        candidates = []
        
        # Only check combinations of 2-3 columns due to computational cost
        for size in range(2, min(self.max_composite_key_size + 1, len(df.columns) + 1)):
            for cols in combinations(df.columns, size):
                # Skip if any column has nulls
                if any(df[col].isnull().any() for col in cols):
                    continue
                
                # Check combined uniqueness
                combined = df[list(cols)].astype(str).agg('-'.join, axis=1)
                unique_ratio = combined.nunique() / len(df)
                
                if unique_ratio >= self.uniqueness_threshold:
                    candidates.append(list(cols))
        
        return candidates
    
    def find_numeric_correlations(self, df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find strong correlations between numeric columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to list of (correlated_column, correlation) tuples
        """
        correlations = {}
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return correlations
        
        # Compute correlation matrix
        try:
            corr_matrix = df[numeric_cols].corr(method='pearson')
        except:
            return correlations
        
        # Extract strong correlations
        for col in numeric_cols:
            strong_corrs = []
            for other_col in numeric_cols:
                if col != other_col:
                    corr_value = corr_matrix.loc[col, other_col]
                    # Report if strong positive or negative correlation
                    if abs(corr_value) >= self.correlation_threshold:
                        strong_corrs.append((other_col, corr_value))
            
            if strong_corrs:
                # Sort by absolute correlation strength
                strong_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
                correlations[col] = strong_corrs
        
        return correlations
    
    def find_categorical_associations(self, df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find strong associations between categorical columns using Cramér's V.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to list of (associated_column, cramers_v) tuples
        """
        associations = {}
        
        # Get categorical columns (max 50 unique values to be practical)
        categorical_cols = [
            col for col in df.columns
            if df[col].dtype == 'object' or df[col].nunique() <= 50
        ]
        
        if len(categorical_cols) < 2:
            return associations
        
        # Compute pairwise associations
        for col in categorical_cols:
            strong_assocs = []
            for other_col in categorical_cols:
                if col != other_col:
                    try:
                        cramers_v = self._cramers_v(df[col], df[other_col])
                        if cramers_v >= self.association_threshold:
                            strong_assocs.append((other_col, cramers_v))
                    except:
                        continue
            
            if strong_assocs:
                # Sort by association strength
                strong_assocs.sort(key=lambda x: x[1], reverse=True)
                associations[col] = strong_assocs
        
        return associations
    
    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calculate Cramér's V statistic for categorical association.
        
        Args:
            x: First categorical variable
            y: Second categorical variable
            
        Returns:
            Cramér's V value (0 to 1)
        """
        # Create contingency table
        confusion_matrix = pd.crosstab(x, y)
        
        # Perform chi-square test
        chi2, _, _, _ = chi2_contingency(confusion_matrix)
        
        # Calculate Cramér's V
        n = len(x)
        min_dim = min(confusion_matrix.shape[0], confusion_matrix.shape[1]) - 1
        
        if min_dim == 0 or n == 0:
            return 0.0
        
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        return cramers_v
    
    def identify_schema_type(self, df: pd.DataFrame) -> Tuple[str, float, Dict[str, Any]]:
        """
        Identify the type of schema/data structure.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Tuple of (schema_type, confidence, features_dict)
        """
        features = {
            'has_timestamp': False,
            'has_id_column': False,
            'has_many_categories': False,
            'temporal_columns': [],
            'id_columns': [],
            'numeric_ratio': 0.0,
            'categorical_ratio': 0.0
        }
        
        # Detect temporal columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                features['temporal_columns'].append(col)
                features['has_timestamp'] = True
            elif df[col].dtype == 'object':
                # Try parsing as datetime
                try:
                    pd.to_datetime(df[col].head(100), errors='raise')
                    features['temporal_columns'].append(col)
                    features['has_timestamp'] = True
                except:
                    pass
        
        # Detect ID columns
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['id', '_id', 'key', 'uuid', 'guid']):
                if df[col].nunique() / len(df) > 0.9:  # High uniqueness
                    features['id_columns'].append(col)
                    features['has_id_column'] = True
        
        # Calculate column type ratios
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        features['numeric_ratio'] = numeric_count / len(df.columns)
        features['categorical_ratio'] = 1 - features['numeric_ratio']
        
        # Count high-cardinality categorical columns
        high_card_cat = sum(
            1 for col in df.columns
            if df[col].dtype == 'object' and df[col].nunique() > 20
        )
        features['has_many_categories'] = high_card_cat > 0
        
        # Schema type classification
        schema_type = "unknown"
        confidence = 0.0
        
        # Time series detection
        if features['has_timestamp'] and features['numeric_ratio'] > 0.5:
            schema_type = "time_series"
            confidence = 0.8 if len(features['temporal_columns']) > 0 else 0.6
        
        # Transactional data detection
        elif features['has_id_column'] and features['has_many_categories']:
            schema_type = "transactional"
            confidence = 0.75 if len(features['id_columns']) > 0 else 0.6
        
        # Hierarchical/star schema detection
        elif features['has_id_column'] and features['categorical_ratio'] > 0.6:
            schema_type = "hierarchical"
            confidence = 0.7
        
        # Wide/feature-rich dataset
        elif features['numeric_ratio'] > 0.7 and len(df.columns) > 20:
            schema_type = "feature_rich"
            confidence = 0.65
        
        return schema_type, confidence, features
    
    def analyze_foreign_key_candidates(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Analyze potential foreign key relationships.
        
        A column might be a foreign key if:
        1. It has an ID-like name pattern
        2. It has moderate cardinality (not unique, but not too few values)
        3. It might reference another column in the dataset
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping potential foreign key columns to referenced columns
        """
        fk_candidates = {}
        
        # Look for ID-ish columns that aren't primary keys
        for col in df.columns:
            # Check for foreign key naming patterns
            if any(pattern in col.lower() for pattern in ['_id', 'ref_', 'fk_', '_key']):
                # Should have moderate cardinality (5-95% of rows)
                unique_ratio = df[col].nunique() / len(df)
                if 0.05 < unique_ratio < 0.95:
                    # Look for potential referenced column
                    # (Simple heuristic: another column with similar name or high overlap)
                    for other_col in df.columns:
                        if col != other_col:
                            # Check if other column might be the referenced primary key
                            if self._might_reference(col, other_col, df):
                                fk_candidates[col] = other_col
                                break
        
        return fk_candidates
    
    def _might_reference(self, fk_col: str, pk_col: str, df: pd.DataFrame) -> bool:
        """
        Check if fk_col might reference pk_col.
        
        Args:
            fk_col: Potential foreign key column
            pk_col: Potential referenced column
            df: DataFrame
            
        Returns:
            True if reference relationship is likely
        """
        # Check name similarity (simple heuristic)
        fk_base = fk_col.lower().replace('_id', '').replace('_key', '').replace('fk_', '')
        pk_base = pk_col.lower().replace('_id', '').replace('_key', '').replace('pk_', '')
        
        if fk_base in pk_base or pk_base in fk_base:
            # Check value overlap
            fk_values = set(df[fk_col].dropna().unique())
            pk_values = set(df[pk_col].dropna().unique())
            
            if len(fk_values) > 0:
                overlap = len(fk_values & pk_values) / len(fk_values)
                return overlap > 0.8  # 80% of FK values exist in PK
        
        return False


# Singleton instance
_dataset_analyzer_instance: Optional[DatasetAnalyzer] = None


def get_dataset_analyzer() -> DatasetAnalyzer:
    """Get the global dataset analyzer instance."""
    global _dataset_analyzer_instance
    if _dataset_analyzer_instance is None:
        _dataset_analyzer_instance = DatasetAnalyzer()
    return _dataset_analyzer_instance
