"""
MetaFeatureExtractor for the Hybrid Preprocessing Oracle.
Extracts exactly 40 meta-features from a column for ML-based preprocessing recommendation.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MetaFeatures:
    """
    Meta-features for hybrid preprocessing oracle (40 features total).
    
    These features are used by the XGBoost + LightGBM ensemble to predict
    the best preprocessing action for a given column.
    """
    
    # Basic stats (5 features)
    missing_ratio: float
    unique_ratio: float
    unique_count_norm: float
    row_count_norm: float
    is_complete: float
    
    # Type indicators (5 features)
    is_numeric: float
    is_bool: float
    is_datetime: float
    is_object: float
    is_categorical: float
    
    # Numeric stats (15 features)
    mean: float
    std: float
    min: float
    max: float
    median: float
    skewness: float
    kurtosis: float
    outlier_ratio: float
    positive_ratio: float
    negative_ratio: float
    zero_ratio: float
    can_log: float
    can_sqrt: float
    has_range: float
    has_variance: float
    
    # Categorical stats (10 features)
    avg_length: float
    max_length: float
    min_length: float
    length_std: float
    cardinality_low: float
    cardinality_medium: float
    cardinality_high: float
    cardinality_unique: float
    entropy: float
    mode_frequency: float
    
    # Name features (5 features)
    has_id: float
    has_name: float
    has_date: float
    has_price: float
    has_count: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input (40 features)."""
        return np.array([
            # Basic stats (5)
            self.missing_ratio,
            self.unique_ratio,
            self.unique_count_norm,
            self.row_count_norm,
            self.is_complete,
            
            # Type indicators (5)
            self.is_numeric,
            self.is_bool,
            self.is_datetime,
            self.is_object,
            self.is_categorical,
            
            # Numeric stats (15)
            self.mean,
            self.std,
            self.min,
            self.max,
            self.median,
            self.skewness,
            self.kurtosis,
            self.outlier_ratio,
            self.positive_ratio,
            self.negative_ratio,
            self.zero_ratio,
            self.can_log,
            self.can_sqrt,
            self.has_range,
            self.has_variance,
            
            # Categorical stats (10)
            self.avg_length,
            self.max_length,
            self.min_length,
            self.length_std,
            self.cardinality_low,
            self.cardinality_medium,
            self.cardinality_high,
            self.cardinality_unique,
            self.entropy,
            self.mode_frequency,
            
            # Name features (5)
            self.has_id,
            self.has_name,
            self.has_date,
            self.has_price,
            self.has_count,
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'missing_ratio': self.missing_ratio,
            'unique_ratio': self.unique_ratio,
            'unique_count_norm': self.unique_count_norm,
            'row_count_norm': self.row_count_norm,
            'is_complete': self.is_complete,
            'is_numeric': self.is_numeric,
            'is_bool': self.is_bool,
            'is_datetime': self.is_datetime,
            'is_object': self.is_object,
            'is_categorical': self.is_categorical,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'median': self.median,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'outlier_ratio': self.outlier_ratio,
            'positive_ratio': self.positive_ratio,
            'negative_ratio': self.negative_ratio,
            'zero_ratio': self.zero_ratio,
            'can_log': self.can_log,
            'can_sqrt': self.can_sqrt,
            'has_range': self.has_range,
            'has_variance': self.has_variance,
            'avg_length': self.avg_length,
            'max_length': self.max_length,
            'min_length': self.min_length,
            'length_std': self.length_std,
            'cardinality_low': self.cardinality_low,
            'cardinality_medium': self.cardinality_medium,
            'cardinality_high': self.cardinality_high,
            'cardinality_unique': self.cardinality_unique,
            'entropy': self.entropy,
            'mode_frequency': self.mode_frequency,
            'has_id': self.has_id,
            'has_name': self.has_name,
            'has_date': self.has_date,
            'has_price': self.has_price,
            'has_count': self.has_count,
        }


class MetaFeatureExtractor:
    """
    Extracts 40 meta-features from a column for hybrid preprocessing oracle.
    
    Features are designed to work with XGBoost + LightGBM ensemble model
    trained on diverse datasets.
    """
    
    def __init__(self):
        """Initialize the meta-feature extractor."""
        pass
    
    def extract(
        self,
        column: pd.Series,
        column_name: str = ""
    ) -> MetaFeatures:
        """
        Extract 40 meta-features from a column.
        
        Args:
            column: The column data
            column_name: Name of the column
            
        Returns:
            MetaFeatures object with 40 features
        """
        # Basic stats (5 features)
        n_rows = len(column)
        n_missing = column.isnull().sum()
        n_unique = column.nunique()
        
        missing_ratio = n_missing / n_rows if n_rows > 0 else 0.0
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0.0
        unique_count_norm = np.log1p(n_unique) / 10.0  # Normalized
        row_count_norm = np.log1p(n_rows) / 15.0  # Normalized
        is_complete = 1.0 if missing_ratio == 0.0 else 0.0
        
        # Type indicators (5 features)
        is_numeric = float(pd.api.types.is_numeric_dtype(column))
        is_bool = float(pd.api.types.is_bool_dtype(column))
        is_datetime = float(pd.api.types.is_datetime64_any_dtype(column))
        is_object = float(pd.api.types.is_object_dtype(column))
        is_categorical = float(isinstance(column.dtype, pd.CategoricalDtype) or 
                              (is_object and unique_ratio < 0.5))
        
        # Numeric stats (15 features)
        mean = std = min_val = max_val = median = 0.0
        skewness = kurtosis = outlier_ratio = 0.0
        positive_ratio = negative_ratio = zero_ratio = 0.0
        can_log = can_sqrt = has_range = has_variance = 0.0
        
        if is_numeric and n_missing < n_rows:
            non_null = column.dropna().values.astype(np.float64)
            
            if len(non_null) > 0:
                mean = float(np.mean(non_null))
                std = float(np.std(non_null))
                min_val = float(np.min(non_null))
                max_val = float(np.max(non_null))
                median = float(np.median(non_null))
                
                # Skewness and kurtosis
                if std > 0 and len(non_null) > 2:
                    centered = (non_null - mean) / std
                    skewness = float(np.mean(centered ** 3))
                    if len(non_null) > 3:
                        kurtosis = float(np.mean(centered ** 4) - 3)
                
                # Outlier ratio (IQR method)
                if len(non_null) > 3:
                    q1 = np.percentile(non_null, 25)
                    q3 = np.percentile(non_null, 75)
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = np.sum((non_null < lower_bound) | (non_null > upper_bound))
                        outlier_ratio = float(outliers / len(non_null))
                
                # Ratio features
                positive_ratio = float(np.sum(non_null > 0) / len(non_null))
                negative_ratio = float(np.sum(non_null < 0) / len(non_null))
                zero_ratio = float(np.sum(non_null == 0) / len(non_null))
                
                # Transformation feasibility
                can_log = 1.0 if np.all(non_null > 0) else 0.0
                can_sqrt = 1.0 if np.all(non_null >= 0) else 0.0
                has_range = 1.0 if (max_val - min_val) > 1e-10 else 0.0
                has_variance = 1.0 if std > 1e-10 else 0.0
        
        # Categorical stats (10 features)
        avg_length = max_length = min_length = length_std = 0.0
        cardinality_low = cardinality_medium = cardinality_high = cardinality_unique = 0.0
        entropy = mode_frequency = 0.0
        
        # String length features
        if is_object or is_categorical:
            non_null = column.dropna().astype(str)
            if len(non_null) > 0:
                lengths = non_null.str.len()
                avg_length = float(lengths.mean())
                max_length = float(lengths.max())
                min_length = float(lengths.min())
                length_std = float(lengths.std())
        
        # Cardinality buckets
        if unique_ratio < 0.01:
            cardinality_low = 1.0
        elif unique_ratio < 0.5:
            cardinality_medium = 1.0
        elif unique_ratio < 0.95:
            cardinality_high = 1.0
        else:
            cardinality_unique = 1.0
        
        # Entropy
        if n_rows > 0:
            value_counts = column.value_counts()
            probabilities = value_counts / n_rows
            entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-10)))
        
        # Mode frequency
        if n_rows > 0:
            mode_count = column.value_counts().iloc[0] if len(column.value_counts()) > 0 else 0
            mode_frequency = float(mode_count / n_rows)
        
        # Name features (5 features)
        name_lower = column_name.lower()
        has_id = 1.0 if any(kw in name_lower for kw in ['id', 'key', 'uuid', 'index']) else 0.0
        has_name = 1.0 if any(kw in name_lower for kw in ['name', 'title', 'label']) else 0.0
        has_date = 1.0 if any(kw in name_lower for kw in ['date', 'time', 'timestamp', 'day', 'month', 'year']) else 0.0
        has_price = 1.0 if any(kw in name_lower for kw in ['price', 'cost', 'amount', 'revenue', 'salary']) else 0.0
        has_count = 1.0 if any(kw in name_lower for kw in ['count', 'number', 'quantity', 'total']) else 0.0
        
        return MetaFeatures(
            # Basic stats (5)
            missing_ratio=missing_ratio,
            unique_ratio=unique_ratio,
            unique_count_norm=unique_count_norm,
            row_count_norm=row_count_norm,
            is_complete=is_complete,
            
            # Type indicators (5)
            is_numeric=is_numeric,
            is_bool=is_bool,
            is_datetime=is_datetime,
            is_object=is_object,
            is_categorical=is_categorical,
            
            # Numeric stats (15)
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            median=median,
            skewness=skewness,
            kurtosis=kurtosis,
            outlier_ratio=outlier_ratio,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            zero_ratio=zero_ratio,
            can_log=can_log,
            can_sqrt=can_sqrt,
            has_range=has_range,
            has_variance=has_variance,
            
            # Categorical stats (10)
            avg_length=avg_length,
            max_length=max_length,
            min_length=min_length,
            length_std=length_std,
            cardinality_low=cardinality_low,
            cardinality_medium=cardinality_medium,
            cardinality_high=cardinality_high,
            cardinality_unique=cardinality_unique,
            entropy=entropy,
            mode_frequency=mode_frequency,
            
            # Name features (5)
            has_id=has_id,
            has_name=has_name,
            has_date=has_date,
            has_price=has_price,
            has_count=has_count,
        )


# Singleton instance for global use
_meta_extractor_instance: Optional[MetaFeatureExtractor] = None


def get_meta_feature_extractor() -> MetaFeatureExtractor:
    """Get the global meta-feature extractor instance."""
    global _meta_extractor_instance
    if _meta_extractor_instance is None:
        _meta_extractor_instance = MetaFeatureExtractor()
    return _meta_extractor_instance
