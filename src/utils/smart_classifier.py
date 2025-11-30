"""
Smart Column Classifier - Simple keyword-based classifier using name patterns + basic content checks.

This classifier uses common sense heuristics to classify columns for preprocessing,
preventing catastrophic errors like dropping target variables or scaling text columns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class SmartColumnClassifier:
    """Simple column classifier using name patterns + basic content checks."""
    
    # Keyword patterns for common column types
    TARGET_KEYWORDS = ['price', 'cost', 'value', 'selling_price', 'sale_price', 'target', 'label', 'y']
    YEAR_KEYWORDS = ['year', 'yr', 'model_year', 'year_built', 'birth_year']
    DISTANCE_KEYWORDS = ['mileage', 'milage', 'miles', 'km', 'odometer', 'distance']
    BINARY_KEYWORDS = ['accident', 'title', 'clean', 'salvage', 'warranty', 'certified', 
                       'is_', 'has_', 'flag', 'indicator']
    DROP_KEYWORDS = ['id', 'vin', 'stock', 'url', 'photo', 'image', 'listing_id', 
                     'index', 'unnamed', 'row_id']
    COLOR_KEYWORDS = ['color', 'colour', 'ext_col', 'exterior']
    
    @classmethod
    def classify(cls, column_name: str, column: pd.Series) -> Dict[str, Any]:
        """
        Classify a column and return the recommended preprocessing action.
        
        Args:
            column_name: Name of the column
            column: The pandas Series containing column data
            
        Returns:
            Dict containing:
                - action: Recommended preprocessing action
                - confidence: Confidence score (0.0 to 1.0)
                - reason: Human-readable explanation
        """
        name_lower = column_name.lower().strip()
        
        # Get column properties
        is_object = column.dtype == 'object' or column.dtype == 'O'
        is_numeric = pd.api.types.is_numeric_dtype(column)
        unique_count = column.nunique()
        total_count = len(column)
        null_count = column.isnull().sum()
        
        # Rule 0: All null columns → drop_column (0.95)
        if null_count == total_count:
            return {
                'action': 'drop_column',
                'confidence': 0.95,
                'reason': f"Column '{column_name}' is entirely null - no information value"
            }
        
        # Rule 0b: Constant columns (1 unique value) → drop_column (0.92)
        if unique_count <= 1:
            return {
                'action': 'drop_column',
                'confidence': 0.92,
                'reason': f"Column '{column_name}' has constant value - no predictive power"
            }
        
        # Rule 1: Target keywords (price, cost) → keep_as_is (1.0 confidence)
        if cls._matches_keywords(name_lower, cls.TARGET_KEYWORDS):
            return {
                'action': 'keep_as_is',
                'confidence': 1.0,
                'reason': f"Column '{column_name}' matches target variable pattern - preserving as-is to prevent data loss"
            }
        
        # Rule 2: Year keywords + numeric + range 1900-2100 → standard_scale (0.95)
        if cls._matches_keywords(name_lower, cls.YEAR_KEYWORDS):
            if is_numeric:
                non_null = column.dropna()
                if len(non_null) > 0:
                    min_val = non_null.min()
                    max_val = non_null.max()
                    if 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100:
                        return {
                            'action': 'standard_scale',
                            'confidence': 0.95,
                            'reason': f"Column '{column_name}' is a numeric year column (range {min_val:.0f}-{max_val:.0f}) - using standard scaling"
                        }
        
        # Rule 3: Distance keywords + numeric → log1p_transform (0.95)
        if cls._matches_keywords(name_lower, cls.DISTANCE_KEYWORDS):
            if is_numeric:
                non_null = column.dropna()
                if len(non_null) > 0 and non_null.min() >= 0:
                    return {
                        'action': 'log1p_transform',
                        'confidence': 0.95,
                        'reason': f"Column '{column_name}' is a distance/mileage column - using log1p transform for skewed distribution"
                    }
        
        # Rule 4: Binary keywords + 2-3 unique values → keep_as_is (0.95)
        if cls._matches_keywords(name_lower, cls.BINARY_KEYWORDS):
            if unique_count <= 3:
                return {
                    'action': 'keep_as_is',
                    'confidence': 0.95,
                    'reason': f"Column '{column_name}' is a critical binary/indicator column - preserving as-is"
                }
        
        # Rule 5: Drop keywords (id, url) → drop_column (0.90)
        if cls._matches_keywords(name_lower, cls.DROP_KEYWORDS):
            return {
                'action': 'drop_column',
                'confidence': 0.90,
                'reason': f"Column '{column_name}' appears to be an ID/metadata column with no predictive value"
            }
        
        # Rule 5b: Color/exterior columns with limited categories → drop or onehot
        if cls._matches_keywords(name_lower, cls.COLOR_KEYWORDS):
            if is_object:
                if unique_count > 10:
                    return {
                        'action': 'drop_column',
                        'confidence': 0.80,
                        'reason': f"Column '{column_name}' is a color column with high cardinality ({unique_count} values) - low predictive value"
                    }
                else:
                    return {
                        'action': 'onehot_encode',
                        'confidence': 0.80,
                        'reason': f"Column '{column_name}' is a color column with {unique_count} categories - using one-hot encoding"
                    }
        
        # Rule 6: Object dtype + ≤10 unique → onehot_encode (0.85)
        if is_object and unique_count <= 10:
            return {
                'action': 'onehot_encode',
                'confidence': 0.85,
                'reason': f"Column '{column_name}' is categorical with {unique_count} unique values - using one-hot encoding"
            }
        
        # Rule 7: Object dtype + 11-50 unique → label_encode (0.80)
        if is_object and 11 <= unique_count <= 50:
            return {
                'action': 'label_encode',
                'confidence': 0.80,
                'reason': f"Column '{column_name}' is categorical with medium cardinality ({unique_count} values) - using label encoding"
            }
        
        # Rule 8: Object dtype + >50 unique → drop_column (0.75)
        if is_object and unique_count > 50:
            unique_ratio = unique_count / total_count if total_count > 0 else 1
            if unique_ratio > 0.5:
                return {
                    'action': 'drop_column',
                    'confidence': 0.75,
                    'reason': f"Column '{column_name}' has very high cardinality ({unique_count} unique, {unique_ratio:.1%} ratio) - likely ID or free text"
                }
            else:
                return {
                    'action': 'frequency_encode',
                    'confidence': 0.70,
                    'reason': f"Column '{column_name}' has high cardinality ({unique_count} values) - using frequency encoding"
                }
        
        # Rule 9: Numeric + ≤10 unique → onehot_encode (0.80)
        if is_numeric and unique_count <= 10:
            return {
                'action': 'onehot_encode',
                'confidence': 0.80,
                'reason': f"Column '{column_name}' is numeric with low cardinality ({unique_count} values) - treating as categorical"
            }
        
        # Rule 10: Numeric + skew > 1.0 → log1p_transform (0.85)
        if is_numeric:
            non_null = column.dropna()
            if len(non_null) > 0:
                try:
                    skewness = abs(float(non_null.skew()))
                    min_val = non_null.min()
                    if skewness > 1.0 and min_val >= 0:
                        return {
                            'action': 'log1p_transform',
                            'confidence': 0.85,
                            'reason': f"Column '{column_name}' is highly skewed (skew={skewness:.2f}) with non-negative values - using log1p transform"
                        }
                except Exception:
                    pass
        
        # Rule 11: Numeric → standard_scale (0.85)
        if is_numeric:
            return {
                'action': 'standard_scale',
                'confidence': 0.85,
                'reason': f"Column '{column_name}' is numeric - using standard scaling for normalization"
            }
        
        # Rule 12: Default → keep_as_is (0.50)
        return {
            'action': 'keep_as_is',
            'confidence': 0.50,
            'reason': f"Column '{column_name}' does not match any specific pattern - keeping as-is for safety"
        }
    
    @staticmethod
    def _matches_keywords(name: str, keywords: list) -> bool:
        """Check if column name matches any of the keywords."""
        for keyword in keywords:
            # Exact match
            if name == keyword:
                return True
            # Starts with keyword (handles prefixes like "is_")
            if keyword.endswith('_') and name.startswith(keyword):
                return True
            # Contains keyword as word boundary
            if keyword in name.split('_'):
                return True
            # Simple contains for single-word keywords
            if len(keyword) >= 4 and keyword in name:
                return True
        return False


def classify_column(column_name: str, column: pd.Series) -> Dict[str, Any]:
    """Convenience function to classify a column."""
    return SmartColumnClassifier.classify(column_name, column)
