"""
Safety Validator - Quick validation to prevent crashes from type mismatches.

This validator ensures that preprocessing actions are safe to apply to columns,
preventing crashes from operations like scaling text or parsing numeric years as dates.
"""

import pandas as pd

from typing import Tuple


class SafetyValidator:
    """Quick validation to prevent crashes from type mismatches."""
    
    @staticmethod
    def can_apply(column: pd.Series, column_name: str, action: str) -> Tuple[bool, str]:
        """
        Validate if an action can be safely applied to a column.
        
        Args:
            column: The pandas Series to validate
            column_name: Name of the column (for error messages)
            action: The preprocessing action to validate
            
        Returns:
            Tuple of (is_safe, error_message)
            - is_safe: True if action can be safely applied
            - error_message: Empty string if safe, description of issue if not
        """
        is_numeric = pd.api.types.is_numeric_dtype(column)
        is_object = column.dtype == 'object' or column.dtype == 'O'
        unique_count = column.nunique()
        
        # Check 1: standard_scale requires numeric dtype (not object)
        if action == 'standard_scale':
            if is_object:
                return (False, f"Cannot apply standard_scale to '{column_name}': column has object/text dtype. Use encoding first.")
            if not is_numeric:
                return (False, f"Cannot apply standard_scale to '{column_name}': column is not numeric (dtype={column.dtype}).")
        
        # Check 2: log1p_transform requires numeric dtype AND no negative values
        if action in ['log1p_transform', 'log_transform']:
            if is_object:
                return (False, f"Cannot apply {action} to '{column_name}': column has object/text dtype.")
            if not is_numeric:
                return (False, f"Cannot apply {action} to '{column_name}': column is not numeric (dtype={column.dtype}).")
            non_null = column.dropna()
            if len(non_null) > 0:
                min_val = non_null.min()
                if min_val < 0:
                    return (False, f"Cannot apply {action} to '{column_name}': column contains negative values (min={min_val}).")
        
        # Check 3: parse_datetime should NOT be applied to numeric years (1900-2100 range)
        if action == 'parse_datetime':
            if is_numeric:
                non_null = column.dropna()
                if len(non_null) > 0:
                    min_val = non_null.min()
                    max_val = non_null.max()
                    # Check if values look like years (4-digit integers in plausible range)
                    if 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100:
                        return (False, f"Cannot apply parse_datetime to '{column_name}': column appears to be numeric years ({min_val:.0f}-{max_val:.0f}). Use standard_scale instead.")
        
        # Check 4: hash_encode should NOT be applied to continuous numeric (>50 unique)
        if action == 'hash_encode':
            if is_numeric and unique_count > 50:
                return (False, f"Cannot apply hash_encode to '{column_name}': column is continuous numeric with {unique_count} unique values. Use log1p_transform or standard_scale instead.")
        
        # Check 5: onehot_encode should NOT be applied to high cardinality (>50 unique)
        if action == 'onehot_encode':
            if unique_count > 50:
                return (False, f"Cannot apply onehot_encode to '{column_name}': column has {unique_count} unique values (too high). Use label_encode or frequency_encode instead.")
        
        # Check 6: robust_scale requires numeric
        if action == 'robust_scale':
            if is_object:
                return (False, f"Cannot apply robust_scale to '{column_name}': column has object/text dtype.")
            if not is_numeric:
                return (False, f"Cannot apply robust_scale to '{column_name}': column is not numeric (dtype={column.dtype}).")
        
        # Check 7: minmax_scale requires numeric
        if action == 'minmax_scale':
            if is_object:
                return (False, f"Cannot apply minmax_scale to '{column_name}': column has object/text dtype.")
            if not is_numeric:
                return (False, f"Cannot apply minmax_scale to '{column_name}': column is not numeric (dtype={column.dtype}).")
        
        # Action is safe
        return (True, "")
    
    @classmethod
    def validate_action(cls, column: pd.Series, column_name: str, action: str) -> Tuple[bool, str, str]:
        """
        Validate action and suggest a fallback if unsafe.
        
        Args:
            column: The pandas Series to validate
            column_name: Name of the column
            action: The preprocessing action to validate
            
        Returns:
            Tuple of (is_safe, error_message, suggested_action)
        """
        is_safe, error_msg = cls.can_apply(column, column_name, action)
        
        if is_safe:
            return (True, "", action)
        
        # Suggest fallback based on column type
        is_numeric = pd.api.types.is_numeric_dtype(column)
        is_object = column.dtype == 'object' or column.dtype == 'O'
        unique_count = column.nunique()
        
        # Suggest appropriate fallback
        if is_object:
            if unique_count <= 10:
                return (False, error_msg, 'onehot_encode')
            elif unique_count <= 50:
                return (False, error_msg, 'label_encode')
            else:
                return (False, error_msg, 'keep_as_is')
        elif is_numeric:
            return (False, error_msg, 'standard_scale')
        else:
            return (False, error_msg, 'keep_as_is')


def validate_action(column: pd.Series, column_name: str, action: str) -> Tuple[bool, str]:
    """Convenience function to validate an action."""
    return SafetyValidator.can_apply(column, column_name, action)
