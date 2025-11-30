"""
Preprocessing Integration - Integrates smart classification with safety validation.

This module provides the main entry point for getting preprocessing decisions,
combining the SmartColumnClassifier with SafetyValidator to ensure robust decisions.
"""

import pandas as pd
from typing import Dict, Any, Optional

from .smart_classifier import SmartColumnClassifier
from .safety_validator import SafetyValidator


class PreprocessingIntegration:
    """Integrates smart classification with safety validation."""
    
    @staticmethod
    def get_preprocessing_decision(
        column: pd.Series, 
        column_name: str, 
        position: int = -1, 
        total_columns: int = -1
    ) -> Dict[str, Any]:
        """
        Two-step decision process:
        1. Get recommendation from SmartColumnClassifier
        2. Validate with SafetyValidator
        3. If unsafe, fallback to keep_as_is with warning
        
        Args:
            column: The pandas Series to process
            column_name: Name of the column
            position: Column position in dataset (optional)
            total_columns: Total number of columns (optional)
            
        Returns:
            Dict containing:
                - action: The recommended preprocessing action
                - confidence: Confidence score (0.0 to 1.0)
                - source: 'smart_classifier' or 'safety_fallback'
                - explanation: Human-readable explanation
                - warning: Optional warning message if safety validation failed
        """
        # Step 1: Get recommendation from SmartColumnClassifier
        smart_result = SmartColumnClassifier.classify(column_name, column)
        
        recommended_action = smart_result['action']
        confidence = smart_result['confidence']
        reason = smart_result['reason']
        
        # Step 2: Validate with SafetyValidator
        is_safe, error_msg = SafetyValidator.can_apply(column, column_name, recommended_action)
        
        if is_safe:
            # Action is safe, return the recommendation
            return {
                'action': recommended_action,
                'confidence': confidence,
                'source': 'smart_classifier',
                'explanation': reason,
                'warning': None
            }
        else:
            # Action is unsafe, use fallback
            is_safe_fb, _, fallback_action = SafetyValidator.validate_action(
                column, column_name, recommended_action
            )
            
            # Determine appropriate fallback
            if not is_safe_fb:
                fallback_action = 'keep_as_is'
            
            return {
                'action': fallback_action,
                'confidence': min(confidence * 0.8, 0.70),  # Reduce confidence for fallback
                'source': 'safety_fallback',
                'explanation': f"Original action '{recommended_action}' was unsafe. {reason}",
                'warning': f"Safety check failed: {error_msg}. Using '{fallback_action}' instead."
            }
    
    @classmethod
    def classify_dataframe_columns(
        cls, 
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Classify all columns in a dataframe.
        
        Args:
            df: The pandas DataFrame to process
            target_column: Optional name of target column to skip
            
        Returns:
            Dict mapping column names to their preprocessing decisions
        """
        results = {}
        total_columns = len(df.columns)
        
        for position, column_name in enumerate(df.columns):
            if column_name == target_column:
                results[column_name] = {
                    'action': 'keep_as_is',
                    'confidence': 1.0,
                    'source': 'target_column',
                    'explanation': 'Target column - preserved as-is',
                    'warning': None
                }
                continue
            
            decision = cls.get_preprocessing_decision(
                df[column_name], 
                column_name, 
                position, 
                total_columns
            )
            results[column_name] = decision
        
        return results
    
    @classmethod
    def get_decision_summary(cls, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of preprocessing decisions.
        
        Args:
            decisions: Dict of column decisions from classify_dataframe_columns
            
        Returns:
            Summary statistics about the decisions
        """
        action_counts = {}
        source_counts = {'smart_classifier': 0, 'safety_fallback': 0, 'target_column': 0}
        warnings = []
        avg_confidence = 0.0
        
        for col_name, decision in decisions.items():
            action = decision['action']
            source = decision['source']
            
            action_counts[action] = action_counts.get(action, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            avg_confidence += decision['confidence']
            
            if decision.get('warning'):
                warnings.append({
                    'column': col_name,
                    'warning': decision['warning']
                })
        
        num_columns = len(decisions)
        
        return {
            'total_columns': num_columns,
            'action_breakdown': action_counts,
            'source_breakdown': source_counts,
            'average_confidence': avg_confidence / num_columns if num_columns > 0 else 0,
            'safety_fallbacks': len(warnings),
            'warnings': warnings
        }


def get_preprocessing_decision(column: pd.Series, column_name: str) -> Dict[str, Any]:
    """Convenience function to get preprocessing decision for a column."""
    return PreprocessingIntegration.get_preprocessing_decision(column, column_name)
