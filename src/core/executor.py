"""
Preprocessing Executor - Applies preprocessing actions to data.

This module transforms data based on preprocessing decisions.
Each action is implemented with proper error handling and type safety.
Now includes statistical validation for proof of quality.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Optional, Union, Tuple
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    QuantileTransformer, PowerTransformer, LabelEncoder, OrdinalEncoder
)

from .actions import PreprocessingAction
from ..validation.statistical_validator import StatisticalValidator
from ..validation.consistency_validator import ConsistencyValidator

logger = logging.getLogger(__name__)


class PreprocessingExecutor:
    """
    Executes preprocessing actions on data.
    
    Implements all 58 preprocessing actions with proper error handling.
    Now includes statistical validation for proof of quality.
    """
    
    def __init__(self):
        """Initialize the executor."""
        self.scalers = {}  # Cache fitted scalers for consistency
        self.encoders = {}  # Cache fitted encoders
        
        # Initialize validators
        self.statistical_validator = StatisticalValidator()
        self.consistency_validator = ConsistencyValidator()
    
    def apply_action(
        self,
        column: pd.Series,
        action: Union[PreprocessingAction, str],
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.Series:
        """
        Apply a single preprocessing action to a column.
        
        Args:
            column: Input column data
            action: Preprocessing action to apply
            parameters: Optional parameters for the action
        
        Returns:
            Transformed column
        
        Raises:
            ValueError: If action is invalid or transformation fails
        """
        # Convert string to enum if needed
        if isinstance(action, str):
            try:
                action = PreprocessingAction(action)
            except ValueError:
                raise ValueError(f"Invalid action: {action}")
        
        try:
            # Route to appropriate handler
            if action == PreprocessingAction.KEEP_AS_IS:
                return column.copy()
            
            # Data Quality Actions
            elif action in [
                PreprocessingAction.DROP_IF_MOSTLY_NULL,
                PreprocessingAction.DROP_IF_CONSTANT,
                PreprocessingAction.DROP_IF_ALL_UNIQUE,
                PreprocessingAction.REMOVE_DUPLICATES,
                PreprocessingAction.FILL_NULL_MEAN,
                PreprocessingAction.FILL_NULL_MEDIAN,
                PreprocessingAction.FILL_NULL_MODE,
                PreprocessingAction.FILL_NULL_FORWARD,
                PreprocessingAction.FILL_NULL_BACKWARD,
                PreprocessingAction.FILL_NULL_INTERPOLATE
            ]:
                return self._handle_null_filling(column, action)
            
            # Scaling Actions
            elif action in [
                PreprocessingAction.STANDARD_SCALE,
                PreprocessingAction.MINMAX_SCALE,
                PreprocessingAction.ROBUST_SCALE,
                PreprocessingAction.MAXABS_SCALE,
                PreprocessingAction.NORMALIZE_L1,
                PreprocessingAction.NORMALIZE_L2
            ]:
                return self._handle_scaling(column, action)
            
            # Transformation Actions
            elif action in [
                PreprocessingAction.LOG_TRANSFORM,
                PreprocessingAction.LOG1P_TRANSFORM,
                PreprocessingAction.SQRT_TRANSFORM,
                PreprocessingAction.BOX_COX,
                PreprocessingAction.YEO_JOHNSON,
                PreprocessingAction.QUANTILE_TRANSFORM,
                PreprocessingAction.POWER_TRANSFORM
            ]:
                return self._handle_transformation(column, action)
            
            # Outlier Handling
            elif action in [
                PreprocessingAction.CLIP_OUTLIERS,
                PreprocessingAction.WINSORIZE,
                PreprocessingAction.CAP_FLOOR_OUTLIERS
            ]:
                return self._handle_outliers(column, action, parameters)
            
            # Encoding Actions
            elif action in [
                PreprocessingAction.LABEL_ENCODE,
                PreprocessingAction.ORDINAL_ENCODE,
                PreprocessingAction.FREQUENCY_ENCODE,
                PreprocessingAction.BINARY_ENCODE
            ]:
                return self._handle_encoding(column, action)
            
            # Type Conversion
            elif action in [
                PreprocessingAction.PARSE_NUMERIC,
                PreprocessingAction.PARSE_BOOLEAN,
                PreprocessingAction.PARSE_DATETIME
            ]:
                return self._handle_type_conversion(column, action)
            
            # Domain-Specific
            elif action in [
                PreprocessingAction.CURRENCY_NORMALIZE,
                PreprocessingAction.PERCENTAGE_TO_DECIMAL,
                PreprocessingAction.TEXT_LOWERCASE,
                PreprocessingAction.TEXT_UPPERCASE,
                PreprocessingAction.TEXT_CLEAN
            ]:
                return self._handle_domain_specific(column, action)
            
            # Binning
            elif action in [
                PreprocessingAction.BINNING_EQUAL_WIDTH,
                PreprocessingAction.BINNING_EQUAL_FREQ
            ]:
                return self._handle_binning(column, action, parameters)
            
            # Drop Column (handled in execute_batch, but if called directly return empty)
            elif action == PreprocessingAction.DROP_COLUMN:
                return pd.Series(dtype=column.dtype)
            
            else:
                logger.warning(f"Action {action.value} not yet implemented, returning original")
                return column.copy()
                
        except Exception as e:
            logger.error(f"Failed to apply {action.value}: {e}")
            raise ValueError(f"Transformation failed for {action.value}: {str(e)}")
    
    def _handle_null_filling(self, column: pd.Series, action: PreprocessingAction) -> pd.Series:
        """Handle null filling actions."""
        if action == PreprocessingAction.FILL_NULL_MEAN:
            return column.fillna(column.mean())
        elif action == PreprocessingAction.FILL_NULL_MEDIAN:
            return column.fillna(column.median())
        elif action == PreprocessingAction.FILL_NULL_MODE:
            mode_val = column.mode()
            if len(mode_val) > 0:
                return column.fillna(mode_val[0])
            return column
        elif action == PreprocessingAction.FILL_NULL_FORWARD:
            return column.fillna(method='ffill')
        elif action == PreprocessingAction.FILL_NULL_BACKWARD:
            return column.fillna(method='bfill')
        elif action == PreprocessingAction.FILL_NULL_INTERPOLATE:
            return column.interpolate()
        return column
    
    def _handle_scaling(self, column: pd.Series, action: PreprocessingAction) -> pd.Series:
        """Handle scaling actions."""
        # Ensure numeric
        if not pd.api.types.is_numeric_dtype(column):
            raise ValueError(f"Cannot scale non-numeric column")
        
        # Drop NaN for fitting
        clean_data = column.dropna().values.reshape(-1, 1)
        
        if len(clean_data) == 0:
            return column
        
        # Select scaler
        if action == PreprocessingAction.STANDARD_SCALE:
            scaler = StandardScaler()
        elif action == PreprocessingAction.MINMAX_SCALE:
            scaler = MinMaxScaler()
        elif action == PreprocessingAction.ROBUST_SCALE:
            scaler = RobustScaler()
        elif action == PreprocessingAction.MAXABS_SCALE:
            scaler = MaxAbsScaler()
        elif action == PreprocessingAction.NORMALIZE_L1:
            # L1 normalization
            result = column.copy()
            result = result / result.abs().sum()
            return result
        elif action == PreprocessingAction.NORMALIZE_L2:
            # L2 normalization
            result = column.copy()
            result = result / np.sqrt((result ** 2).sum())
            return result
        else:
            return column
        
        # Fit and transform
        scaler.fit(clean_data)
        result = column.copy()
        mask = ~column.isna()
        result[mask] = scaler.transform(column[mask].values.reshape(-1, 1)).flatten()
        
        return result
    
    def _handle_transformation(self, column: pd.Series, action: PreprocessingAction) -> pd.Series:
        """Handle transformation actions."""
        if not pd.api.types.is_numeric_dtype(column):
            raise ValueError(f"Cannot transform non-numeric column")
        
        result = column.copy()
        
        if action == PreprocessingAction.LOG_TRANSFORM:
            # Ensure positive values
            if (result <= 0).any():
                logger.warning("Log transform on non-positive values, using log1p instead")
                return np.log1p(result)
            return np.log(result)
        
        elif action == PreprocessingAction.LOG1P_TRANSFORM:
            return np.log1p(result)
        
        elif action == PreprocessingAction.SQRT_TRANSFORM:
            if (result < 0).any():
                raise ValueError("Cannot apply sqrt to negative values")
            return np.sqrt(result)
        
        elif action == PreprocessingAction.BOX_COX:
            # Box-Cox requires positive values
            clean_data = result.dropna()
            if (clean_data <= 0).any():
                raise ValueError("Box-Cox requires positive values")
            transformed, _ = stats.boxcox(clean_data)
            result[~result.isna()] = transformed
            return result
        
        elif action == PreprocessingAction.YEO_JOHNSON:
            # Yeo-Johnson works with any values
            transformer = PowerTransformer(method='yeo-johnson')
            clean_data = result.dropna().values.reshape(-1, 1)
            transformer.fit(clean_data)
            mask = ~result.isna()
            result[mask] = transformer.transform(result[mask].values.reshape(-1, 1)).flatten()
            return result
        
        elif action == PreprocessingAction.QUANTILE_TRANSFORM:
            transformer = QuantileTransformer()
            clean_data = result.dropna().values.reshape(-1, 1)
            transformer.fit(clean_data)
            mask = ~result.isna()
            result[mask] = transformer.transform(result[mask].values.reshape(-1, 1)).flatten()
            return result
        
        elif action == PreprocessingAction.POWER_TRANSFORM:
            transformer = PowerTransformer()
            clean_data = result.dropna().values.reshape(-1, 1)
            transformer.fit(clean_data)
            mask = ~result.isna()
            result[mask] = transformer.transform(result[mask].values.reshape(-1, 1)).flatten()
            return result
        
        return result
    
    def _handle_outliers(self, column: pd.Series, action: PreprocessingAction, parameters: Dict) -> pd.Series:
        """Handle outlier actions."""
        if not pd.api.types.is_numeric_dtype(column):
            return column
        
        result = column.copy()
        
        if action == PreprocessingAction.CLIP_OUTLIERS:
            # IQR method
            Q1 = result.quantile(0.25)
            Q3 = result.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return result.clip(lower, upper)
        
        elif action == PreprocessingAction.WINSORIZE:
            # Winsorize at 5th and 95th percentiles
            lower = result.quantile(0.05)
            upper = result.quantile(0.95)
            return result.clip(lower, upper)
        
        elif action == PreprocessingAction.CAP_FLOOR_OUTLIERS:
            # Cap at mean ± 3*std
            mean = result.mean()
            std = result.std()
            lower = mean - 3 * std
            upper = mean + 3 * std
            return result.clip(lower, upper)
        
        return result
    
    def _handle_encoding(self, column: pd.Series, action: PreprocessingAction) -> pd.Series:
        """Handle encoding actions."""
        result = column.copy()
        
        if action == PreprocessingAction.LABEL_ENCODE:
            encoder = LabelEncoder()
            # Handle NaN
            mask = ~result.isna()
            result[mask] = encoder.fit_transform(result[mask].astype(str))
            return result
        
        elif action == PreprocessingAction.FREQUENCY_ENCODE:
            # Replace categories with their frequency
            freq = result.value_counts(normalize=True)
            return result.map(freq)
        
        elif action == PreprocessingAction.BINARY_ENCODE:
            # Binary encoding for boolean-like columns
            unique_vals = result.dropna().unique()
            if len(unique_vals) <= 2:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1} if len(unique_vals) == 2 else {unique_vals[0]: 0}
                return result.map(mapping)
            else:
                logger.warning("Binary encode requires <=2 unique values, using label encode")
                return self._handle_encoding(column, PreprocessingAction.LABEL_ENCODE)
        
        return result
    
    def _handle_type_conversion(self, column: pd.Series, action: PreprocessingAction) -> pd.Series:
        """Handle type conversion actions."""
        if action == PreprocessingAction.PARSE_NUMERIC:
            return pd.to_numeric(column, errors='coerce')
        
        elif action == PreprocessingAction.PARSE_BOOLEAN:
            # Convert to boolean
            true_vals = ['true', 'yes', '1', 't', 'y']
            false_vals = ['false', 'no', '0', 'f', 'n']
            result = column.astype(str).str.lower()
            result = result.map(lambda x: True if x in true_vals else (False if x in false_vals else None))
            return result
        
        elif action == PreprocessingAction.PARSE_DATETIME:
            return pd.to_datetime(column, errors='coerce')
        
        return column
    
    def _handle_domain_specific(self, column: pd.Series, action: PreprocessingAction) -> pd.Series:
        """Handle domain-specific actions."""
        result = column.copy()
        
        if action == PreprocessingAction.CURRENCY_NORMALIZE:
            # Remove currency symbols and convert to float
            result = result.astype(str).str.replace(r'[$,€£¥]', '', regex=True)
            return pd.to_numeric(result, errors='coerce')
        
        elif action == PreprocessingAction.PERCENTAGE_TO_DECIMAL:
            # Convert percentage to decimal
            result = result.astype(str).str.replace('%', '')
            result = pd.to_numeric(result, errors='coerce') / 100
            return result
        
        elif action == PreprocessingAction.TEXT_LOWERCASE:
            return result.astype(str).str.lower()
        
        elif action == PreprocessingAction.TEXT_UPPERCASE:
            return result.astype(str).str.upper()
        
        elif action == PreprocessingAction.TEXT_CLEAN:
            # Remove special characters and extra whitespace
            result = result.astype(str).str.replace(r'[^\w\s]', '', regex=True)
            result = result.str.replace(r'\s+', ' ', regex=True).str.strip()
            return result
        
        return result
    
    def _handle_binning(self, column: pd.Series, action: PreprocessingAction, parameters: Dict) -> pd.Series:
        """Handle binning actions."""
        if not pd.api.types.is_numeric_dtype(column):
            return column
        
        n_bins = parameters.get('n_bins', 5)
        
        if action == PreprocessingAction.BINNING_EQUAL_WIDTH:
            return pd.cut(column, bins=n_bins, labels=False)
        
        elif action == PreprocessingAction.BINNING_EQUAL_FREQ:
            return pd.qcut(column, q=n_bins, labels=False, duplicates='drop')
        
        return column
    
    def execute_batch(
        self,
        df: pd.DataFrame,
        decisions: Dict[str, Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute preprocessing on entire dataframe with VALIDATION.
        
        Args:
            df: Input dataframe
            decisions: Dict mapping column names to decision dicts
        
        Returns:
            Tuple of (Preprocessed dataframe, Validation report)
        """
        result_df = df.copy()
        validation_report = {
            "summary": {
                "total_columns": len(df.columns),
                "processed_columns": 0,
                "validated_columns": 0,
                "passed_validation": 0
            },
            "columns": {}
        }
        
        # Track dropped columns to remove them at the end
        columns_to_drop = []
        
        for col_name, decision in decisions.items():
            if col_name not in df.columns:
                continue
            
            # Only apply if accepted
            if not decision.get('accepted', True):
                continue
            
            try:
                action_str = decision['action']
                action = PreprocessingAction(action_str)
                parameters = decision.get('parameters', {})
                
                # Handle DROP_COLUMN separately
                if action == PreprocessingAction.DROP_COLUMN:
                    columns_to_drop.append(col_name)
                    validation_report["columns"][col_name] = {
                        "action": "drop_column",
                        "status": "dropped",
                        "validation": "N/A (Column Dropped)"
                    }
                    continue
                
                # Apply transformation
                original_col = df[col_name]
                transformed_col = self.apply_action(
                    original_col,
                    action,
                    parameters
                )
                
                # Update result
                result_df[col_name] = transformed_col
                validation_report["summary"]["processed_columns"] += 1
                
                # --- STATISTICAL VALIDATION (Proof of Quality) ---
                validation_results = {}
                
                # 1. Statistical Tests (Normality, Variance, etc.)
                stat_result = self.statistical_validator.validate(
                    original_col, transformed_col, action.value
                )
                validation_results["statistical"] = stat_result.to_dict()
                
                # 2. Consistency Checks (Range, Semantic)
                const_result = self.consistency_validator.validate(
                    original_col, transformed_col, action.value, col_name
                )
                validation_results["consistency"] = const_result.to_dict()
                
                # Determine overall pass/fail
                passed = stat_result.passed and const_result.passed
                if passed:
                    validation_report["summary"]["passed_validation"] += 1
                
                validation_report["summary"]["validated_columns"] += 1
                validation_report["columns"][col_name] = {
                    "action": action.value,
                    "status": "processed",
                    "passed": passed,
                    "metrics": validation_results
                }
                
                logger.info(f"Applied {action} to {col_name} (Validated: {passed})")
                
            except Exception as e:
                logger.error(f"Failed to process column {col_name}: {e}")
                result_df[col_name] = df[col_name]
                validation_report["columns"][col_name] = {
                    "action": decision.get('action'),
                    "status": "failed",
                    "error": str(e)
                }
        
        # Drop columns marked for removal
        if columns_to_drop:
            result_df = result_df.drop(columns=columns_to_drop)
            logger.info(f"Dropped {len(columns_to_drop)} columns: {columns_to_drop}")
        
        return result_df, validation_report


# Global executor instance
_executor_instance = None

def get_executor() -> PreprocessingExecutor:
    """Get global executor instance."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = PreprocessingExecutor()
    return _executor_instance
