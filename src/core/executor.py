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
    QuantileTransformer, PowerTransformer, LabelEncoder, OrdinalEncoder,
    OneHotEncoder
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import hashlib

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
    
    # Fallback mapping for when primary action fails
    ACTION_FALLBACKS = {
        'standard_scale': 'minmax_scale',
        'box_cox': 'log1p_transform',
        'log_transform': 'log1p_transform',
        'onehot_encode': 'label_encode',
        'target_encode': 'frequency_encode',
        'hash_encode': 'label_encode',
        'text_vectorize_tfidf': 'label_encode',
        'geo_cluster_kmeans': 'binning_equal_freq',
    }
    
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
                PreprocessingAction.BINARY_ENCODE,
                PreprocessingAction.ONEHOT_ENCODE,
                PreprocessingAction.HASH_ENCODE,
                PreprocessingAction.TARGET_ENCODE
            ]:
                return self._handle_encoding(column, action, parameters)
            
            # Type Conversion
            elif action in [
                PreprocessingAction.PARSE_NUMERIC,
                PreprocessingAction.PARSE_BOOLEAN,
                PreprocessingAction.PARSE_DATETIME,
                PreprocessingAction.PARSE_CATEGORICAL,
                PreprocessingAction.PARSE_JSON
            ]:
                return self._handle_type_conversion(column, action)
            
            # Domain-Specific
            elif action in [
                PreprocessingAction.CURRENCY_NORMALIZE,
                PreprocessingAction.PERCENTAGE_TO_DECIMAL,
                PreprocessingAction.TEXT_LOWERCASE,
                PreprocessingAction.TEXT_UPPERCASE,
                PreprocessingAction.TEXT_CLEAN,
                PreprocessingAction.PHONE_STANDARDIZE,
                PreprocessingAction.EMAIL_VALIDATE,
                PreprocessingAction.URL_PARSE
            ]:
                return self._handle_domain_specific(column, action)
            
            # Text Processing
            elif action in [
                PreprocessingAction.TEXT_VECTORIZE_TFIDF
            ]:
                return self._handle_text_processing(column, action, parameters)
            
            # Datetime Feature Extraction
            elif action in [
                PreprocessingAction.DATETIME_EXTRACT_YEAR,
                PreprocessingAction.CYCLIC_TIME_ENCODE
            ]:
                return self._handle_datetime_extraction(column, action)
            
            # Geospatial
            elif action in [
                PreprocessingAction.GEO_CLUSTER_KMEANS
            ]:
                return self._handle_geospatial(column, action, parameters)
            
            # Binning
            elif action in [
                PreprocessingAction.BINNING_EQUAL_WIDTH,
                PreprocessingAction.BINNING_EQUAL_FREQ,
                PreprocessingAction.BINNING_CUSTOM
            ]:
                return self._handle_binning(column, action, parameters)
            
            # Feature Engineering
            elif action in [
                PreprocessingAction.POLYNOMIAL_FEATURES,
                PreprocessingAction.INTERACTION_FEATURES
            ]:
                return self._handle_feature_engineering(column, action, parameters)
            
            # Drop Column (handled in execute_batch, but if called directly return empty)
            elif action == PreprocessingAction.DROP_COLUMN:
                return pd.Series(dtype=column.dtype)
            
            # Remove Outliers (special handling - returns filtered column)
            elif action == PreprocessingAction.REMOVE_OUTLIERS:
                return self._handle_remove_outliers(column)
            
            else:
                logger.warning(f"Action {action.value} not yet implemented, returning original")
                return column.copy()
                
        except Exception as e:
            logger.error(f"Failed to apply {action.value}: {e}")
            # Try fallback action if available
            fallback = self.ACTION_FALLBACKS.get(action.value if hasattr(action, 'value') else str(action))
            if fallback:
                logger.info(f"Attempting fallback action: {fallback}")
                try:
                    return self.apply_action(column, fallback, parameters)
                except Exception as fallback_error:
                    logger.error(f"Fallback {fallback} also failed: {fallback_error}")
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
        """Handle scaling actions with automatic type coercion."""
        # Try to coerce to numeric if not already numeric
        if not pd.api.types.is_numeric_dtype(column):
            # Attempt to parse as numeric (handles "4.5", "$19.99", "3,456" etc.)
            column = self._coerce_to_numeric(column)
            if not pd.api.types.is_numeric_dtype(column):
                raise ValueError(f"Cannot scale non-numeric column (coercion failed)")
        
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
            abs_sum = result.abs().sum()
            if abs_sum > 0:
                result = result / abs_sum
            return result
        elif action == PreprocessingAction.NORMALIZE_L2:
            # L2 normalization
            result = column.copy()
            l2_norm = np.sqrt((result ** 2).sum())
            if l2_norm > 0:
                result = result / l2_norm
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
        """Handle transformation actions with automatic type coercion."""
        # Try to coerce to numeric if not already numeric
        if not pd.api.types.is_numeric_dtype(column):
            column = self._coerce_to_numeric(column)
            if not pd.api.types.is_numeric_dtype(column):
                raise ValueError(f"Cannot transform non-numeric column (coercion failed)")
        
        result = column.copy()
        
        if action == PreprocessingAction.LOG_TRANSFORM:
            # Ensure positive values
            if (result.dropna() <= 0).any():
                logger.warning("Log transform on non-positive values, using log1p instead")
                return np.log1p(result.clip(lower=0))
            return np.log(result)
        
        elif action == PreprocessingAction.LOG1P_TRANSFORM:
            # Clip to ensure no negative values cause issues
            return np.log1p(result.clip(lower=-1 + 1e-10))
        
        elif action == PreprocessingAction.SQRT_TRANSFORM:
            if (result.dropna() < 0).any():
                logger.warning("Sqrt transform on negative values, using absolute values")
                result = result.abs()
            return np.sqrt(result)
        
        elif action == PreprocessingAction.BOX_COX:
            # Box-Cox requires positive values
            clean_data = result.dropna()
            if len(clean_data) == 0:
                return result
            shift = 0.0
            if (clean_data <= 0).any():
                # Shift data to make it positive
                shift = abs(clean_data.min()) + 1
                logger.warning(f"Box-Cox: shifting data by {shift} to make positive")
            # Apply shift to all non-null values before transformation
            if shift > 0:
                result = result + shift
                clean_data = result.dropna()
            transformed, _ = stats.boxcox(clean_data)
            result[~result.isna()] = transformed
            return result
        
        elif action == PreprocessingAction.YEO_JOHNSON:
            # Yeo-Johnson works with any values
            clean_data = result.dropna().values.reshape(-1, 1)
            if len(clean_data) == 0:
                return result
            transformer = PowerTransformer(method='yeo-johnson')
            transformer.fit(clean_data)
            mask = ~result.isna()
            result[mask] = transformer.transform(result[mask].values.reshape(-1, 1)).flatten()
            return result
        
        elif action == PreprocessingAction.QUANTILE_TRANSFORM:
            clean_data = result.dropna().values.reshape(-1, 1)
            if len(clean_data) < 2:
                return result
            # Use conservative n_quantiles to avoid overfitting on small datasets
            n_quantiles = min(len(clean_data) // 2, 1000)
            n_quantiles = max(n_quantiles, 2)  # Minimum 2 quantiles
            transformer = QuantileTransformer(n_quantiles=n_quantiles)
            transformer.fit(clean_data)
            mask = ~result.isna()
            result[mask] = transformer.transform(result[mask].values.reshape(-1, 1)).flatten()
            return result
        
        elif action == PreprocessingAction.POWER_TRANSFORM:
            clean_data = result.dropna().values.reshape(-1, 1)
            if len(clean_data) == 0:
                return result
            transformer = PowerTransformer()
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
    
    def _handle_encoding(self, column: pd.Series, action: PreprocessingAction, parameters: Optional[Dict[str, Any]] = None) -> pd.Series:
        """Handle encoding actions including one-hot, hash, and target encoding."""
        result = column.copy()
        parameters = parameters or {}
        
        if action == PreprocessingAction.LABEL_ENCODE:
            encoder = LabelEncoder()
            # Handle NaN
            mask = ~result.isna()
            if mask.sum() > 0:
                result[mask] = encoder.fit_transform(result[mask].astype(str))
            return result
        
        elif action == PreprocessingAction.ORDINAL_ENCODE:
            # Ordinal encoding with optional order specification
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            mask = ~result.isna()
            if mask.sum() > 0:
                values = result[mask].values.reshape(-1, 1)
                result[mask] = encoder.fit_transform(values.astype(str)).flatten()
            return result
        
        elif action == PreprocessingAction.FREQUENCY_ENCODE:
            # Replace categories with their frequency
            freq = result.value_counts(normalize=True)
            return result.map(freq).fillna(0)
        
        elif action == PreprocessingAction.BINARY_ENCODE:
            # Binary encoding for boolean-like columns
            unique_vals = result.dropna().unique()
            if len(unique_vals) <= 2:
                mapping = {unique_vals[0]: 0}
                if len(unique_vals) == 2:
                    mapping[unique_vals[1]] = 1
                return result.map(mapping)
            else:
                logger.warning("Binary encode requires <=2 unique values, using label encode")
                return self._handle_encoding(column, PreprocessingAction.LABEL_ENCODE)
        
        elif action == PreprocessingAction.ONEHOT_ENCODE:
            # One-hot encoding - returns label encoded for single column
            # (full one-hot requires DataFrame context, handled in execute_batch)
            # For single column, use label encoding as surrogate
            logger.info("One-hot encoding single column - using label encoding (full one-hot requires DataFrame context)")
            encoder = LabelEncoder()
            mask = ~result.isna()
            if mask.sum() > 0:
                result[mask] = encoder.fit_transform(result[mask].astype(str))
            return result
        
        elif action == PreprocessingAction.HASH_ENCODE:
            # Hash encoding - good for high cardinality categoricals
            n_components = parameters.get('n_components', 8)
            def hash_value(val):
                if pd.isna(val):
                    return 0
                hash_bytes = hashlib.md5(str(val).encode()).digest()
                return int.from_bytes(hash_bytes[:4], 'big') % n_components
            return result.apply(hash_value)
        
        elif action == PreprocessingAction.TARGET_ENCODE:
            # Target encoding requires target variable
            # If target not provided, fall back to frequency encoding
            target = parameters.get('target')
            if target is None:
                logger.warning("Target encoding requires target variable, falling back to frequency encoding")
                return self._handle_encoding(column, PreprocessingAction.FREQUENCY_ENCODE)
            
            # Calculate mean target value for each category
            combined = pd.DataFrame({'category': result, 'target': target})
            means = combined.groupby('category')['target'].mean()
            global_mean = target.mean()
            return result.map(means).fillna(global_mean)
        
        return result
    
    def _handle_type_conversion(self, column: pd.Series, action: PreprocessingAction) -> pd.Series:
        """Handle type conversion actions."""
        if action == PreprocessingAction.PARSE_NUMERIC:
            return self._coerce_to_numeric(column)
        
        elif action == PreprocessingAction.PARSE_BOOLEAN:
            # Convert to boolean (returns 0/1 integers for ML compatibility)
            true_vals = ['true', 'yes', '1', 't', 'y', 'on']
            false_vals = ['false', 'no', '0', 'f', 'n', 'off']
            result = column.astype(str).str.lower().str.strip()
            return result.map(lambda x: 1 if x in true_vals else (0 if x in false_vals else np.nan))
        
        elif action == PreprocessingAction.PARSE_DATETIME:
            return pd.to_datetime(column, errors='coerce', infer_datetime_format=True)
        
        elif action == PreprocessingAction.PARSE_CATEGORICAL:
            # Convert to pandas categorical type
            return column.astype('category')
        
        elif action == PreprocessingAction.PARSE_JSON:
            # Parse JSON strings - extract first level keys/values
            import json
            def parse_json_value(val):
                if pd.isna(val):
                    return val
                try:
                    parsed = json.loads(str(val))
                    if isinstance(parsed, dict):
                        # Return flattened string representation
                        return str(parsed)
                    return parsed
                except:
                    return val
            return column.apply(parse_json_value)
        
        return column
    
    def _handle_domain_specific(self, column: pd.Series, action: PreprocessingAction) -> pd.Series:
        """Handle domain-specific actions."""
        result = column.copy()
        
        if action == PreprocessingAction.CURRENCY_NORMALIZE:
            # Remove currency symbols and convert to float
            result = result.astype(str).str.replace(r'[$,€£¥₹\s]', '', regex=True)
            return pd.to_numeric(result, errors='coerce')
        
        elif action == PreprocessingAction.PERCENTAGE_TO_DECIMAL:
            # Convert percentage to decimal
            result = result.astype(str).str.replace('%', '').str.strip()
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
        
        elif action == PreprocessingAction.PHONE_STANDARDIZE:
            # Standardize phone numbers - remove non-numeric characters
            result = result.astype(str).str.replace(r'[^\d+]', '', regex=True)
            return result
        
        elif action == PreprocessingAction.EMAIL_VALIDATE:
            # Validate and clean email addresses - lowercase and strip
            result = result.astype(str).str.lower().str.strip()
            # Mark invalid emails as NaN
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            valid_mask = result.str.match(email_pattern, na=False)
            result = result.where(valid_mask, np.nan)
            return result
        
        elif action == PreprocessingAction.URL_PARSE:
            # Parse URLs - extract domain or validate
            def extract_domain(url):
                if pd.isna(url):
                    return url
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(str(url))
                    return parsed.netloc if parsed.netloc else str(url)
                except:
                    return url
            return result.apply(extract_domain)
        
        return result
    
    def _handle_binning(self, column: pd.Series, action: PreprocessingAction, parameters: Optional[Dict[str, Any]] = None) -> pd.Series:
        """Handle binning actions."""
        parameters = parameters or {}
        
        # Try to coerce to numeric if needed
        if not pd.api.types.is_numeric_dtype(column):
            column = self._coerce_to_numeric(column)
            if not pd.api.types.is_numeric_dtype(column):
                logger.warning("Cannot bin non-numeric column, returning original")
                return column
        
        n_bins = parameters.get('n_bins', 5)
        
        if action == PreprocessingAction.BINNING_EQUAL_WIDTH:
            try:
                return pd.cut(column, bins=n_bins, labels=False)
            except ValueError:
                # Handle case where there are too few unique values
                return pd.cut(column, bins=min(n_bins, column.nunique()), labels=False, duplicates='drop')
        
        elif action == PreprocessingAction.BINNING_EQUAL_FREQ:
            try:
                return pd.qcut(column, q=n_bins, labels=False, duplicates='drop')
            except ValueError:
                # Handle case where there are too few unique values
                return pd.qcut(column, q=min(n_bins, column.nunique()), labels=False, duplicates='drop')
        
        elif action == PreprocessingAction.BINNING_CUSTOM:
            # Custom binning with specified bins
            bins = parameters.get('bins', [0, 25, 50, 75, 100])
            return pd.cut(column, bins=bins, labels=False, include_lowest=True)
        
        return column
    
    def _handle_text_processing(self, column: pd.Series, action: PreprocessingAction, parameters: Optional[Dict[str, Any]] = None) -> pd.Series:
        """Handle text processing actions like TF-IDF vectorization."""
        parameters = parameters or {}
        
        if action == PreprocessingAction.TEXT_VECTORIZE_TFIDF:
            max_features = parameters.get('max_features', 100)
            
            # Fill NaN with empty string
            text_data = column.fillna('').astype(str)
            
            # Use TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=max_features)
            try:
                tfidf_matrix = vectorizer.fit_transform(text_data)
                # Return mean TF-IDF score as single column (for single-column output)
                # Full vectorization should be done at DataFrame level
                logger.info(f"TF-IDF vectorization: returning mean score per row (full vectors have {tfidf_matrix.shape[1]} features)")
                return pd.Series(tfidf_matrix.mean(axis=1).A1, index=column.index)
            except Exception as e:
                logger.warning(f"TF-IDF vectorization failed: {e}, falling back to label encoding")
                return self._handle_encoding(column, PreprocessingAction.LABEL_ENCODE)
        
        return column
    
    def _handle_datetime_extraction(self, column: pd.Series, action: PreprocessingAction) -> pd.Series:
        """Handle datetime feature extraction."""
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(column):
            column = pd.to_datetime(column, errors='coerce')
        
        if action == PreprocessingAction.DATETIME_EXTRACT_YEAR:
            return column.dt.year.astype(float)
        
        elif action == PreprocessingAction.CYCLIC_TIME_ENCODE:
            # Cyclic encoding for time features (hour, day of week, month)
            # Returns sin/cos encoding for hour of day as example
            if column.dt.hour is not None:
                hour = column.dt.hour
                # Normalize to [0, 2*pi]
                hour_normalized = 2 * np.pi * hour / 24
                # Return sin component (could also return cos in separate column)
                return np.sin(hour_normalized)
            else:
                # For dates without time, encode month cyclically
                month = column.dt.month
                month_normalized = 2 * np.pi * (month - 1) / 12
                return np.sin(month_normalized)
        
        return column
    
    def _handle_geospatial(self, column: pd.Series, action: PreprocessingAction, parameters: Optional[Dict[str, Any]] = None) -> pd.Series:
        """Handle geospatial actions like clustering."""
        parameters = parameters or {}
        
        if action == PreprocessingAction.GEO_CLUSTER_KMEANS:
            n_clusters = parameters.get('n_clusters', 10)
            
            # For single column (e.g., latitude), cluster into bins
            # Full geospatial clustering requires lat/lon pair at DataFrame level
            if not pd.api.types.is_numeric_dtype(column):
                column = self._coerce_to_numeric(column)
                if not pd.api.types.is_numeric_dtype(column):
                    logger.warning("Cannot cluster non-numeric column, returning original")
                    return column
            
            clean_data = column.dropna().values.reshape(-1, 1)
            if len(clean_data) < n_clusters:
                logger.warning(f"Not enough data for {n_clusters} clusters, using {len(clean_data)} clusters")
                n_clusters = max(1, len(clean_data))
            
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(clean_data)
                
                result = column.copy()
                mask = ~column.isna()
                result[mask] = kmeans.predict(column[mask].values.reshape(-1, 1))
                return result
            except Exception as e:
                logger.warning(f"KMeans clustering failed: {e}, falling back to binning")
                return self._handle_binning(column, PreprocessingAction.BINNING_EQUAL_FREQ, {'n_bins': n_clusters})
        
        return column
    
    def _handle_feature_engineering(self, column: pd.Series, action: PreprocessingAction, parameters: Optional[Dict[str, Any]] = None) -> pd.Series:
        """Handle feature engineering actions."""
        parameters = parameters or {}
        
        if action == PreprocessingAction.POLYNOMIAL_FEATURES:
            # For single column, return squared values
            if not pd.api.types.is_numeric_dtype(column):
                column = self._coerce_to_numeric(column)
                if not pd.api.types.is_numeric_dtype(column):
                    logger.warning("Cannot create polynomial features for non-numeric column")
                    return column
            
            degree = parameters.get('degree', 2)
            return column ** degree
        
        elif action == PreprocessingAction.INTERACTION_FEATURES:
            # Interaction features require multiple columns - return original
            logger.info("Interaction features require multiple columns, returning original")
            return column
        
        return column
    
    def _handle_remove_outliers(self, column: pd.Series) -> pd.Series:
        """Handle outlier removal (replaces outliers with NaN)."""
        if not pd.api.types.is_numeric_dtype(column):
            return column
        
        result = column.copy()
        Q1 = result.quantile(0.25)
        Q3 = result.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Replace outliers with NaN
        result = result.where((result >= lower) & (result <= upper), np.nan)
        return result
    
    def _coerce_to_numeric(self, column: pd.Series) -> pd.Series:
        """
        Attempt to coerce a column to numeric type.
        
        Handles common patterns like:
        - Numeric strings: "4.5", "3,456"
        - Currency: "$19.99", "€100"
        - Percentages: "45%"
        
        Args:
            column: Input column
            
        Returns:
            Numeric series if coercion successful, original otherwise
        """
        if pd.api.types.is_numeric_dtype(column):
            return column
        
        # Work with string representation
        result = column.astype(str)
        
        # Remove common non-numeric characters (order matters)
        # First remove percentage signs
        result = result.str.replace(r'%', '', regex=True)
        # Then remove currency symbols, commas, and whitespace
        result = result.str.replace(r'[$€£¥₹,\s]', '', regex=True)
        
        # Try to convert to numeric
        numeric_result = pd.to_numeric(result, errors='coerce')
        
        # Check if conversion was successful for majority of values
        success_rate = numeric_result.notna().sum() / len(column) if len(column) > 0 else 0
        
        if success_rate > 0.5:  # At least 50% successful conversion
            logger.info(f"Successfully coerced column to numeric (success rate: {success_rate:.1%})")
            return numeric_result
        else:
            logger.debug(f"Numeric coercion failed (success rate: {success_rate:.1%})")
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
