"""
Curriculum Meta-Learner for Neural Oracle Training
====================================================

This module implements a curriculum-based meta-learning approach for training
the neural oracle ensemble. The curriculum progresses through stages:

1. Deterministic Stage: Handle obvious cases (datetimes, nulls, constants)
2. Numeric Stage: Learn numeric preprocessing in isolation
3. Categorical Stage: Learn categorical preprocessing with numeric context
4. Text Stage: Learn text preprocessing with full context

Key Innovation:
- Uses actual ML model performance to generate ground truth labels
- Tries ALL preprocessing actions on each column
- Measures performance improvement to determine best action
- Generates high-quality training data with confidence scores

Usage:
    from scripts.curriculum_meta_learner import CurriculumMetaLearner
    
    learner = CurriculumMetaLearner(n_datasets=40, cv_folds=3)
    training_data = learner.run_curriculum()
    
    # Training data format:
    # {
    #     'features': np.ndarray (62 features),
    #     'label': str (action name),
    #     'confidence': float (0-1),
    #     'column_type': str ('numeric', 'categorical', 'text'),
    #     'performance_score': float
    # }
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress specific warnings for cleaner training output
# - ConvergenceWarning: LogisticRegression may not converge on small datasets (expected)
# - FutureWarning: Pandas/sklearn deprecation warnings (not actionable during training)
# - UserWarning: sklearn feature name warnings (expected with numeric arrays)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', message='.*X does not have valid feature names.*')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """A single training sample for the neural oracle."""
    features: np.ndarray
    label: str
    confidence: float
    column_type: str
    column_name: str
    dataset_name: str
    performance_score: float
    performance_baseline: float
    all_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    n_datasets: int = 40
    cv_folds: int = 3
    max_samples_per_dataset: int = 5000
    min_samples_for_cv: int = 50
    random_state: int = 42
    
    # Stage-specific settings
    numeric_actions: List[str] = field(default_factory=lambda: [
        'keep_as_is',
        'standard_scale',
        'minmax_scale',
        'robust_scale',
        'log_transform',
        'log1p_transform',
        'sqrt_transform',
        'clip_outliers',
    ])
    
    categorical_actions: List[str] = field(default_factory=lambda: [
        'keep_as_is',
        'onehot_encode',
        'label_encode',
        'ordinal_encode',
        'frequency_encode',
        'drop_column',
    ])
    
    text_actions: List[str] = field(default_factory=lambda: [
        'keep_as_is',
        'drop_column',
        'label_encode',
    ])
    
    deterministic_rules: Dict[str, str] = field(default_factory=lambda: {
        'all_null': 'drop_column',
        'constant': 'drop_column',
        'all_unique_id': 'drop_column',
        'datetime': 'parse_datetime',
        'boolean': 'parse_boolean',
    })


class PreprocessingExecutor:
    """Executes preprocessing actions on columns."""
    
    @staticmethod
    def apply_action(
        column: pd.Series,
        action: str,
        fit: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """
        Apply a preprocessing action to a column.
        
        Args:
            column: Input column
            action: Action name
            fit: Whether to fit transformers
        
        Returns:
            Tuple of (transformed_column, transformer_object)
        """
        non_null = column.dropna()
        if len(non_null) == 0:
            return None, None
        
        try:
            if action == 'keep_as_is':
                return column.values.reshape(-1, 1), None
            
            elif action == 'standard_scale':
                if not pd.api.types.is_numeric_dtype(column):
                    return None, None
                scaler = StandardScaler()
                values = column.fillna(column.mean()).values.reshape(-1, 1)
                return scaler.fit_transform(values), scaler
            
            elif action == 'minmax_scale':
                if not pd.api.types.is_numeric_dtype(column):
                    return None, None
                scaler = MinMaxScaler()
                values = column.fillna(column.mean()).values.reshape(-1, 1)
                return scaler.fit_transform(values), scaler
            
            elif action == 'robust_scale':
                if not pd.api.types.is_numeric_dtype(column):
                    return None, None
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                values = column.fillna(column.median()).values.reshape(-1, 1)
                return scaler.fit_transform(values), scaler
            
            elif action == 'log_transform':
                if not pd.api.types.is_numeric_dtype(column):
                    return None, None
                values = column.fillna(0)
                if (values <= 0).any():
                    return None, None
                return np.log(values).values.reshape(-1, 1), None
            
            elif action == 'log1p_transform':
                if not pd.api.types.is_numeric_dtype(column):
                    return None, None
                values = column.fillna(0)
                if (values < 0).any():
                    return None, None
                return np.log1p(values).values.reshape(-1, 1), None
            
            elif action == 'sqrt_transform':
                if not pd.api.types.is_numeric_dtype(column):
                    return None, None
                values = column.fillna(0)
                if (values < 0).any():
                    return None, None
                return np.sqrt(values).values.reshape(-1, 1), None
            
            elif action == 'clip_outliers':
                if not pd.api.types.is_numeric_dtype(column):
                    return None, None
                values = column.fillna(column.median())
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                clipped = values.clip(lower, upper)
                return clipped.values.reshape(-1, 1), None
            
            elif action == 'onehot_encode':
                values = column.fillna('__MISSING__').astype(str)
                dummies = pd.get_dummies(values, prefix='', prefix_sep='')
                return dummies.values, None
            
            elif action == 'label_encode':
                values = column.fillna('__MISSING__').astype(str)
                le = LabelEncoder()
                encoded = le.fit_transform(values)
                return encoded.reshape(-1, 1), le
            
            elif action == 'ordinal_encode':
                from sklearn.preprocessing import OrdinalEncoder
                values = column.fillna('__MISSING__').astype(str).values.reshape(-1, 1)
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                return enc.fit_transform(values), enc
            
            elif action == 'frequency_encode':
                values = column.fillna('__MISSING__').astype(str)
                freq = values.value_counts(normalize=True)
                encoded = values.map(freq)
                return encoded.values.reshape(-1, 1), freq
            
            elif action == 'drop_column':
                return None, 'DROP'
            
            else:
                return None, None
                
        except Exception as e:
            logger.debug(f"Action {action} failed: {e}")
            return None, None


class PerformanceMeasurer:
    """Measures ML model performance with different preprocessing."""
    
    def __init__(self, cv_folds: int = 3, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.executor = PreprocessingExecutor()
    
    def measure_action_performance(
        self,
        df: pd.DataFrame,
        column_name: str,
        target: pd.Series,
        action: str,
        other_columns: Optional[pd.DataFrame] = None
    ) -> Optional[float]:
        """
        Measure CV performance when applying an action to a column.
        
        Args:
            df: DataFrame containing the column
            column_name: Name of column to preprocess
            target: Target variable
            action: Action to apply
            other_columns: Pre-processed other columns to include
        
        Returns:
            Cross-validation accuracy score, or None if action fails
        """
        column = df[column_name]
        
        # Apply action
        transformed, _ = self.executor.apply_action(column, action)
        
        if action == 'drop_column' or transformed is None:
            # For drop, use only other columns
            if other_columns is not None and len(other_columns.columns) > 0:
                X = other_columns.values
            else:
                return 0.0  # Cannot evaluate with no features
        else:
            # Combine with other columns
            if other_columns is not None and len(other_columns.columns) > 0:
                X = np.hstack([other_columns.values, transformed])
            else:
                X = transformed
        
        # Handle missing values in target
        mask = ~target.isna()
        X = X[mask]
        y = target[mask].values
        
        if len(y) < self.cv_folds * 2:
            return None
        
        # Encode target if needed
        if not pd.api.types.is_numeric_dtype(target):
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Use stratified CV for classification
        try:
            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
            
            # Use a simple model for speed
            model = LogisticRegression(
                max_iter=500,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )
            
            return float(scores.mean())
            
        except Exception as e:
            logger.debug(f"CV failed for action {action}: {e}")
            return None
    
    def find_best_action(
        self,
        df: pd.DataFrame,
        column_name: str,
        target: pd.Series,
        candidate_actions: List[str],
        other_columns: Optional[pd.DataFrame] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Find the best action for a column by trying all candidates.
        
        Returns:
            Tuple of (best_action, best_score, all_scores)
        """
        scores = {}
        
        for action in candidate_actions:
            score = self.measure_action_performance(
                df, column_name, target, action, other_columns
            )
            if score is not None:
                scores[action] = score
        
        if not scores:
            return 'keep_as_is', 0.0, {}
        
        best_action = max(scores, key=scores.get)
        best_score = scores[best_action]
        
        return best_action, best_score, scores


class CurriculumMetaLearner:
    """
    Curriculum-based meta-learner for neural oracle training.
    
    Implements staged learning:
    1. Deterministic: Handle obvious cases
    2. Numeric: Learn numeric preprocessing
    3. Categorical: Learn categorical with numeric context
    4. Text: Learn text with full context
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()
        self.measurer = PerformanceMeasurer(
            cv_folds=self.config.cv_folds,
            random_state=self.config.random_state
        )
        self.training_samples: List[TrainingSample] = []
        
        # Import feature extractor
        try:
            from src.features.enhanced_extractor import get_meta_learning_extractor
            self.extractor = get_meta_learning_extractor()
        except ImportError:
            logger.warning("MetaLearningFeatureExtractor not available, using minimal")
            from src.features.minimal_extractor import get_feature_extractor
            self.extractor = get_feature_extractor()
    
    def classify_column(self, column: pd.Series, column_name: str) -> str:
        """Classify a column as numeric, categorical, text, or deterministic."""
        # Check for deterministic cases
        if column.isna().all():
            return 'deterministic:all_null'
        
        if column.nunique() <= 1:
            return 'deterministic:constant'
        
        if column.nunique() == len(column) and 'id' in column_name.lower():
            return 'deterministic:all_unique_id'
        
        if pd.api.types.is_datetime64_any_dtype(column):
            return 'deterministic:datetime'
        
        # Check for boolean
        if pd.api.types.is_bool_dtype(column):
            return 'deterministic:boolean'
        
        unique_vals = set(column.dropna().astype(str).str.lower().unique())
        if unique_vals.issubset({'true', 'false', 'yes', 'no', '0', '1', 't', 'f', 'y', 'n'}):
            if len(unique_vals) <= 3:
                return 'deterministic:boolean'
        
        # Numeric vs categorical vs text
        if pd.api.types.is_numeric_dtype(column):
            return 'numeric'
        
        # String/object columns
        avg_len = column.dropna().astype(str).str.len().mean()
        unique_ratio = column.nunique() / len(column.dropna()) if len(column.dropna()) > 0 else 0
        
        if unique_ratio > 0.5 and avg_len > 30:
            return 'text'
        
        return 'categorical'
    
    def process_deterministic(
        self,
        df: pd.DataFrame,
        column_name: str,
        rule_type: str
    ) -> Optional[TrainingSample]:
        """Process a deterministic column (no ML needed)."""
        column = df[column_name]
        
        # Get action from rules
        action = self.config.deterministic_rules.get(rule_type, 'keep_as_is')
        
        # Extract features
        if hasattr(self.extractor, 'extract_meta_features'):
            features = self.extractor.extract_meta_features(column, column_name)
        else:
            features = self.extractor.extract(column, column_name)
        
        return TrainingSample(
            features=features.to_array(),
            label=action,
            confidence=1.0,  # Deterministic = high confidence
            column_type='deterministic',
            column_name=column_name,
            dataset_name='',
            performance_score=1.0,
            performance_baseline=0.0,
        )
    
    def process_numeric_column(
        self,
        df: pd.DataFrame,
        column_name: str,
        target: pd.Series,
        other_columns: Optional[pd.DataFrame] = None,
        dataset_name: str = ""
    ) -> Optional[TrainingSample]:
        """Process a numeric column using performance-based learning."""
        column = df[column_name]
        
        # Find best action through actual performance measurement
        best_action, best_score, all_scores = self.measurer.find_best_action(
            df, column_name, target,
            self.config.numeric_actions,
            other_columns
        )
        
        if not all_scores:
            return None
        
        # Calculate confidence based on score gap
        scores_sorted = sorted(all_scores.values(), reverse=True)
        if len(scores_sorted) > 1:
            gap = scores_sorted[0] - scores_sorted[1]
            confidence = min(1.0, 0.5 + gap * 5)  # Scale gap to confidence
        else:
            confidence = 0.7
        
        # Baseline is keep_as_is score
        baseline = all_scores.get('keep_as_is', 0.0)
        
        # Extract features
        if hasattr(self.extractor, 'extract_meta_features'):
            features = self.extractor.extract_meta_features(column, column_name)
        else:
            features = self.extractor.extract(column, column_name)
        
        return TrainingSample(
            features=features.to_array(),
            label=best_action,
            confidence=confidence,
            column_type='numeric',
            column_name=column_name,
            dataset_name=dataset_name,
            performance_score=best_score,
            performance_baseline=baseline,
            all_scores=all_scores,
        )
    
    def process_categorical_column(
        self,
        df: pd.DataFrame,
        column_name: str,
        target: pd.Series,
        other_columns: Optional[pd.DataFrame] = None,
        dataset_name: str = ""
    ) -> Optional[TrainingSample]:
        """Process a categorical column using performance-based learning."""
        column = df[column_name]
        
        # Find best action
        best_action, best_score, all_scores = self.measurer.find_best_action(
            df, column_name, target,
            self.config.categorical_actions,
            other_columns
        )
        
        if not all_scores:
            return None
        
        # Calculate confidence
        scores_sorted = sorted(all_scores.values(), reverse=True)
        if len(scores_sorted) > 1:
            gap = scores_sorted[0] - scores_sorted[1]
            confidence = min(1.0, 0.5 + gap * 5)
        else:
            confidence = 0.7
        
        baseline = all_scores.get('keep_as_is', 0.0)
        
        # Extract features
        if hasattr(self.extractor, 'extract_meta_features'):
            features = self.extractor.extract_meta_features(column, column_name)
        else:
            features = self.extractor.extract(column, column_name)
        
        return TrainingSample(
            features=features.to_array(),
            label=best_action,
            confidence=confidence,
            column_type='categorical',
            column_name=column_name,
            dataset_name=dataset_name,
            performance_score=best_score,
            performance_baseline=baseline,
            all_scores=all_scores,
        )
    
    def process_text_column(
        self,
        df: pd.DataFrame,
        column_name: str,
        target: pd.Series,
        other_columns: Optional[pd.DataFrame] = None,
        dataset_name: str = ""
    ) -> Optional[TrainingSample]:
        """Process a text column."""
        column = df[column_name]
        
        # Find best action (limited options for text)
        best_action, best_score, all_scores = self.measurer.find_best_action(
            df, column_name, target,
            self.config.text_actions,
            other_columns
        )
        
        if not all_scores:
            # Default to drop for high-cardinality text
            best_action = 'drop_column'
            best_score = 0.0
            all_scores = {'drop_column': 0.0}
        
        # Calculate confidence
        scores_sorted = sorted(all_scores.values(), reverse=True)
        if len(scores_sorted) > 1:
            gap = scores_sorted[0] - scores_sorted[1]
            confidence = min(1.0, 0.5 + gap * 5)
        else:
            confidence = 0.6
        
        baseline = all_scores.get('keep_as_is', 0.0)
        
        # Extract features
        if hasattr(self.extractor, 'extract_meta_features'):
            features = self.extractor.extract_meta_features(column, column_name)
        else:
            features = self.extractor.extract(column, column_name)
        
        return TrainingSample(
            features=features.to_array(),
            label=best_action,
            confidence=confidence,
            column_type='text',
            column_name=column_name,
            dataset_name=dataset_name,
            performance_score=best_score,
            performance_baseline=baseline,
            all_scores=all_scores,
        )
    
    def process_dataset(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_name: str = ""
    ) -> List[TrainingSample]:
        """
        Process a dataset through the curriculum stages.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            dataset_name: Name for logging
        
        Returns:
            List of training samples
        """
        samples = []
        
        if target_column not in df.columns:
            logger.warning(f"Target column '{target_column}' not found in {dataset_name}")
            return samples
        
        target = df[target_column]
        feature_columns = [c for c in df.columns if c != target_column]
        
        # Classify all columns first
        column_types = {}
        for col in feature_columns:
            column_types[col] = self.classify_column(df[col], col)
        
        logger.info(f"Processing {dataset_name}: {len(feature_columns)} columns")
        
        # Stage 1: Deterministic
        for col, col_type in column_types.items():
            if col_type.startswith('deterministic:'):
                rule_type = col_type.split(':')[1]
                sample = self.process_deterministic(df, col, rule_type)
                if sample:
                    sample.dataset_name = dataset_name
                    samples.append(sample)
        
        # Prepare numeric columns first (for context in later stages)
        numeric_cols = [c for c, t in column_types.items() if t == 'numeric']
        processed_numeric = pd.DataFrame()
        
        # Stage 2: Numeric
        for col in numeric_cols:
            # Use already processed numeric columns as context
            sample = self.process_numeric_column(
                df, col, target,
                other_columns=processed_numeric if len(processed_numeric.columns) > 0 else None,
                dataset_name=dataset_name
            )
            if sample:
                samples.append(sample)
                # Add to processed (using standard scaling)
                scaled, _ = PreprocessingExecutor.apply_action(df[col], 'standard_scale')
                if scaled is not None:
                    processed_numeric[col] = scaled.flatten()
        
        # Stage 3: Categorical
        categorical_cols = [c for c, t in column_types.items() if t == 'categorical']
        for col in categorical_cols:
            sample = self.process_categorical_column(
                df, col, target,
                other_columns=processed_numeric if len(processed_numeric.columns) > 0 else None,
                dataset_name=dataset_name
            )
            if sample:
                samples.append(sample)
        
        # Stage 4: Text
        text_cols = [c for c, t in column_types.items() if t == 'text']
        for col in text_cols:
            sample = self.process_text_column(
                df, col, target,
                other_columns=processed_numeric if len(processed_numeric.columns) > 0 else None,
                dataset_name=dataset_name
            )
            if sample:
                samples.append(sample)
        
        logger.info(f"  Generated {len(samples)} training samples")
        return samples
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get training data in array format.
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        if not self.training_samples:
            raise ValueError("No training samples. Run process_dataset first.")
        
        X = np.vstack([s.features for s in self.training_samples])
        y = np.array([s.label for s in self.training_samples])
        
        # Get feature names
        try:
            from src.features.enhanced_extractor import MetaLearningFeatures
            feature_names = MetaLearningFeatures.get_feature_names()
        except ImportError:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        return X, y, feature_names
    
    def get_sample_weights(self) -> np.ndarray:
        """Get sample weights based on confidence scores."""
        return np.array([s.confidence for s in self.training_samples])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected training data."""
        if not self.training_samples:
            return {'total_samples': 0}
        
        labels = [s.label for s in self.training_samples]
        types = [s.column_type for s in self.training_samples]
        confidences = [s.confidence for s in self.training_samples]
        
        from collections import Counter
        
        return {
            'total_samples': len(self.training_samples),
            'label_distribution': dict(Counter(labels)),
            'type_distribution': dict(Counter(types)),
            'avg_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'unique_labels': len(set(labels)),
            'unique_datasets': len(set(s.dataset_name for s in self.training_samples)),
        }
    
    def save_training_data(self, path: Path):
        """Save training data to file."""
        data = {
            'samples': [
                {
                    'features': s.features.tolist(),
                    'label': s.label,
                    'confidence': s.confidence,
                    'column_type': s.column_type,
                    'column_name': s.column_name,
                    'dataset_name': s.dataset_name,
                    'performance_score': s.performance_score,
                    'performance_baseline': s.performance_baseline,
                }
                for s in self.training_samples
            ],
            'statistics': self.get_statistics(),
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved training data to {path}")


def demo():
    """Demo of curriculum meta-learner with sample data."""
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'id': range(n),
        'age': np.random.randint(18, 80, n),
        'income': np.random.lognormal(10, 1, n),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'rating': np.random.uniform(0, 5, n),
        'description': ['Sample text ' * np.random.randint(1, 10) for _ in range(n)],
        'target': np.random.choice([0, 1], n),
    })
    
    # Initialize learner
    config = CurriculumConfig(cv_folds=3)
    learner = CurriculumMetaLearner(config)
    
    # Process dataset
    samples = learner.process_dataset(df, 'target', 'demo_dataset')
    learner.training_samples.extend(samples)
    
    # Print statistics
    stats = learner.get_statistics()
    print("\n=== Curriculum Meta-Learner Demo ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Label distribution: {stats['label_distribution']}")
    print(f"Type distribution: {stats['type_distribution']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")


if __name__ == '__main__':
    demo()
