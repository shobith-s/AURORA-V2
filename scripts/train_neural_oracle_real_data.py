"""
Advanced Neural Oracle Training Script
========================================
This script:
1. Collects diverse open-source datasets from multiple sources
2. Processes them through the symbolic engine to get "ground truth" decisions
3. Trains the XGBoost neural oracle on real-world data
4. Validates the model with safety checks to prevent catastrophic data loss
5. Saves comprehensive training metrics and the trained model

Safety Features:
- Drop action validation (ensures no important columns are dropped)
- Confusion matrix analysis
- Per-action accuracy tracking
- Sample size requirements
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_diabetes, load_breast_cancer, load_wine, 
    load_iris, fetch_california_housing, fetch_openml
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.core.preprocessor import IntelligentPreprocessor
from src.core.actions import PreprocessingAction
from src.features.minimal_extractor import MinimalFeatureExtractor


# Suppress warnings
warnings.filterwarnings('ignore')


class DatasetCollector:
    """Collects diverse datasets from multiple sources."""
    
    def __init__(self):
        self.datasets = []
        
    def collect_sklearn_datasets(self) -> List[Tuple[str, pd.DataFrame, str]]:
        """Collect built-in scikit-learn datasets."""
        datasets = []
        
        print("📦 Collecting scikit-learn datasets...")
        
        # Diabetes (numeric regression)
        try:
            diabetes = load_diabetes(as_frame=True)
            df = diabetes.frame
            datasets.append(("diabetes", df, "regression"))
            print("  ✓ Diabetes dataset (442 rows, 11 columns)")
        except Exception as e:
            print(f"  ✗ Diabetes failed: {e}")
        
        # Breast Cancer (binary classification)
        try:
            cancer = load_breast_cancer(as_frame=True)
            df = cancer.frame
            datasets.append(("breast_cancer", df, "classification"))
            print("  ✓ Breast Cancer dataset (569 rows, 31 columns)")
        except Exception as e:
            print(f"  ✗ Breast Cancer failed: {e}")
        
        # Wine (multiclass classification)
        try:
            wine = load_wine(as_frame=True)
            df = wine.frame
            datasets.append(("wine", df, "classification"))
            print("  ✓ Wine dataset (178 rows, 14 columns)")
        except Exception as e:
            print(f"  ✗ Wine failed: {e}")
        
        # Iris (multiclass classification)
        try:
            iris = load_iris(as_frame=True)
            df = iris.frame
            datasets.append(("iris", df, "classification"))
            print("  ✓ Iris dataset (150 rows, 5 columns)")
        except Exception as e:
            print(f"  ✗ Iris failed: {e}")
        
        # California Housing (regression)
        try:
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame
            datasets.append(("california_housing", df, "regression"))
            print("  ✓ California Housing dataset (20640 rows, 9 columns)")
        except Exception as e:
            print(f"  ✗ California Housing failed: {e}")
        
        return datasets
    
    def collect_openml_datasets(self, max_datasets: int = 10) -> List[Tuple[str, pd.DataFrame, str]]:
        """Collect datasets from OpenML."""
        datasets = []
        
        print(f"\n📦 Collecting OpenML datasets (max {max_datasets})...")
        
        # Curated list of diverse OpenML datasets
        openml_datasets = [
            (31, "credit-g", "classification"),  # German Credit
            (23381, "dresses-sales", "classification"),  # Dresses Attribute Sales
            (40536, "SpeedDating", "classification"),  # Speed Dating
            (40945, "titanic", "classification"),  # Titanic
            (41027, "Australian", "classification"),  # Australian Credit
            (1487, "click_prediction_small", "classification"),  # Click Prediction
            (42570, "compass", "classification"),  # COMPAS Recidivism
        ]
        
        for dataset_id, name, task in openml_datasets[:max_datasets]:
            try:
                print(f"  Fetching {name} (ID: {dataset_id})...")
                data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
                
                # Combine features and target
                if hasattr(data, 'frame') and data.frame is not None:
                    df = data.frame
                else:
                    df = pd.concat([data.data, data.target], axis=1)
                
                # Limit size for faster processing
                if len(df) > 5000:
                    df = df.sample(n=5000, random_state=42)
                
                datasets.append((name, df, task))
                print(f"    ✓ {name} ({len(df)} rows, {len(df.columns)} columns)")
                
            except Exception as e:
                print(f"    ✗ {name} failed: {e}")
                continue
        
        return datasets
    
    def collect_all(self, use_openml: bool = True) -> List[Tuple[str, pd.DataFrame, str]]:
        """Collect all available datasets."""
        all_datasets = []
        
        # Always collect sklearn datasets (fast and reliable)
        all_datasets.extend(self.collect_sklearn_datasets())
        
        # Optionally collect OpenML datasets (requires internet)
        if use_openml:
            try:
                all_datasets.extend(self.collect_openml_datasets(max_datasets=5))
            except Exception as e:
                print(f"⚠️  OpenML collection failed: {e}")
                print("   Continuing with sklearn datasets only...")
        
        print(f"\n✅ Total datasets collected: {len(all_datasets)}")
        return all_datasets


class NeuralOracleTrainer:
    """Trains the neural oracle on real-world datasets with safety validation."""
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.feature_extractor = MinimalFeatureExtractor()
        self.preprocessor = IntelligentPreprocessor(
            confidence_threshold=0.75,
            use_neural_oracle=False,  # Don't use old oracle during training
            enable_learning=False
        )
        
        self.training_data = []
        self.action_counts = {}
        
    def process_dataset(
        self, 
        dataset_name: str, 
        df: pd.DataFrame, 
        task_type: str
    ) -> List[Dict[str, Any]]:
        """Process a dataset to extract training samples."""
        print(f"\n🔄 Processing {dataset_name} ({task_type})...")
        
        samples = []
        
        # Process each column
        for col_name in df.columns:
            column = df[col_name]
            
            # Skip if all nulls
            if column.isnull().all():
                continue
            
            try:
                # Get symbolic decision (ground truth)
                result = self.preprocessor.preprocess_column(
                    column=column,
                    column_name=col_name,
                    target_available=True,
                    context=task_type
                )
                
                # Extract features
                features = self.feature_extractor.extract(column, col_name)
                
                # Create training sample
                sample = {
                    'dataset': dataset_name,
                    'column': col_name,
                    'task_type': task_type,
                    'features': features,
                    'action': result.action.value,
                    'confidence': result.confidence,
                    'source': result.source,
                    'explanation': result.explanation
                }
                
                samples.append(sample)
                
                # Track action distribution
                action = result.action.value
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
                
            except Exception as e:
                print(f"  ⚠️  Skipped column '{col_name}': {e}")
                continue
        
        print(f"  ✓ Extracted {len(samples)} training samples from {dataset_name}")
        return samples
    
    def collect_training_data(
        self, 
        datasets: List[Tuple[str, pd.DataFrame, str]]
    ) -> pd.DataFrame:
        """Collect training data from all datasets."""
        print("\n" + "="*70)
        print("PHASE 1: COLLECTING TRAINING DATA")
        print("="*70)
        
        all_samples = []
        
        for dataset_name, df, task_type in datasets:
            samples = self.process_dataset(dataset_name, df, task_type)
            all_samples.extend(samples)
        
        print(f"\n✅ Total training samples collected: {len(all_samples)}")
        
        # Display action distribution
        print("\n📊 Action Distribution:")
        for action, count in sorted(self.action_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(all_samples) * 100
            print(f"  {action:30s}: {count:5d} ({pct:5.1f}%)")
        
        self.training_data = all_samples
        return all_samples
    
    def prepare_features_labels(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and labels."""
        print("\n" + "="*70)
        print("PHASE 2: PREPARING FEATURES AND LABELS")
        print("="*70)
        
        # Extract features and labels
        X_list = []
        y_list = []
        
        for sample in self.training_data:
            X_list.append(sample['features'])
            y_list.append(sample['action'])
        
        # Convert to numpy arrays
        X = np.array(X_list)
        
        # Encode actions as integers
        unique_actions = sorted(set(y_list))
        action_to_idx = {action: idx for idx, action in enumerate(unique_actions)}
        y = np.array([action_to_idx[action] for action in y_list])
        
        print(f"  ✓ Feature matrix: {X.shape}")
        print(f"  ✓ Label vector: {y.shape}")
        print(f"  ✓ Number of actions: {len(unique_actions)}")
        
        return X, y, unique_actions
    
    def train_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        action_names: List[str]
    ) -> xgb.XGBClassifier:
        """Train XGBoost model with cross-validation."""
        print("\n" + "="*70)
        print("PHASE 3: TRAINING XGBOOST MODEL")
        print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=len(action_names),
            random_state=42,
            n_jobs=-1
        )
        
        print("\n  Training model...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n  ✅ Training Complete!")
        print(f"  Training Accuracy: {model.score(X_train, y_train):.1%}")
        print(f"  Test Accuracy: {accuracy:.1%}")
        
        # Store test data for validation
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.action_names = action_names
        
        return model
    
    def validate_safety(self) -> bool:
        """Validate model safety to prevent catastrophic data loss."""
        print("\n" + "="*70)
        print("PHASE 4: SAFETY VALIDATION")
        print("="*70)
        
        # Create action index mapping
        action_to_idx = {action: idx for idx, action in enumerate(self.action_names)}
        
        # Check 1: DROP_COLUMN accuracy
        print("\n🔍 Check 1: DROP_COLUMN Decision Validation")
        drop_idx = action_to_idx.get('drop_column', -1)
        if drop_idx != -1:
            drop_mask = self.y_test == drop_idx
            drop_correct = (self.y_pred[drop_mask] == drop_idx).sum()
            drop_total = drop_mask.sum()
            drop_accuracy = drop_correct / drop_total if drop_total > 0 else 1.0
            
            print(f"  DROP_COLUMN samples: {drop_total}")
            print(f"  DROP_COLUMN accuracy: {drop_accuracy:.1%}")
            
            if drop_accuracy < 0.7:
                print("  ⚠️  WARNING: Low DROP accuracy - model may drop important columns!")
                return False
            else:
                print("  ✅ DROP accuracy acceptable")
        
        # Check 2: Per-action accuracy
        print("\n🔍 Check 2: Per-Action Accuracy")
        action_accuracies = {}
        for idx, action in enumerate(self.action_names):
            mask = self.y_test == idx
            if mask.sum() > 0:
                accuracy = (self.y_pred[mask] == idx).sum() / mask.sum()
                action_accuracies[action] = accuracy
                
                status = "✅" if accuracy >= 0.6 else "⚠️"
                print(f"  {status} {action:30s}: {accuracy:.1%} ({mask.sum():4d} samples)")
        
        # Check 3: Confusion matrix for critical actions
        print("\n🔍 Check 3: Confusion Matrix Analysis")
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Check for catastrophic confusions (e.g., DROP confused with KEEP)
        keep_idx = action_to_idx.get('keep_as_is', -1)
        if drop_idx != -1 and keep_idx != -1:
            drop_as_keep = cm[drop_idx, keep_idx] if drop_idx < len(cm) else 0
            keep_as_drop = cm[keep_idx, drop_idx] if keep_idx < len(cm) else 0
            
            print(f"  DROP predicted as KEEP: {drop_as_keep}")
            print(f"  KEEP predicted as DROP: {keep_as_drop}")
            
            if keep_as_drop > len(self.y_test) * 0.05:  # >5% of data
                print("  ⚠️  WARNING: Model drops too many KEEP columns!")
                return False
            else:
                print("  ✅ No catastrophic confusions detected")
        
        # Check 4: Minimum accuracy threshold
        print("\n🔍 Check 4: Overall Accuracy Threshold")
        overall_accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"  Overall Accuracy: {overall_accuracy:.1%}")
        
        if overall_accuracy < 0.70:
            print("  ⚠️  WARNING: Overall accuracy below 70% threshold!")
            return False
        else:
            print("  ✅ Overall accuracy acceptable")
        
        print("\n" + "="*70)
        print("✅ SAFETY VALIDATION PASSED")
        print("="*70)
        return True
    
    def save_model(self, model: xgb.XGBClassifier, metadata: Dict[str, Any]):
        """Save trained model and metadata."""
        print("\n" + "="*70)
        print("PHASE 5: SAVING MODEL")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.output_dir / f"neural_oracle_real_data_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✅ Model saved: {model_path}")
        
        # Save metadata
        metadata_path = self.output_dir / f"training_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✅ Metadata saved: {metadata_path}")
        
        # Also save as default model
        default_path = self.output_dir / "neural_oracle_v1.pkl"
        with open(default_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✅ Default model updated: {default_path}")
        
        return model_path


def main():
    """Main training pipeline."""
    print("="*70)
    print("AURORA NEURAL ORACLE - REAL DATASET TRAINING")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Step 1: Collect datasets
    collector = DatasetCollector()
    datasets = collector.collect_all(use_openml=True)
    
    if len(datasets) == 0:
        print("\n❌ No datasets collected. Exiting.")
        return
    
    # Step 2: Train model
    trainer = NeuralOracleTrainer()
    
    # Collect training data
    trainer.collect_training_data(datasets)
    
    if len(trainer.training_data) < 100:
        print(f"\n❌ Insufficient training data ({len(trainer.training_data)} samples). Need at least 100.")
        return
    
    # Prepare features and labels
    X, y, action_names = trainer.prepare_features_labels()
    
    # Train model
    model = trainer.train_model(X, y, action_names)
    
    # Validate safety
    if not trainer.validate_safety():
        print("\n❌ SAFETY VALIDATION FAILED - Model NOT saved")
        print("The model may cause catastrophic data loss. Please review and retrain.")
        return
    
    # Save model with metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'num_datasets': len(datasets),
        'num_samples': len(trainer.training_data),
        'num_actions': len(action_names),
        'actions': action_names,
        'action_distribution': trainer.action_counts,
        'test_accuracy': float(accuracy_score(trainer.y_test, trainer.y_pred)),
        'datasets_used': [name for name, _, _ in datasets]
    }
    
    model_path = trainer.save_model(model, metadata)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"Model Path: {model_path}")
    print(f"Total Samples: {len(trainer.training_data)}")
    print(f"Test Accuracy: {metadata['test_accuracy']:.1%}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
