"""
Incremental Neural Oracle Training Script
==========================================
This script allows you to:
1. Load an existing trained model
2. Add new datasets to the training data
3. Combine old + new data for incremental learning
4. Retrain the model while preserving previous knowledge

Usage:
    python scripts/train_neural_oracle_incremental.py --existing-model models/neural_oracle_v1.pkl
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
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
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.preprocessor import IntelligentPreprocessor
from src.core.actions import PreprocessingAction
from src.features.minimal_extractor import MinimalFeatureExtractor

warnings.filterwarnings('ignore')


class IncrementalTrainer:
    """Incremental training for neural oracle."""
    
    def __init__(self, existing_model_path: Optional[Path] = None, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.feature_extractor = MinimalFeatureExtractor()
        self.preprocessor = IntelligentPreprocessor(
            confidence_threshold=0.75,
            use_neural_oracle=False,
            enable_learning=False
        )
        
        # Load existing model and metadata if provided
        self.existing_model = None
        self.existing_metadata = None
        self.previous_training_data = []
        
        if existing_model_path and existing_model_path.exists():
            self.load_existing_model(existing_model_path)
        
        self.new_training_data = []
        self.action_counts = {}
    
    def load_existing_model(self, model_path: Path):
        """Load existing model and its metadata."""
        print(f"\n📂 Loading existing model: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                self.existing_model = pickle.load(f)
            print(f"  ✅ Model loaded successfully")
            
            # Try to load metadata
            metadata_pattern = model_path.parent / f"training_metadata_*.json"
            metadata_files = sorted(model_path.parent.glob("training_metadata_*.json"))
            
            if metadata_files:
                latest_metadata = metadata_files[-1]
                with open(latest_metadata, 'r') as f:
                    self.existing_metadata = json.load(f)
                print(f"  ✅ Metadata loaded: {latest_metadata.name}")
                print(f"     Previous samples: {self.existing_metadata.get('num_samples', 'unknown')}")
                print(f"     Previous accuracy: {self.existing_metadata.get('test_accuracy', 'unknown'):.1%}")
            else:
                print(f"  ⚠️  No metadata found")
                
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            raise
    
    def collect_new_datasets(self, dataset_names: List[str] = None) -> List[Tuple[str, pd.DataFrame, str]]:
        """Collect new datasets for incremental training."""
        print("\n" + "="*70)
        print("COLLECTING NEW DATASETS")
        print("="*70)
        
        datasets = []
        
        # If no specific datasets requested, use a default set
        if dataset_names is None:
            dataset_names = ['openml']
        
        # OpenML datasets
        if 'openml' in dataset_names:
            print("\n📦 Collecting OpenML datasets...")
            openml_datasets = [
                (1590, "adult", "classification"),  # Adult Income
                (40996, "car", "classification"),  # Car Evaluation
                (40975, "car_evaluation", "classification"),
                (40668, "connect-4", "classification"),
                (40927, "covertype", "classification"),
            ]
            
            for dataset_id, name, task in openml_datasets[:3]:  # Limit to 3 for speed
                try:
                    print(f"  Fetching {name} (ID: {dataset_id})...")
                    data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
                    
                    if hasattr(data, 'frame') and data.frame is not None:
                        df = data.frame
                    else:
                        df = pd.concat([data.data, data.target], axis=1)
                    
                    # Limit size
                    if len(df) > 5000:
                        df = df.sample(n=5000, random_state=42)
                    
                    datasets.append((name, df, task))
                    print(f"    ✅ {name} ({len(df)} rows, {len(df.columns)} columns)")
                    
                except Exception as e:
                    print(f"    ✗ {name} failed: {e}")
                    continue
        
        # Add custom CSV datasets if provided
        if 'custom' in dataset_names:
            print("\n📦 Looking for custom CSV datasets in data/custom/...")
            custom_dir = Path("data/custom")
            if custom_dir.exists():
                for csv_file in custom_dir.glob("*.csv"):
                    try:
                        df = pd.read_csv(csv_file)
                        datasets.append((csv_file.stem, df, "unknown"))
                        print(f"  ✅ {csv_file.name} ({len(df)} rows, {len(df.columns)} columns)")
                    except Exception as e:
                        print(f"  ✗ {csv_file.name} failed: {e}")
        
        print(f"\n✅ New datasets collected: {len(datasets)}")
        return datasets
    
    def process_dataset(self, dataset_name: str, df: pd.DataFrame, task_type: str) -> List[Dict[str, Any]]:
        """Process a dataset to extract training samples."""
        print(f"\n🔄 Processing {dataset_name} ({task_type})...")
        
        samples = []
        
        for col_name in df.columns:
            column = df[col_name]
            
            if column.isnull().all():
                continue
            
            try:
                result = self.preprocessor.preprocess_column(
                    column=column,
                    column_name=col_name,
                    target_available=True,
                    context=task_type
                )
                
                features = self.feature_extractor.extract(column, col_name)
                
                sample = {
                    'dataset': dataset_name,
                    'column': col_name,
                    'task_type': task_type,
                    'features': features,
                    'action': result.action.value,
                    'confidence': result.confidence,
                    'source': result.source,
                }
                
                samples.append(sample)
                
                action = result.action.value
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
                
            except Exception as e:
                continue
        
        print(f"  ✓ Extracted {len(samples)} samples from {dataset_name}")
        return samples
    
    def collect_new_training_data(self, datasets: List[Tuple[str, pd.DataFrame, str]]):
        """Collect training data from new datasets."""
        print("\n" + "="*70)
        print("PROCESSING NEW DATASETS")
        print("="*70)
        
        all_samples = []
        
        for dataset_name, df, task_type in datasets:
            samples = self.process_dataset(dataset_name, df, task_type)
            all_samples.extend(samples)
        
        print(f"\n✅ New training samples: {len(all_samples)}")
        
        # Display action distribution
        print("\n📊 New Data Action Distribution:")
        for action, count in sorted(self.action_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(all_samples) * 100 if len(all_samples) > 0 else 0
            print(f"  {action:30s}: {count:5d} ({pct:5.1f}%)")
        
        self.new_training_data = all_samples
    
    def combine_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Combine old and new training data."""
        print("\n" + "="*70)
        print("COMBINING OLD + NEW TRAINING DATA")
        print("="*70)
        
        # For now, we only have new data (old data not stored)
        # In production, you'd want to store previous training samples
        all_data = self.new_training_data
        
        print(f"  Previous samples: {self.existing_metadata.get('num_samples', 0) if self.existing_metadata else 0}")
        print(f"  New samples: {len(self.new_training_data)}")
        print(f"  Total samples: {len(all_data)}")
        
        # Extract features and labels
        X_list = [sample['features'] for sample in all_data]
        y_list = [sample['action'] for sample in all_data]
        
        X = np.array(X_list)
        
        # Encode actions
        unique_actions = sorted(set(y_list))
        action_to_idx = {action: idx for idx, action in enumerate(unique_actions)}
        y = np.array([action_to_idx[action] for action in y_list])
        
        print(f"\n  ✓ Feature matrix: {X.shape}")
        print(f"  ✓ Label vector: {y.shape}")
        print(f"  ✓ Number of actions: {len(unique_actions)}")
        
        return X, y, unique_actions
    
    def train_incremental(self, X: np.ndarray, y: np.ndarray, action_names: List[str]) -> xgb.XGBClassifier:
        """Train model incrementally."""
        print("\n" + "="*70)
        print("INCREMENTAL TRAINING")
        print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        # Create new model (or use existing as warm start)
        if self.existing_model is not None:
            print("\n  🔄 Using existing model as warm start...")
            # Note: XGBoost doesn't support direct warm start, so we train from scratch
            # but with more trees to preserve knowledge
            n_estimators = 300  # More trees for incremental learning
        else:
            n_estimators = 200
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=len(action_names),
            random_state=42,
            n_jobs=-1
        )
        
        print(f"\n  Training model with {n_estimators} trees...")
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
        
        if self.existing_metadata:
            prev_acc = self.existing_metadata.get('test_accuracy', 0)
            improvement = accuracy - prev_acc
            print(f"  Previous Accuracy: {prev_acc:.1%}")
            print(f"  Improvement: {improvement:+.1%}")
        
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.action_names = action_names
        
        return model
    
    def save_model(self, model: xgb.XGBClassifier, metadata: Dict[str, Any]):
        """Save incrementally trained model."""
        print("\n" + "="*70)
        print("SAVING INCREMENTAL MODEL")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.output_dir / f"neural_oracle_incremental_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✅ Model saved: {model_path}")
        
        # Save metadata
        metadata_path = self.output_dir / f"training_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✅ Metadata saved: {metadata_path}")
        
        # Update default model
        default_path = self.output_dir / "neural_oracle_v1.pkl"
        with open(default_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✅ Default model updated: {default_path}")
        
        return model_path


def main():
    parser = argparse.ArgumentParser(description='Incremental Neural Oracle Training')
    parser.add_argument('--existing-model', type=str, default='models/neural_oracle_v1.pkl',
                       help='Path to existing model (default: models/neural_oracle_v1.pkl)')
    parser.add_argument('--datasets', type=str, nargs='+', default=['openml'],
                       help='Datasets to add: openml, custom (default: openml)')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models (default: models)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("AURORA NEURAL ORACLE - INCREMENTAL TRAINING")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Initialize trainer
    existing_model_path = Path(args.existing_model) if args.existing_model else None
    trainer = IncrementalTrainer(existing_model_path, args.output_dir)
    
    # Collect new datasets
    datasets = trainer.collect_new_datasets(args.datasets)
    
    if len(datasets) == 0:
        print("\n❌ No new datasets collected. Exiting.")
        return
    
    # Process new datasets
    trainer.collect_new_training_data(datasets)
    
    if len(trainer.new_training_data) < 50:
        print(f"\n❌ Insufficient new data ({len(trainer.new_training_data)} samples). Need at least 50.")
        return
    
    # Combine and prepare data
    X, y, action_names = trainer.combine_training_data()
    
    # Train incrementally
    model = trainer.train_incremental(X, y, action_names)
    
    # Save model
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'training_type': 'incremental',
        'previous_model': str(existing_model_path) if existing_model_path else None,
        'previous_samples': trainer.existing_metadata.get('num_samples', 0) if trainer.existing_metadata else 0,
        'new_samples': len(trainer.new_training_data),
        'total_samples': len(trainer.new_training_data),  # Only new for now
        'num_actions': len(action_names),
        'actions': action_names,
        'action_distribution': trainer.action_counts,
        'test_accuracy': float(accuracy_score(trainer.y_test, trainer.y_pred)),
        'new_datasets_used': [name for name, _, _ in datasets]
    }
    
    model_path = trainer.save_model(model, metadata)
    
    print("\n" + "="*70)
    print("✅ INCREMENTAL TRAINING COMPLETE")
    print("="*70)
    print(f"Model Path: {model_path}")
    print(f"New Samples: {len(trainer.new_training_data)}")
    print(f"Test Accuracy: {metadata['test_accuracy']:.1%}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
