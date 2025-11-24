"""
Real-World Data Training for AURORA Neural Oracle
Downloads and trains on actual datasets from various domains.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neural.oracle import NeuralOracle
from src.features.minimal_extractor import MinimalFeatureExtractor
from src.core.actions import PreprocessingAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_real_world_datasets() -> List[Tuple[str, pd.DataFrame]]:
    """
    Download real-world datasets from various sources.
    Returns list of (dataset_name, dataframe) tuples.
    """
    datasets = []
    
    logger.info("Downloading real-world datasets...")
    
    # 1. Titanic Dataset (Classic ML dataset)
    try:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        datasets.append(("Titanic", df))
        logger.info(f"✓ Downloaded Titanic dataset: {df.shape}")
    except Exception as e:
        logger.warning(f"✗ Failed to download Titanic: {e}")
    
    # 2. Wine Quality Dataset
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=';')
        datasets.append(("Wine Quality", df))
        logger.info(f"✓ Downloaded Wine Quality dataset: {df.shape}")
    except Exception as e:
        logger.warning(f"✗ Failed to download Wine Quality: {e}")
    
    # 3. Iris Dataset (Simple but good for testing)
    try:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        df = pd.read_csv(url)
        datasets.append(("Iris", df))
        logger.info(f"✓ Downloaded Iris dataset: {df.shape}")
    except Exception as e:
        logger.warning(f"✗ Failed to download Iris: {e}")
    
    # 4. Tips Dataset (Mixed types)
    try:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
        df = pd.read_csv(url)
        datasets.append(("Tips", df))
        logger.info(f"✓ Downloaded Tips dataset: {df.shape}")
    except Exception as e:
        logger.warning(f"✗ Failed to download Tips: {e}")
    
    # 5. Diamonds Dataset (Large, mixed types)
    try:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
        df = pd.read_csv(url)
        # Sample to avoid memory issues
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)
        datasets.append(("Diamonds", df))
        logger.info(f"✓ Downloaded Diamonds dataset: {df.shape}")
    except Exception as e:
        logger.warning(f"✗ Failed to download Diamonds: {e}")
    
    # 6. Car Evaluation Dataset
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        df = pd.read_csv(url, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
        datasets.append(("Car Evaluation", df))
        logger.info(f"✓ Downloaded Car Evaluation dataset: {df.shape}")
    except Exception as e:
        logger.warning(f"✗ Failed to download Car Evaluation: {e}")
    
    logger.info(f"\nSuccessfully downloaded {len(datasets)} datasets")
    return datasets


def create_expert_labels(column: pd.Series, column_name: str) -> PreprocessingAction:
    """
    Create expert labels for real-world data based on domain knowledge.
    This simulates what a data scientist would recommend.
    """
    # Try numeric conversion
    try:
        numeric_col = pd.to_numeric(column, errors='coerce')
        numeric_ratio = numeric_col.notna().sum() / len(column)
    except:
        numeric_ratio = 0.0
    
    # Numeric columns
    if numeric_ratio > 0.8:
        # Check for skewness
        if numeric_ratio == 1.0:
            skew = numeric_col.skew()
            if abs(skew) > 2:
                return PreprocessingAction.LOG_TRANSFORM
            elif numeric_col.std() / numeric_col.mean() > 2:  # High CV
                return PreprocessingAction.ROBUST_SCALE
            else:
                return PreprocessingAction.STANDARD_SCALE
        else:
            # Mixed numeric/text - parse first
            return PreprocessingAction.PARSE_NUMERIC
    
    # Categorical columns
    unique_count = column.nunique()
    total_count = len(column)
    unique_ratio = unique_count / total_count if total_count > 0 else 0
    
    # ID columns (nearly all unique)
    if unique_ratio > 0.95 and total_count > 50:
        return PreprocessingAction.DROP_COLUMN
    
    # Constant columns
    if unique_count == 1:
        return PreprocessingAction.DROP_IF_CONSTANT
    
    # High cardinality categorical
    if unique_count > 50:
        return PreprocessingAction.HASH_ENCODE
    
    # Medium cardinality
    if unique_count > 10:
        return PreprocessingAction.TARGET_ENCODE
    
    # Low cardinality
    if unique_count <= 10:
        return PreprocessingAction.ONEHOT_ENCODE
    
    # Text columns
    if column.dtype == 'object':
        # Check if it looks like free text
        avg_length = column.astype(str).str.len().mean()
        if avg_length > 50:
            return PreprocessingAction.TEXT_CLEAN
        else:
            return PreprocessingAction.LABEL_ENCODE
    
    # Default
    return PreprocessingAction.KEEP_AS_IS


def extract_training_data_from_datasets(datasets: List[Tuple[str, pd.DataFrame]]):
    """
    Extract training data from real-world datasets.
    Returns (features, labels, extractor) tuple.
    """
    logger.info("\nExtracting training data from real-world datasets...")
    logger.info("=" * 70)
    
    extractor = MinimalFeatureExtractor()
    all_features = []
    all_labels = []
    
    for dataset_name, df in datasets:
        logger.info(f"\nProcessing {dataset_name}...")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
        
        for col_name in df.columns:
            column = df[col_name]
            
            # Extract features
            features = extractor.extract(column, col_name)
            
            # Create expert label
            label = create_expert_labels(column, col_name)
            
            all_features.append(features)
            all_labels.append(label)
            
            logger.info(f"    {col_name:20s} → {label.value:20s}")
    
    logger.info(f"\n✓ Extracted {len(all_features)} training examples from real-world data")
    return all_features, all_labels, extractor


def main():
    logger.info("=" * 70)
    logger.info("AURORA Neural Oracle - Real-World Data Training")
    logger.info("=" * 70)
    
    # Step 1: Download datasets
    datasets = download_real_world_datasets()
    
    if not datasets:
        logger.error("No datasets downloaded! Check your internet connection.")
        return 1
    
    # Step 2: Extract training data
    features, labels, extractor = extract_training_data_from_datasets(datasets)
    
    # Step 3: Combine with synthetic data for robustness
    logger.info("\n" + "=" * 70)
    logger.info("Combining with synthetic data for better coverage...")
    logger.info("=" * 70)
    
    from src.data.generator import SyntheticDataGenerator
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate synthetic data for edge cases
    synthetic_columns, synthetic_labels, _ = generator.generate_training_data(
        n_samples=2000,
        ambiguous_only=True
    )
    
    # Extract features from synthetic data (reuse extractor from earlier)
    for col, label in zip(synthetic_columns, synthetic_labels):
        feat = extractor.extract(col)
        features.append(feat)
        labels.append(label)
    
    logger.info(f"✓ Combined dataset: {len(features)} total examples")
    logger.info(f"  - Real-world: {len(datasets) * 10} examples (approx)")
    logger.info(f"  - Synthetic: 2000 examples")
    
    # Step 4: Train model
    logger.info("\n" + "=" * 70)
    logger.info("Training XGBoost model on combined dataset...")
    logger.info("=" * 70)
    
    oracle = NeuralOracle()
    
    try:
        metrics = oracle.train(
            features=features,
            labels=labels,
            validation_split=0.2
        )
        
        logger.info(f"\n✓ Training Complete!")
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"  Training Accuracy:   {metrics['train_accuracy']:6.2%}")
        logger.info(f"  Validation Accuracy: {metrics['val_accuracy']:6.2%}")
        logger.info(f"  Number of Trees:     {metrics['num_trees']:6d}")
        logger.info(f"  Number of Classes:   {metrics['num_classes']:6d}")
        
        # Save model
        model_path = Path(__file__).parent.parent / "models" / "neural_oracle_realworld.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        oracle.save(model_path)
        
        logger.info(f"\n✓ Model saved to: {model_path}")
        logger.info(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")
        
        # Final instructions
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 70)
        logger.info("\nNext Steps:")
        logger.info("  1. Copy the model to replace the default:")
        logger.info(f"     cp {model_path} models/neural_oracle_v1.pkl")
        logger.info("\n  2. Restart backend server:")
        logger.info("     uvicorn src.api.server:app --reload")
        logger.info("\n  3. Test with your own datasets!")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
