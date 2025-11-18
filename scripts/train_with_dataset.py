#!/usr/bin/env python3
"""
Train Neural Oracle with custom datasets.

This script allows you to train the AURORA Neural Oracle with your own datasets
or open-source datasets from UCI, Kaggle, scikit-learn, etc.

Usage:
    python scripts/train_with_dataset.py --file data/titanic.csv --type csv
    python scripts/train_with_dataset.py --file data/housing.pkl --output models/housing_oracle.pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from typing import List
from src.neural.oracle import NeuralOracle
from src.features.minimal_extractor import MinimalFeatureExtractor
from src.core.actions import PreprocessingAction


def load_dataset(file_path: str, file_type: str = 'csv') -> pd.DataFrame:
    """
    Load dataset from file.

    Args:
        file_path: Path to dataset file
        file_type: Type of file (csv, json, pkl)

    Returns:
        Loaded DataFrame
    """
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'json':
        return pd.read_json(file_path)
    elif file_type in ['pkl', 'pickle']:
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def infer_preprocessing_action(column: pd.Series) -> PreprocessingAction:
    """
    Infer ground truth preprocessing action based on column characteristics.

    This is a heuristic-based approach. In production, you'd ideally have
    expert-labeled ground truth.

    Args:
        column: Pandas Series to analyze

    Returns:
        Recommended preprocessing action
    """
    # Handle null values
    null_ratio = column.isna().mean()
    if null_ratio > 0.7:
        return PreprocessingAction.DROP

    # Numeric columns
    if pd.api.types.is_numeric_dtype(column):
        clean_col = column.dropna()

        if len(clean_col) == 0:
            return PreprocessingAction.DROP

        # Constant column
        if clean_col.nunique() == 1:
            return PreprocessingAction.DROP

        # Unique ID (every value is unique)
        unique_ratio = clean_col.nunique() / len(clean_col)
        if unique_ratio > 0.95:
            return PreprocessingAction.DROP

        # Calculate outlier ratio using IQR method
        q1, q3 = clean_col.quantile([0.25, 0.75])
        iqr = q3 - q1

        if iqr == 0:  # No variance
            return PreprocessingAction.DROP

        outlier_ratio = ((clean_col < q1 - 1.5*iqr) | (clean_col > q3 + 1.5*iqr)).mean()

        # High outlier ratio -> robust scaling
        if outlier_ratio > 0.15:
            return PreprocessingAction.ROBUST_SCALE

        # Calculate skewness
        try:
            skewness = abs(clean_col.skew())
        except:
            skewness = 0

        # Highly skewed -> log transform
        if skewness > 2.0 and clean_col.min() > 0:
            return PreprocessingAction.LOG_TRANSFORM
        elif skewness > 1.0:
            return PreprocessingAction.YEO_JOHNSON

        # Normal-ish distribution -> standard scaling
        return PreprocessingAction.STANDARD_SCALE

    # Categorical/Object columns
    elif pd.api.types.is_object_dtype(column) or pd.api.types.is_categorical_dtype(column):
        clean_col = column.dropna()

        if len(clean_col) == 0:
            return PreprocessingAction.DROP

        n_unique = clean_col.nunique()
        unique_ratio = n_unique / len(clean_col)

        # Unique ID
        if unique_ratio > 0.95:
            return PreprocessingAction.DROP

        # Constant
        if n_unique == 1:
            return PreprocessingAction.DROP

        # High cardinality (>50 unique values)
        if n_unique > 50:
            return PreprocessingAction.TARGET_ENCODE

        # Low cardinality (<=10 unique values)
        if n_unique <= 10:
            return PreprocessingAction.ONE_HOT_ENCODE

        # Medium cardinality (11-50)
        return PreprocessingAction.LABEL_ENCODE

    # Boolean columns
    elif pd.api.types.is_bool_dtype(column):
        return PreprocessingAction.KEEP

    # DateTime columns
    elif pd.api.types.is_datetime64_any_dtype(column):
        return PreprocessingAction.DROP  # Extract features in preprocessing

    # Default
    return PreprocessingAction.KEEP


def train_from_dataset(dataset_path: str,
                      file_type: str = 'csv',
                      output_model: str = 'models/neural_oracle_custom.pkl',
                      exclude_columns: List[str] = None,
                      validation_split: float = 0.2):
    """
    Train Neural Oracle from a real dataset.

    Args:
        dataset_path: Path to dataset file
        file_type: Type of file (csv, json, pkl)
        output_model: Path to save trained model
        exclude_columns: Columns to exclude (e.g., target variable)
        validation_split: Fraction of data for validation
    """
    print("\n" + "="*70)
    print("Training Neural Oracle from Custom Dataset")
    print("="*70 + "\n")

    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    df = load_dataset(dataset_path, file_type)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns\n")

    # Exclude specified columns
    if exclude_columns:
        print(f"Excluding columns: {', '.join(exclude_columns)}")
        df = df.drop(columns=exclude_columns, errors='ignore')

    # Extract features and labels
    print("Extracting features and inferring preprocessing actions...")
    print("-" * 70)

    extractor = MinimalFeatureExtractor()

    features_list = []
    labels_list = []
    column_info = []

    for col_name in df.columns:
        column = df[col_name]

        try:
            # Extract 10 lightweight features
            features = extractor.extract(column)

            # Infer ground truth action
            label = infer_preprocessing_action(column)

            features_list.append(features)
            labels_list.append(label)
            column_info.append({
                'name': col_name,
                'action': label.value,
                'dtype': str(column.dtype),
                'null_pct': column.isna().mean() * 100,
                'unique': column.nunique()
            })

            # Print progress
            status = "✓" if label != PreprocessingAction.KEEP else "•"
            print(f"  {status} {col_name:30s} → {label.value:20s} "
                  f"(null: {column.isna().mean()*100:5.1f}%, "
                  f"unique: {column.nunique():5d})")

        except Exception as e:
            print(f"  ✗ {col_name:30s} → SKIPPED (error: {str(e)[:30]})")

    print(f"\nProcessed {len(features_list)} columns successfully\n")

    if len(features_list) == 0:
        print("ERROR: No columns could be processed!")
        return

    # Print action distribution
    print("Action Distribution:")
    print("-" * 70)
    action_counts = pd.Series([l.value for l in labels_list]).value_counts()
    total = len(labels_list)

    for action, count in action_counts.items():
        pct = count / total * 100
        bar_length = int(pct / 2)  # 50 chars = 100%
        bar = "█" * bar_length
        print(f"  {action:25s}: {count:3d} ({pct:5.1f}%)  {bar}")

    # Check if we have enough diversity
    if len(action_counts) < 3:
        print("\n⚠️  WARNING: Low action diversity. Consider using a more diverse dataset.")

    # Train model
    print(f"\nTraining Neural Oracle...")
    print("-" * 70)

    oracle = NeuralOracle()

    try:
        metrics = oracle.train(
            features=features_list,
            labels=labels_list,
            validation_split=validation_split
        )

        print(f"\nTraining Results:")
        print(f"  Training Accuracy:   {metrics['train_accuracy']:6.2%}")
        print(f"  Validation Accuracy: {metrics['val_accuracy']:6.2%}")
        print(f"  Number of Trees:     {metrics['num_trees']:6d}")
        print(f"  Number of Features:  {metrics['num_features']:6d}")
        print(f"  Number of Classes:   {metrics['num_classes']:6d}")

        # Benchmark inference
        print(f"\nBenchmarking inference speed...")
        test_features = features_list[0]
        avg_time_ms = oracle.benchmark_inference(test_features, num_iterations=1000)
        print(f"  Average inference time: {avg_time_ms:.2f}ms")

        if avg_time_ms < 5.0:
            print(f"    ✓ Target: <5ms - ACHIEVED!")
        else:
            print(f"    ⚠ Target: <5ms - Consider simplifying model")

        # Feature importance
        print(f"\nTop 10 Feature Importance:")
        print("-" * 70)
        top_features = oracle.get_top_features(top_k=10)
        max_importance = max(dict(top_features).values())

        for i, (feature, importance) in enumerate(top_features, 1):
            bar_length = int(importance / max_importance * 40)
            bar = "█" * bar_length
            print(f"  {i:2d}. {feature:25s} {bar} {importance:.1f}")

        # Save model
        print(f"\nSaving model...")
        print("-" * 70)

        output_path = Path(output_model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        oracle.save(output_path)

        file_size_kb = output_path.stat().st_size / 1024
        print(f"  Model saved to: {output_path}")
        print(f"  File size: {file_size_kb:.1f} KB")

        # Save training metadata
        metadata = {
            'dataset_path': dataset_path,
            'num_columns': len(features_list),
            'action_distribution': action_counts.to_dict(),
            'train_accuracy': metrics['train_accuracy'],
            'val_accuracy': metrics['val_accuracy'],
            'avg_inference_ms': avg_time_ms,
            'model_size_kb': file_size_kb
        }

        metadata_path = output_path.with_suffix('.metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved to: {metadata_path}")

        # Summary
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print("\nNext Steps:")
        print("  1. Test the model:")
        print(f"     from src.neural.oracle import NeuralOracle")
        print(f"     oracle = NeuralOracle.load('{output_model}')")
        print(f"     prediction = oracle.predict(features)")
        print("\n  2. Use in production:")
        print(f"     Update NEURAL_ORACLE_PATH in .env to '{output_model}'")
        print(f"     Restart the API server")
        print("\n  3. Evaluate accuracy:")
        print(f"     python scripts/evaluate_system.py --model {output_model}")
        print("\n" + "="*70 + "\n")

        return oracle

    except Exception as e:
        print(f"\n✗ Training failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        print(f"\nPlease check:")
        print(f"  1. Dataset has sufficient diversity")
        print(f"  2. At least 3 different actions in the dataset")
        print(f"  3. Minimum 20 samples for training")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Train Neural Oracle with custom datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with Titanic dataset
  python scripts/train_with_dataset.py --file data/titanic.csv --exclude Survived

  # Train with housing data
  python scripts/train_with_dataset.py --file data/housing.csv --output models/housing_oracle.pkl

  # Train with pickle file
  python scripts/train_with_dataset.py --file data/credit.pkl --type pkl

Popular datasets to try:
  - Titanic: https://www.kaggle.com/c/titanic/data
  - House Prices: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
  - UCI Adult Income: https://archive.ics.uci.edu/ml/datasets/adult
  - Credit Card Default: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
        """
    )

    parser.add_argument(
        '--file', '-f',
        required=True,
        help='Path to dataset file'
    )
    parser.add_argument(
        '--type', '-t',
        default='csv',
        choices=['csv', 'json', 'pkl', 'pickle'],
        help='File type (default: csv)'
    )
    parser.add_argument(
        '--output', '-o',
        default='models/neural_oracle_custom.pkl',
        help='Output model path (default: models/neural_oracle_custom.pkl)'
    )
    parser.add_argument(
        '--exclude', '-e',
        nargs='+',
        default=None,
        help='Columns to exclude (e.g., target variable, ID columns)'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Validation split fraction (default: 0.2)'
    )

    args = parser.parse_args()

    # Validate file exists
    if not Path(args.file).exists():
        print(f"ERROR: File not found: {args.file}")
        return 1

    # Train model
    oracle = train_from_dataset(
        dataset_path=args.file,
        file_type=args.type,
        output_model=args.output,
        exclude_columns=args.exclude,
        validation_split=args.validation_split
    )

    return 0 if oracle is not None else 1


if __name__ == '__main__':
    exit(main())
