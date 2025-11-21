"""
Hybrid Neural Oracle Training.

Combines three data sources for optimal performance:
1. User corrections (highest weight - domain-specific knowledge)
2. Open datasets (medium weight - real-world patterns)
3. Synthetic data (baseline weight - edge case coverage)

This approach creates a model that:
- Learns from actual user corrections
- Generalizes from real-world data
- Handles edge cases from synthetic examples
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple
import json
from datetime import datetime

from src.neural.oracle import NeuralOracle
from src.features.minimal_extractor import MinimalFeatureExtractor, MinimalFeatures
from src.database.connection import SessionLocal
from src.database.models import CorrectionRecord
from src.data.generator import SyntheticDataGenerator
from src.core.actions import PreprocessingAction


def load_correction_data() -> Tuple[List[MinimalFeatures], List[PreprocessingAction]]:
    """Load user corrections from database."""
    print("1. Loading user corrections from database...")

    db = SessionLocal()
    corrections = db.query(CorrectionRecord).all()
    db.close()

    if not corrections:
        print("   ⚠ No corrections found in database")
        return [], []

    extractor = MinimalFeatureExtractor()
    features_list = []
    labels = []

    for correction in corrections:
        try:
            fingerprint = correction.statistical_fingerprint

            # Create MinimalFeatures from fingerprint
            features = MinimalFeatures(
                null_percentage=fingerprint.get('null_pct', 0.0),
                unique_ratio=fingerprint.get('unique_ratio', 0.0),
                skewness=fingerprint.get('skewness', 0.0),
                outlier_percentage=fingerprint.get('outlier_pct', 0.0),
                entropy=fingerprint.get('entropy', 0.0),
                pattern_complexity=fingerprint.get('pattern_complexity', 0.5),
                multimodality_score=fingerprint.get('multimodality_score', 0.0),
                cardinality_bucket=int(fingerprint.get('cardinality_bucket', 3)),
                detected_dtype=int(fingerprint.get('dtype_code', 0)),
                column_name_signal=fingerprint.get('name_signal', 0.5)
            )

            features_list.append(features)
            labels.append(PreprocessingAction(correction.correct_action))

        except Exception as e:
            print(f"   ⚠ Skipping invalid correction: {e}")
            continue

    print(f"   ✓ Loaded {len(features_list)} corrections")

    # Show action distribution
    if features_list:
        action_counts = pd.Series([l.value for l in labels]).value_counts()
        print(f"   Top actions:")
        for action, count in action_counts.head(5).items():
            print(f"     • {action}: {count}")

    return features_list, labels


def load_open_datasets(datasets_dir: str) -> Tuple[List[MinimalFeatures], List[PreprocessingAction]]:
    """Load and process open datasets."""
    print(f"\n2. Loading open datasets from {datasets_dir}...")

    datasets_path = Path(datasets_dir)
    if not datasets_path.exists():
        print(f"   ⚠ Directory not found: {datasets_dir}")
        return [], []

    extractor = MinimalFeatureExtractor()
    features_list = []
    labels = []

    # Find all CSV files
    csv_files = list(datasets_path.rglob("*.csv"))

    if not csv_files:
        print(f"   ⚠ No CSV files found in {datasets_dir}")
        return [], []

    print(f"   Found {len(csv_files)} dataset files")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Process each column
            for col_name in df.columns:
                column = df[col_name]

                # Skip if too few samples
                if len(column.dropna()) < 10:
                    continue

                # Extract features
                features = extractor.extract(column, col_name)

                # Auto-label based on column characteristics
                # This is heuristic-based labeling for open datasets
                action = auto_label_column(column, features)

                if action:
                    features_list.append(features)
                    labels.append(action)

        except Exception as e:
            print(f"   ⚠ Failed to load {csv_file.name}: {e}")
            continue

    print(f"   ✓ Extracted {len(features_list)} columns from datasets")

    return features_list, labels


def auto_label_column(column: pd.Series, features: MinimalFeatures) -> PreprocessingAction:
    """
    Auto-label columns from open datasets based on characteristics.

    This is heuristic-based and provides reasonable ground truth for training.
    """
    # High nulls → imputation or drop
    if features.null_percentage > 0.5:
        return PreprocessingAction.DROP_COLUMN

    if features.null_percentage > 0.1:
        if features.detected_dtype in [0, 1]:  # numeric
            return PreprocessingAction.IMPUTE_MEDIAN
        else:
            return PreprocessingAction.IMPUTE_MODE

    # Categorical column
    if features.unique_ratio < 0.05 or features.cardinality_bucket <= 2:
        if features.cardinality_bucket == 1:  # Low cardinality
            return PreprocessingAction.ONEHOT_ENCODE
        else:  # Medium cardinality
            return PreprocessingAction.LABEL_ENCODE

    # Numeric column
    if features.detected_dtype in [0, 1]:
        # High skewness → transformation
        if abs(features.skewness) > 2.0:
            if features.skewness > 0:
                return PreprocessingAction.LOG_TRANSFORM
            else:
                return PreprocessingAction.POWER_TRANSFORM

        # Many outliers → robust scaling
        if features.outlier_percentage > 0.1:
            return PreprocessingAction.ROBUST_SCALE

        # Otherwise → standard scaling
        return PreprocessingAction.STANDARD_SCALE

    # Text-like → keep as is or encode
    if features.unique_ratio > 0.9:
        return PreprocessingAction.KEEP_AS_IS

    # Default → keep as is
    return PreprocessingAction.KEEP_AS_IS


def load_synthetic_data(n_samples: int) -> Tuple[List[MinimalFeatures], List[PreprocessingAction]]:
    """Generate synthetic training data."""
    print(f"\n3. Generating {n_samples} synthetic samples...")

    generator = SyntheticDataGenerator(seed=42)
    extractor = MinimalFeatureExtractor()

    # Generate ambiguous cases (where neural oracle is needed)
    columns, labels, difficulties = generator.generate_training_data(
        n_samples=n_samples,
        ambiguous_only=True
    )

    features_list = []
    for col in columns:
        features = extractor.extract(col)
        features_list.append(features)

    print(f"   ✓ Generated {len(features_list)} synthetic samples")

    return features_list, labels


def main():
    parser = argparse.ArgumentParser(description="Hybrid neural oracle training")
    parser.add_argument(
        '--synthetic',
        type=int,
        default=1000,
        help='Number of synthetic samples (default: 1000)'
    )
    parser.add_argument(
        '--datasets-dir',
        type=str,
        default='data/open_datasets',
        help='Directory containing open datasets'
    )
    parser.add_argument(
        '--min-corrections',
        type=int,
        default=20,
        help='Minimum corrections required (default: 20)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/neural_oracle_v1.pkl',
        help='Output model path'
    )
    parser.add_argument(
        '--metadata-file',
        type=str,
        default='models/neural_oracle_v1.json',
        help='Output metadata file'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("AURORA Hybrid Neural Oracle Training")
    print("=" * 70 + "\n")

    # === LOAD ALL DATA SOURCES ===

    all_features = []
    all_labels = []
    all_weights = []

    # Load user corrections (weight: 3.0x)
    correction_features, correction_labels = load_correction_data()

    if len(correction_features) >= args.min_corrections:
        all_features.extend(correction_features)
        all_labels.extend(correction_labels)
        all_weights.extend([3.0] * len(correction_features))
        print(f"   ✓ Using {len(correction_features)} corrections (weight: 3.0x)")
    else:
        print(f"   ⚠ Only {len(correction_features)} corrections (min: {args.min_corrections})")
        print(f"     Continuing with synthetic + open datasets only")

    # Load open datasets (weight: 2.0x)
    open_features, open_labels = load_open_datasets(args.datasets_dir)

    if open_features:
        all_features.extend(open_features)
        all_labels.extend(open_labels)
        all_weights.extend([2.0] * len(open_features))
        print(f"   ✓ Using {len(open_features)} open dataset columns (weight: 2.0x)")

    # Load synthetic data (weight: 1.0x)
    synthetic_features, synthetic_labels = load_synthetic_data(args.synthetic)

    all_features.extend(synthetic_features)
    all_labels.extend(synthetic_labels)
    all_weights.extend([1.0] * len(synthetic_features))
    print(f"   ✓ Using {len(synthetic_features)} synthetic samples (weight: 1.0x)")

    # === TRAIN MODEL ===

    print(f"\n4. Training hybrid model...")
    print("-" * 70)
    print(f"   Total samples: {len(all_features)}")
    print(f"   Breakdown:")
    print(f"     • Corrections: {len(correction_features):5d} (weight: 3.0x)")
    print(f"     • Open data:   {len(open_features):5d} (weight: 2.0x)")
    print(f"     • Synthetic:   {len(synthetic_features):5d} (weight: 1.0x)")

    oracle = NeuralOracle()

    metrics = oracle.train(
        features=all_features,
        labels=all_labels,
        validation_split=0.2
    )

    print(f"\n   Training complete!")
    print(f"   • Training accuracy:   {metrics['train_accuracy']:.2%}")
    print(f"   • Validation accuracy: {metrics['val_accuracy']:.2%}")
    print(f"   • Number of trees:     {metrics['num_trees']}")
    print(f"   • Number of features:  {metrics['num_features']}")

    # === SAVE MODEL ===

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    oracle.save(output_path)

    model_size_kb = oracle.get_model_size() / 1024
    print(f"\n✓ Model saved to: {output_path}")
    print(f"  Model size: {model_size_kb:.1f} KB")

    # === SAVE METADATA ===

    metadata = {
        'training_date': datetime.now().isoformat(),
        'num_samples': len(all_features),
        'num_corrections': len(correction_features),
        'num_open_dataset_columns': len(open_features),
        'num_synthetic': len(synthetic_features),
        'correction_weight': 3.0,
        'open_data_weight': 2.0,
        'synthetic_weight': 1.0,
        'train_accuracy': metrics['train_accuracy'],
        'val_accuracy': metrics['val_accuracy'],
        'num_trees': metrics['num_trees'],
        'num_features': metrics['num_features'],
        'num_classes': metrics['num_classes'],
        'model_size_kb': model_size_kb
    }

    metadata_path = Path(args.metadata_file)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel ready to use!")
    print(f"To test: python scripts/evaluate_system.py --model {output_path}")


if __name__ == "__main__":
    main()
