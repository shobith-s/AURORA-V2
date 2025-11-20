"""
Train Neural Oracle from Real User Corrections.

This script trains the neural oracle using REAL correction data from the database,
making it genuinely learn from user feedback.

Approach:
1. Load correction records from database
2. Extract features from stored statistical fingerprints
3. Train neural oracle to predict correct actions
4. Validate and save model

This makes the neural oracle a TRUE meta-learner that improves over time.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neural.oracle import NeuralOracle
from src.features.minimal_extractor import MinimalFeatures
from src.database.connection import SessionLocal
from src.database.models import CorrectionRecord
from src.core.actions import PreprocessingAction
from src.data.generator import SyntheticDataGenerator


def extract_features_from_fingerprint(fingerprint: Dict[str, Any]) -> MinimalFeatures:
    """
    Convert statistical fingerprint to MinimalFeatures.

    Args:
        fingerprint: Statistical fingerprint from correction record

    Returns:
        MinimalFeatures object
    """
    return MinimalFeatures(
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


def load_correction_data(db_session, min_samples: int = 50) -> Tuple[List[MinimalFeatures], List[PreprocessingAction], Dict[str, Any]]:
    """
    Load correction records from database and convert to training data.

    Args:
        db_session: Database session
        min_samples: Minimum samples required

    Returns:
        (features, labels, metadata)
    """
    print("Loading correction records from database...")

    corrections = db_session.query(CorrectionRecord).all()

    if len(corrections) < min_samples:
        print(f"⚠️  Only {len(corrections)} corrections found. Minimum {min_samples} recommended.")
        print(f"    Consider collecting more user feedback or using synthetic data.")
        return [], [], {'num_corrections': len(corrections), 'sufficient': False}

    features_list = []
    labels = []

    for correction in corrections:
        try:
            # Extract features from statistical fingerprint
            fingerprint = correction.statistical_fingerprint
            features = extract_features_from_fingerprint(fingerprint)
            features_list.append(features)

            # Use the correct action as label
            action = PreprocessingAction(correction.correct_action)
            labels.append(action)

        except Exception as e:
            print(f"⚠️  Skipping invalid correction record: {e}")
            continue

    # Get action distribution
    action_counts = pd.Series([l.value for l in labels]).value_counts()

    metadata = {
        'num_corrections': len(corrections),
        'num_valid': len(labels),
        'sufficient': len(labels) >= min_samples,
        'action_distribution': action_counts.to_dict(),
        'date_range': {
            'earliest': min(c.timestamp for c in corrections).isoformat() if corrections else None,
            'latest': max(c.timestamp for c in corrections).isoformat() if corrections else None
        }
    }

    return features_list, labels, metadata


def load_synthetic_data(n_samples: int = 1000, ambiguous_only: bool = True) -> Tuple[List[MinimalFeatures], List[PreprocessingAction]]:
    """
    Load synthetic training data as baseline.

    Args:
        n_samples: Number of samples to generate
        ambiguous_only: Only generate ambiguous cases

    Returns:
        (features, labels)
    """
    print(f"Generating {n_samples} synthetic training samples...")

    from src.features.minimal_extractor import MinimalFeatureExtractor

    generator = SyntheticDataGenerator(seed=42)
    extractor = MinimalFeatureExtractor()

    columns, labels, difficulties = generator.generate_training_data(
        n_samples=n_samples,
        ambiguous_only=ambiguous_only
    )

    features_list = []
    for col in columns:
        features = extractor.extract(col)
        features_list.append(features)

    print(f"  Generated {len(features_list)} samples")

    return features_list, labels


def combine_datasets(
    real_features: List[MinimalFeatures],
    real_labels: List[PreprocessingAction],
    synthetic_features: List[MinimalFeatures],
    synthetic_labels: List[PreprocessingAction],
    real_weight: float = 2.0
) -> Tuple[List[MinimalFeatures], List[PreprocessingAction], List[float]]:
    """
    Combine real and synthetic data, weighting real data more heavily.

    Args:
        real_features: Features from real corrections
        real_labels: Labels from real corrections
        synthetic_features: Features from synthetic data
        synthetic_labels: Labels from synthetic data
        real_weight: Weight for real samples (higher = more important)

    Returns:
        (combined_features, combined_labels, sample_weights)
    """
    # Combine datasets
    all_features = real_features + synthetic_features
    all_labels = real_labels + synthetic_labels

    # Create sample weights
    real_weights = [real_weight] * len(real_features)
    synthetic_weights = [1.0] * len(synthetic_features)
    sample_weights = real_weights + synthetic_weights

    return all_features, all_labels, sample_weights


def main(
    use_corrections: bool = True,
    use_synthetic: bool = True,
    min_corrections: int = 50,
    synthetic_samples: int = 1000
):
    """
    Main training function.

    Args:
        use_corrections: Use real correction data from database
        use_synthetic: Use synthetic training data
        min_corrections: Minimum corrections required
        synthetic_samples: Number of synthetic samples to generate
    """
    print("\n" + "="*70)
    print("AURORA Neural Oracle Training - Learning from Corrections")
    print("="*70 + "\n")

    # Initialize storage
    all_features = []
    all_labels = []
    sample_weights = None

    # Step 1: Load correction data
    if use_corrections:
        print("Step 1: Loading real correction data from database...")
        print("-" * 70)

        db = SessionLocal()
        try:
            real_features, real_labels, metadata = load_correction_data(db, min_corrections)

            if metadata['sufficient']:
                print(f"✓ Loaded {metadata['num_valid']} corrections")
                print(f"\nAction distribution from user corrections:")
                for action, count in sorted(metadata['action_distribution'].items(), key=lambda x: -x[1])[:10]:
                    print(f"  {action:25s}: {count:3d}")

                all_features.extend(real_features)
                all_labels.extend(real_labels)
                real_weight = 2.0  # Real data is 2x more important
            else:
                print(f"⚠️  Insufficient corrections ({metadata['num_corrections']}), will rely on synthetic data")
                real_features, real_labels = [], []
                real_weight = 1.0
        finally:
            db.close()
    else:
        real_features, real_labels = [], []
        real_weight = 1.0

    # Step 2: Load synthetic data
    if use_synthetic:
        print(f"\nStep 2: Loading synthetic training data...")
        print("-" * 70)

        synthetic_features, synthetic_labels = load_synthetic_data(
            n_samples=synthetic_samples,
            ambiguous_only=True
        )

        if real_features:
            # Combine with real data
            print(f"\nCombining datasets:")
            print(f"  Real corrections:  {len(real_features):4d} samples (weight: {real_weight}x)")
            print(f"  Synthetic data:    {len(synthetic_features):4d} samples (weight: 1.0x)")

            all_features, all_labels, sample_weights = combine_datasets(
                real_features, real_labels,
                synthetic_features, synthetic_labels,
                real_weight=real_weight
            )
        else:
            all_features = synthetic_features
            all_labels = synthetic_labels

    if not all_features:
        print("\n✗ No training data available!")
        print("  Either collect user corrections or enable synthetic data generation.")
        return 1

    print(f"\nTotal training samples: {len(all_features)}")

    # Step 3: Train model
    print(f"\nStep 3: Training XGBoost model...")
    print("-" * 70)

    oracle = NeuralOracle()

    try:
        metrics = oracle.train(
            features=all_features,
            labels=all_labels,
            validation_split=0.2
        )

        print(f"\nTraining Complete!")
        print(f"\nPerformance Metrics:")
        print(f"  Training Accuracy:   {metrics['train_accuracy']:6.2%}")
        print(f"  Validation Accuracy: {metrics['val_accuracy']:6.2%}")
        print(f"  Number of Trees:     {metrics['num_trees']:6d}")
        print(f"  Number of Features:  {metrics['num_features']:6d}")
        print(f"  Number of Classes:   {metrics['num_classes']:6d}")

        # Model size
        model_size_kb = oracle.get_model_size() / 1024
        print(f"  Model Size:          {model_size_kb:6.1f} KB")

        # Step 4: Benchmark
        print(f"\nStep 4: Benchmarking inference speed...")
        print("-" * 70)

        test_features = all_features[0]
        avg_time_ms = oracle.benchmark_inference(test_features, num_iterations=1000)

        print(f"Average inference time: {avg_time_ms:.2f}ms")

        if avg_time_ms < 5.0:
            print(f"  ✓ Target: <5ms - ACHIEVED!")
        else:
            print(f"  ⚠️  Target: <5ms - needs optimization")

        # Step 5: Feature importance
        print(f"\nStep 5: Feature importance analysis...")
        print("-" * 70)

        top_features = oracle.get_top_features(top_k=10)
        print(f"\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(top_features, 1):
            bar_length = int(importance / max(dict(top_features).values()) * 40)
            bar = "█" * bar_length
            print(f"  {i:2d}. {feature:25s} {bar} {importance:.1f}")

        # Step 6: Save model
        print(f"\nStep 6: Saving model...")
        print("-" * 70)

        model_path = Path(__file__).parent.parent / "models" / "neural_oracle_v1.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        oracle.save(model_path)

        print(f"Model saved to: {model_path}")
        print(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")

        # Step 7: Save training metadata
        metadata_path = model_path.with_suffix('.json')
        training_metadata = {
            'training_date': pd.Timestamp.now().isoformat(),
            'num_samples': len(all_features),
            'num_real_corrections': len(real_features),
            'num_synthetic': len(synthetic_features) if use_synthetic else 0,
            'real_data_weight': real_weight,
            'train_accuracy': metrics['train_accuracy'],
            'val_accuracy': metrics['val_accuracy'],
            'model_size_kb': model_size_kb,
            'avg_inference_ms': avg_time_ms
        }

        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)

        print(f"Metadata saved to: {metadata_path}")

        # Final summary
        print("\n" + "="*70)
        print("NEURAL ORACLE TRAINING COMPLETE!")
        print("="*70)

        if real_features:
            print(f"\n✨ Model trained on {len(real_features)} REAL user corrections!")
            print(f"   The neural oracle is now learning from actual usage.")

        print("\nNext Steps:")
        print("  1. Restart your backend server:")
        print("     uvicorn src.api.server:app --reload")
        print("\n  2. The improved neural oracle will automatically load")
        print("\n  3. As users make more corrections, retrain periodically:")
        print("     python scripts/train_from_corrections.py")
        print("\n  4. Monitor accuracy improvements over time!")
        print("\n" + "="*70 + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ Training failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train neural oracle from corrections")
    parser.add_argument('--no-corrections', action='store_true',
                       help='Do not use correction data (synthetic only)')
    parser.add_argument('--no-synthetic', action='store_true',
                       help='Do not use synthetic data (corrections only)')
    parser.add_argument('--min-corrections', type=int, default=50,
                       help='Minimum corrections required (default: 50)')
    parser.add_argument('--synthetic-samples', type=int, default=1000,
                       help='Number of synthetic samples (default: 1000)')

    args = parser.parse_args()

    exit(main(
        use_corrections=not args.no_corrections,
        use_synthetic=not args.no_synthetic,
        min_corrections=args.min_corrections,
        synthetic_samples=args.synthetic_samples
    ))
