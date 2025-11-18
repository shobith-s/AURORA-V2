"""
Training script for AURORA Neural Oracle.
Trains XGBoost model on synthetic edge cases.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neural.oracle import NeuralOracle
from src.features.minimal_extractor import MinimalFeatureExtractor
from src.data.generator import SyntheticDataGenerator
from src.core.actions import PreprocessingAction


def main():
    print("\n" + "="*70)
    print("AURORA Neural Oracle Training")
    print("="*70 + "\n")

    # Step 1: Generate training data
    print("Step 1: Generating training data...")
    print("-" * 70)

    generator = SyntheticDataGenerator(seed=42)

    # Generate 5000 samples (only ambiguous cases for neural oracle)
    columns, labels, difficulties = generator.generate_training_data(
        n_samples=5000,
        ambiguous_only=True
    )

    print(f"Generated {len(columns)} training samples")
    print(f"\nDifficulty breakdown:")
    difficulty_counts = pd.Series(difficulties).value_counts()
    for diff, count in difficulty_counts.items():
        print(f"  {diff:8s}: {count:4d} samples")

    print(f"\nAction breakdown:")
    action_counts = pd.Series([l.value for l in labels]).value_counts()
    for action, count in action_counts.head(10).items():
        print(f"  {action:25s}: {count:3d}")

    # Step 2: Extract features
    print(f"\nStep 2: Extracting minimal features...")
    print("-" * 70)

    extractor = MinimalFeatureExtractor()
    features_list = []

    for i, col in enumerate(columns):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(columns)} columns...")
        features = extractor.extract(col)
        features_list.append(features)

    print(f"Extracted features for {len(features_list)} samples")
    print(f"  Feature dimensions: 10 features per sample")

    # Step 3: Train model
    print(f"\nStep 3: Training XGBoost model...")
    print("-" * 70)

    oracle = NeuralOracle()

    try:
        metrics = oracle.train(
            features=features_list,
            labels=labels,
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

        # Step 4: Benchmark inference
        print(f"\nStep 4: Benchmarking inference speed...")
        print("-" * 70)

        test_features = features_list[0]
        avg_time_ms = oracle.benchmark_inference(test_features, num_iterations=1000)

        print(f"Average inference time: {avg_time_ms:.2f}ms")

        if avg_time_ms < 5.0:
            print(f"  ✓ Target: <5ms - ACHIEVED!")
        else:
            print(f"  ✗ Target: <5ms - needs optimization")

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

        # Step 7: Validation test
        print(f"\nStep 7: Quick validation test...")
        print("-" * 70)

        # Test on a few samples
        test_samples = features_list[:5]
        test_labels = labels[:5]

        print("\nTesting 5 random samples:")
        for i, (features, true_label) in enumerate(zip(test_samples, test_labels), 1):
            pred = oracle.predict(features, return_probabilities=False)
            match = "✓" if pred.action == true_label else "✗"
            print(f"  {i}. Predicted: {pred.action.value:20s} "
                  f"(conf: {pred.confidence:.2f}) | "
                  f"True: {true_label.value:20s} {match}")

        # Final summary
        print("\n" + "="*70)
        print("NEURAL ORACLE TRAINING COMPLETE!")
        print("="*70)
        print("\nNext Steps:")
        print("  1. Restart your backend server:")
        print("     uvicorn src.api.server:app --reload")
        print("\n  2. The neural oracle will automatically load on startup")
        print("\n  3. Check /health endpoint to verify:")
        print("     curl http://localhost:8000/health")
        print("\n  4. Test preprocessing with edge cases!")
        print("\n" + "="*70 + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ Training failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        print(f"\nPlease check:")
        print(f"  1. XGBoost is installed: pip install xgboost")
        print(f"  2. All dependencies are available")
        print(f"  3. Sufficient memory (at least 1GB free)")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
