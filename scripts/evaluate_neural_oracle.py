"""
Evaluate Neural Oracle Performance.

Tests the newly trained neural oracle model with diverse test cases.
Shows layer usage, confidence distribution, and accuracy.

Usage:
    python scripts/evaluate_neural_oracle.py
    python scripts/evaluate_neural_oracle.py --model models/neural_oracle_v1.pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import json

from src.core.preprocessor import IntelligentPreprocessor
from src.core.actions import PreprocessingAction
from src.data.generator import SyntheticDataGenerator


class NeuralOracleEvaluator:
    """Evaluate neural oracle integration and performance."""

    def __init__(self, model_path: Path = None):
        """
        Initialize evaluator.

        Args:
            model_path: Path to neural oracle model
        """
        self.model_path = model_path

        # Initialize preprocessor with neural oracle
        self.preprocessor = IntelligentPreprocessor(
            use_neural_oracle=True,
            enable_learning=True,
            enable_cache=False,  # Disable for consistent testing
            enable_meta_learning=True
        )

        # Load model info if available
        self.model_metadata = None
        if model_path:
            metadata_path = model_path.parent / f"{model_path.stem}.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)

        self.results = []
        self.layer_usage = Counter()
        self.confidence_scores = []

    def create_test_cases(self) -> List[Tuple[str, pd.Series, str]]:
        """
        Create diverse test cases.

        Returns:
            List of (name, series, description) tuples
        """
        test_cases = []

        # 1. High skewness (should trigger symbolic or neural)
        test_cases.append((
            "High Positive Skewness",
            pd.Series([1, 2, 3, 100, 200, 500, 1000], name="revenue"),
            "Numeric column with extreme positive skew"
        ))

        # 2. High nulls (should trigger imputation or drop)
        test_cases.append((
            "High Null Rate",
            pd.Series([1, None, None, None, 5, None, None, 8, None, None], name="sparse"),
            "Column with >50% missing values"
        ))

        # 3. Low cardinality categorical
        test_cases.append((
            "Low Cardinality Categorical",
            pd.Series(['A', 'B', 'C', 'A', 'B', 'C'] * 10, name="category"),
            "Categorical with 3 unique values"
        ))

        # 4. High cardinality categorical
        test_cases.append((
            "High Cardinality Categorical",
            pd.Series([f"Cat_{i}" for i in range(100)], name="high_card"),
            "Categorical with 100 unique values"
        ))

        # 5. Normal distribution
        test_cases.append((
            "Normal Distribution",
            pd.Series(np.random.normal(50, 10, 100), name="normal"),
            "Well-behaved normal distribution"
        ))

        # 6. Many outliers
        data = list(np.random.normal(50, 5, 90))
        data.extend([200, 300, -100, -200, 500, 600, -300, -400, 700, 800])
        test_cases.append((
            "Many Outliers",
            pd.Series(data, name="outliers"),
            "Normal data with 10% extreme outliers"
        ))

        # 7. Low variance (almost constant)
        test_cases.append((
            "Low Variance",
            pd.Series([10, 10, 10, 10, 11, 10, 10, 10] * 10, name="constant_ish"),
            "Nearly constant values"
        ))

        # 8. Negative skewness
        test_cases.append((
            "Negative Skewness",
            pd.Series([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1, 1, 1, 1, 1], name="neg_skew"),
            "Left-skewed distribution"
        ))

        # 9. Bimodal distribution
        data1 = list(np.random.normal(10, 2, 50))
        data2 = list(np.random.normal(50, 2, 50))
        test_cases.append((
            "Bimodal Distribution",
            pd.Series(data1 + data2, name="bimodal"),
            "Two distinct peaks"
        ))

        # 10. Medium nulls + moderate skew (ambiguous)
        data = list(np.random.exponential(2, 70))
        data.extend([None] * 30)
        test_cases.append((
            "Ambiguous Case",
            pd.Series(data, name="ambiguous"),
            "Medium nulls + moderate skew (needs neural oracle)"
        ))

        # 11. Text-like high uniqueness
        test_cases.append((
            "High Uniqueness Text",
            pd.Series([f"Text_{i}_unique" for i in range(100)], name="text_ids"),
            "95%+ unique values (text-like)"
        ))

        # 12. Uniform distribution
        test_cases.append((
            "Uniform Distribution",
            pd.Series(np.random.uniform(0, 100, 100), name="uniform"),
            "Evenly distributed values"
        ))

        return test_cases

    def evaluate_test_case(
        self,
        name: str,
        series: pd.Series,
        description: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single test case.

        Args:
            name: Test case name
            series: Test data
            description: Test description

        Returns:
            Evaluation result
        """
        # Get preprocessing recommendation
        result = self.preprocessor.preprocess_column(series, series.name)

        # Track layer usage
        self.layer_usage[result.source] += 1

        # Track confidence
        self.confidence_scores.append(result.confidence)

        # Calculate statistics
        stats = {
            'null_pct': series.isna().mean() * 100,
            'unique_ratio': series.nunique() / len(series) if len(series) > 0 else 0,
            'dtype': str(series.dtype)
        }

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                stats['mean'] = float(non_null.mean())
                stats['std'] = float(non_null.std())
                stats['skewness'] = float(non_null.skew()) if len(non_null) > 2 else 0.0

        return {
            'test_case': name,
            'description': description,
            'action': result.action.value,
            'confidence': result.confidence,
            'source': result.source,
            'explanation': result.explanation[:100] + "..." if len(result.explanation) > 100 else result.explanation,
            'warning': result.warning,
            'require_manual_review': result.require_manual_review,
            'statistics': stats
        }

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation suite.

        Returns:
            Evaluation results
        """
        print("=" * 70)
        print("AURORA V2 - Neural Oracle Evaluation")
        print("=" * 70)

        if self.model_metadata:
            print("\nModel Information:")
            print(f"  Training Date: {self.model_metadata.get('training_date', 'N/A')}")
            print(f"  Training Samples: {self.model_metadata.get('num_samples', 'N/A')}")
            print(f"  - Corrections: {self.model_metadata.get('num_corrections', 'N/A')}")
            print(f"  - Open Datasets: {self.model_metadata.get('num_open_dataset_columns', 'N/A')}")
            print(f"  - Synthetic: {self.model_metadata.get('num_synthetic', 'N/A')}")
            print(f"  Training Accuracy: {self.model_metadata.get('train_accuracy', 0)*100:.2f}%")
            print(f"  Validation Accuracy: {self.model_metadata.get('val_accuracy', 0)*100:.2f}%")
            print(f"  Model Size: {self.model_metadata.get('model_size_kb', 0):.1f} KB")

        print("\n" + "=" * 70)
        print("Running Test Cases...")
        print("=" * 70)

        # Get test cases
        test_cases = self.create_test_cases()

        # Run each test
        for i, (name, series, description) in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {name}")
            print(f"    Description: {description}")

            result = self.evaluate_test_case(name, series, description)
            self.results.append(result)

            print(f"    → Action: {result['action']}")
            print(f"    → Confidence: {result['confidence']:.2%}")
            print(f"    → Source: {result['source']}")

            if result['warning']:
                print(f"    → Warning: {result['warning']}")

        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        # Layer usage
        total_decisions = sum(self.layer_usage.values())
        print("\nLayer Usage:")
        for layer in ['learned', 'symbolic', 'neural', 'meta_learning']:
            count = self.layer_usage[layer]
            pct = (count / total_decisions * 100) if total_decisions > 0 else 0
            print(f"  {layer:15s}: {count:3d} decisions ({pct:5.1f}%)")

        # Confidence distribution
        confidences = np.array(self.confidence_scores)
        print("\nConfidence Statistics:")
        print(f"  Mean:   {np.mean(confidences):.3f}")
        print(f"  Median: {np.median(confidences):.3f}")
        print(f"  Std:    {np.std(confidences):.3f}")
        print(f"  Min:    {np.min(confidences):.3f}")
        print(f"  Max:    {np.max(confidences):.3f}")

        # Confidence buckets
        print("\nConfidence Distribution:")
        high = np.sum(confidences >= 0.9)
        medium = np.sum((confidences >= 0.7) & (confidences < 0.9))
        low = np.sum((confidences >= 0.5) & (confidences < 0.7))
        very_low = np.sum(confidences < 0.5)

        print(f"  High (≥0.9):     {high:3d} ({high/len(confidences)*100:5.1f}%)")
        print(f"  Medium (0.7-0.9): {medium:3d} ({medium/len(confidences)*100:5.1f}%)")
        print(f"  Low (0.5-0.7):    {low:3d} ({low/len(confidences)*100:5.1f}%)")
        print(f"  Very Low (<0.5):  {very_low:3d} ({very_low/len(confidences)*100:5.1f}%)")

        # Warning counts
        warnings = sum(1 for r in self.results if r['warning'])
        manual_review = sum(1 for r in self.results if r['require_manual_review'])
        print(f"\nWarnings Issued: {warnings}/{len(self.results)}")
        print(f"Manual Review Required: {manual_review}/{len(self.results)}")

        # Action diversity
        action_counts = Counter(r['action'] for r in self.results)
        print("\nAction Distribution:")
        for action, count in action_counts.most_common():
            pct = (count / len(self.results) * 100)
            print(f"  {action:25s}: {count:3d} ({pct:5.1f}%)")

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)

        # Neural oracle usage
        neural_usage = self.layer_usage['neural']
        neural_pct = (neural_usage / total_decisions * 100) if total_decisions > 0 else 0

        print(f"\n✓ Neural Oracle Used: {neural_usage}/{total_decisions} times ({neural_pct:.1f}%)")

        if neural_usage == 0:
            print("  ℹ️  Neural oracle not triggered in these test cases.")
            print("     This is normal - symbolic rules handle most cases well.")
            print("     Neural oracle only activates when symbolic confidence < 0.9")

        return {
            'total_tests': len(self.results),
            'layer_usage': dict(self.layer_usage),
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'median': float(np.median(confidences)),
                'std': float(np.std(confidences))
            },
            'warnings_issued': warnings,
            'manual_review_required': manual_review,
            'results': self.results
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate neural oracle")
    parser.add_argument(
        '--model',
        type=str,
        default='models/neural_oracle_v1.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results'
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"⚠ Model not found: {model_path}")
        print(f"  Train first: python scripts/train_hybrid.py")
        return 1

    # Run evaluation
    evaluator = NeuralOracleEvaluator(model_path=model_path)
    results = evaluator.run_evaluation()

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
