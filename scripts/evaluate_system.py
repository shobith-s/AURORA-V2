#!/usr/bin/env python3
"""
System evaluation script for AURORA.
Evaluates accuracy, coverage, and quality of preprocessing recommendations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime
from collections import defaultdict

from src.core.preprocessor import AuroraPreprocessor
from src.core.actions import PreprocessingAction
from src.data.generator import generate_synthetic_data


class SystemEvaluator:
    """Evaluate AURORA preprocessing system."""

    def __init__(self, output_dir: str = "./evaluation_results"):
        """
        Initialize evaluator.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = AuroraPreprocessor()
        self.results = []

    def evaluate_missing_value_handling(self) -> Dict[str, Any]:
        """
        Evaluate missing value detection and handling recommendations.

        Returns:
            Evaluation results
        """
        print("=" * 70)
        print("Evaluating Missing Value Handling")
        print("=" * 70)

        test_cases = [
            {
                'name': 'High Missing Rate - Numeric',
                'type': 'numeric',
                'missing_rate': 0.9,
                'expected_actions': [PreprocessingAction.REMOVE_COLUMN]
            },
            {
                'name': 'Medium Missing Rate - Numeric Normal',
                'type': 'numeric',
                'missing_rate': 0.2,
                'expected_actions': [PreprocessingAction.FILL_MEAN, PreprocessingAction.FILL_MEDIAN]
            },
            {
                'name': 'Low Missing Rate - Categorical',
                'type': 'categorical',
                'missing_rate': 0.1,
                'expected_actions': [PreprocessingAction.FILL_MODE]
            },
        ]

        results = []

        for case in test_cases:
            print(f"\nTest: {case['name']}")

            # Generate test data
            if case['type'] == 'numeric':
                data = pd.DataFrame({
                    'test_column': np.random.randn(1000)
                })
            else:
                data = pd.DataFrame({
                    'test_column': np.random.choice(['A', 'B', 'C'], size=1000)
                })

            # Add missing values
            mask = np.random.random(1000) < case['missing_rate']
            data.loc[mask, 'test_column'] = None

            # Get recommendations
            recommendations = self.preprocessor.analyze(data)

            if recommendations:
                action = recommendations[0]['action']
                confidence = recommendations[0]['confidence']

                # Check if action is expected
                is_correct = action in case['expected_actions']

                result = {
                    'test_case': case['name'],
                    'actual_missing_rate': data['test_column'].isna().mean(),
                    'recommended_action': action,
                    'confidence': confidence,
                    'expected_actions': [a.value for a in case['expected_actions']],
                    'is_correct': is_correct
                }

                print(f"  Recommended: {action} (confidence: {confidence:.2f})")
                print(f"  Expected: {[a.value for a in case['expected_actions']]}")
                print(f"  Result: {' PASS' if is_correct else ' FAIL'}")

            else:
                result = {
                    'test_case': case['name'],
                    'error': 'No recommendations'
                }
                print(f"   No recommendations generated")

            results.append(result)

        # Calculate accuracy
        correct = sum(1 for r in results if r.get('is_correct', False))
        total = len(results)
        accuracy = correct / total if total > 0 else 0

        print(f"\nOverall Accuracy: {accuracy*100:.1f}% ({correct}/{total})")

        return {
            'evaluation_type': 'missing_value_handling',
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'results': results
        }

    def evaluate_scaling_recommendations(self) -> Dict[str, Any]:
        """
        Evaluate scaling and normalization recommendations.

        Returns:
            Evaluation results
        """
        print("\n" + "=" * 70)
        print("Evaluating Scaling Recommendations")
        print("=" * 70)

        test_cases = [
            {
                'name': 'Normal Distribution',
                'data_fn': lambda: np.random.randn(1000),
                'expected_actions': [PreprocessingAction.STANDARDIZE, PreprocessingAction.NORMALIZE]
            },
            {
                'name': 'Skewed Distribution',
                'data_fn': lambda: np.random.exponential(2, 1000),
                'expected_actions': [PreprocessingAction.LOG_TRANSFORM]
            },
            {
                'name': 'Uniform Range',
                'data_fn': lambda: np.random.uniform(0, 100, 1000),
                'expected_actions': [PreprocessingAction.NORMALIZE, PreprocessingAction.STANDARDIZE]
            },
        ]

        results = []

        for case in test_cases:
            print(f"\nTest: {case['name']}")

            # Generate test data
            data = pd.DataFrame({
                'test_column': case['data_fn']()
            })

            # Get recommendations
            recommendations = self.preprocessor.analyze(data)

            if recommendations:
                action = recommendations[0]['action']
                confidence = recommendations[0]['confidence']

                is_correct = action in case['expected_actions']

                result = {
                    'test_case': case['name'],
                    'recommended_action': action,
                    'confidence': confidence,
                    'expected_actions': [a.value for a in case['expected_actions']],
                    'is_correct': is_correct
                }

                print(f"  Recommended: {action} (confidence: {confidence:.2f})")
                print(f"  Expected: {[a.value for a in case['expected_actions']]}")
                print(f"  Result: {' PASS' if is_correct else ' FAIL'}")

            else:
                result = {
                    'test_case': case['name'],
                    'error': 'No recommendations'
                }
                print(f"   No recommendations generated")

            results.append(result)

        correct = sum(1 for r in results if r.get('is_correct', False))
        total = len(results)
        accuracy = correct / total if total > 0 else 0

        print(f"\nOverall Accuracy: {accuracy*100:.1f}% ({correct}/{total})")

        return {
            'evaluation_type': 'scaling_recommendations',
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'results': results
        }

    def evaluate_encoding_recommendations(self) -> Dict[str, Any]:
        """
        Evaluate categorical encoding recommendations.

        Returns:
            Evaluation results
        """
        print("\n" + "=" * 70)
        print("Evaluating Encoding Recommendations")
        print("=" * 70)

        test_cases = [
            {
                'name': 'Low Cardinality',
                'data': ['A', 'B', 'C'] * 333 + ['A'],
                'expected_actions': [PreprocessingAction.ENCODE_ONEHOT]
            },
            {
                'name': 'High Cardinality',
                'data': [f'Cat_{i}' for i in range(1000)],
                'expected_actions': [PreprocessingAction.ENCODE_LABEL, PreprocessingAction.REMOVE_COLUMN]
            },
        ]

        results = []

        for case in test_cases:
            print(f"\nTest: {case['name']}")

            # Generate test data
            data = pd.DataFrame({
                'test_column': case['data']
            })

            # Get recommendations
            recommendations = self.preprocessor.analyze(data)

            if recommendations:
                action = recommendations[0]['action']
                confidence = recommendations[0]['confidence']

                is_correct = action in case['expected_actions']

                result = {
                    'test_case': case['name'],
                    'unique_values': data['test_column'].nunique(),
                    'recommended_action': action,
                    'confidence': confidence,
                    'expected_actions': [a.value for a in case['expected_actions']],
                    'is_correct': is_correct
                }

                print(f"  Unique values: {data['test_column'].nunique()}")
                print(f"  Recommended: {action} (confidence: {confidence:.2f})")
                print(f"  Expected: {[a.value for a in case['expected_actions']]}")
                print(f"  Result: {' PASS' if is_correct else ' FAIL'}")

            else:
                result = {
                    'test_case': case['name'],
                    'error': 'No recommendations'
                }
                print(f"   No recommendations generated")

            results.append(result)

        correct = sum(1 for r in results if r.get('is_correct', False))
        total = len(results)
        accuracy = correct / total if total > 0 else 0

        print(f"\nOverall Accuracy: {accuracy*100:.1f}% ({correct}/{total})")

        return {
            'evaluation_type': 'encoding_recommendations',
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'results': results
        }

    def evaluate_coverage(self) -> Dict[str, Any]:
        """
        Evaluate recommendation coverage across diverse datasets.

        Returns:
            Evaluation results
        """
        print("\n" + "=" * 70)
        print("Evaluating Recommendation Coverage")
        print("=" * 70)

        num_datasets = 10
        coverage_stats = defaultdict(int)
        total_columns = 0

        for i in range(num_datasets):
            print(f"\nDataset {i+1}/{num_datasets}")

            # Generate diverse dataset
            data = generate_synthetic_data(
                num_rows=np.random.randint(500, 2000),
                num_numeric=np.random.randint(5, 20),
                num_categorical=np.random.randint(5, 15),
                missing_rate=np.random.uniform(0, 0.3)
            )

            # Get recommendations
            recommendations = self.preprocessor.analyze(data)

            total_columns += len(data.columns)

            # Count action types
            for rec in recommendations:
                coverage_stats[rec['action']] += 1

            # Check coverage
            coverage = len(recommendations) / len(data.columns) if len(data.columns) > 0 else 0
            print(f"  Columns: {len(data.columns)}, Recommendations: {len(recommendations)}")
            print(f"  Coverage: {coverage*100:.1f}%")

        # Calculate overall coverage
        overall_coverage = sum(coverage_stats.values()) / total_columns if total_columns > 0 else 0

        print("\n" + "-" * 70)
        print("Action Distribution:")
        for action, count in sorted(coverage_stats.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_columns) * 100 if total_columns > 0 else 0
            print(f"  {action:<30}: {count:>5} ({pct:>5.1f}%)")

        print(f"\nOverall Coverage: {overall_coverage*100:.1f}%")

        return {
            'evaluation_type': 'coverage',
            'timestamp': datetime.now().isoformat(),
            'total_columns': total_columns,
            'overall_coverage': overall_coverage,
            'action_distribution': dict(coverage_stats)
        }

    def evaluate_confidence_calibration(self) -> Dict[str, Any]:
        """
        Evaluate confidence score calibration.

        Returns:
            Evaluation results
        """
        print("\n" + "=" * 70)
        print("Evaluating Confidence Calibration")
        print("=" * 70)

        # Generate test datasets
        num_datasets = 20
        all_confidences = []

        for i in range(num_datasets):
            data = generate_synthetic_data(
                num_rows=1000,
                num_numeric=10,
                num_categorical=5,
                missing_rate=np.random.uniform(0, 0.3)
            )

            recommendations = self.preprocessor.analyze(data)

            for rec in recommendations:
                all_confidences.append(rec['confidence'])

        # Analyze confidence distribution
        confidences = np.array(all_confidences)

        stats = {
            'mean_confidence': float(np.mean(confidences)),
            'median_confidence': float(np.median(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'num_samples': len(confidences)
        }

        # Confidence buckets
        buckets = {
            'very_high (>0.9)': np.sum(confidences > 0.9),
            'high (0.7-0.9)': np.sum((confidences >= 0.7) & (confidences <= 0.9)),
            'medium (0.5-0.7)': np.sum((confidences >= 0.5) & (confidences < 0.7)),
            'low (<0.5)': np.sum(confidences < 0.5)
        }

        print("\nConfidence Statistics:")
        print(f"  Mean: {stats['mean_confidence']:.3f}")
        print(f"  Median: {stats['median_confidence']:.3f}")
        print(f"  Std: {stats['std_confidence']:.3f}")
        print(f"  Range: [{stats['min_confidence']:.3f}, {stats['max_confidence']:.3f}]")

        print("\nConfidence Distribution:")
        for bucket, count in buckets.items():
            pct = (count / len(confidences)) * 100 if len(confidences) > 0 else 0
            print(f"  {bucket:<20}: {count:>5} ({pct:>5.1f}%)")

        return {
            'evaluation_type': 'confidence_calibration',
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'distribution': buckets
        }

    def run_all_evaluations(self) -> Dict[str, Any]:
        """
        Run all evaluation tests.

        Returns:
            Complete evaluation results
        """
        print("\n" + "=" * 70)
        print("AURORA SYSTEM EVALUATION SUITE")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'evaluations': []
        }

        # Run evaluations
        evaluations = [
            ('missing_value_handling', self.evaluate_missing_value_handling),
            ('scaling_recommendations', self.evaluate_scaling_recommendations),
            ('encoding_recommendations', self.evaluate_encoding_recommendations),
            ('coverage', self.evaluate_coverage),
            ('confidence_calibration', self.evaluate_confidence_calibration),
        ]

        for name, eval_fn in evaluations:
            try:
                result = eval_fn()
                all_results['evaluations'].append(result)
            except Exception as e:
                print(f"\n Evaluation '{name}' failed: {e}")
                all_results['evaluations'].append({
                    'evaluation_type': name,
                    'success': False,
                    'error': str(e)
                })

        # Calculate overall score
        accuracies = [
            e.get('accuracy', 0)
            for e in all_results['evaluations']
            if 'accuracy' in e
        ]

        overall_accuracy = np.mean(accuracies) if accuracies else 0
        all_results['overall_accuracy'] = overall_accuracy

        # Save results
        output_file = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        print(f"Overall Accuracy: {overall_accuracy*100:.1f}%")
        print(f"Results saved to: {output_file}")

        return all_results


def main():
    """Run evaluation suite."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate AURORA system')
    parser.add_argument(
        '--output-dir',
        default='./evaluation_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    evaluator = SystemEvaluator(output_dir=args.output_dir)
    results = evaluator.run_all_evaluations()


if __name__ == '__main__':
    main()
