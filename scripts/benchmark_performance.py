#!/usr/bin/env python3
"""
Performance benchmarking script for AURORA preprocessing system.
Tests throughput, latency, and accuracy across different dataset sizes and types.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import json
from datetime import datetime

from src.core.preprocessor import AuroraPreprocessor
from src.data.generator import generate_synthetic_data
from src.utils.monitor import PerformanceMonitor


class PerformanceBenchmark:
    """Benchmark the AURORA preprocessing system."""

    def __init__(self, output_dir: str = "./benchmark_results"):
        """
        Initialize benchmark.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = AuroraPreprocessor()
        self.monitor = PerformanceMonitor()

        self.results = []

    def benchmark_dataset_sizes(
        self,
        sizes: List[int] = [100, 1000, 10000, 50000, 100000]
    ) -> Dict[str, Any]:
        """
        Benchmark performance across different dataset sizes.

        Args:
            sizes: List of dataset sizes (number of rows)

        Returns:
            Benchmark results
        """
        print("=" * 70)
        print("Benchmarking Dataset Sizes")
        print("=" * 70)

        results = []

        for size in sizes:
            print(f"\nTesting with {size:,} rows...")

            # Generate test data
            data = generate_synthetic_data(
                num_rows=size,
                num_numeric=10,
                num_categorical=5,
                missing_rate=0.1
            )

            # Benchmark preprocessing
            start_time = time.time()
            self.monitor.reset()

            try:
                recommendations = self.preprocessor.analyze(data)
                end_time = time.time()

                elapsed_time = end_time - start_time
                throughput = size / elapsed_time if elapsed_time > 0 else 0

                result = {
                    'dataset_size': size,
                    'num_columns': len(data.columns),
                    'elapsed_time_sec': elapsed_time,
                    'throughput_rows_per_sec': throughput,
                    'memory_usage_mb': self.monitor.get_memory_usage() / (1024 * 1024),
                    'num_recommendations': len(recommendations),
                    'success': True
                }

                print(f"   Time: {elapsed_time:.2f}s")
                print(f"   Throughput: {throughput:,.0f} rows/sec")
                print(f"   Memory: {result['memory_usage_mb']:.1f} MB")

            except Exception as e:
                print(f"   Error: {e}")
                result = {
                    'dataset_size': size,
                    'success': False,
                    'error': str(e)
                }

            results.append(result)

        return {
            'benchmark_type': 'dataset_sizes',
            'timestamp': datetime.now().isoformat(),
            'results': results
        }

    def benchmark_column_types(self) -> Dict[str, Any]:
        """
        Benchmark performance for different column type distributions.

        Returns:
            Benchmark results
        """
        print("\n" + "=" * 70)
        print("Benchmarking Column Type Distributions")
        print("=" * 70)

        test_cases = [
            {'name': 'Mostly Numeric', 'num_numeric': 50, 'num_categorical': 5},
            {'name': 'Mostly Categorical', 'num_numeric': 5, 'num_categorical': 50},
            {'name': 'Balanced', 'num_numeric': 25, 'num_categorical': 25},
            {'name': 'Many Columns', 'num_numeric': 100, 'num_categorical': 100},
        ]

        results = []

        for case in test_cases:
            print(f"\nTesting: {case['name']}")
            print(f"  Numeric: {case['num_numeric']}, Categorical: {case['num_categorical']}")

            # Generate test data
            data = generate_synthetic_data(
                num_rows=10000,
                num_numeric=case['num_numeric'],
                num_categorical=case['num_categorical'],
                missing_rate=0.1
            )

            # Benchmark
            start_time = time.time()
            try:
                recommendations = self.preprocessor.analyze(data)
                end_time = time.time()

                elapsed_time = end_time - start_time

                result = {
                    'case_name': case['name'],
                    'num_numeric': case['num_numeric'],
                    'num_categorical': case['num_categorical'],
                    'total_columns': case['num_numeric'] + case['num_categorical'],
                    'elapsed_time_sec': elapsed_time,
                    'time_per_column_ms': (elapsed_time / (case['num_numeric'] + case['num_categorical'])) * 1000,
                    'success': True
                }

                print(f"   Time: {elapsed_time:.2f}s")
                print(f"   Time per column: {result['time_per_column_ms']:.1f}ms")

            except Exception as e:
                print(f"   Error: {e}")
                result = {
                    'case_name': case['name'],
                    'success': False,
                    'error': str(e)
                }

            results.append(result)

        return {
            'benchmark_type': 'column_types',
            'timestamp': datetime.now().isoformat(),
            'results': results
        }

    def benchmark_data_quality(self) -> Dict[str, Any]:
        """
        Benchmark performance with different data quality levels.

        Returns:
            Benchmark results
        """
        print("\n" + "=" * 70)
        print("Benchmarking Data Quality Levels")
        print("=" * 70)

        missing_rates = [0.0, 0.1, 0.3, 0.5, 0.7]
        results = []

        for missing_rate in missing_rates:
            print(f"\nTesting with {missing_rate*100:.0f}% missing values...")

            # Generate test data
            data = generate_synthetic_data(
                num_rows=10000,
                num_numeric=20,
                num_categorical=10,
                missing_rate=missing_rate
            )

            # Benchmark
            start_time = time.time()
            try:
                recommendations = self.preprocessor.analyze(data)
                end_time = time.time()

                elapsed_time = end_time - start_time

                # Count actions
                action_counts = {}
                for rec in recommendations:
                    action = rec['action']
                    action_counts[action] = action_counts.get(action, 0) + 1

                result = {
                    'missing_rate': missing_rate,
                    'elapsed_time_sec': elapsed_time,
                    'action_distribution': action_counts,
                    'success': True
                }

                print(f"   Time: {elapsed_time:.2f}s")
                print(f"   Actions: {action_counts}")

            except Exception as e:
                print(f"   Error: {e}")
                result = {
                    'missing_rate': missing_rate,
                    'success': False,
                    'error': str(e)
                }

            results.append(result)

        return {
            'benchmark_type': 'data_quality',
            'timestamp': datetime.now().isoformat(),
            'results': results
        }

    def benchmark_neural_oracle(self) -> Dict[str, Any]:
        """
        Compare performance with and without neural oracle.

        Returns:
            Benchmark results
        """
        print("\n" + "=" * 70)
        print("Benchmarking Neural Oracle Impact")
        print("=" * 70)

        data = generate_synthetic_data(
            num_rows=10000,
            num_numeric=20,
            num_categorical=10,
            missing_rate=0.1
        )

        results = []

        # Test with neural oracle
        print("\nWith Neural Oracle...")
        self.preprocessor.use_neural = True
        start_time = time.time()
        try:
            recommendations_with = self.preprocessor.analyze(data)
            time_with = time.time() - start_time

            results.append({
                'mode': 'with_neural',
                'elapsed_time_sec': time_with,
                'num_recommendations': len(recommendations_with),
                'success': True
            })

            print(f"   Time: {time_with:.2f}s")

        except Exception as e:
            print(f"   Error: {e}")
            results.append({
                'mode': 'with_neural',
                'success': False,
                'error': str(e)
            })

        # Test without neural oracle
        print("\nWithout Neural Oracle...")
        self.preprocessor.use_neural = False
        start_time = time.time()
        try:
            recommendations_without = self.preprocessor.analyze(data)
            time_without = time.time() - start_time

            results.append({
                'mode': 'without_neural',
                'elapsed_time_sec': time_without,
                'num_recommendations': len(recommendations_without),
                'success': True
            })

            print(f"   Time: {time_without:.2f}s")

            if results[0]['success'] and results[1]['success']:
                speedup = time_with / time_without if time_without > 0 else 1.0
                print(f"\nSpeedup factor: {speedup:.2f}x")

        except Exception as e:
            print(f"   Error: {e}")
            results.append({
                'mode': 'without_neural',
                'success': False,
                'error': str(e)
            })

        return {
            'benchmark_type': 'neural_oracle',
            'timestamp': datetime.now().isoformat(),
            'results': results
        }

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all benchmarks.

        Returns:
            Complete benchmark results
        """
        print("\n" + "=" * 70)
        print("AURORA PERFORMANCE BENCHMARK SUITE")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': []
        }

        # Run benchmarks
        benchmarks = [
            ('dataset_sizes', lambda: self.benchmark_dataset_sizes()),
            ('column_types', lambda: self.benchmark_column_types()),
            ('data_quality', lambda: self.benchmark_data_quality()),
            ('neural_oracle', lambda: self.benchmark_neural_oracle()),
        ]

        for name, benchmark_fn in benchmarks:
            try:
                result = benchmark_fn()
                all_results['benchmarks'].append(result)
            except Exception as e:
                print(f"\n Benchmark '{name}' failed: {e}")
                all_results['benchmarks'].append({
                    'benchmark_type': name,
                    'success': False,
                    'error': str(e)
                })

        # Save results
        output_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {output_file}")

        return all_results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable benchmark report.

        Args:
            results: Benchmark results

        Returns:
            Formatted report
        """
        lines = [
            "=" * 70,
            "AURORA PERFORMANCE BENCHMARK REPORT",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        for benchmark in results.get('benchmarks', []):
            benchmark_type = benchmark.get('benchmark_type', 'Unknown')
            lines.append(f"\n{benchmark_type.upper()}")
            lines.append("-" * 70)

            if benchmark_type == 'dataset_sizes':
                for result in benchmark.get('results', []):
                    if result.get('success'):
                        lines.append(
                            f"  {result['dataset_size']:>8,} rows: "
                            f"{result['elapsed_time_sec']:>6.2f}s "
                            f"({result['throughput_rows_per_sec']:>8,.0f} rows/s)"
                        )

            elif benchmark_type == 'column_types':
                for result in benchmark.get('results', []):
                    if result.get('success'):
                        lines.append(
                            f"  {result['case_name']:<20}: "
                            f"{result['elapsed_time_sec']:>6.2f}s "
                            f"({result['time_per_column_ms']:>5.1f}ms/col)"
                        )

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)


def main():
    """Run benchmark suite."""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark AURORA performance')
    parser.add_argument(
        '--output-dir',
        default='./benchmark_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmark with smaller datasets'
    )

    args = parser.parse_args()

    benchmark = PerformanceBenchmark(output_dir=args.output_dir)

    if args.quick:
        print("Running quick benchmark...")
        results = benchmark.benchmark_dataset_sizes(sizes=[100, 1000, 5000])
    else:
        results = benchmark.run_all_benchmarks()

    # Print report
    report = benchmark.generate_report({'benchmarks': [results]} if args.quick else results)
    print("\n" + report)


if __name__ == '__main__':
    main()
