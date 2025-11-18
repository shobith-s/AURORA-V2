#!/usr/bin/env python3
"""
Synthetic data generation script for testing AURORA.
Generates diverse datasets with controlled characteristics for evaluation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json

from src.data.generator import (
    generate_synthetic_data,
    SyntheticDataGenerator
)


def save_dataset(data: pd.DataFrame, output_path: Path, metadata: dict = None):
    """
    Save dataset with metadata.

    Args:
        data: DataFrame to save
        output_path: Path to save file
        metadata: Optional metadata dictionary
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save data
    if output_path.suffix == '.csv':
        data.to_csv(output_path, index=False)
    elif output_path.suffix == '.json':
        data.to_json(output_path, orient='records', indent=2)
    elif output_path.suffix in ['.pkl', '.pickle']:
        data.to_pickle(output_path)
    else:
        # Default to CSV
        output_path = output_path.with_suffix('.csv')
        data.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")
    print(f"  Rows: {len(data):,}")
    print(f"  Columns: {len(data.columns)}")

    # Save metadata
    if metadata:
        metadata_path = output_path.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_path}")


def generate_basic_dataset(args):
    """Generate a basic dataset with specified parameters."""
    print("\n" + "="*70)
    print("Generating Basic Dataset")
    print("="*70)

    data = generate_synthetic_data(
        num_rows=args.rows,
        num_numeric=args.numeric,
        num_categorical=args.categorical,
        missing_rate=args.missing_rate,
        add_outliers=args.add_outliers,
        seed=args.seed
    )

    metadata = {
        'type': 'basic',
        'generated_at': datetime.now().isoformat(),
        'num_rows': args.rows,
        'num_numeric': args.numeric,
        'num_categorical': args.categorical,
        'missing_rate': args.missing_rate,
        'has_outliers': args.add_outliers,
        'seed': args.seed
    }

    save_dataset(data, Path(args.output), metadata)


def generate_edge_cases(args):
    """Generate dataset with edge cases for testing."""
    print("\n" + "="*70)
    print("Generating Edge Case Dataset")
    print("="*70)

    generator = SyntheticDataGenerator(seed=args.seed)

    # Generate various edge cases
    edge_cases = []

    # 1. High missing rate columns
    print("\nGenerating columns with high missing rates...")
    for i in range(5):
        col = generator.generate_numeric_column(1000, missing_rate=0.8)
        edge_cases.append((f'high_missing_{i}', col))

    # 2. Highly skewed columns
    print("Generating highly skewed columns...")
    for i in range(5):
        col = generator.generate_skewed_numeric_column(1000, skewness=3.0)
        edge_cases.append((f'high_skew_{i}', col))

    # 3. High cardinality categorical
    print("Generating high cardinality categorical columns...")
    for i in range(3):
        col = generator.generate_categorical_column(1000, num_categories=500)
        edge_cases.append((f'high_cardinality_{i}', col))

    # 4. Constant columns
    print("Generating constant columns...")
    for i in range(3):
        col = pd.Series([42.0] * 1000)
        edge_cases.append((f'constant_{i}', col))

    # 5. Unique ID columns
    print("Generating unique ID columns...")
    for i in range(3):
        col = pd.Series(range(1000))
        edge_cases.append((f'unique_id_{i}', col))

    # 6. Outlier-heavy columns
    print("Generating outlier-heavy columns...")
    for i in range(5):
        col = generator.generate_outlier_column(1000, outlier_ratio=0.2)
        edge_cases.append((f'outliers_{i}', col))

    # Combine into DataFrame
    data = pd.DataFrame({name: col for name, col in edge_cases})

    metadata = {
        'type': 'edge_cases',
        'generated_at': datetime.now().isoformat(),
        'num_rows': 1000,
        'num_columns': len(edge_cases),
        'categories': [
            'high_missing',
            'high_skew',
            'high_cardinality',
            'constant',
            'unique_id',
            'outliers'
        ],
        'seed': args.seed
    }

    save_dataset(data, Path(args.output), metadata)
    print(f"\nGenerated {len(edge_cases)} edge case columns")


def generate_realistic_dataset(args):
    """Generate a realistic dataset mimicking real-world data."""
    print("\n" + "="*70)
    print("Generating Realistic Dataset")
    print("="*70)

    generator = SyntheticDataGenerator(seed=args.seed)

    data = {}

    # User data
    print("\nGenerating user-related columns...")
    data['user_id'] = pd.Series(range(args.rows))
    data['age'] = generator.generate_numeric_column(args.rows, mean=35, std=12, missing_rate=0.02)
    data['gender'] = generator.generate_categorical_column(args.rows, num_categories=3)
    data['country'] = generator.generate_categorical_column(args.rows, num_categories=50)

    # Transaction data
    print("Generating transaction columns...")
    data['purchase_amount'] = generator.generate_skewed_numeric_column(args.rows, skewness=2.0)
    data['transaction_count'] = generator.generate_numeric_column(args.rows, mean=15, std=10)
    data['last_purchase_days'] = generator.generate_numeric_column(args.rows, mean=30, std=60)

    # Engagement metrics
    print("Generating engagement metrics...")
    data['login_count'] = generator.generate_numeric_column(args.rows, mean=20, std=15)
    data['session_duration_min'] = generator.generate_skewed_numeric_column(args.rows, skewness=1.5)
    data['pages_viewed'] = generator.generate_numeric_column(args.rows, mean=50, std=30)

    # Features with issues
    print("Adding realistic data quality issues...")
    data['incomplete_field'] = generator.generate_numeric_column(args.rows, missing_rate=0.4)
    data['sparse_category'] = generator.generate_categorical_column(args.rows, num_categories=200)
    data['outlier_metric'] = generator.generate_outlier_column(args.rows, outlier_ratio=0.1)

    df = pd.DataFrame(data)

    metadata = {
        'type': 'realistic',
        'generated_at': datetime.now().isoformat(),
        'description': 'Realistic e-commerce user dataset',
        'num_rows': args.rows,
        'num_columns': len(df.columns),
        'column_groups': {
            'user_info': ['user_id', 'age', 'gender', 'country'],
            'transactions': ['purchase_amount', 'transaction_count', 'last_purchase_days'],
            'engagement': ['login_count', 'session_duration_min', 'pages_viewed'],
            'quality_issues': ['incomplete_field', 'sparse_category', 'outlier_metric']
        },
        'seed': args.seed
    }

    save_dataset(df, Path(args.output), metadata)
    print(f"\nGenerated realistic dataset with {len(df.columns)} columns")


def generate_training_dataset(args):
    """Generate training dataset for neural oracle."""
    print("\n" + "="*70)
    print("Generating Neural Oracle Training Dataset")
    print("="*70)

    generator = SyntheticDataGenerator(seed=args.seed)

    # Generate training samples
    print(f"\nGenerating {args.samples} training samples...")
    columns, labels, difficulties = generator.generate_training_data(
        n_samples=args.samples,
        ambiguous_only=args.ambiguous_only
    )

    # Convert to DataFrame for easier handling
    data = []
    for col, label, difficulty in zip(columns, labels, difficulties):
        # Store column statistics and label
        stats = {
            'mean': col.mean() if col.dtype in [np.float64, np.int64] else None,
            'std': col.std() if col.dtype in [np.float64, np.int64] else None,
            'null_pct': col.isna().mean(),
            'unique_ratio': col.nunique() / len(col),
            'label': label.value,
            'difficulty': difficulty
        }
        data.append(stats)

    df = pd.DataFrame(data)

    metadata = {
        'type': 'training',
        'generated_at': datetime.now().isoformat(),
        'num_samples': args.samples,
        'ambiguous_only': args.ambiguous_only,
        'difficulty_distribution': df['difficulty'].value_counts().to_dict(),
        'label_distribution': df['label'].value_counts().to_dict(),
        'seed': args.seed
    }

    save_dataset(df, Path(args.output), metadata)

    print(f"\nDifficulty breakdown:")
    for diff, count in df['difficulty'].value_counts().items():
        print(f"  {diff:10s}: {count:5d} samples ({count/len(df)*100:.1f}%)")

    print(f"\nTop 10 actions:")
    for action, count in df['label'].value_counts().head(10).items():
        print(f"  {action:25s}: {count:4d} samples")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic datasets for AURORA testing and training'
    )

    # Common arguments
    parser.add_argument(
        '--output', '-o',
        default='./data/synthetic_dataset.csv',
        help='Output file path (CSV, JSON, or PKL)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Dataset type to generate')

    # Basic dataset
    basic_parser = subparsers.add_parser('basic', help='Generate basic dataset')
    basic_parser.add_argument('--rows', type=int, default=1000, help='Number of rows')
    basic_parser.add_argument('--numeric', type=int, default=10, help='Number of numeric columns')
    basic_parser.add_argument('--categorical', type=int, default=5, help='Number of categorical columns')
    basic_parser.add_argument('--missing-rate', type=float, default=0.1, help='Missing value rate (0-1)')
    basic_parser.add_argument('--add-outliers', action='store_true', help='Add outliers to numeric columns')

    # Edge cases
    edge_parser = subparsers.add_parser('edge-cases', help='Generate edge case dataset')

    # Realistic dataset
    realistic_parser = subparsers.add_parser('realistic', help='Generate realistic dataset')
    realistic_parser.add_argument('--rows', type=int, default=5000, help='Number of rows')

    # Training dataset
    training_parser = subparsers.add_parser('training', help='Generate training dataset for neural oracle')
    training_parser.add_argument('--samples', type=int, default=5000, help='Number of training samples')
    training_parser.add_argument('--ambiguous-only', action='store_true', help='Only generate ambiguous cases')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate generator
    if args.command == 'basic':
        generate_basic_dataset(args)
    elif args.command == 'edge-cases':
        generate_edge_cases(args)
    elif args.command == 'realistic':
        generate_realistic_dataset(args)
    elif args.command == 'training':
        generate_training_dataset(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

    print("\n" + "="*70)
    print("Generation Complete!")
    print("="*70 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())
