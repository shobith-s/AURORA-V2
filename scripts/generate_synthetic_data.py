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

from src.data.generator import SyntheticDataGenerator


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

    generator = SyntheticDataGenerator(seed=args.seed)

    # Generate multiple columns with different characteristics
    columns_dict = {}

    # Generate numeric columns
    for i in range(args.numeric):
        if args.add_outliers and i % 3 == 0:
            col = generator.generate_outlier_heavy(args.rows, outlier_ratio=0.1)
        elif i % 3 == 1:
            col = generator.generate_skewed_numeric(args.rows, skewness_target=2.0)
        else:
            col = generator.generate_normal_scaled(args.rows)
        columns_dict[col.name or f'numeric_{i}'] = col.data

    # Generate categorical columns
    for i in range(args.categorical):
        if i % 2 == 0:
            col = generator.generate_low_cardinality_categorical(args.rows)
        else:
            col = generator.generate_high_cardinality_categorical(args.rows, n_categories=50)
        columns_dict[col.name or f'categorical_{i}'] = col.data

    data = pd.DataFrame(columns_dict)

    # Add missing values if requested
    if args.missing_rate > 0:
        mask = np.random.random(data.shape) < args.missing_rate
        data = data.mask(mask)

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

    # Use the built-in edge case dataset generator
    print("\nGenerating comprehensive edge case dataset...")
    data = generator.generate_edge_case_dataset(
        n_samples_per_case=args.rows,
        include_hard_cases=True
    )

    # Print ground truth if available
    if 'ground_truth' in data.attrs:
        print("\nGenerated columns with ground truth labels:")
        for col_name, action in data.attrs['ground_truth'].items():
            difficulty = data.attrs['difficulties'].get(col_name, 'unknown')
            description = data.attrs['descriptions'].get(col_name, '')
            print(f"  {col_name}: {action.value} [{difficulty}]")
            if description:
                print(f"    → {description}")

    metadata = {
        'type': 'edge_cases',
        'generated_at': datetime.now().isoformat(),
        'num_rows': len(data),
        'num_columns': len(data.columns),
        'seed': args.seed,
        'ground_truth': {k: v.value for k, v in data.attrs.get('ground_truth', {}).items()},
        'difficulties': data.attrs.get('difficulties', {})
    }

    save_dataset(data, Path(args.output), metadata)
    print(f"\nGenerated {len(data.columns)} edge case columns with {len(data)} rows")


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
    # Generate age with normal distribution
    age_data = np.random.normal(35, 12, args.rows)
    age_data = np.clip(age_data, 18, 100)  # Reasonable age range
    mask = np.random.random(args.rows) < 0.02
    age_series = pd.Series(age_data)
    age_series[mask] = np.nan
    data['age'] = age_series

    # Generate categorical columns
    data['gender'] = pd.Series(np.random.choice(['M', 'F', 'Other'], args.rows))
    countries = [f'Country_{i}' for i in range(50)]
    data['country'] = pd.Series(np.random.choice(countries, args.rows))

    # Transaction data
    print("Generating transaction columns...")
    data['purchase_amount'] = generator.generate_skewed_numeric(args.rows, skewness_target=2.0).data
    data['transaction_count'] = pd.Series(np.abs(np.random.normal(15, 10, args.rows)).astype(int))
    data['last_purchase_days'] = pd.Series(np.abs(np.random.normal(30, 60, args.rows)))

    # Engagement metrics
    print("Generating engagement metrics...")
    data['login_count'] = pd.Series(np.abs(np.random.normal(20, 15, args.rows)).astype(int))
    data['session_duration_min'] = generator.generate_skewed_numeric(args.rows, skewness_target=1.5).data
    data['pages_viewed'] = pd.Series(np.abs(np.random.normal(50, 30, args.rows)).astype(int))

    # Features with issues
    print("Adding realistic data quality issues...")
    data['incomplete_field'] = generator.generate_mostly_null(args.rows, null_fraction=0.4).data
    data['sparse_category'] = generator.generate_high_cardinality_categorical(args.rows, n_categories=200).data
    data['outlier_metric'] = generator.generate_outlier_heavy(args.rows, outlier_ratio=0.1).data

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
    edge_parser.add_argument('--rows', type=int, default=1000, help='Number of rows per case')

    # Realistic dataset
    realistic_parser = subparsers.add_parser('realistic', help='Generate realistic dataset')
    realistic_parser.add_argument('--rows', type=int, default=5000, help='Number of rows')

    # Training dataset
    training_parser = subparsers.add_parser('training', help='Generate training dataset for neural oracle')
    training_parser.add_argument('--samples', type=int, default=5000, help='Number of training samples')
    training_parser.add_argument('--ambiguous-only', action='store_true', help='Only generate ambiguous cases')

    args = parser.parse_args()

    # If no command provided, generate a sample dataset by default
    if not args.command:
        print("\nNo command specified. Generating sample dataset...")
        print("(Use --help to see all available commands)\n")

        # Create a mock args object for edge-cases with default values
        class SampleArgs:
            rows = 1000
            seed = 42
            output = './data/sample_dataset.csv'

        sample_args = SampleArgs()
        generate_edge_cases(sample_args)
        return 0

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
