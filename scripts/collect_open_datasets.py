"""
Collect open-source datasets for training the Neural Oracle.

Downloads popular datasets from:
- Scikit-learn built-in datasets
- UCI ML Repository (manual download)

Run this script to build a diverse training corpus.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn import datasets
import requests
import argparse


def collect_sklearn_datasets(output_dir: Path) -> list:
    """Collect all scikit-learn built-in datasets."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Regression datasets
    datasets_to_fetch = [
        ('diabetes', datasets.load_diabetes),
        ('california_housing', datasets.fetch_california_housing),
    ]

    # Classification datasets
    datasets_to_fetch += [
        ('iris', datasets.load_iris),
        ('wine', datasets.load_wine),
        ('breast_cancer', datasets.load_breast_cancer),
        ('digits', datasets.load_digits),
    ]

    collected = []

    for name, loader in datasets_to_fetch:
        try:
            print(f"Fetching {name}...")
            data = loader()

            # Convert to DataFrame
            if hasattr(data, 'feature_names'):
                df = pd.DataFrame(data.data, columns=data.feature_names)
            else:
                df = pd.DataFrame(data.data)

            if hasattr(data, 'target'):
                df['target'] = data.target

            # Save
            output_file = output_dir / f"{name}.csv"
            df.to_csv(output_file, index=False)

            collected.append(name)
            print(f"  ✓ Saved to {output_file} ({len(df)} rows, {len(df.columns)} columns)")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    return collected


def collect_uci_datasets(output_dir: Path) -> list:
    """
    Download popular UCI datasets.

    Note: Some UCI datasets require manual download due to licensing.
    This function downloads the publicly available ones.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # UCI datasets with direct download links
    uci_datasets = [
        {
            'name': 'adult',
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            'columns': ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                       'marital_status', 'occupation', 'relationship', 'race', 'sex',
                       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
        },
        {
            'name': 'heart_disease',
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
            'columns': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        }
    ]

    collected = []

    for dataset in uci_datasets:
        try:
            print(f"Downloading {dataset['name']}...")
            response = requests.get(dataset['url'], timeout=30)
            response.raise_for_status()

            # Parse CSV
            from io import StringIO
            df = pd.read_csv(
                StringIO(response.text),
                names=dataset['columns'],
                na_values=['?', '']
            )

            # Save
            output_file = output_dir / f"{dataset['name']}.csv"
            df.to_csv(output_file, index=False)

            collected.append(dataset['name'])
            print(f"  ✓ Downloaded to {output_file} ({len(df)} rows, {len(df.columns)} columns)")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    return collected


def create_synthetic_edge_cases(output_dir: Path) -> list:
    """Create synthetic datasets with specific edge cases."""

    output_dir.mkdir(parents=True, exist_ok=True)

    collected = []

    # 1. High skewness dataset
    try:
        print("Creating high_skewness dataset...")
        import numpy as np

        df = pd.DataFrame({
            'skewed_values': np.concatenate([
                np.random.normal(10, 2, 900),
                np.random.normal(100, 10, 100)
            ]),
            'normal_values': np.random.normal(50, 10, 1000),
            'uniform_values': np.random.uniform(0, 100, 1000)
        })

        output_file = output_dir / "high_skewness.csv"
        df.to_csv(output_file, index=False)
        collected.append('high_skewness')
        print(f"  ✓ Created {output_file}")

    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # 2. High null percentage dataset
    try:
        print("Creating high_nulls dataset...")

        df = pd.DataFrame({
            'sparse_column': [np.random.choice([1, 2, 3, None], p=[0.1, 0.1, 0.1, 0.7]) for _ in range(1000)],
            'medium_nulls': [np.random.choice([10, 20, None], p=[0.4, 0.4, 0.2]) for _ in range(1000)],
            'low_nulls': [np.random.choice(['A', 'B', 'C', None], p=[0.3, 0.3, 0.3, 0.1]) for _ in range(1000)]
        })

        output_file = output_dir / "high_nulls.csv"
        df.to_csv(output_file, index=False)
        collected.append('high_nulls')
        print(f"  ✓ Created {output_file}")

    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # 3. Mixed types dataset
    try:
        print("Creating mixed_types dataset...")

        df = pd.DataFrame({
            'numeric_int': np.random.randint(0, 100, 1000),
            'numeric_float': np.random.random(1000) * 100,
            'categorical_low': np.random.choice(['A', 'B', 'C'], 1000),
            'categorical_high': np.random.choice([f'cat_{i}' for i in range(50)], 1000),
            'binary': np.random.choice([0, 1], 1000),
            'text_like': [f"item_{i}" for i in np.random.randint(0, 100, 1000)]
        })

        output_file = output_dir / "mixed_types.csv"
        df.to_csv(output_file, index=False)
        collected.append('mixed_types')
        print(f"  ✓ Created {output_file}")

    except Exception as e:
        print(f"  ✗ Failed: {e}")

    return collected


def main():
    """Main collection workflow."""

    parser = argparse.ArgumentParser(description="Collect open datasets for training")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/open_datasets',
        help='Output directory for datasets'
    )
    parser.add_argument(
        '--sklearn-only',
        action='store_true',
        help='Only collect scikit-learn datasets'
    )
    parser.add_argument(
        '--uci-only',
        action='store_true',
        help='Only collect UCI datasets'
    )
    parser.add_argument(
        '--synthetic-only',
        action='store_true',
        help='Only create synthetic datasets'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("AURORA Open Dataset Collection")
    print("=" * 70 + "\n")

    # Create datasets directory
    datasets_dir = Path(args.output_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    sklearn_datasets = []
    uci_datasets = []
    synthetic_datasets = []

    # Collect datasets based on flags
    if args.sklearn_only:
        print("\n1. Collecting scikit-learn datasets...")
        sklearn_datasets = collect_sklearn_datasets(datasets_dir / "sklearn")

    elif args.uci_only:
        print(f"\n2. Collecting UCI datasets...")
        uci_datasets = collect_uci_datasets(datasets_dir / "uci")

    elif args.synthetic_only:
        print(f"\n3. Creating synthetic edge case datasets...")
        synthetic_datasets = create_synthetic_edge_cases(datasets_dir / "synthetic")

    else:
        # Collect all
        print("\n1. Collecting scikit-learn datasets...")
        sklearn_datasets = collect_sklearn_datasets(datasets_dir / "sklearn")

        print(f"\n2. Collecting UCI datasets...")
        uci_datasets = collect_uci_datasets(datasets_dir / "uci")

        print(f"\n3. Creating synthetic edge case datasets...")
        synthetic_datasets = create_synthetic_edge_cases(datasets_dir / "synthetic")

    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\nCollected {len(sklearn_datasets) + len(uci_datasets) + len(synthetic_datasets)} datasets:")
    print(f"  • Scikit-learn: {len(sklearn_datasets)}")
    print(f"  • UCI ML: {len(uci_datasets)}")
    print(f"  • Synthetic: {len(synthetic_datasets)}")

    print(f"\nDatasets saved to: {datasets_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Review datasets for quality")
    print("  2. Train neural oracle:")
    print(f"     python scripts/train_hybrid.py --datasets-dir {datasets_dir}")


if __name__ == "__main__":
    main()
