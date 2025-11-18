"""
Synthetic data generator for training and testing.
Generates edge cases and ambiguous scenarios for NeuralOracle training.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import string

from ..core.actions import PreprocessingAction


@dataclass
class SyntheticColumn:
    """A synthetic column with ground truth label."""
    data: pd.Series
    name: str
    ground_truth_action: PreprocessingAction
    difficulty: str  # 'easy', 'medium', 'hard'
    description: str


class SyntheticDataGenerator:
    """
    Generate synthetic data for training and testing.
    Focuses on edge cases and ambiguous scenarios.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_skewed_numeric(
        self,
        n_samples: int = 1000,
        skewness_target: float = 3.0,
        noise: float = 0.1
    ) -> SyntheticColumn:
        """Generate highly skewed numeric data (good for log transform)."""
        # Use gamma distribution for positive skew
        shape = 2.0
        scale = 1.0 / (skewness_target / 2)
        data = np.random.gamma(shape, scale, n_samples)

        # Add some noise
        data += np.random.normal(0, noise, n_samples)
        data = np.maximum(data, 0.01)  # Ensure positive

        return SyntheticColumn(
            data=pd.Series(data, name="skewed_revenue"),
            name="skewed_revenue",
            ground_truth_action=PreprocessingAction.LOG_TRANSFORM,
            difficulty='easy',
            description=f"Highly skewed positive data (skew ~{skewness_target})"
        )

    def generate_bimodal_distribution(
        self,
        n_samples: int = 1000,
        mode1_mean: float = 10.0,
        mode2_mean: float = 50.0
    ) -> SyntheticColumn:
        """Generate bimodal distribution (good for quantile transform)."""
        # Generate two modes
        n1 = n_samples // 2
        n2 = n_samples - n1

        mode1 = np.random.normal(mode1_mean, 2, n1)
        mode2 = np.random.normal(mode2_mean, 5, n2)

        data = np.concatenate([mode1, mode2])
        np.random.shuffle(data)

        return SyntheticColumn(
            data=pd.Series(data, name="bimodal_metric"),
            name="bimodal_metric",
            ground_truth_action=PreprocessingAction.QUANTILE_TRANSFORM,
            difficulty='medium',
            description="Bimodal distribution with two distinct peaks"
        )

    def generate_outlier_heavy(
        self,
        n_samples: int = 1000,
        outlier_pct: float = 0.15
    ) -> SyntheticColumn:
        """Generate data with many outliers (good for robust scaling/clipping)."""
        # Generate normal data
        n_normal = int(n_samples * (1 - outlier_pct))
        normal_data = np.random.normal(50, 10, n_normal)

        # Add outliers
        n_outliers = n_samples - n_normal
        outliers = np.random.uniform(150, 300, n_outliers)

        data = np.concatenate([normal_data, outliers])
        np.random.shuffle(data)

        action = PreprocessingAction.CLIP_OUTLIERS if outlier_pct > 0.1 else PreprocessingAction.WINSORIZE

        return SyntheticColumn(
            data=pd.Series(data, name="outlier_heavy_values"),
            name="outlier_heavy_values",
            ground_truth_action=action,
            difficulty='easy',
            description=f"Data with {outlier_pct:.0%} outliers"
        )

    def generate_constant_column(
        self,
        n_samples: int = 1000,
        value: Any = 42
    ) -> SyntheticColumn:
        """Generate constant column (should be dropped)."""
        data = [value] * n_samples

        return SyntheticColumn(
            data=pd.Series(data, name="constant_column"),
            name="constant_column",
            ground_truth_action=PreprocessingAction.DROP_COLUMN,
            difficulty='easy',
            description="Constant column with single value"
        )

    def generate_mostly_null(
        self,
        n_samples: int = 1000,
        null_pct: float = 0.7
    ) -> SyntheticColumn:
        """Generate column with high null percentage (should be dropped)."""
        n_nulls = int(n_samples * null_pct)
        n_values = n_samples - n_nulls

        data = [None] * n_nulls + list(np.random.normal(0, 1, n_values))
        random.shuffle(data)

        return SyntheticColumn(
            data=pd.Series(data, name="mostly_null"),
            name="mostly_null",
            ground_truth_action=PreprocessingAction.DROP_COLUMN,
            difficulty='easy',
            description=f"Column with {null_pct:.0%} null values"
        )

    def generate_high_cardinality_categorical(
        self,
        n_samples: int = 1000,
        n_categories: int = 200
    ) -> SyntheticColumn:
        """Generate high-cardinality categorical (good for target encoding)."""
        categories = [f"CAT_{i:04d}" for i in range(n_categories)]
        data = np.random.choice(categories, n_samples)

        return SyntheticColumn(
            data=pd.Series(data, name="user_id_category"),
            name="user_id_category",
            ground_truth_action=PreprocessingAction.TARGET_ENCODE,
            difficulty='medium',
            description=f"High-cardinality categorical with {n_categories} categories"
        )

    def generate_low_cardinality_categorical(
        self,
        n_samples: int = 1000,
        n_categories: int = 5
    ) -> SyntheticColumn:
        """Generate low-cardinality categorical (good for one-hot encoding)."""
        categories = [f"Type_{chr(65+i)}" for i in range(n_categories)]  # Type_A, Type_B, etc.
        data = np.random.choice(categories, n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])

        return SyntheticColumn(
            data=pd.Series(data, name="category_type"),
            name="category_type",
            ground_truth_action=PreprocessingAction.ONEHOT_ENCODE,
            difficulty='easy',
            description=f"Low-cardinality categorical with {n_categories} categories"
        )

    def generate_date_strings(
        self,
        n_samples: int = 1000,
        format_type: str = "iso"
    ) -> SyntheticColumn:
        """Generate date strings (should be parsed as datetime)."""
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_samples)]

        if format_type == "iso":
            data = [d.strftime("%Y-%m-%d") for d in dates]
        elif format_type == "us":
            data = [d.strftime("%m/%d/%Y") for d in dates]
        else:
            data = [d.strftime("%d-%m-%Y") for d in dates]

        return SyntheticColumn(
            data=pd.Series(data, name="date_column"),
            name="date_column",
            ground_truth_action=PreprocessingAction.PARSE_DATETIME,
            difficulty='easy',
            description=f"Date strings in {format_type} format"
        )

    def generate_currency_strings(
        self,
        n_samples: int = 1000
    ) -> SyntheticColumn:
        """Generate currency strings (should be normalized)."""
        values = np.random.uniform(10, 10000, n_samples)
        data = [f"${v:,.2f}" for v in values]

        return SyntheticColumn(
            data=pd.Series(data, name="price"),
            name="price",
            ground_truth_action=PreprocessingAction.CURRENCY_NORMALIZE,
            difficulty='easy',
            description="Currency strings with $ symbols"
        )

    def generate_percentage_strings(
        self,
        n_samples: int = 1000
    ) -> SyntheticColumn:
        """Generate percentage strings (should be converted to decimal)."""
        values = np.random.uniform(0, 100, n_samples)
        data = [f"{v:.1f}%" for v in values]

        return SyntheticColumn(
            data=pd.Series(data, name="completion_rate"),
            name="completion_rate",
            ground_truth_action=PreprocessingAction.PERCENTAGE_TO_DECIMAL,
            difficulty='easy',
            description="Percentage strings with % symbol"
        )

    def generate_mixed_types(
        self,
        n_samples: int = 1000,
        contamination: float = 0.2
    ) -> SyntheticColumn:
        """Generate mostly numeric with some string contamination (edge case)."""
        n_numeric = int(n_samples * (1 - contamination))
        n_strings = n_samples - n_numeric

        numeric_data = [str(int(x)) for x in np.random.randint(0, 100, n_numeric)]
        string_data = [''.join(random.choices(string.ascii_letters, k=5)) for _ in range(n_strings)]

        data = numeric_data + string_data
        random.shuffle(data)

        return SyntheticColumn(
            data=pd.Series(data, name="mixed_column"),
            name="mixed_column",
            ground_truth_action=PreprocessingAction.PARSE_NUMERIC,
            difficulty='hard',
            description=f"Mixed types: {contamination:.0%} non-numeric contamination"
        )

    def generate_normal_scaled(
        self,
        n_samples: int = 1000
    ) -> SyntheticColumn:
        """Generate normally distributed data (good for standard scaling)."""
        data = np.random.normal(50, 15, n_samples)

        return SyntheticColumn(
            data=pd.Series(data, name="normal_feature"),
            name="normal_feature",
            ground_truth_action=PreprocessingAction.STANDARD_SCALE,
            difficulty='easy',
            description="Normally distributed numeric data"
        )

    def generate_boolean_variants(
        self,
        n_samples: int = 1000,
        variant: str = "tf"
    ) -> SyntheticColumn:
        """Generate boolean-like strings (should be parsed as boolean)."""
        if variant == "tf":
            data = np.random.choice(["True", "False"], n_samples)
        elif variant == "yn":
            data = np.random.choice(["Yes", "No"], n_samples)
        else:  # 01
            data = np.random.choice(["0", "1"], n_samples)

        return SyntheticColumn(
            data=pd.Series(data, name="boolean_flag"),
            name="boolean_flag",
            ground_truth_action=PreprocessingAction.PARSE_BOOLEAN,
            difficulty='easy',
            description=f"Boolean strings ({variant} variant)"
        )

    def generate_edge_case_dataset(
        self,
        n_samples_per_case: int = 1000,
        include_hard_cases: bool = True
    ) -> pd.DataFrame:
        """
        Generate a complete dataset with various edge cases.

        Args:
            n_samples_per_case: Number of samples for each case
            include_hard_cases: Whether to include difficult edge cases

        Returns:
            DataFrame with multiple columns covering different scenarios
        """
        columns = []

        # Easy cases
        columns.append(self.generate_skewed_numeric(n_samples_per_case))
        columns.append(self.generate_constant_column(n_samples_per_case))
        columns.append(self.generate_mostly_null(n_samples_per_case))
        columns.append(self.generate_low_cardinality_categorical(n_samples_per_case))
        columns.append(self.generate_normal_scaled(n_samples_per_case))
        columns.append(self.generate_currency_strings(n_samples_per_case))
        columns.append(self.generate_percentage_strings(n_samples_per_case))
        columns.append(self.generate_date_strings(n_samples_per_case))
        columns.append(self.generate_boolean_variants(n_samples_per_case, "tf"))

        # Medium difficulty
        columns.append(self.generate_bimodal_distribution(n_samples_per_case))
        columns.append(self.generate_high_cardinality_categorical(n_samples_per_case))
        columns.append(self.generate_outlier_heavy(n_samples_per_case, 0.12))

        # Hard cases
        if include_hard_cases:
            columns.append(self.generate_mixed_types(n_samples_per_case, 0.15))
            columns.append(self.generate_outlier_heavy(n_samples_per_case, 0.08))  # Borderline

        # Create DataFrame
        df = pd.DataFrame({col.name: col.data for col in columns})

        # Store ground truth
        df.attrs['ground_truth'] = {
            col.name: col.ground_truth_action for col in columns
        }
        df.attrs['difficulties'] = {
            col.name: col.difficulty for col in columns
        }
        df.attrs['descriptions'] = {
            col.name: col.description for col in columns
        }

        return df

    def generate_training_data(
        self,
        n_samples: int = 10000,
        ambiguous_only: bool = True
    ) -> Tuple[List[pd.Series], List[PreprocessingAction], List[str]]:
        """
        Generate training data for NeuralOracle.

        Args:
            n_samples: Total number of samples to generate
            ambiguous_only: Only generate ambiguous/edge cases

        Returns:
            (columns, labels, difficulties) tuple
        """
        columns = []
        labels = []
        difficulties = []

        generators = {
            'easy': [
                (self.generate_skewed_numeric, {}),
                (self.generate_constant_column, {}),
                (self.generate_low_cardinality_categorical, {}),
                (self.generate_normal_scaled, {}),
            ],
            'medium': [
                (self.generate_bimodal_distribution, {}),
                (self.generate_high_cardinality_categorical, {'n_categories': 150}),
                (self.generate_outlier_heavy, {'outlier_pct': 0.12}),
            ],
            'hard': [
                (self.generate_mixed_types, {'contamination': 0.15}),
                (self.generate_outlier_heavy, {'outlier_pct': 0.08}),
            ]
        }

        # Choose distribution based on mode
        if ambiguous_only:
            distribution = {'medium': 0.6, 'hard': 0.4}
        else:
            distribution = {'easy': 0.4, 'medium': 0.4, 'hard': 0.2}

        generated = 0
        while generated < n_samples:
            # Choose difficulty level
            difficulty = np.random.choice(
                list(distribution.keys()),
                p=list(distribution.values())
            )

            # Choose random generator from that difficulty
            generator_fn, kwargs = random.choice(generators[difficulty])

            # Generate column
            col = generator_fn(**kwargs)

            columns.append(col.data)
            labels.append(col.ground_truth_action)
            difficulties.append(difficulty)

            generated += 1

        return columns, labels, difficulties


def main():
    """Generate and save sample datasets."""
    generator = SyntheticDataGenerator(seed=42)

    # Generate edge case dataset
    print("Generating edge case dataset...")
    df = generator.generate_edge_case_dataset(n_samples_per_case=1000)

    # Save to CSV
    output_path = Path(__file__).parent.parent.parent / "data" / "synthetic" / "edge_cases.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved edge case dataset to {output_path}")

    # Print ground truth
    print("\nGround truth labels:")
    for col_name, action in df.attrs['ground_truth'].items():
        difficulty = df.attrs['difficulties'][col_name]
        print(f"  {col_name}: {action.value} [{difficulty}]")


if __name__ == "__main__":
    main()
