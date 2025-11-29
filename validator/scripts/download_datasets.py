"""
Download real datasets for neural oracle training
Focus: Quality over quantity - diverse, real-world datasets
"""
import os
import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_diabetes,
    fetch_california_housing, fetch_covtype
)
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_sklearn_datasets(output_dir: Path) -> dict:
    """Download built-in sklearn datasets (real-world, high quality)"""
    
    datasets = {}
    
    logger.info("Downloading sklearn datasets...")
    
    # Dataset 1: Iris (classic, multi-class)
    try:
        iris = load_iris(as_frame=True)
        datasets['iris'] = iris['frame']
        logger.info("✅ Iris: 150 rows, 5 columns")
    except Exception as e:
        logger.error(f"❌ Iris failed: {e}")
    
    # Dataset 2: Wine (chemical analysis)
    try:
        wine = load_wine(as_frame=True)
        datasets['wine'] = wine['frame']
        logger.info("✅ Wine: 178 rows, 14 columns")
    except Exception as e:
        logger.error(f"❌ Wine failed: {e}")
    
    # Dataset 3: Breast Cancer (medical)
    try:
        cancer = load_breast_cancer(as_frame=True)
        datasets['breast_cancer'] = cancer['frame']
        logger.info("✅ Breast Cancer: 569 rows, 31 columns")
    except Exception as e:
        logger.error(f"❌ Breast Cancer failed: {e}")
    
    # Dataset 4: Diabetes (medical, regression)
    try:
        diabetes = load_diabetes(as_frame=True)
        datasets['diabetes'] = diabetes['frame']
        logger.info("✅ Diabetes: 442 rows, 11 columns")
    except Exception as e:
        logger.error(f"❌ Diabetes failed: {e}")
    
    # Dataset 5: California Housing (real estate)
    try:
        housing = fetch_california_housing(as_frame=True)
        datasets['california_housing'] = housing['frame']
        logger.info("✅ California Housing: 20640 rows, 9 columns")
    except Exception as e:
        logger.error(f"❌ California Housing failed: {e}")
    
    return datasets


def create_diverse_synthetic_datasets() -> dict:
    """
    Create diverse synthetic datasets that mimic real-world patterns
    Focus: Different domains, data types, quality issues
    """
    
    datasets = {}
    
    logger.info("Creating diverse synthetic datasets...")
    
    # E-commerce dataset
    datasets['ecommerce'] = pd.DataFrame({
        'order_id': range(1000),
        'user_id': np.random.randint(1, 200, 1000),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 1000),
        'price': np.random.lognormal(4, 1.5, 1000),
        'quantity': np.random.poisson(2, 1000),
        'discount_pct': np.random.beta(2, 5, 1000) * 100,
        'rating': np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
        'is_verified': np.random.choice([0, 1], 1000, p=[0.3, 0.7])
    })
    logger.info("✅ E-commerce: 1000 rows, 8 columns")
    
    # Healthcare dataset
    datasets['healthcare'] = pd.DataFrame({
        'patient_id': range(500),
        'age': np.random.normal(50, 15, 500).clip(18, 90).astype(int),
        'bmi': np.random.normal(25, 5, 500).clip(15, 50),
        'blood_pressure': np.random.normal(120, 20, 500).clip(80, 180).astype(int),
        'cholesterol': np.random.choice(['Normal', 'Borderline', 'High'], 500),
        'smoker': np.random.choice([0, 1], 500, p=[0.7, 0.3]),
        'diagnosis': np.random.choice(['Healthy', 'At Risk', 'Diseased'], 500)
    })
    logger.info("✅ Healthcare: 500 rows, 7 columns")
    
    # Finance dataset
    datasets['finance'] = pd.DataFrame({
        'account_id': range(800),
        'balance': np.random.lognormal(8, 2, 800),
        'transaction_count': np.random.poisson(15, 800),
        'credit_score': np.random.normal(700, 100, 800).clip(300, 850).astype(int),
        'account_type': np.random.choice(['Checking', 'Savings', 'Credit'], 800),
        'has_loan': np.random.choice([0, 1], 800, p=[0.6, 0.4])
    })
    logger.info("✅ Finance: 800 rows, 6 columns")
    
    return datasets


def load_local_datasets(data_dir: Path) -> dict:
    """Load local CSV files (like Books.csv)"""
    
    datasets = {}
    
    logger.info(f"Loading local datasets from {data_dir}...")
    
    # Books.csv
    books_path = data_dir / 'Books.csv'
    if books_path.exists():
        try:
            books = pd.read_csv(books_path, low_memory=False, nrows=2000)
            datasets['books'] = books
            logger.info(f"✅ Books: {books.shape[0]} rows, {books.shape[1]} columns")
        except Exception as e:
            logger.error(f"❌ Books.csv failed: {e}")
    
    return datasets


def main():
    """Download all datasets"""
    
    output_dir = Path('validator/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_datasets = {}
    
    # 1. Sklearn datasets (real-world, high quality)
    sklearn_datasets = download_sklearn_datasets(output_dir)
    all_datasets.update(sklearn_datasets)
    
    # 2. Synthetic datasets (diverse domains)
    synthetic_datasets = create_diverse_synthetic_datasets()
    all_datasets.update(synthetic_datasets)
    
    # 3. Local datasets (Books.csv)
    local_datasets = load_local_datasets(Path('datas'))
    all_datasets.update(local_datasets)
    
    # Save all datasets
    logger.info(f"\n{'='*60}")
    logger.info("SAVING DATASETS")
    logger.info(f"{'='*60}")
    
    for name, df in all_datasets.items():
        output_path = output_dir / f"{name}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved {name}: {df.shape}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TOTAL: {len(all_datasets)} datasets")
    logger.info(f"Total columns: {sum(df.shape[1] for df in all_datasets.values())}")
    logger.info(f"Total rows: {sum(df.shape[0] for df in all_datasets.values())}")
    logger.info(f"{'='*60}")
    
    return all_datasets


if __name__ == "__main__":
    datasets = main()
