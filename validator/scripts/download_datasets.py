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
    
    # NEW: Titanic-like dataset
    datasets['passenger_survival'] = pd.DataFrame({
        'passenger_id': range(600),
        'age': np.random.normal(35, 15, 600).clip(1, 80),
        'fare': np.random.lognormal(3, 1.5, 600),
        'pclass': np.random.choice([1, 2, 3], 600, p=[0.2, 0.3, 0.5]),
        'sex': np.random.choice(['male', 'female'], 600),
        'embarked': np.random.choice(['C', 'Q', 'S'], 600, p=[0.2, 0.1, 0.7]),
        'survived': np.random.choice([0, 1], 600, p=[0.6, 0.4])
    })
    logger.info("✅ Passenger Survival: 600 rows, 7 columns")
    
    # NEW: House prices dataset
    datasets['real_estate'] = pd.DataFrame({
        'property_id': range(700),
        'price': np.random.lognormal(12, 0.8, 700),
        'sqft': np.random.normal(2000, 800, 700).clip(500, 10000).astype(int),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], 700, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3], 700, p=[0.15, 0.2, 0.35, 0.2, 0.1]),
        'year_built': np.random.normal(1990, 20, 700).clip(1950, 2023).astype(int),
        'property_type': np.random.choice(['House', 'Condo', 'Townhouse'], 700),
        'has_garage': np.random.choice([0, 1], 700, p=[0.3, 0.7]),
        'lot_size': np.random.lognormal(8, 0.5, 700)
    })
    logger.info("✅ Real Estate: 700 rows, 9 columns")
    
    # NEW: Customer churn dataset
    datasets['customer_churn'] = pd.DataFrame({
        'customer_id': range(900),
        'tenure_months': np.random.poisson(24, 900),
        'monthly_charges': np.random.normal(70, 30, 900).clip(20, 200),
        'total_charges': np.random.lognormal(7, 1, 900),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 900),
        'payment_method': np.random.choice(['Electronic', 'Mailed', 'Bank transfer', 'Credit card'], 900),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 900),
        'churn': np.random.choice([0, 1], 900, p=[0.7, 0.3])
    })
    logger.info("✅ Customer Churn: 900 rows, 8 columns")
    
    # NEW: Student performance dataset
    datasets['student_performance'] = pd.DataFrame({
        'student_id': range(500),
        'study_hours': np.random.gamma(3, 2, 500).clip(0, 40),
        'attendance_pct': np.random.beta(9, 1, 500) * 100,
        'previous_score': np.random.normal(75, 15, 500).clip(0, 100),
        'parent_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 500),
        'test_preparation': np.random.choice(['none', 'completed'], 500, p=[0.6, 0.4]),
        'final_score': np.random.normal(70, 20, 500).clip(0, 100)
    })
    logger.info("✅ Student Performance: 500 rows, 7 columns")
    
    # NEW: Credit card transactions
    datasets['credit_transactions'] = pd.DataFrame({
        'transaction_id': range(1200),
        'amount': np.random.lognormal(4, 2, 1200),
        'merchant_category': np.random.choice(['Retail', 'Food', 'Gas', 'Online', 'Travel'], 1200),
        'hour_of_day': np.random.randint(0, 24, 1200),
        'day_of_week': np.random.randint(0, 7, 1200),
        'distance_from_home': np.random.gamma(2, 10, 1200),
        'is_fraud': np.random.choice([0, 1], 1200, p=[0.98, 0.02])
    })
    logger.info("✅ Credit Transactions: 1200 rows, 7 columns")
    
    # NEW: Employee attrition
    datasets['employee_attrition'] = pd.DataFrame({
        'employee_id': range(600),
        'age': np.random.normal(35, 10, 600).clip(22, 65).astype(int),
        'years_at_company': np.random.gamma(2, 3, 600).clip(0, 40).astype(int),
        'salary': np.random.lognormal(11, 0.5, 600),
        'job_satisfaction': np.random.choice([1, 2, 3, 4], 600),
        'work_life_balance': np.random.choice([1, 2, 3, 4], 600),
        'department': np.random.choice(['Sales', 'R&D', 'HR', 'IT'], 600),
        'attrition': np.random.choice([0, 1], 600, p=[0.8, 0.2])
    })
    logger.info("✅ Employee Attrition: 600 rows, 8 columns")
    
    # NEW: Insurance claims
    datasets['insurance_claims'] = pd.DataFrame({
        'claim_id': range(800),
        'age': np.random.normal(40, 15, 800).clip(18, 80).astype(int),
        'bmi': np.random.normal(28, 6, 800).clip(15, 50),
        'children': np.random.poisson(1, 800).clip(0, 5),
        'smoker': np.random.choice(['yes', 'no'], 800, p=[0.2, 0.8]),
        'region': np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], 800),
        'claim_amount': np.random.lognormal(9, 1, 800)
    })
    logger.info("✅ Insurance Claims: 800 rows, 7 columns")
    
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
