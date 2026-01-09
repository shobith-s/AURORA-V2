#!/usr/bin/env python3
"""
AURORA V2 - Adaptive Learning Demo
Creates sample corrections to demonstrate the adaptive learning system
"""

import sys
sys.path.insert(0, '/home/shobiths/Desktop/AURORA-V2')

from src.core.preprocessor import IntelligentPreprocessor
import pandas as pd
import numpy as np

def create_sample_corrections():
    """Create sample corrections for demonstration"""
    
    print('='*80)
    print('AURORA V2 - ADAPTIVE LEARNING DEMO')
    print('='*80)
    
    # Initialize preprocessor with learning enabled
    print('\n📝 Initializing preprocessor with adaptive learning...')
    preprocessor = IntelligentPreprocessor(
        enable_learning=True,
        use_neural_oracle=False
    )
    
    print('✅ Preprocessor initialized')
    
    # Create sample data patterns
    print('\n📊 Creating sample data patterns...')
    
    # Pattern 1: Highly skewed revenue data (5 corrections)
    print('\n1️⃣  Revenue Pattern (Highly Skewed → Log Transform)')
    for i in range(5):
        revenue = pd.Series(
            np.random.lognormal(mean=10, sigma=2, size=100),
            name=f'revenue_{i}'
        )
        result = preprocessor.preprocess_column(revenue, f'revenue_{i}', apply_action=False)
        correction = preprocessor.submit_correction(
            column_data=revenue,
            column_name=f'revenue_{i}',
            wrong_action=result.action.value,
            correct_action='log_transform',
            confidence=result.confidence
        )
        print(f'   ✓ Correction {i+1}: {result.action.value} → log_transform')
    
    # Pattern 2: Priority/Rating columns (5 corrections)
    print('\n2️⃣  Priority Pattern (Categorical → Ordinal Encode)')
    for i in range(5):
        priority = pd.Series(
            np.random.choice(['Low', 'Medium', 'High', 'Critical'], size=100),
            name=f'priority_{i}'
        )
        result = preprocessor.preprocess_column(priority, f'priority_{i}', apply_action=False)
        correction = preprocessor.submit_correction(
            column_data=priority,
            column_name=f'priority_{i}',
            wrong_action=result.action.value,
            correct_action='ordinal_encode',
            confidence=result.confidence
        )
        print(f'   ✓ Correction {i+1}: {result.action.value} → ordinal_encode')
    
    # Pattern 3: Price data (3 corrections)
    print('\n3️⃣  Price Pattern (Skewed → Log1p Transform)')
    for i in range(3):
        price = pd.Series(
            np.random.lognormal(mean=4, sigma=1.5, size=100),
            name=f'price_{i}'
        )
        result = preprocessor.preprocess_column(price, f'price_{i}', apply_action=False)
        correction = preprocessor.submit_correction(
            column_data=price,
            column_name=f'price_{i}',
            wrong_action=result.action.value,
            correct_action='log1p_transform',
            confidence=result.confidence
        )
        print(f'   ✓ Correction {i+1}: {result.action.value} → log1p_transform')
    
    print('\n' + '='*80)
    print('✅ DEMO COMPLETE - 13 SAMPLE CORRECTIONS CREATED')
    print('='*80)
    
    print('\n📊 Summary:')
    print('   - Revenue pattern: 5 corrections (need 5 more for rule generation)')
    print('   - Priority pattern: 5 corrections (need 5 more for rule generation)')
    print('   - Price pattern: 3 corrections (need 7 more for rule generation)')
    
    print('\n💡 To view corrections, run:')
    print('   python3 view_database.py')
    print('\n')

if __name__ == '__main__':
    create_sample_corrections()
