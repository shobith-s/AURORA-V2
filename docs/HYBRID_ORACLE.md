# Hybrid Preprocessing Oracle

The Hybrid Preprocessing Oracle is an advanced ML-based system for automatic preprocessing recommendations that combines machine learning predictions with rule-based fallbacks.

## Overview

The hybrid oracle uses:
- **XGBoost + LightGBM ensemble** for ML predictions (74.7% accuracy)
- **Rule-based fallbacks** for edge cases the ML model hasn't seen enough examples of
- **40 meta-features** for comprehensive column analysis
- **10 preprocessing actions** learned from diverse datasets

## Architecture

```
Column Data
    ↓
MetaFeatureExtractor (40 features)
    ↓
HybridPreprocessingOracle
    ├─→ ML Prediction (XGBoost + LightGBM)
    └─→ Rule-Based Fallback (8 rules)
    ↓
Preprocessing Action + Confidence
```

## Features Extracted (40 total)

### Basic Stats (5 features)
- `missing_ratio`: Percentage of missing values
- `unique_ratio`: Ratio of unique values to total
- `unique_count_norm`: Normalized unique count (log scale)
- `row_count_norm`: Normalized row count (log scale)
- `is_complete`: Binary flag for no missing values

### Type Indicators (5 features)
- `is_numeric`: Is numeric type
- `is_bool`: Is boolean type
- `is_datetime`: Is datetime type
- `is_object`: Is object/string type
- `is_categorical`: Is categorical type

### Numeric Stats (15 features)
- `mean`, `std`, `min`, `max`, `median`: Basic statistics
- `skewness`, `kurtosis`: Distribution shape
- `outlier_ratio`: Percentage of outliers (IQR method)
- `positive_ratio`, `negative_ratio`, `zero_ratio`: Value ratios
- `can_log`: Can apply log transform (all positive)
- `can_sqrt`: Can apply sqrt transform (all non-negative)
- `has_range`: Has non-zero range
- `has_variance`: Has non-zero variance

### Categorical Stats (10 features)
- `avg_length`, `max_length`, `min_length`, `length_std`: String length stats
- `cardinality_low`, `cardinality_medium`, `cardinality_high`, `cardinality_unique`: Cardinality buckets
- `entropy`: Information entropy
- `mode_frequency`: Frequency of most common value

### Name Features (5 features)
- `has_id`: Contains 'id', 'key', 'uuid', 'index' in name
- `has_name`: Contains 'name', 'title', 'label' in name
- `has_date`: Contains 'date', 'time', 'timestamp' in name
- `has_price`: Contains 'price', 'cost', 'amount', 'revenue' in name
- `has_count`: Contains 'count', 'number', 'quantity' in name

## Supported Actions (10)

1. `clip_outliers`: Clip outliers to IQR bounds
2. `drop_column`: Remove column entirely
3. `frequency_encode`: Encode by frequency
4. `keep_as_is`: No transformation needed
5. `log1p_transform`: Log(1+x) transform (for data with zeros)
6. `log_transform`: Log transform (for positive data)
7. `minmax_scale`: Scale to [0, 1] range
8. `robust_scale`: Scale using median and IQR
9. `sqrt_transform`: Square root transform
10. `standard_scale`: Standardize to mean=0, std=1

## Rule-Based Fallbacks

The oracle uses these rules when ML confidence < 0.6 or rule confidence > 0.85:

### Rule 1: Constant Columns
- **Condition**: No variance (std = 0)
- **Action**: `drop_column`
- **Confidence**: 95%

### Rule 2: ID-like Columns
- **Condition**: Name contains 'id' AND unique_ratio > 95%
- **Action**: `drop_column`
- **Confidence**: 90%

### Rule 3: High Cardinality Categorical
- **Condition**: Categorical with unique_ratio > 70%
- **Action**: `drop_column` (if >95%) or `frequency_encode` (70-95%)
- **Confidence**: 80-85%

### Rule 4: Highly Skewed Data
- **Condition**: Numeric, positive, |skewness| > 2.0
- **Action**: `log1p_transform` (if zeros present) or `log_transform`
- **Confidence**: 87-88%

### Rule 5: Many Outliers
- **Condition**: outlier_ratio > 15%
- **Action**: `clip_outliers`
- **Confidence**: 83%

### Rule 6: Large Range Data
- **Condition**: Numeric with std > 100 or range > 1000
- **Action**: `standard_scale`
- **Confidence**: 82%

### Rule 7: Boolean-like Data
- **Condition**: Only 2 unique values or very low cardinality
- **Action**: `keep_as_is`
- **Confidence**: 75%

### Rule 8: High Missing Ratio
- **Condition**: missing_ratio > 60%
- **Action**: `drop_column`
- **Confidence**: 80%

## Usage

### Basic Usage

```python
from src.neural.hybrid_oracle import HybridPreprocessingOracle
from src.features.meta_extractor import MetaFeatureExtractor
import pandas as pd

# Create oracle (without ML models, uses rules only)
oracle = HybridPreprocessingOracle()
extractor = MetaFeatureExtractor()

# Analyze a single column
column = pd.Series([1, 2, 3, 100, 200, 300])
features = extractor.extract(column, 'revenue')
prediction = oracle.predict_column(column, 'revenue')

print(f"Action: {prediction.action}")
print(f"Confidence: {prediction.confidence:.2%}")
print(f"Source: {prediction.source}")  # 'ml' or 'rule'
print(f"Reason: {prediction.reason}")
```

### Analyzing a DataFrame

```python
# Analyze all columns in a DataFrame
df = pd.DataFrame({
    'id': range(100),
    'revenue': [100, 200, 300, ...],
    'category': ['A', 'B', 'C', ...]
})

results = oracle.predict_dataframe(df, target_column='revenue')
print(results)
```

Output:
```
   column_name         action  confidence  source                       reason
0           id    drop_column        0.90    rule  ID-like column with high...
1     category  keep_as_is           0.75    rule  Boolean or binary data
```

### Using with Trained Models

When you have trained XGBoost and LightGBM models:

```python
import pickle

# Load trained model package
with open('models/aurora_preprocessing_oracle_20251201.pkl', 'rb') as f:
    model_package = pickle.load(f)

# Create oracle with trained models
oracle = HybridPreprocessingOracle(
    xgb_model=model_package['xgb_model'],
    lgb_model=model_package['lgb_model'],
    label_encoder=model_package['label_encoder'],
    feature_extractor=model_package['feature_extractor'],
    config=model_package.get('config', {})
)

# Now predictions use both ML and rules
prediction = oracle.predict_column(column, 'column_name')
```

### Via NeuralOracle (Unified Interface)

The `NeuralOracle` class automatically detects and uses hybrid models:

```python
from src.neural.oracle import NeuralOracle

# Automatically loads the best available model
# Priority: hybrid → ensemble → legacy
oracle = NeuralOracle()

# Or specify path explicitly
oracle = NeuralOracle(model_path='models/aurora_preprocessing_oracle_20251201.pkl')

# Check if hybrid model is loaded
if oracle.is_hybrid:
    print("Hybrid model loaded successfully!")
```

## Model Package Format

When training a new hybrid model, save it in this format:

```python
model_package = {
    'hybrid_model': HybridPreprocessingOracle,  # Optional: pre-configured oracle
    'xgb_model': XGBClassifier,                 # Trained XGBoost model
    'lgb_model': LGBMClassifier,                # Trained LightGBM model
    'label_encoder': LabelEncoder,              # Label encoder for actions
    'feature_extractor': MetaFeatureExtractor,  # Feature extractor instance
    'config': {                                  # Training configuration
        'ml_confidence_threshold': 0.6,
        'rule_confidence_threshold': 0.85,
        # ... other config
    },
    'removed_classes': [...],                   # Classes removed during training
    'metadata': {                               # Training metadata
        'train_date': '2025-12-01',
        'accuracy': 0.747,
        'n_datasets': 40,
        # ... other metadata
    }
}

# Save to models directory
import pickle
with open('models/aurora_preprocessing_oracle_20251201_120000.pkl', 'wb') as f:
    pickle.dump(model_package, f)
```

## Model File Naming Convention

Use this pattern for hybrid model files:
```
aurora_preprocessing_oracle_YYYYMMDD_HHMMSS.pkl
```

Examples:
- `aurora_preprocessing_oracle_20251201_120000.pkl`
- `aurora_preprocessing_oracle_20251215_093045.pkl`

The system will automatically pick the most recent hybrid model.

## Backward Compatibility

The implementation maintains full backward compatibility:

- **Old models** (neural_oracle_v2_improved_*.pkl) continue to work
- **Legacy API** remains unchanged
- **MinimalFeatures** (20 features) still supported for old models
- **MetaFeatures** (40 features) used only for new hybrid models

## Performance

- **ML Accuracy**: 74.7% on test data
- **Inference Time**: ~5ms per column (with models)
- **Rule Inference**: <1ms per column (without models)
- **Memory**: ~10MB for hybrid model package

## Training a New Model

See the Colab notebook at `colab/meta_learning_training.ipynb` for:
1. Collecting diverse datasets from OpenML
2. Generating training labels
3. Training XGBoost + LightGBM ensemble
4. Exporting model package

Expected training time: ~30-45 minutes

## Testing

Run the comprehensive test suite:

```bash
# Test hybrid oracle only
pytest tests/test_hybrid_oracle.py -v

# Test integration with existing code
pytest tests/test_integration.py -v
```

All 21 hybrid oracle tests should pass ✅

## Troubleshooting

### Model not found
If you see "No pre-trained model loaded":
- Check that model file exists in `models/` directory
- Verify filename matches pattern `aurora_preprocessing_oracle_*.pkl`
- Ensure model package has required keys

### Low confidence predictions
If predictions have low confidence:
- Check if column type matches model training data
- Review rule-based fallbacks for edge cases
- Consider retraining with more diverse datasets

### Import errors
If you get import errors:
- Ensure `xgboost` and `lightgbm` are installed
- Run `pip install -r requirements.txt`

## Future Enhancements

Potential improvements:
- [ ] Add more sophisticated rules (e.g., datetime detection)
- [ ] Support custom rule definitions
- [ ] Adaptive confidence thresholds based on column type
- [ ] Online learning for model updates
- [ ] Explainability for rule-based decisions
- [ ] Performance profiling and optimization

## References

- Original Neural Oracle: `src/neural/oracle.py`
- Feature Extractors: `src/features/`
- Training Script: `colab/meta_learning_training.ipynb`
- Tests: `tests/test_hybrid_oracle.py`
