# Meta-Learning Guide for Neural Oracle Training

This guide explains how to train the Neural Oracle (Layer 2) using meta-learning with the Google Colab notebook.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [How Meta-Learning Works](#how-meta-learning-works)
4. [Using the Colab Notebook](#using-the-colab-notebook)
5. [Feature Engineering](#feature-engineering)
6. [Deploying the Trained Model](#deploying-the-trained-model)
7. [Troubleshooting](#troubleshooting)
8. [Benchmarking](#benchmarking)

---

## Overview

### What is the Neural Oracle?

The Neural Oracle is Layer 2 of AURORA's three-layer architecture:

```
Layer 1: Symbolic Engine (180+ rules) → Primary decision maker
Layer 2: Neural Oracle (XGBoost + LightGBM) → Handles ambiguous cases
Layer 3: Adaptive Learning → Learns from user corrections
```

When the Symbolic Engine's confidence is below 0.75, the Neural Oracle provides preprocessing recommendations based on column features.

### Why Meta-Learning?

Meta-learning trains the Neural Oracle by:
1. Collecting diverse datasets from OpenML
2. Trying ALL preprocessing actions on each column
3. Measuring actual ML model performance
4. Learning which actions work best for which column types

This produces high-quality training data with performance-based ground truth labels.

---

## Quick Start

### Option 1: Google Colab (Recommended)

1. Open the notebook: [colab/meta_learning_training.ipynb](../colab/meta_learning_training.ipynb)
2. Click "Runtime" → "Run all"
3. Wait ~30-45 minutes
4. Download the trained model file
5. Copy to `models/` directory

### Option 2: Local Training

```bash
# Clone repository
git clone https://github.com/shobith-s/AURORA-V2.git
cd AURORA-V2

# Install dependencies
pip install -r requirements.txt

# Run training script
python scripts/curriculum_meta_learner.py
```

---

## How Meta-Learning Works

### Curriculum Learning Stages

The training follows a curriculum that progresses through data types:

#### Stage 1: Deterministic Cases
- **All null columns** → DROP_COLUMN
- **Constant columns** → DROP_COLUMN  
- **ID columns** (all unique) → DROP_COLUMN
- **Datetime columns** → PARSE_DATETIME
- **Boolean columns** → PARSE_BOOLEAN

These are handled by rules, not ML, with 100% confidence.

#### Stage 2: Numeric Columns
Learn optimal preprocessing for numeric data:
- `keep_as_is` - Already suitable
- `standard_scale` - Standardization to mean=0, std=1
- `minmax_scale` - Scale to [0, 1]
- `log_transform` - For skewed distributions
- `log1p_transform` - For skewed with zeros
- `robust_scale` - For outlier-heavy data
- `clip_outliers` - Cap extreme values

#### Stage 3: Categorical Columns
Learn with numeric context already processed:
- `onehot_encode` - Low cardinality (≤10 unique)
- `label_encode` - Medium cardinality
- `ordinal_encode` - Ordered categories
- `frequency_encode` - High cardinality
- `drop_column` - Too noisy/uninformative

#### Stage 4: Text Columns
Learn with full context:
- `drop_column` - High cardinality text (names, descriptions)
- `label_encode` - Can be treated as categorical
- `keep_as_is` - Already processed or embeddings

### Performance-Based Ground Truth

For each column, we:
1. Try each candidate action
2. Run 3-fold cross-validation with a classifier
3. Measure accuracy for each action
4. Select the action with highest accuracy as ground truth

```python
# Pseudocode for ground truth generation
for column in dataset:
    scores = {}
    for action in candidate_actions:
        X_transformed = apply_action(column, action)
        score = cross_val_score(model, X_transformed, y)
        scores[action] = score.mean()
    
    best_action = max(scores, key=scores.get)
    confidence = score_gap_to_confidence(scores)
    
    training_sample = {
        'features': extract_features(column),
        'label': best_action,
        'confidence': confidence
    }
```

### Confidence Calculation

Confidence is based on the performance gap between best and second-best actions:

```
confidence = 0.5 + (gap × 5)
confidence = min(confidence, 1.0)
```

Large gaps → high confidence (clear winner)
Small gaps → low confidence (multiple good options)

---

## Using the Colab Notebook

### Cell Structure

| Cell | Purpose | Time |
|------|---------|------|
| 1-4 | Setup & Configuration | 2 min |
| 5 | Dataset Collection (40 OpenML) | 5 min |
| 6-9 | Curriculum Meta-Learning | 25 min |
| 10-11 | Train XGBoost + LightGBM | 5 min |
| 12 | Evaluate on Test Set | 1 min |
| 13-14 | Export & Download | 1 min |

### Configuration Options

```python
# In Cell 4
config = {
    'n_datasets': 40,        # Number of OpenML datasets
    'cv_folds': 3,           # Cross-validation folds
    'random_state': 42,      # For reproducibility
    'max_samples': 5000,     # Max rows per dataset
    'min_confidence': 0.5,   # Filter low-confidence samples
}
```

### Expected Output

```
Training Summary:
================
Total samples: 847
Training accuracy: 93.2%
Test accuracy: 90.5%
Labels: ['drop_column', 'keep_as_is', 'label_encode', ...]

Model saved to: neural_oracle_meta_v3_20241201_120000.pkl
Size: 4.8 MB
```

---

## Feature Engineering

### Feature Categories (62 total)

#### Basic Features (10)
From MinimalFeatureExtractor:
- `null_percentage` - Ratio of null values
- `unique_ratio` - Unique values / total values
- `skewness` - Distribution asymmetry
- `outlier_percentage` - Percentage of IQR outliers
- `entropy` - Information content
- `pattern_complexity` - Detected patterns count
- `multimodality_score` - Bimodality indicator
- `cardinality_bucket` - 0=low, 1=med, 2=high, 3=unique
- `detected_dtype` - 0=num, 1=cat, 2=text, 3=temp
- `column_name_signal` - Keyword-based signal

#### Column Name Semantics (20)
Name pattern detection:
- `name_contains_id` - "id", "key", "uuid" in name
- `name_contains_date` - "date", "created" in name
- `name_contains_price` - "price", "cost" in name
- `name_contains_rating` - "rating", "score" in name
- `name_contains_count` - "count", "num", "qty" in name
- ... and 15 more

#### Domain Patterns (15)
Value pattern detection:
- `looks_like_year` - Values in 1900-2100
- `looks_like_rating` - Values in 0-5 or 0-10
- `looks_like_percentage` - Values in 0-100
- `looks_like_currency` - Has $ or € symbols
- `looks_like_boolean` - True/False patterns
- `looks_like_email` - Email pattern detected
- `looks_like_url` - URL pattern detected
- ... and 8 more

#### Distribution Features (8)
- `distribution_type` - Normal/uniform/exp/bimodal
- `mode_spike_ratio` - Most common value frequency
- `has_outliers` - Binary outlier indicator
- `quantile_25`, `quantile_75` - Quartiles
- `coefficient_dispersion` - IQR / mean
- `range_normalized` - (max-min) / mean
- `has_zeros` - Contains zero values

#### Text/Categorical (5)
- `avg_string_length` - Mean string length
- `char_diversity` - Unique chars / total chars
- `numeric_string_ratio` - Strings that are numbers
- `has_mixed_case` - Contains upper and lower
- `word_count_avg` - Average words per value

#### Temporal Features (4)
- `autocorrelation_lag1` - Time series correlation
- `is_monotonic_score` - Increasing/decreasing trend
- `seasonality_score` - Periodic pattern strength
- `trend_strength` - Linear trend correlation

### Feature Importance

Top features typically include:
1. `unique_ratio` - Cardinality indicator
2. `detected_dtype` - Data type
3. `name_contains_id` - ID column detection
4. `skewness` - Distribution shape
5. `looks_like_categorical` - Category indicator

---

## Deploying the Trained Model

### Step 1: Download Model

From Colab, the model downloads automatically. Or manually:
```python
from google.colab import files
files.download('neural_oracle_meta_v3_TIMESTAMP.pkl')
```

### Step 2: Copy to Models Directory

```bash
cp neural_oracle_meta_v3_TIMESTAMP.pkl /path/to/AURORA-V2/models/
```

### Step 3: Update NeuralOracle (Optional)

The NeuralOracle auto-detects models. To force a specific model:

```python
# In src/neural/oracle.py
DEFAULT_MODEL = "models/neural_oracle_meta_v3_TIMESTAMP.pkl"
```

### Step 4: Verify Deployment

```python
from src.neural.oracle import NeuralOracle

oracle = NeuralOracle()
print(f"Model loaded: {oracle.model is not None}")

# Test prediction
from src.features.minimal_extractor import get_feature_extractor
import pandas as pd

extractor = get_feature_extractor()
test_col = pd.Series([1.5, 2.3, 3.1, 4.2, 5.0])
features = extractor.extract(test_col, 'test_column')

prediction = oracle.predict(features)
print(f"Predicted action: {prediction.action}")
print(f"Confidence: {prediction.confidence:.2f}")
```

### Step 5: Run Validation

```bash
python scripts/model_integration_utils.py
```

Expected output:
```
Bestsellers test:
  Model loaded: True
  Accuracy: 91.4%
  Meets threshold: True
```

---

## Troubleshooting

### Common Issues

#### "Module not found: openml"
```bash
pip install openml
```

#### "Memory error during training"
Reduce dataset count or sample size:
```python
config['n_datasets'] = 20
config['max_samples'] = 2000
```

#### "Model accuracy below threshold"
1. Increase training datasets
2. Check for data quality issues
3. Verify feature extraction works

#### "Model doesn't load in NeuralOracle"
Check model format:
```python
import pickle
with open('model.pkl', 'rb') as f:
    loaded = pickle.load(f)
print(type(loaded))  # Should be VotingClassifier or dict
```

#### "Classes not recognized"
The model uses action strings that must map to `PreprocessingAction` enum:
```python
# Valid classes
['keep_as_is', 'drop_column', 'standard_scale', 'label_encode', ...]
```

### Debug Logging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Benchmarking

### Metrics to Track

| Metric | Target | Current |
|--------|--------|---------|
| Training Accuracy | 93%+ | - |
| Test Accuracy | 90%+ | - |
| Bestsellers Accuracy | 90%+ | 71% (baseline) |
| Inference Time | <5ms | ~2ms |
| Model Size | <10MB | ~5MB |
| Training Time | <45min | ~35min |

### Running Benchmarks

```bash
# Full comprehensive benchmark
python benchmark_comprehensive.py

# Quick validation
python scripts/validate_against_ground_truth.py

# Model comparison
python -c "
from scripts.model_integration_utils import compare_with_current
results = compare_with_current('models/new_model.pkl')
print(results)
"
```

### Expected Improvements

After meta-learning training:
- Bestsellers.csv: 71% → 90%+
- Ambiguous cases: Better confidence calibration
- Edge cases: Improved handling of rare patterns

---

## Advanced Topics

### Custom Datasets

Add your own training datasets:
```python
# In Colab notebook
custom_df = pd.read_csv('your_data.csv')
samples = learner.process_dataset(
    custom_df, 
    target_column='your_target',
    dataset_name='custom'
)
```

### Hyperparameter Tuning

Adjust XGBoost parameters:
```python
xgb_params = {
    'n_estimators': 300,     # More trees
    'max_depth': 8,          # Deeper trees
    'learning_rate': 0.03,   # Slower learning
    'subsample': 0.8,        # Row sampling
    'colsample_bytree': 0.8, # Feature sampling
}
```

### Adding New Actions

1. Add action to `CurriculumConfig`:
```python
numeric_actions = [..., 'new_action']
```

2. Implement in `PreprocessingExecutor`:
```python
elif action == 'new_action':
    return transform_new(column), transformer
```

3. Add to `PreprocessingAction` enum (src/core/actions.py)

---

## References

- [AURORA Architecture](ARCHITECTURE.md)
- [Neural Oracle Documentation](NEURAL_ORACLE.md)
- [Transformation Decisions](TRANSFORMATION_DECISIONS.md)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [OpenML Datasets](https://www.openml.org/)
