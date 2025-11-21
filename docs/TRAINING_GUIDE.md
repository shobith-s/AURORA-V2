# AURORA V2 - Hybrid Training Guide

## Quick Start

### Step 1: Collect Datasets

```bash
python scripts/collect_open_datasets.py
```

This will download:
- 6 sklearn datasets (diabetes, housing, iris, wine, breast_cancer, digits)
- 2 UCI datasets (adult, heart_disease)
- 3 synthetic edge cases

**Expected output**: ~11 CSV files in `data/open_datasets/`

---

### Step 2: Verify Setup

```bash
python scripts/verify_training_setup.py
```

This checks:
- ✓ All dependencies installed
- ✓ Datasets collected successfully
- ✓ Database accessible
- ✓ Output directory ready

**If all checks pass**, proceed to training!

---

### Step 3: Train the Model

**Option A: Use the Windows Wrapper (Easiest)**
```bash
scripts\train_hybrid_windows.bat
```

**Option B: Direct Command (Use Forward Slashes!)**
```bash
python scripts/train_hybrid.py --datasets-dir data/open_datasets
```

**⚠️ IMPORTANT for Windows Users:**
- **DO USE** forward slashes: `data/open_datasets` ✓
- **DON'T USE** backslashes: `data\open_datasets` ✗ (causes path concatenation issue)
- Forward slashes work correctly on Windows!

---

## Understanding the Training Process

### Data Sources (Weighted Combination)

The hybrid training combines three data sources:

1. **User Corrections** (weight: **3.0x**) - Highest priority
   - Loaded from `aurora.db`
   - Domain-specific knowledge from actual usage
   - Requires at least 20 corrections (configurable)

2. **Open Datasets** (weight: **2.0x**) - Medium priority
   - Real-world CSV files from sklearn, UCI, etc.
   - Provides broad generalization
   - Auto-labeled using heuristics

3. **Synthetic Data** (weight: **1.0x**) - Baseline
   - Generated edge cases
   - Handles rare scenarios
   - Default: 1,000 samples

### Training Parameters

```bash
python scripts/train_hybrid.py \
  --datasets-dir data/open_datasets \     # Path to collected datasets
  --synthetic 1000 \                       # Number of synthetic samples
  --min-corrections 20 \                   # Min user corrections required
  --output models/neural_oracle_v1.pkl \   # Output model path
  --metadata-file models/neural_oracle_v1.json  # Training metadata
```

### Expected Training Output

```
======================================================================
AURORA Hybrid Neural Oracle Training
======================================================================

1. Loading user corrections from database...
   ✓ Loaded 45 corrections
   Top actions:
     • STANDARD_SCALE: 15
     • IMPUTE_MEDIAN: 12
     • LOG_TRANSFORM: 8

2. Loading open datasets from data/open_datasets...
   Found 11 dataset files
   ✓ Extracted 287 columns from datasets

3. Generating 1000 synthetic samples...
   ✓ Generated 1000 synthetic samples

4. Training hybrid model...
----------------------------------------------------------------------
   Total samples: 1332
   Breakdown:
     • Corrections:    45 (weight: 3.0x)
     • Open data:     287 (weight: 2.0x)
     • Synthetic:    1000 (weight: 1.0x)

   Training complete!
   • Training accuracy:   92.45%
   • Validation accuracy: 89.23%
   • Number of trees:     50
   • Number of features:  10

✓ Model saved to: models/neural_oracle_v1.pkl
  Model size: 147.3 KB

✓ Metadata saved to: models/neural_oracle_v1.json

======================================================================
TRAINING COMPLETE
======================================================================

Model ready to use!
To test: python scripts/evaluate_system.py --model models/neural_oracle_v1.pkl
```

---

## Troubleshooting

### Issue: "Directory not found: ...dataopen_datasets"

**Problem**: Path concatenation error on Windows when using backslashes

**Solution**: Use forward slashes instead:
```bash
# ✗ Wrong (backslash gets consumed)
python scripts/train_hybrid.py --datasets-dir data\open_datasets

# ✓ Correct (forward slash works on Windows)
python scripts/train_hybrid.py --datasets-dir data/open_datasets
```

**Alternative**: Use the batch wrapper:
```bash
scripts\train_hybrid_windows.bat
```

---

### Issue: "Only 0 corrections (min: 20)"

**Problem**: Not enough user corrections in database

**Solution**: This is normal for new installations. Training will proceed with synthetic + open data only.

To accumulate corrections:
1. Start the API server: `uvicorn src.api.server:app --reload`
2. Use the system and submit corrections via the UI
3. Re-train once you have 20+ corrections

---

### Issue: "No CSV files found in data/open_datasets"

**Problem**: Datasets not collected yet

**Solution**: Run the collection script first:
```bash
python scripts/collect_open_datasets.py
```

---

### Issue: "ModuleNotFoundError: No module named 'xgboost'"

**Problem**: Missing dependencies

**Solution**: Install requirements:
```bash
pip install -r requirements.txt
```

Or install specific package:
```bash
pip install xgboost>=2.0.0
```

---

## Training Metadata

After training, check `models/neural_oracle_v1.json` for details:

```json
{
  "training_date": "2025-11-21T10:30:00",
  "num_samples": 1332,
  "num_corrections": 45,
  "num_open_dataset_columns": 287,
  "num_synthetic": 1000,
  "train_accuracy": 0.9245,
  "val_accuracy": 0.8923,
  "num_trees": 50,
  "model_size_kb": 147.3
}
```

This metadata is useful for:
- Tracking training history
- Comparing different training runs
- Debugging model performance

---

## Advanced: Custom Training

### Train with More Synthetic Data

```bash
python scripts/train_hybrid.py \
  --datasets-dir data/open_datasets \
  --synthetic 5000
```

### Train with Lower Correction Threshold

```bash
python scripts/train_hybrid.py \
  --datasets-dir data/open_datasets \
  --min-corrections 10
```

### Train to Custom Location

```bash
python scripts/train_hybrid.py \
  --datasets-dir data/open_datasets \
  --output models/custom_model.pkl \
  --metadata-file models/custom_metadata.json
```

---

## Testing the Trained Model

### Option 1: Run Complete System Tests

```bash
pytest tests/test_complete_system.py -v -s
```

Tests all 5 phases:
1. SHAP explainability
2. Training integration
3. Confidence warnings
4. Layer metrics
5. Complete system

### Option 2: Run Evaluation Script

```bash
python scripts/evaluate_system.py --model models/neural_oracle_v1.pkl
```

### Option 3: Use in Production

The model is automatically loaded by the preprocessor when available:

```python
from src.core.preprocessor import IntelligentPreprocessor

# Automatically loads models/neural_oracle_v1.pkl if exists
preprocessor = IntelligentPreprocessor(use_neural_oracle=True)
```

---

## Training Best Practices

### 1. **Collect User Corrections Regularly**
   - User corrections have 3x weight
   - Most valuable training data
   - Reflects actual use cases

### 2. **Retrain Periodically**
   - After accumulating 100+ new corrections
   - When adding new preprocessing actions
   - When performance degrades

### 3. **Monitor Training Metrics**
   - Target validation accuracy: >85%
   - Model size should stay <5MB
   - Training should complete in <2 minutes

### 4. **Version Your Models**
   - Keep training metadata
   - Tag models with dates
   - A/B test new models before deployment

---

## Integration with Main System

Once trained, the neural oracle is used when:
1. Symbolic engine confidence < 0.9
2. No learned patterns match
3. Meta-learning is inconclusive

The system architecture:
```
User Input
    ↓
[1. Learned Patterns]  ← User corrections
    ↓ (if no match)
[2. Symbolic Rules]     ← Hand-crafted rules
    ↓ (if confidence < 0.9)
[3. Neural Oracle]      ← This trained model
    ↓ (if confidence < 0.7)
[4. Meta-Learning]      ← Heuristics
    ↓
Final Decision
```

---

## Next Steps

After successful training:

1. **Test the system**: `pytest tests/test_complete_system.py -v -s`
2. **Start the API**: `uvicorn src.api.server:app --reload`
3. **Use the UI**: Navigate to `http://localhost:8000`
4. **Monitor metrics**: Check `/api/metrics/layers` endpoint
5. **Collect corrections**: System improves over time!

---

## Need Help?

- Check `docs/IMPLEMENTATION_GUIDE.md` for architecture details
- Review `docs/HANDOVER.md` for system overview
- Run `python scripts/verify_training_setup.py` to diagnose issues
- Check training logs in console output
