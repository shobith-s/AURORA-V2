# Neural Oracle Training Guide

## 🧠 What is the Neural Oracle?

The Neural Oracle is AURORA's **meta-learner** - a lightweight XGBoost model that handles ambiguous preprocessing decisions when symbolic rules have low confidence (<0.9).

### Current Role in Architecture

```
Column Data → Preprocessing Pipeline
                    ↓
    1️⃣ Learned Patterns (from corrections) → High conf? ✓ Done
                    ↓ Low conf (<0.9)
    2️⃣ Symbolic Engine (165+ rules)       → High conf? ✓ Done
                    ↓ Low conf (<0.9)
    3️⃣ Neural Oracle (XGBoost)             → ✓ Final decision
```

**Key Stats:**
- Model Size: ~50KB (compressed)
- Inference Time: <5ms
- Training Time: ~2 minutes
- Accuracy Target: >85% on ambiguous cases

---

## 🎯 Two Training Approaches

### **Option 1: Train from Synthetic Data (Baseline)**

Use this when you have **no or few user corrections yet**.

```bash
# Generate 5000 synthetic edge cases and train
python scripts/train_neural_oracle.py
```

**Pros:**
- ✅ Works immediately without any user data
- ✅ Covers wide variety of edge cases
- ✅ Good baseline performance (~80% accuracy)

**Cons:**
- ❌ Not tailored to your specific data patterns
- ❌ May not handle domain-specific cases well

---

### **Option 2: Train from User Corrections (Recommended)**

Use this when you have **50+ user corrections** in the database.

```bash
# Train from real corrections + synthetic data
python scripts/train_from_corrections.py
```

**Pros:**
- ✅ Learns from REAL user feedback
- ✅ Improves with every correction
- ✅ Tailored to your specific use cases
- ✅ Genuinely gets smarter over time

**Cons:**
- ⚠️  Requires 50+ corrections for good performance
- ⚠️  Quality depends on correction quality

---

## 📊 Training Options

### Basic Training (Synthetic Only)

```bash
python scripts/train_neural_oracle.py
```

**Output:**
- Model: `models/neural_oracle_v1.pkl`
- Performance: ~80% validation accuracy
- Training samples: 5000 synthetic edge cases

### Hybrid Training (Corrections + Synthetic)

```bash
# Use both corrections and synthetic data (recommended)
python scripts/train_from_corrections.py

# Corrections only (if you have 200+)
python scripts/train_from_corrections.py --no-synthetic

# Synthetic only (fallback if no corrections)
python scripts/train_from_corrections.py --no-corrections

# Adjust parameters
python scripts/train_from_corrections.py \
    --min-corrections 100 \
    --synthetic-samples 2000
```

**Hybrid Training Benefits:**
- Real corrections are weighted 2x more heavily
- Synthetic data fills gaps in correction coverage
- Best balance of specificity and generalization

---

## 🔄 Retraining Workflow

### Initial Setup (Day 1)

```bash
# 1. Train baseline model
python scripts/train_neural_oracle.py

# 2. Start the server
uvicorn src.api.server:app --reload

# 3. Verify model loaded
curl http://localhost:8000/metrics/neural_oracle
```

### Weekly Retraining (Recommended)

```bash
# Check how many corrections you have
curl http://localhost:8000/metrics/learning | jq '.corrections.total'

# If >= 50 corrections, retrain!
python scripts/train_from_corrections.py

# Restart server to load new model
# (or implement hot-reload in production)
```

### Automated Retraining (Production)

Set up a cron job or scheduled task:

```bash
# crontab -e
# Retrain every Sunday at 2 AM
0 2 * * 0 cd /path/to/AURORA-V2 && python scripts/train_from_corrections.py >> logs/training.log 2>&1
```

---

## 📈 Monitoring Training Progress

### Check Correction Count

```bash
# Via API
curl http://localhost:8000/metrics/learning

# Via Database
python -c "
from src.database.connection import SessionLocal
from src.database.models import CorrectionRecord
db = SessionLocal()
count = db.query(CorrectionRecord).count()
print(f'Total corrections: {count}')
db.close()
"
```

### Evaluate Model Performance

```python
from pathlib import Path
from src.neural.oracle import NeuralOracle

# Load model
model_path = Path("models/neural_oracle_v1.pkl")
oracle = NeuralOracle(model_path)

# Check feature importance
top_features = oracle.get_top_features(top_k=10)
for name, importance in top_features:
    print(f"{name:25s}: {importance:.2f}")

# Check model size
size_kb = oracle.get_model_size() / 1024
print(f"\nModel size: {size_kb:.1f} KB")
```

### View Training History

```bash
# Training metadata is saved alongside model
cat models/neural_oracle_v1.json
```

**Example Output:**
```json
{
  "training_date": "2024-11-20T12:30:45",
  "num_samples": 1250,
  "num_real_corrections": 250,
  "num_synthetic": 1000,
  "real_data_weight": 2.0,
  "train_accuracy": 0.89,
  "val_accuracy": 0.85,
  "model_size_kb": 48.3,
  "avg_inference_ms": 3.2
}
```

---

## 🎓 Understanding the Training Data

### What Features Does It Use?

The neural oracle uses 10 minimal features:

1. **null_percentage** - % of null values
2. **unique_ratio** - Unique values / total values
3. **skewness** - Distribution skew
4. **outlier_percentage** - % of outliers (IQR method)
5. **entropy** - Information entropy
6. **pattern_complexity** - String pattern complexity
7. **multimodality_score** - Multiple peaks in distribution
8. **cardinality_bucket** - Binned unique count
9. **detected_dtype** - Inferred data type
10. **column_name_signal** - Signal from column name

### What Actions Can It Predict?

All preprocessing actions from `PreprocessingAction` enum:

- `KEEP` - No transformation needed
- `DROP` - Remove column
- `STANDARD_SCALE` - Mean=0, std=1
- `MINMAX_SCALE` - Scale to [0, 1]
- `LOG_TRANSFORM` - Log(x + 1) for skewed data
- `QUANTILE_TRANSFORM` - Map to uniform distribution
- `ROBUST_SCALE` - Scale using median/IQR
- `CLIP_OUTLIERS` - Cap extreme values
- `ONE_HOT_ENCODE` - Create binary columns
- `LABEL_ENCODE` - Map to integers
- `FILL_MEAN` - Impute with mean
- `FILL_MEDIAN` - Impute with median
- `FILL_MODE` - Impute with mode
- And more...

---

## 🔧 Troubleshooting

### "Only X corrections found. Minimum 50 recommended"

**Solution:** Use hybrid training or collect more feedback.

```bash
# Still train with synthetic data as supplement
python scripts/train_from_corrections.py
```

### "ModuleNotFoundError: No module named 'xgboost'"

**Solution:** Install XGBoost.

```bash
pip install xgboost
```

### "Training accuracy < 70%"

**Possible causes:**
1. **Noisy corrections** - Users making inconsistent corrections
2. **Insufficient data** - Need more samples
3. **Feature mismatch** - Corrections don't match feature extraction

**Solutions:**
- Add more synthetic data to stabilize training
- Filter out low-confidence corrections
- Validate correction quality

### Model not loading on server startup

**Check:**
```bash
# Does model file exist?
ls -lh models/neural_oracle_v1.pkl

# Is it valid?
python -c "
from src.neural.oracle import NeuralOracle
from pathlib import Path
oracle = NeuralOracle(Path('models/neural_oracle_v1.pkl'))
print('Model loaded successfully!')
"
```

---

## 🚀 Advanced: Improving Neural Oracle Performance

### 1. Increase Training Data

```bash
# Generate more synthetic samples
python scripts/train_from_corrections.py --synthetic-samples 5000
```

### 2. Adjust Correction Weight

Edit `scripts/train_from_corrections.py`:

```python
# Line ~156: Increase weight for real corrections
real_weight = 3.0  # Default is 2.0
```

### 3. Tune XGBoost Parameters

Edit `src/neural/oracle.py` training params:

```python
params = {
    'max_depth': 8,        # Default: 6 (deeper trees)
    'eta': 0.05,           # Default: 0.1 (slower learning)
    'num_boost_round': 100 # Default: 50 (more trees)
}
```

**Trade-off:** Better accuracy but larger model and slower inference.

### 4. Implement A/B Testing

Compare old vs. new model:

```python
# Save old model
mv models/neural_oracle_v1.pkl models/neural_oracle_v1_old.pkl

# Train new model
python scripts/train_from_corrections.py

# Implement A/B testing in preprocessor
# Route 50% traffic to each model, compare accuracy
```

---

## 📊 Performance Benchmarks

### Expected Performance

| Metric | Target | Typical | Excellent |
|--------|--------|---------|-----------|
| Validation Accuracy | >75% | 80-85% | >90% |
| Model Size | <100KB | 40-60KB | <40KB |
| Inference Time | <5ms | 2-4ms | <2ms |
| Training Time | <5min | 1-3min | <1min |

### When to Retrain

✅ **Good reasons to retrain:**
- Collected 50+ new corrections
- Validation accuracy dropped >5%
- New preprocessing actions added
- Weekly/monthly maintenance

❌ **Bad reasons to retrain:**
- Only 5-10 new corrections
- Model is performing well
- Daily retraining (overkill)

---

## 🎯 Next Steps

1. **Train baseline model** → `python scripts/train_neural_oracle.py`
2. **Start server** → `uvicorn src.api.server:app --reload`
3. **Collect feedback** → Users make corrections via `/correct` endpoint
4. **Monitor progress** → Check `/metrics/learning` regularly
5. **Retrain when ready** → `python scripts/train_from_corrections.py`
6. **Repeat** → Continuous improvement!

---

## 📚 Related Documentation

- **Architecture Overview**: `docs/ARCHITECTURE_V3_PROPOSAL.md`
- **Implementation Status**: `IMPLEMENTATION_STATUS.md`
- **Adaptive Learning**: `src/learning/adaptive_engine.py`
- **Database Models**: `src/database/models.py`

---

**💡 Pro Tip:** The real power comes from the combination of all three layers:
1. **Learned Patterns** catch frequent domain-specific patterns (fastest)
2. **Symbolic Rules** handle deterministic cases (fast)
3. **Neural Oracle** handles truly ambiguous edge cases (smart)

This creates a system that's **both fast AND smart**! 🚀
